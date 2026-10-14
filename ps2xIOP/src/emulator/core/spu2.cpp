#include "spu2.h"
#include "spu2_gaussian.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace ps2x::iop::detail
{
    namespace
    {
        constexpr uint32_t kAddressMask = Spu2::kRamHalfwords - 1u;
        constexpr uint32_t kVolumeBlock = 0x760u; // core 0's MVOL / EVOL / AVOL / BVOL; core 1's at +0x28
        constexpr uint32_t kVolumeStride = 0x28u;
        constexpr size_t kInputStartFrames = 512u; // AutoDMA frames held before the input plays (one transfer)

        bool voiceLogEnabled()
        {
            static const bool enabled = std::getenv("VENDETTA_SPULOG") != nullptr;
            return enabled;
        }

        // VENDETTA_SPU_INTERP=linear: straight lines between samples (as before) instead of the hardware's Gaussian
        bool gaussianInterpolation()
        {
            static const bool gaussian = []
            {
                const char *value = std::getenv("VENDETTA_SPU_INTERP");
                return !(value && std::string(value) == "linear");
            }();
            return gaussian;
        }

        int32_t clamp16(int32_t value)
        {
            return std::clamp(value, -32768, 32767);
        }

        // The envelope step the voices' ADSR and the volume sweeps share: a rate is a shift (0-31, fast..slow) and
        // a step (+7..+4 rising, -8..-5 falling); the level moves by step << (11 - shift) every 1 << (shift - 11)
        // samples; exponential rising slows to a quarter above 0x6000, exponential falling scales by the level.
        void envelopeStep(int32_t &level, uint32_t &counter, bool exponential, bool decrease, uint32_t shift, uint32_t stepCode)
        {
            int32_t step = decrease ? -8 + static_cast<int32_t>(stepCode) : 7 - static_cast<int32_t>(stepCode);
            uint32_t cycles = 1u << (shift > 11u ? shift - 11u : 0u);
            if (shift < 11u)
                step *= 1 << (11u - shift);
            if (exponential && !decrease && level > 0x6000)
                cycles *= 4u;
            if (exponential && decrease)
                step = (step * level) >> 15;
            if (++counter < cycles)
                return;
            counter = 0u;
            level = std::clamp(level + step, 0, 0x7FFF);
        }

        uint32_t composeAddress(uint16_t high, uint16_t low)
        {
            return ((static_cast<uint32_t>(high) << 16u) | low) & kAddressMask;
        }
    }

    void Spu2::Volume::set(uint16_t value)
    {
        const bool wasSweep = (reg & 0x8000u) != 0u;
        reg = value;
        if ((value & 0x8000u) == 0u)
        {
            level = static_cast<int16_t>(static_cast<uint16_t>(value << 1u)); // fixed: 0x3FFF = full, bit 14 = inverted
            return;
        }
        if (!wasSweep)
            level = std::min(std::abs(level), 0x7FFF); // a sweep starts from the current level
        counter = 0u;
    }

    void Spu2::Volume::step()
    {
        if ((reg & 0x8000u) == 0u)
            return;
        envelopeStep(level, counter, (reg & 0x4000u) != 0u, (reg & 0x2000u) != 0u, (reg >> 2u) & 0x1Fu, reg & 3u);
    }

    int32_t Spu2::Volume::value() const
    {
        if ((reg & 0x8000u) == 0u)
            return level;
        return (reg & 0x1000u) != 0u ? -level : level;
    }

    Spu2::Spu2()
        : m_ram(kRamHalfwords)
    {
        reset();
    }

    void Spu2::reset()
    {
        std::fill(m_ram.begin(), m_ram.end(), uint16_t{0});
        m_regs.fill(0u);
        for (Core &core : m_cores)
            core = Core{};
        m_nextSampleCycle = kIopCyclesPerSample;
    }

    uint16_t Spu2::read16(uint32_t offset) const
    {
        offset &= kRegisterBytes - 2u;
        if (offset >= kVolumeBlock && offset < kVolumeBlock + 2u * kVolumeStride)
        {
            const Core &core = m_cores[(offset - kVolumeBlock) / kVolumeStride];
            const uint32_t field = (offset - kVolumeBlock) % kVolumeStride;
            if (field == 0x10u)
                return static_cast<uint16_t>(core.masterLeft.value());
            if (field == 0x12u)
                return static_cast<uint16_t>(core.masterRight.value());
            return reg(offset);
        }
        const Core &core = m_cores[(offset >> 10u) & 1u];
        const uint32_t local = offset & 0x3FFu;
        if (local < 0x180u)
        {
            const Voice &voice = core.voices[local >> 4u];
            switch (local & 0xFu)
            {
            case 0xAu: return static_cast<uint16_t>(voice.phase == Phase::Off ? 0 : voice.envelope); // ENVX
            case 0xCu: return static_cast<uint16_t>(voice.left.value());                             // VOLXL
            case 0xEu: return static_cast<uint16_t>(voice.right.value());                            // VOLXR
            default: return reg(offset);
            }
        }
        if (local >= 0x1C0u && local < 0x1C0u + kVoices * 12u)
        {
            const Voice &voice = core.voices[(local - 0x1C0u) / 12u];
            const uint32_t now = voice.blockLoaded ? voice.blockAddress : voice.nextAddress;
            switch ((local - 0x1C0u) % 12u)
            {
            case 8u: return static_cast<uint16_t>(now >> 16u); // NAX
            case 10u: return static_cast<uint16_t>(now);
            default: return reg(offset);
            }
        }
        if (local == 0x340u)
            return static_cast<uint16_t>(core.endFlags); // ENDX
        if (local == 0x342u)
            return static_cast<uint16_t>(core.endFlags >> 16u);
        return reg(offset);
    }

    void Spu2::write16(uint32_t offset, uint16_t value)
    {
        offset &= kRegisterBytes - 2u;
        m_regs[offset >> 1u] = value;
        if (offset >= kVolumeBlock && offset < kVolumeBlock + 2u * kVolumeStride)
        {
            Core &core = m_cores[(offset - kVolumeBlock) / kVolumeStride];
            const uint32_t field = (offset - kVolumeBlock) % kVolumeStride;
            if (field == 0u)
                core.masterLeft.set(value);
            else if (field == 2u)
                core.masterRight.set(value);
            return;
        }
        const int coreIndex = static_cast<int>((offset >> 10u) & 1u);
        Core &core = m_cores[coreIndex];
        const uint32_t base = static_cast<uint32_t>(coreIndex) * 0x400u;
        const uint32_t local = offset & 0x3FFu;
        if (local < 0x180u)
        {
            Voice &voice = core.voices[local >> 4u];
            switch (local & 0xFu)
            {
            case 0x0u: voice.left.set(value); break;
            case 0x2u: voice.right.set(value); break;
            case 0xAu: voice.envelope = value & 0x7FFFu; break;
            default: break;
            }
            return;
        }
        if (local >= 0x1C0u && local < 0x1C0u + kVoices * 12u)
        {
            const uint32_t index = (local - 0x1C0u) / 12u;
            const uint32_t field = (local - 0x1C0u) % 12u;
            Voice &voice = core.voices[index];
            const uint32_t fieldBase = base + 0x1C0u + index * 12u;
            if (field == 4u || field == 6u) // LSAX: the loop address the block flags must not move
            {
                voice.loopAddress = composeAddress(reg(fieldBase + 4u), reg(fieldBase + 6u));
                voice.loopWritten = true;
                if (voice.phase != Phase::Off)
                    voice.loopLocked = true;
            }
            else if (field == 8u || field == 10u) // NAX
            {
                voice.nextAddress = composeAddress(reg(fieldBase + 8u), reg(fieldBase + 10u));
                voice.blockLoaded = false;
                voice.position = 28u;
            }
            return;
        }
        switch (local)
        {
        case 0x1A0u: keyOn(coreIndex, value); break;
        case 0x1A2u: keyOn(coreIndex, static_cast<uint32_t>(value & 0xFFu) << 16u); break;
        case 0x1A4u: keyOff(coreIndex, value); break;
        case 0x1A6u: keyOff(coreIndex, static_cast<uint32_t>(value & 0xFFu) << 16u); break;
        case 0x1A8u:
        case 0x1AAu:
            core.transferAddress = composeAddress(reg(base + 0x1A8u), reg(base + 0x1AAu));
            if (voiceLogEnabled() && local == 0x1AAu)
            {
                static int tsaLines = 0; // HCTP: upload start addresses (bank-load diagnosis)
                if (tsaLines++ < 3000)
                    std::fprintf(stderr, "[sputsa] core %d TSA 0x%05X%c", coreIndex, core.transferAddress, 10);
            }
            break;
        case 0x2E0u: // ESA: a new effects area starts at its beginning
        case 0x2E2u:
            core.reverbPosition = 0u;
            break;
        case 0x1ACu: // the data port: a manual transfer into sound memory
            m_ram[core.transferAddress] = value;
            core.transferAddress = (core.transferAddress + 1u) & kAddressMask;
            break;
        case 0x340u: core.endFlags &= 0xFF0000u; break;
        case 0x342u: core.endFlags &= 0xFFFFu; break;
        default: break;
        }
    }

    void Spu2::keyOn(int coreIndex, uint32_t mask)
    {
        Core &core = m_cores[coreIndex];
        const uint32_t base = static_cast<uint32_t>(coreIndex) * 0x400u;
        for (int index = 0; index < kVoices; ++index)
        {
            if ((mask & (1u << index)) == 0u)
                continue;
            Voice &voice = core.voices[index];
            const uint32_t fieldBase = base + 0x1C0u + static_cast<uint32_t>(index) * 12u;
            voice.nextAddress = composeAddress(reg(fieldBase), reg(fieldBase + 2u)); // SSA
            voice.phase = Phase::Attack;
            voice.envelope = 0;
            voice.envelopeCounter = 0u;
            voice.blockLoaded = false;
            voice.position = 28u;
            voice.history1 = voice.history2 = 0;
            voice.recent.fill(0);
            voice.counter = 0u;
            voice.loopLocked = voice.loopWritten;
            voice.loopWritten = false;
            voice.playedSamples = 0u;
            voice.peak = 0;
            core.endFlags &= ~(1u << index);
            if (voiceLogEnabled())
                std::fprintf(stderr, "[spu] at sample %llu core %d voice %d key on: start 0x%05X loop 0x%05X%s pitch 0x%04X adsr 0x%04X/0x%04X vol 0x%04X/0x%04X nz %u/256\n",
                             static_cast<unsigned long long>(m_nextSampleCycle / kIopCyclesPerSample),
                             coreIndex, index, voice.nextAddress, voice.loopAddress, voice.loopLocked ? " (set)" : "",
                             reg(base + static_cast<uint32_t>(index) * 0x10u + 4u), reg(base + static_cast<uint32_t>(index) * 0x10u + 6u),
                             reg(base + static_cast<uint32_t>(index) * 0x10u + 8u), voice.left.reg, voice.right.reg,
                             [&] { unsigned nz = 0; for (uint32_t i = 0; i < 256u; ++i) nz += m_ram[(voice.nextAddress + i) & kAddressMask] != 0u; return nz; }());
        }
    }

    void Spu2::keyOff(int coreIndex, uint32_t mask)
    {
        Core &core = m_cores[coreIndex];
        if (voiceLogEnabled() && mask != 0u)
            std::fprintf(stderr, "[spu] core %d key off 0x%06X at sample %llu\n", coreIndex, mask,
                         static_cast<unsigned long long>(m_nextSampleCycle / kIopCyclesPerSample));
        for (int index = 0; index < kVoices; ++index)
        {
            Voice &voice = core.voices[index];
            if ((mask & (1u << index)) != 0u && voice.phase != Phase::Off)
            {
                voice.phase = Phase::Release;
                voice.envelopeCounter = 0u;
            }
        }
    }

    // One 16-byte PS-ADPCM block: a header halfword (shift | filter << 4, flags << 8: 1 = loop end, 2 = repeat,
    // 4 = loop start), then 28 4-bit samples, each shifted and predicted from the two before it.
    void Spu2::decodeBlock(Voice &voice)
    {
        static constexpr int32_t kFilter0[5] = {0, 60, 115, 98, 122};
        static constexpr int32_t kFilter1[5] = {0, 0, -52, -55, -60};
        const uint32_t address = voice.nextAddress & kAddressMask;
        const uint16_t header = m_ram[address];
        voice.blockAddress = address;
        voice.blockFlags = static_cast<uint8_t>(header >> 8u);
        if ((voice.blockFlags & 4u) != 0u && !voice.loopLocked)
            voice.loopAddress = address;
        uint32_t shift = header & 0xFu;
        if (shift > 12u)
            shift = 9u;
        const uint32_t filter = std::min<uint32_t>((header >> 4u) & 7u, 4u);
        for (uint32_t i = 0; i < 28u; ++i)
        {
            const uint16_t word = m_ram[(address + 1u + i / 4u) & kAddressMask];
            const uint32_t nibble = (word >> ((i & 3u) * 4u)) & 0xFu;
            int32_t sample = static_cast<int16_t>(static_cast<uint16_t>(nibble << 12u)) >> shift;
            sample += (voice.history1 * kFilter0[filter] + voice.history2 * kFilter1[filter] + 32) >> 6;
            sample = clamp16(sample);
            voice.history2 = voice.history1;
            voice.history1 = sample;
            voice.samples[i] = static_cast<int16_t>(sample);
        }
        voice.position = 0u;
        voice.blockLoaded = true;
        voice.nextAddress = (address + 8u) & kAddressMask;
    }

    // The voice's sample for this output tick: the pitch (0x1000 = 48 kHz) moves through the decoded samples; a
    // block's loop-end flag sets ENDX and jumps to the loop address (repeat) or ends the voice. The output is the
    // hardware's 4-point Gaussian interpolation over the last four samples (the pitch counter's fraction picks the
    // weights), which softens the high end as the PS2 does.
    int32_t Spu2::nextSample(Core &core, int index, Voice &voice, uint32_t pitch)
    {
        voice.counter += pitch;
        while (voice.counter >= 0x1000u)
        {
            voice.counter -= 0x1000u;
            if (voice.position >= 28u)
            {
                if (voice.blockLoaded && (voice.blockFlags & 1u) != 0u)
                {
                    core.endFlags |= 1u << index;
                    if ((voice.blockFlags & 2u) == 0u)
                    {
                        voice.phase = Phase::Off;
                        voice.envelope = 0;
                        voice.blockLoaded = false;
                        voice.recent.fill(0);
                        return 0;
                    }
                    voice.nextAddress = voice.loopAddress;
                }
                decodeBlock(voice);
            }
            voice.recent = {voice.recent[1], voice.recent[2], voice.recent[3], voice.samples[voice.position++]};
        }
        if (!gaussianInterpolation())
            return voice.recent[2] + (((voice.recent[3] - voice.recent[2]) * static_cast<int32_t>(voice.counter)) >> 12);
        const std::array<int16_t, 4> &weights = kSpu2Gaussian[(voice.counter >> 4u) & 0xFFu];
        int32_t out = 0;
        for (int i = 0; i < 4; ++i)
            out += (weights[i] * voice.recent[i]) >> 15;
        return out;
    }

    // ADSR: attack to the top, decay to the sustain level, sustain (rising or falling) until key off, release to 0
    void Spu2::stepEnvelope(int coreIndex, int index, Voice &voice)
    {
        const uint32_t voiceBase = static_cast<uint32_t>(coreIndex) * 0x400u + static_cast<uint32_t>(index) * 0x10u;
        const uint16_t adsr1 = reg(voiceBase + 6u);
        const uint16_t adsr2 = reg(voiceBase + 8u);
        switch (voice.phase)
        {
        case Phase::Attack:
            envelopeStep(voice.envelope, voice.envelopeCounter, (adsr1 & 0x8000u) != 0u, false, (adsr1 >> 10u) & 0x1Fu, (adsr1 >> 8u) & 3u);
            if (voice.envelope >= 0x7FFF)
            {
                voice.envelope = 0x7FFF;
                voice.phase = Phase::Decay;
                voice.envelopeCounter = 0u;
            }
            break;
        case Phase::Decay:
        {
            envelopeStep(voice.envelope, voice.envelopeCounter, true, true, (adsr1 >> 4u) & 0xFu, 0u);
            const int32_t sustain = std::min<int32_t>(0x7FFF, static_cast<int32_t>((adsr1 & 0xFu) + 1u) << 11u);
            if (voice.envelope <= sustain)
            {
                voice.phase = Phase::Sustain;
                voice.envelopeCounter = 0u;
            }
            break;
        }
        case Phase::Sustain:
            envelopeStep(voice.envelope, voice.envelopeCounter, (adsr2 & 0x8000u) != 0u, (adsr2 & 0x4000u) != 0u, (adsr2 >> 8u) & 0x1Fu, (adsr2 >> 6u) & 3u);
            break;
        case Phase::Release:
            envelopeStep(voice.envelope, voice.envelopeCounter, (adsr2 & 0x20u) != 0u, true, adsr2 & 0x1Fu, 0u);
            if (voice.envelope <= 0)
            {
                voice.envelope = 0;
                voice.phase = Phase::Off;
            }
            break;
        case Phase::Off:
            break;
        }
    }

    void Spu2::mixCore(int coreIndex, int32_t extLeft, int32_t extRight)
    {
        Core &core = m_cores[coreIndex];
        const uint32_t base = static_cast<uint32_t>(coreIndex) * 0x400u;
        const auto pair = [this, base](uint32_t low)
        { return static_cast<uint32_t>(reg(base + low)) | (static_cast<uint32_t>(reg(base + low + 2u) & 0xFFu) << 16u); };
        const uint32_t pitchModulation = pair(0x180u);
        const uint32_t noise = pair(0x184u);
        const uint32_t dryLeft = pair(0x188u);
        const uint32_t wetVoicesLeft = pair(0x18Cu);
        const uint32_t dryRight = pair(0x190u);
        const uint32_t wetVoicesRight = pair(0x194u);

        // the noise generator (for voices with their NON bit): clock from ATTR bits 8-13
        {
            const uint16_t attr = reg(base + 0x19Au);
            const int32_t noiseStep = static_cast<int32_t>((attr >> 8u) & 3u) + 4;
            const uint32_t noiseShift = (attr >> 10u) & 0xFu;
            const int32_t parity = ((core.noiseLevel >> 15) ^ (core.noiseLevel >> 12) ^ (core.noiseLevel >> 11) ^
                                    (core.noiseLevel >> 10) ^ 1) & 1;
            core.noiseTimer -= noiseStep;
            if (core.noiseTimer < 0)
            {
                core.noiseLevel = (core.noiseLevel * 2 + parity) & 0xFFFF;
                core.noiseTimer += static_cast<int32_t>(0x20000u >> noiseShift);
                if (core.noiseTimer < 0)
                    core.noiseTimer += static_cast<int32_t>(0x20000u >> noiseShift);
            }
        }

        int32_t left = 0;
        int32_t right = 0;
        int32_t wetLeft = 0;
        int32_t wetRight = 0;
        int32_t previousOutput = 0;
        for (int index = 0; index < kVoices; ++index)
        {
            Voice &voice = core.voices[index];
            if (voice.phase == Phase::Off)
            {
                voice.output = 0;
                previousOutput = 0;
                continue;
            }
            uint32_t pitch = reg(base + static_cast<uint32_t>(index) * 0x10u + 4u);
            if (index > 0 && (pitchModulation & (1u << index)) != 0u)
                pitch = static_cast<uint32_t>(std::max(0, (static_cast<int32_t>(pitch) * (0x8000 + previousOutput)) >> 15));
            pitch = std::min<uint32_t>(pitch, 0x3FFFu);
            int32_t sample = nextSample(core, index, voice, pitch);
            if (voice.phase == Phase::Off)
            {
                if (voiceLogEnabled())
                    std::fprintf(stderr, "[spu] core %d voice %d ended (sample end) after %.1f ms, peak %d\n", coreIndex, index,
                                 voice.playedSamples / 48.0, voice.peak);
                voice.output = 0;
                previousOutput = 0;
                continue;
            }
            if ((noise & (1u << index)) != 0u)
                sample = static_cast<int16_t>(static_cast<uint16_t>(core.noiseLevel));
            stepEnvelope(coreIndex, index, voice);
            if (voice.phase == Phase::Off && voiceLogEnabled())
                std::fprintf(stderr, "[spu] core %d voice %d ended (released) after %.1f ms, peak %d\n", coreIndex, index,
                             voice.playedSamples / 48.0, voice.peak);
            sample = (sample * voice.envelope) >> 15;
            voice.output = sample;
            ++voice.playedSamples;
            voice.peak = std::max(voice.peak, std::abs(sample));
            previousOutput = sample;
            voice.left.step();
            voice.right.step();
            const int32_t voiceLeft = (sample * voice.left.value()) >> 15;
            const int32_t voiceRight = (sample * voice.right.value()) >> 15;
            if ((dryLeft & (1u << index)) != 0u)
                left += voiceLeft;
            if ((dryRight & (1u << index)) != 0u)
                right += voiceRight;
            if ((wetVoicesLeft & (1u << index)) != 0u)
                wetLeft += voiceLeft;
            if ((wetVoicesRight & (1u << index)) != 0u)
                wetRight += voiceRight;
        }

        // the sound-data input (AutoDMA), held back one transfer so a refill that comes a little late does not gap
        int32_t inputLeft = 0;
        int32_t inputRight = 0;
        const size_t pending = core.input.size() - core.inputRead;
        if (!core.inputPlaying && pending >= kInputStartFrames * 2u)
            core.inputPlaying = true;
        if (core.inputPlaying)
        {
            if (pending >= 2u)
            {
                inputLeft = core.input[core.inputRead];
                inputRight = core.input[core.inputRead + 1u];
                core.inputRead += 2u;
            }
            else
            {
                core.inputPlaying = false;
            }
        }
        if (core.inputRead >= 1u << 16u)
        {
            core.input.erase(core.input.begin(), core.input.begin() + static_cast<std::ptrdiff_t>(core.inputRead));
            core.inputRead = 0u;
        }

        // MMIX: the dry gates, bits 11 / 10 voices L / R, 7 / 6 the sound-data input, 3 / 2 the external input (core 0's
        // out); the wet gates (into the reverb), bits 9 / 8, 5 / 4, 1 / 0
        const uint32_t volumes = kVolumeBlock + static_cast<uint32_t>(coreIndex) * kVolumeStride;
        const uint16_t mmix = reg(base + 0x198u);
        const int32_t inLeft = (inputLeft * static_cast<int16_t>(reg(volumes + 0xCu))) >> 15;  // BVOLL
        const int32_t inRight = (inputRight * static_cast<int16_t>(reg(volumes + 0xEu))) >> 15; // BVOLR
        const int32_t exLeft = (extLeft * static_cast<int16_t>(reg(volumes + 0x8u))) >> 15;     // AVOLL
        const int32_t exRight = (extRight * static_cast<int16_t>(reg(volumes + 0xAu))) >> 15;   // AVOLR
        int32_t outLeft = 0;
        int32_t outRight = 0;
        if ((mmix & 0x800u) != 0u)
            outLeft += left;
        if ((mmix & 0x400u) != 0u)
            outRight += right;
        if ((mmix & 0x80u) != 0u)
            outLeft += inLeft;
        if ((mmix & 0x40u) != 0u)
            outRight += inRight;
        if ((mmix & 0x8u) != 0u)
            outLeft += exLeft;
        if ((mmix & 0x4u) != 0u)
            outRight += exRight;
        {
            const int32_t intoLeft = ((mmix & 0x200u) != 0u ? wetLeft : 0) + ((mmix & 0x20u) != 0u ? inLeft : 0) +
                                     ((mmix & 0x2u) != 0u ? exLeft : 0);
            const int32_t intoRight = ((mmix & 0x100u) != 0u ? wetRight : 0) + ((mmix & 0x10u) != 0u ? inRight : 0) +
                                      ((mmix & 0x1u) != 0u ? exRight : 0);
            int32_t effectLeft = 0;
            int32_t effectRight = 0;
            static const bool reverbOn = []
            {
                const char *value = std::getenv("VENDETTA_REVERB"); // =0: no reverb (comparing)
                return !(value && value[0] == '0');
            }();
            if (reverbOn)
                reverb(coreIndex, clamp16(intoLeft), clamp16(intoRight), effectLeft, effectRight);
            outLeft += (effectLeft * static_cast<int16_t>(reg(volumes + 0x4u))) >> 15;   // EVOLL
            outRight += (effectRight * static_cast<int16_t>(reg(volumes + 0x6u))) >> 15; // EVOLR
        }
        core.masterLeft.step();
        core.masterRight.step();
        core.outLeft = clamp16((clamp16(outLeft) * core.masterLeft.value()) >> 15);
        core.outRight = clamp16((clamp16(outRight) * core.masterRight.value()) >> 15);
    }

    namespace
    {
        // The half-band filter around the reverb (39 taps; 2x down to the effect's rate and back up), as PCSX2's SPU2.
        constexpr int kReverbTaps = 39;
        constexpr std::array<int32_t, kReverbTaps> kReverbFilter = {
            -1, 0, 2, 0, -10, 0, 35, 0, -103, 0, 266, 0, -616, 0, 1332, 0, -2960, 0, 10246, 16384,
            10246, 0, -2960, 0, 1332, 0, -616, 0, 266, 0, -103, 0, 35, 0, -10, 0, 2, 0, -1};

        int32_t reverbMul(int32_t a, int32_t b) { return (a * b) >> 15; }
    }

    // The SPU2's reverb (the PlayStation's, per core): every output sample the wet input goes into a half-band filter;
    // each sample one channel of the effect runs (left, then right: 24 kHz each) on the effects area between ESA and
    // EEA — two reflection paths ("same side" and "different side") with the wall and IIR volumes, four comb taps, two
    // all-pass filters — and its output is filtered back up to 48 kHz. ATTR bit 7 lets it write the effects area; the
    // registers: the 22 addresses at 0x2E4-0x33A (offsets in halfwords from the current position), EEA 0x33C, the
    // coefficients at 0x774 (core 0) / 0x79C (core 1).
    void Spu2::reverb(int coreIndex, int32_t wetLeft, int32_t wetRight, int32_t &outLeft, int32_t &outRight)
    {
        Core &core = m_cores[coreIndex];
        const uint32_t base = static_cast<uint32_t>(coreIndex) * 0x400u;
        const uint32_t ring = core.reverbRing;
        core.reverbDown[0][ring] = core.reverbDown[0][ring | 64u] = wetLeft;
        core.reverbDown[1][ring] = core.reverbDown[1][ring | 64u] = wetRight;

        const bool right = core.reverbRight;
        const int channel = right ? 1 : 0;
        const uint32_t first = (ring + 64u - (kReverbTaps - 1)) & 63u;

        const uint32_t start = composeAddress(reg(base + 0x2E0u), reg(base + 0x2E2u));
        const uint32_t end = ((static_cast<uint32_t>(reg(base + 0x33Cu)) & 0x3Fu) << 16u) | 0xFFFFu;
        int32_t effect = 0;
        if (end > start && end < kRamHalfwords)
        {
            const uint32_t size = end - start + 1u;
            const auto address = [&](uint32_t offsetRegister, int32_t delta)
            {
                const uint32_t offset = composeAddress(reg(base + offsetRegister), reg(base + offsetRegister + 2u));
                int64_t position = static_cast<int64_t>(core.reverbPosition) + offset + delta;
                // (position mod size), the usual case without a division: within one turn of the buffer either way
                if (position >= 0 && position < 2 * static_cast<int64_t>(size))
                {
                    if (position >= static_cast<int64_t>(size))
                        position -= size;
                }
                else
                {
                    position %= static_cast<int64_t>(size);
                    if (position < 0)
                        position += size;
                }
                return (start + static_cast<uint32_t>(position)) & kAddressMask;
            };
            const auto memory = [&](uint32_t halfword) { return static_cast<int32_t>(static_cast<int16_t>(m_ram[halfword])); };
            const uint32_t coefficients = kVolumeBlock + static_cast<uint32_t>(coreIndex) * kVolumeStride + 0x14u;
            const auto coefficient = [&](uint32_t index) { return static_cast<int32_t>(static_cast<int16_t>(reg(coefficients + index * 2u))); };
            const int32_t iirVolume = coefficient(0), comb1 = coefficient(1), comb2 = coefficient(2), comb3 = coefficient(3),
                          comb4 = coefficient(4), wall = coefficient(5), apf1Volume = coefficient(6), apf2Volume = coefficient(7),
                          inputVolume = coefficient(right ? 9u : 8u);

            // the address registers (each a high / low pair)
            const uint32_t apf1Size = composeAddress(reg(base + 0x2E4u), reg(base + 0x2E6u));
            const uint32_t apf2Size = composeAddress(reg(base + 0x2E8u), reg(base + 0x2EAu));
            const uint32_t sameDst = right ? 0x2F0u : 0x2ECu;
            const uint32_t sameSrc = right ? 0x308u : 0x304u;
            const uint32_t diffDst = right ? 0x310u : 0x30Cu;
            const uint32_t diffSrc = right ? 0x324u : 0x328u; // the other side's
            const uint32_t comb1Src = right ? 0x2F8u : 0x2F4u;
            const uint32_t comb2Src = right ? 0x300u : 0x2FCu;
            const uint32_t comb3Src = right ? 0x318u : 0x314u;
            const uint32_t comb4Src = right ? 0x320u : 0x31Cu;
            const uint32_t apf1Dst = right ? 0x330u : 0x32Cu;
            const uint32_t apf2Dst = right ? 0x338u : 0x334u;

            int32_t down = 0;
            for (int i = 0; i < kReverbTaps; ++i)
                down += core.reverbDown[channel][first + static_cast<uint32_t>(i)] * kReverbFilter[static_cast<size_t>(i)];
            const int32_t in = reverbMul(inputVolume, clamp16(down >> 15));

            const uint32_t samePrevious = address(sameDst, -1);
            const uint32_t diffPrevious = address(diffDst, -1);
            const int32_t same = reverbMul(iirVolume, in + reverbMul(wall, memory(address(sameSrc, 0))) - memory(samePrevious)) +
                                 memory(samePrevious);
            const int32_t diff = reverbMul(iirVolume, in + reverbMul(wall, memory(address(diffSrc, 0))) - memory(diffPrevious)) +
                                 memory(diffPrevious);
            int32_t out = reverbMul(comb1, memory(address(comb1Src, 0))) + reverbMul(comb2, memory(address(comb2Src, 0))) +
                          reverbMul(comb3, memory(address(comb3Src, 0))) + reverbMul(comb4, memory(address(comb4Src, 0)));
            const uint32_t apf1Src = address(apf1Dst, -static_cast<int32_t>(apf1Size));
            const int32_t apf1 = out - reverbMul(apf1Volume, memory(apf1Src));
            out = memory(apf1Src) + reverbMul(apf1Volume, apf1);
            const uint32_t apf2Src = address(apf2Dst, -static_cast<int32_t>(apf2Size));
            const int32_t apf2 = out - reverbMul(apf2Volume, memory(apf2Src));
            out = memory(apf2Src) + reverbMul(apf2Volume, apf2);

            if ((reg(base + 0x19Au) & 0x80u) != 0u) // ATTR: effects on (else the buffer is only read)
            {
                m_ram[address(sameDst, 0)] = static_cast<uint16_t>(clamp16(same));
                m_ram[address(diffDst, 0)] = static_cast<uint16_t>(clamp16(diff));
                m_ram[address(apf1Dst, 0)] = static_cast<uint16_t>(clamp16(apf1));
                m_ram[address(apf2Dst, 0)] = static_cast<uint16_t>(clamp16(apf2));
            }
            effect = clamp16(out);
            if (right)
                core.reverbPosition = (core.reverbPosition + 1u) % size;
        }
        core.reverbUp[channel][ring] = core.reverbUp[channel][ring | 64u] = effect;
        core.reverbUp[channel ^ 1][ring] = core.reverbUp[channel ^ 1][ring | 64u] = 0;
        core.reverbRight = !right;
        core.reverbRing = (ring + 1u) & 63u;

        // back up to 48 kHz: the zero-stuffed stream through the same filter (x2 for the stuffed zeros)
        int32_t upLeft = 0;
        int32_t upRight = 0;
        for (int i = 0; i < kReverbTaps; ++i)
        {
            upLeft += core.reverbUp[0][first + static_cast<uint32_t>(i)] * kReverbFilter[static_cast<size_t>(i)];
            upRight += core.reverbUp[1][first + static_cast<uint32_t>(i)] * kReverbFilter[static_cast<size_t>(i)];
        }
        outLeft = clamp16(upLeft >> 14);
        outRight = clamp16(upRight >> 14);
        core.reverbInPeak = std::max({core.reverbInPeak, std::abs(wetLeft), std::abs(wetRight)});
        core.reverbOutPeak = std::max({core.reverbOutPeak, std::abs(outLeft), std::abs(outRight)});
    }

    void Spu2::advanceTo(uint64_t iopCycle, std::vector<int16_t> &out)
    {
        if (iopCycle >= m_nextSampleCycle + kIopCyclesPerSample * 48000u)
            m_nextSampleCycle = iopCycle - iopCycle % kIopCyclesPerSample + kIopCyclesPerSample; // over a second behind: resync
        while (m_nextSampleCycle <= iopCycle)
        {
            m_nextSampleCycle += kIopCyclesPerSample;
            mixCore(0, 0, 0);
            mixCore(1, m_cores[0].outLeft, m_cores[0].outRight);
            out.push_back(static_cast<int16_t>(m_cores[1].outLeft));
            out.push_back(static_cast<int16_t>(m_cores[1].outRight));
            if (++m_samplesSinceReport >= 48000u)
            {
                m_samplesSinceReport = 0u;
                if (voiceLogEnabled() && (m_cores[0].reverbInPeak | m_cores[1].reverbInPeak) != 0)
                    std::fprintf(stderr, "[spu] reverb: core 0 in %d out %d, core 1 in %d out %d (peaks, last second)\n",
                                 m_cores[0].reverbInPeak, m_cores[0].reverbOutPeak, m_cores[1].reverbInPeak,
                                 m_cores[1].reverbOutPeak);
                for (Core &core : m_cores)
                    core.reverbInPeak = core.reverbOutPeak = 0;
            }
        }
    }

    void Spu2::dmaWrite(int coreIndex, const uint8_t *data, size_t bytes)
    {
        Core &core = m_cores[coreIndex & 1];
        if (voiceLogEnabled() && core.transferAddress >= 0x2800u)
        {
            // HCTP: sound-memory uploads (bank loads) - in-match hit sounds play from empty SPU RAM
            static int lines = 0;
            if (lines < 3000)
            {
                ++lines;
                size_t nz = 0;
                for (size_t i = 0; i < bytes; ++i)
                    nz += data[i] != 0u;
                std::fprintf(stderr, "[spudma] core %d to 0x%05X bytes %zu nonzero %zu\n", coreIndex, core.transferAddress, bytes, nz);
            }
        }
        for (size_t i = 0; i + 1u < bytes; i += 2u)
        {
            m_ram[core.transferAddress] = static_cast<uint16_t>(data[i] | (data[i + 1u] << 8u));
            core.transferAddress = (core.transferAddress + 1u) & kAddressMask;
        }
    }

    void Spu2::dmaRead(int coreIndex, uint8_t *data, size_t bytes)
    {
        Core &core = m_cores[coreIndex & 1];
        for (size_t i = 0; i + 1u < bytes; i += 2u)
        {
            const uint16_t value = m_ram[core.transferAddress];
            data[i] = static_cast<uint8_t>(value);
            data[i + 1u] = static_cast<uint8_t>(value >> 8u);
            core.transferAddress = (core.transferAddress + 1u) & kAddressMask;
        }
    }

    void Spu2::markDmaReady(int coreIndex)
    {
        m_regs[(0x344u + static_cast<uint32_t>(coreIndex & 1) * 0x400u) >> 1u] |= 0x80u;
    }

    bool Spu2::autoDmaEnabled(int coreIndex) const
    {
        return reg(0x1B0u + static_cast<uint32_t>(coreIndex & 1) * 0x400u) != 0u;
    }

    void Spu2::queueInput(int coreIndex, const int16_t *frames, size_t frameCount)
    {
        Core &core = m_cores[coreIndex & 1];
        core.input.insert(core.input.end(), frames, frames + frameCount * 2u);
        const size_t pending = core.input.size() - core.inputRead;
        if (pending > 48000u * 2u) // over a second queued (the input stopped being played): keep the newest
            core.inputRead = core.input.size() - 48000u * 2u;
    }
}
