#include <intrin.h>
#include <bit>
#include <string>
#include <memory>
#include "runtime/ps2_host_profile.h"
#include "runtime/ps2_vu1.h"
#include "runtime/ps2_vu1_compiled.h"
#include "runtime/gs/ps2_gif_arbiter.h"
#include "runtime/gs/gs_frontend.h"
#include "runtime/ps2_memory.h"
#include "runtime/ps2_vu_code_log.h"
#include "ps2_vu1_detail.h"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <ps2_log.h>

namespace
{
    constexpr uint8_t laneForComponent(uint32_t component)
    {
        return static_cast<uint8_t>(1u << (3u - component));
    }

    // The rounding mode of runLoop's caller: VU arithmetic runs round-toward-zero, but the GIF packets it kicks are
    // processed (front end, drawing) in the caller's mode like every other path to the GS.
    thread_local int t_callerRoundingMode = FE_TONEAREST;
}

void VU1Interpreter::addVfRead(InstructionUsage &usage, uint8_t reg, uint8_t lanes)
{
    if (lanes == 0u)
        return;
    for (uint32_t index = 0; index < usage.vfReadCount; ++index)
    {
        if (usage.vfRead[index].reg == reg)
        {
            usage.vfRead[index].lanes |= lanes;
            return;
        }
    }
    if (usage.vfReadCount < usage.vfRead.size())
        usage.vfRead[usage.vfReadCount++] = {reg, lanes};
}

void VU1Interpreter::addVfWrite(InstructionUsage &usage, uint8_t reg, uint8_t lanes)
{
    if (reg == 0u || lanes == 0u)
        return;
    if (usage.vfWrite.reg == 0u)
        usage.vfWrite = {reg, lanes};
    else if (usage.vfWrite.reg == reg)
        usage.vfWrite.lanes |= lanes;
}

uint8_t VU1Interpreter::vfReadLanes(const InstructionUsage &usage, uint8_t reg)
{
    for (uint32_t index = 0; index < usage.vfReadCount; ++index)
    {
        if (usage.vfRead[index].reg == reg)
            return usage.vfRead[index].lanes;
    }
    return 0u;
}

VU1Interpreter::VU1Interpreter(Unit unit)
    : m_unit(unit)
{
    const char *fast = std::getenv("VENDETTA_VU1_FAST");
    m_fastMode = !(fast && fast[0] == '0');
    const char *compiled = std::getenv("VENDETTA_VU1_COMPILED");
    m_compiledEnabled = !(compiled && compiled[0] == '0');
    reset();
}

void VU1Interpreter::resetScheduler()
{
    // field by field / memset (`= {}` on these goes through a zeroed temporary: see FlagPipelineEntry::clear)
    for (FlagPipelineEntry &entry : m_flagPipeline)
        entry.clear();
    m_flagSlotsUsed = 0;
    m_storeSlotsUsed = 0;
    m_registerWritesSettleCycle = 0;
    m_fdiv.clear();
    m_nextCommitCycle = 0;
    for (ScalarPipelineEntry &entry : m_efu)
        entry.clear();
    for (PendingStore &store : m_storePipeline)
        store.clear();
    for (PendingVfWrite &write : m_vfWritePipeline)
        write.clear();
    for (PendingViWrite &write : m_viWritePipeline)
        write.clear();
    for (PendingAccWrite &write : m_accWritePipeline)
        write.clear();
    m_xgkick.clear();
    std::memset(m_vfReady.data(), 0, sizeof(m_vfReady));
    std::memset(m_viReady.data(), 0, sizeof(m_viReady));
    std::memset(m_accReady.data(), 0, sizeof(m_accReady));
    std::memset(m_vfLatestWrite.data(), 0, sizeof(m_vfLatestWrite));
    std::memset(m_viLatestWrite.data(), 0, sizeof(m_viLatestWrite));
    std::memset(m_accLatestWrite.data(), 0, sizeof(m_accLatestWrite));
    m_nextWriteSequence = 0;
    m_efuResourceReady = 0;
    m_workingClip = m_state.clip;
    m_viBranchBackupValue = 0;
    m_viBranchBackupReg = 0;
    m_viBranchBackupValid = false;
    m_stopRequested = false;
    m_pendingHaltD = false;
    m_pendingHaltT = false;
}

void VU1Interpreter::reset()
{
    std::memset(&m_state, 0, sizeof(m_state));
    m_state.vf[0][3] = 1.0f;
    m_state.q = 1.0f;
    m_state.r = 0x3F800000u;
    m_cycle = 0;
    resetScheduler();
}

float VU1Interpreter::broadcast(const float *vf, uint8_t bc)
{
    return normalizeOperand(vf[bc & 3u]);
}

float VU1Interpreter::normalizeOperand(float value) const
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t exponent = (bits >> 23) & 0xFFu;
    if (exponent == 0u)
    {
        bits &= 0x80000000u;
    }
    else if (exponent == 0xFFu)
    {
        bits = (bits & 0x80000000u) | 0x7F7FFFFFu;
    }
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

float VU1Interpreter::normalizeResult(float value, uint32_t &laneFlags) const
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = bits & 0x80000000u;
    const uint32_t magnitude = bits & 0x7FFFFFFFu;
    const uint32_t exponent = (bits >> 23) & 0xFFu;

    laneFlags = sign != 0u ? 0x2u : 0u;
    if (magnitude == 0u)
    {
        laneFlags |= 0x1u;
    }
    else if (exponent == 0u)
    {
        laneFlags |= 0x5u;
        bits = sign;
    }
    else if (exponent == 0xFFu)
    {
        laneFlags |= 0x8u;
        bits = sign | 0x7F7FFFFFu;
    }

    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

uint32_t VU1Interpreter::microAddressMask() const
{
    return m_unit == Unit::VU1 ? 0x3FFFu : 0x0FFFu;
}

int32_t VU1Interpreter::readBranchVi(uint8_t reg) const
{
    if (reg == 0u)
        return 0;
    if (m_viBranchBackupValid &&
        m_viBranchBackupReg == reg)
    {
        return m_viBranchBackupValue;
    }
    return m_state.vi[reg];
}

void VU1Interpreter::recordViWriteForBranch(uint8_t reg, int32_t oldValue)
{
    if (reg == 0u)
        return;
    m_viBranchBackupValue = oldValue;
    m_viBranchBackupReg = reg;
    m_viBranchBackupValid = true;
}

void VU1Interpreter::applyDest(float *dst, const float *result, uint8_t dest)
{
    if (dest & 0x8u)
        dst[0] = result[0];
    if (dest & 0x4u)
        dst[1] = result[1];
    if (dest & 0x2u)
        dst[2] = result[2];
    if (dest & 0x1u)
        dst[3] = result[3];
}

void VU1Interpreter::applyDestAcc(const float *result, uint8_t dest)
{
    applyDest(m_state.acc, result, dest);
}

void VU1Interpreter::normalizeFmacResult(float *result, uint8_t dest,
                                         uint8_t laneFlags[4])
{
    for (uint32_t component = 0; component < 4u; ++component)
    {
        laneFlags[component] = 0u;
        if ((dest & laneForComponent(component)) == 0u)
            continue;

        long double exactResult = 0.0L;
        if (calculateFmacExactResult(component, exactResult))
        {
            laneFlags[component] = normalizeFmacExactResult(result[component], exactResult);
            continue;
        }

        uint32_t flags = 0u;
        result[component] = normalizeResult(result[component], flags);
        laneFlags[component] = static_cast<uint8_t>(flags);
    }
}

bool VU1Interpreter::calculateFmacExactResult(uint32_t component,
                                               long double &result) const
{
    const uint32_t upper = m_currentUpperInstruction;
    const uint8_t op = static_cast<uint8_t>(upper & 0x3Fu);
    const uint8_t special = op >= 0x3Cu
                                ? static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu))
                                : 0xFFu;
    const uint8_t fs = FS(upper);
    const uint8_t ft = FT(upper);

    const auto operand = [this](float value)
    {
        return static_cast<long double>(normalizeOperand(value));
    };
    const auto vs = [&](uint32_t lane)
    {
        return operand(m_state.vf[fs][lane]);
    };
    const auto vt = [&](uint32_t lane)
    {
        return operand(m_state.vf[ft][lane]);
    };
    const auto acc = [&](uint32_t lane)
    {
        return operand(m_state.acc[lane]);
    };

    const long double q = operand(m_state.q);
    const long double i = operand(m_state.i);

    if (op < 0x3Cu)
    {
        if (op <= 0x03u)
            result = vs(component) + vt(op & 3u);
        else if (op <= 0x07u)
            result = vs(component) - vt(op & 3u);
        else if (op <= 0x0Bu)
            result = acc(component) + vs(component) * vt(op & 3u);
        else if (op <= 0x0Fu)
            result = acc(component) - vs(component) * vt(op & 3u);
        else if (op >= 0x18u && op <= 0x1Bu)
            result = vs(component) * vt(op & 3u);
        else
        {
            switch (op)
            {
            case 0x1Cu:
                result = vs(component) * q;
                break;
            case 0x1Eu:
                result = vs(component) * i;
                break;
            case 0x20u:
                result = vs(component) + q;
                break;
            case 0x21u:
                result = acc(component) + vs(component) * q;
                break;
            case 0x22u:
                result = vs(component) + i;
                break;
            case 0x23u:
                result = acc(component) + vs(component) * i;
                break;
            case 0x24u:
                result = vs(component) - q;
                break;
            case 0x25u:
                result = acc(component) - vs(component) * q;
                break;
            case 0x26u:
                result = vs(component) - i;
                break;
            case 0x27u:
                result = acc(component) - vs(component) * i;
                break;
            case 0x28u:
                result = vs(component) + vt(component);
                break;
            case 0x29u:
                result = acc(component) + vs(component) * vt(component);
                break;
            case 0x2Au:
                result = vs(component) * vt(component);
                break;
            case 0x2Cu:
                result = vs(component) - vt(component);
                break;
            case 0x2Du:
                result = acc(component) - vs(component) * vt(component);
                break;
            case 0x2Eu:
            {
                static constexpr uint8_t left[4] = {1u, 2u, 0u, 3u};
                static constexpr uint8_t right[4] = {2u, 0u, 1u, 3u};
                result = component == 3u
                             ? 0.0L
                             : acc(component) - vs(left[component]) * vt(right[component]);
                break;
            }
            default:
                return false;
            }
        }
        return true;
    }

    if (special <= 0x03u)
        result = vs(component) + vt(special & 3u);
    else if (special <= 0x07u)
        result = vs(component) - vt(special & 3u);
    else if (special <= 0x0Bu)
        result = acc(component) + vs(component) * vt(special & 3u);
    else if (special <= 0x0Fu)
        result = acc(component) - vs(component) * vt(special & 3u);
    else if (special >= 0x18u && special <= 0x1Bu)
        result = vs(component) * vt(special & 3u);
    else
    {
        switch (special)
        {
        case 0x1Cu:
            result = vs(component) * q;
            break;
        case 0x1Eu:
            result = vs(component) * i;
            break;
        case 0x20u:
            result = vs(component) + q;
            break;
        case 0x21u:
            result = acc(component) + vs(component) * q;
            break;
        case 0x22u:
            result = vs(component) + i;
            break;
        case 0x23u:
            result = acc(component) + vs(component) * i;
            break;
        case 0x24u:
            result = vs(component) - q;
            break;
        case 0x25u:
            result = acc(component) - vs(component) * q;
            break;
        case 0x26u:
            result = vs(component) - i;
            break;
        case 0x27u:
            result = acc(component) - vs(component) * i;
            break;
        case 0x28u:
            result = vs(component) + vt(component);
            break;
        case 0x29u:
            result = acc(component) + vs(component) * vt(component);
            break;
        case 0x2Au:
            result = vs(component) * vt(component);
            break;
        case 0x2Cu:
            result = vs(component) - vt(component);
            break;
        case 0x2Du:
            result = acc(component) - vs(component) * vt(component);
            break;
        case 0x2Eu:
        {
            static constexpr uint8_t left[4] = {1u, 2u, 0u, 3u};
            static constexpr uint8_t right[4] = {2u, 0u, 1u, 3u};
            result = component == 3u
                         ? 0.0L
                         : vs(left[component]) * vt(right[component]);
            break;
        }
        default:
            return false;
        }
    }
    return true;
}

uint8_t VU1Interpreter::normalizeFmacExactResult(float &value,
                                                  long double exactResult) const
{
    const bool negative = std::signbit(exactResult);
    const long double magnitude = std::fabs(exactResult);
    const long double maximum = static_cast<long double>(std::numeric_limits<float>::max());
    const long double minimum = static_cast<long double>(std::numeric_limits<float>::min());
    uint8_t flags = negative ? 0x2u : 0u;

    uint32_t bits = negative ? 0x80000000u : 0u;
    if (magnitude == 0.0L)
    {
        flags |= 0x1u;
        std::memcpy(&value, &bits, sizeof(value));
    }
    else if (magnitude > maximum)
    {
        flags |= 0x8u;
        bits |= 0x7F7FFFFFu;
        std::memcpy(&value, &bits, sizeof(value));
    }
    else if (magnitude < minimum)
    {
        flags |= 0x5u;
        std::memcpy(&value, &bits, sizeof(value));
    }

    return flags;
}

uint32_t VU1Interpreter::calculateFmacProductSticky(uint8_t dest) const
{
    uint32_t extraSticky = 0u;
    const uint32_t upper = m_currentUpperInstruction;
    const uint8_t op = static_cast<uint8_t>(upper & 0x3Fu);
    const uint8_t special = op >= 0x3Cu ? static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu)) : 0xFFu;
    const bool productSum =
        (op >= 0x08u && op <= 0x0Fu) ||
        op == 0x21u || op == 0x23u || op == 0x25u || op == 0x27u ||
        op == 0x29u || op == 0x2Du || op == 0x2Eu ||
        (special >= 0x08u && special <= 0x0Fu) ||
        special == 0x21u || special == 0x23u || special == 0x25u ||
        special == 0x27u || special == 0x29u || special == 0x2Du;
    if (!productSum)
        return 0u;

    const uint8_t fs = FS(upper);
    const uint8_t ft = FT(upper);
    for (uint32_t component = 0; component < 4u; ++component)
    {
        if ((dest & laneForComponent(component)) == 0u)
            continue;
        static constexpr uint8_t crossLeft[4] = {1u, 2u, 0u, 3u};
        static constexpr uint8_t crossRight[4] = {2u, 0u, 1u, 3u};
        const uint8_t leftComponent = op == 0x2Eu ? crossLeft[component] : static_cast<uint8_t>(component);
        const float left = normalizeOperand(m_state.vf[fs][leftComponent]);
        float right = 0.0f;
        if ((op >= 0x08u && op <= 0x0Fu) || (special >= 0x08u && special <= 0x0Fu))
        {
            right = normalizeOperand(m_state.vf[ft][(op >= 0x08u && op <= 0x0Fu ? op : special) & 3u]);
        }
        else if (op == 0x21u || op == 0x25u || special == 0x21u || special == 0x25u)
        {
            right = normalizeOperand(m_state.q);
        }
        else if (op == 0x23u || op == 0x27u || special == 0x23u || special == 0x27u)
        {
            right = normalizeOperand(m_state.i);
        }
        else if (op == 0x2Eu)
        {
            right = normalizeOperand(m_state.vf[ft][crossRight[component]]);
        }
        else
        {
            right = normalizeOperand(m_state.vf[ft][component]);
        }

        float product = left * right;
        const long double exactProduct = static_cast<long double>(left) * static_cast<long double>(right);
        const uint8_t productFlags = normalizeFmacExactResult(product, exactProduct);
        // Product-sum instructions report Z/S/U/O from the add/subtract result
        // as current flags, while every product condition accumulates into the
        // corresponding sticky flag.
        extraSticky |= productFlags & 0xFu;
    }
    return extraSticky;
}

void VU1Interpreter::updateFmacFlags(const uint8_t laneFlags[4], uint8_t dest,
                                     uint32_t extraSticky)
{
    if (dest == 0u)
        return;

    uint32_t mac = 0u;
    uint32_t status = 0u;
    for (uint32_t component = 0; component < 4u; ++component)
    {
        const uint8_t lane = laneForComponent(component);
        if ((dest & lane) == 0u)
            continue;

        const uint32_t flags = laneFlags[component];
        if ((flags & 0x1u) != 0u)
            mac |= lane;
        if ((flags & 0x2u) != 0u)
            mac |= static_cast<uint32_t>(lane) << 4;
        if ((flags & 0x4u) != 0u)
            mac |= static_cast<uint32_t>(lane) << 8;
        if ((flags & 0x8u) != 0u)
            mac |= static_cast<uint32_t>(lane) << 12;
        status |= flags;
    }

    FlagPipelineEntry *entry = takeFreeSlot(m_flagPipeline, m_flagSlotsUsed);
    if (!entry)
    {
        reportReservedInstruction(true, 0xFFFFFFFFu);
        return;
    }

    entry->clear();
    entry->valid = true;
    entry->issueCycle = m_cycle;
    entry->readyCycle = m_cycle + kFmacLatency;
    noteReadyCycle(entry->readyCycle);
    entry->mac = mac;
    entry->status = status;
    entry->extraSticky = extraSticky;
    entry->writesMac = true;
    entry->writesStatus = true;
}

void VU1Interpreter::applyFmacDest(float *dst, float *result, uint8_t dest)
{
    uint8_t laneFlags[4]{};
    normalizeFmacResult(result, dest, laneFlags);
    updateFmacFlags(laneFlags, dest, calculateFmacProductSticky(dest));
    applyDest(dst, result, dest);
}

void VU1Interpreter::applyFmacDestAcc(float *result, uint8_t dest)
{
    uint8_t laneFlags[4]{};
    normalizeFmacResult(result, dest, laneFlags);
    updateFmacFlags(laneFlags, dest, calculateFmacProductSticky(dest));
    applyDestAcc(result, dest);
}

void VU1Interpreter::queueFsset(uint16_t immediate)
{
    for (FlagPipelineEntry &entry : m_flagPipeline)
    {
        if (entry.valid && entry.issueCycle == m_cycle)
            entry.writesStatus = false;
    }

    if (FlagPipelineEntry *slot = takeFreeSlot(m_flagPipeline, m_flagSlotsUsed))
    {
        FlagPipelineEntry &entry = *slot;
        entry.clear();
        entry.valid = true;
        entry.issueCycle = m_cycle;
        entry.readyCycle = m_cycle + kFmacLatency;
        noteReadyCycle(entry.readyCycle);
        entry.status = static_cast<uint32_t>(immediate) & 0xFC0u;
        entry.writesSticky = true;
        return;
    }
    reportReservedInstruction(false, 0xFFFFFFFEu);
}

void VU1Interpreter::queueClip(uint32_t clip)
{
    m_workingClip = ((m_workingClip << 6) | (clip & 0x3Fu)) & 0xFFFFFFu;
    if (FlagPipelineEntry *slot = takeFreeSlot(m_flagPipeline, m_flagSlotsUsed))
    {
        FlagPipelineEntry &entry = *slot;
        entry.clear();
        entry.valid = true;
        entry.issueCycle = m_cycle;
        entry.readyCycle = m_cycle + kFmacLatency;
        noteReadyCycle(entry.readyCycle);
        entry.clip = m_workingClip;
        entry.writesClip = true;
        return;
    }
    reportReservedInstruction(true, 0xFFFFFFFDu);
}

void VU1Interpreter::queueFcset(uint32_t clip)
{
    m_workingClip = clip & 0xFFFFFFu;
    for (FlagPipelineEntry &entry : m_flagPipeline)
    {
        if (entry.valid && entry.issueCycle == m_cycle)
            entry.writesClip = false;
    }
    if (FlagPipelineEntry *slot = takeFreeSlot(m_flagPipeline, m_flagSlotsUsed))
    {
        FlagPipelineEntry &entry = *slot;
        entry.clear();
        entry.valid = true;
        entry.issueCycle = m_cycle;
        entry.readyCycle = m_cycle + kFmacLatency;
        noteReadyCycle(entry.readyCycle);
        entry.clip = m_workingClip;
        entry.writesClip = true;
        return;
    }
    reportReservedInstruction(false, 0xFFFFFFFAu);
}

void VU1Interpreter::queueQ(float value, uint32_t latency, uint32_t statusDi)
{
    uint32_t ignoredFlags = 0u;
    value = normalizeResult(value, ignoredFlags);
    m_fdiv.valid = true;
    m_fdiv.readyCycle = m_cycle + latency;
    noteReadyCycle(m_fdiv.readyCycle);
    m_fdiv.value = value;
    m_fdiv.statusDi = statusDi & 0x30u;
}

void VU1Interpreter::queueP(float value, uint32_t latency)
{
    uint32_t ignoredFlags = 0u;
    value = normalizeResult(value, ignoredFlags);
    for (ScalarPipelineEntry &entry : m_efu)
    {
        if (!entry.valid)
        {
            entry.valid = true;
            entry.readyCycle = m_cycle + latency;
            noteReadyCycle(entry.readyCycle);
            entry.value = value;
            // EFU throughput is one cycle shorter than result visibility.
            m_efuResourceReady = m_cycle + (latency > 0u ? latency - 1u : 0u);
            return;
        }
    }
    reportReservedInstruction(false, 0xFFFFFFF9u);
}

void VU1Interpreter::queueStore(uint32_t address, const uint32_t words[4], uint8_t laneMask)
{
    if (m_fastMode)
    {
        // Fast mode writes stores at once, as it does register results: what can read VU memory (a later pair, the
        // XGKICK transfer at the next cycle boundary, the end of the program) comes a cycle later, after the queued
        // store would have landed. The program still ends no earlier than that landing (pipelinesPending).
        if (m_activeVuData && address + 16u <= m_activeVuDataSize)
        {
            // only the store's lanes (the others stay as they are), without a merged copy that is read back
            uint8_t *const destination = m_activeVuData + address;
            if (laneMask == 0xFu)
                std::memcpy(destination, words, 16u);
            else
            {
                for (uint32_t component = 0; component < 4u; ++component)
                {
                    if ((laneMask & laneForComponent(component)) != 0u)
                        std::memcpy(destination + component * 4u, &words[component], 4u);
                }
            }
        }
        m_registerWritesSettleCycle = std::max(m_registerWritesSettleCycle, m_cycle + 1u);
        return;
    }
    if (PendingStore *slot = takeFreeSlot(m_storePipeline, m_storeSlotsUsed))
    {
        PendingStore &store = *slot;
        store.valid = true;
        store.readyCycle = m_cycle + 1u;
        noteReadyCycle(store.readyCycle);
        store.address = address;
        store.laneMask = laneMask;
        std::copy(words, words + 4, store.words.begin());
        return;
    }
    reportReservedInstruction(false, 0xFFFFFFFCu);
}

void VU1Interpreter::queueVfWrite(uint8_t reg, uint8_t laneMask,
                                  const float value[4], uint32_t latency)
{
    if (reg == 0u || laneMask == 0u)
        return;
    for (PendingVfWrite &write : m_vfWritePipeline)
    {
        if (!write.valid)
        {
            write.clear();
            write.valid = true;
            write.readyCycle = m_cycle + latency;
            noteReadyCycle(write.readyCycle);
            write.sequence = ++m_nextWriteSequence;
            write.reg = reg;
            write.laneMask = laneMask;
            std::copy(value, value + 4, write.value.begin());
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if ((laneMask & laneForComponent(component)) != 0u)
                    m_vfLatestWrite[reg][component] = write.sequence;
            }
            return;
        }
    }
    reportReservedInstruction(false, 0xFFFFFFF7u);
}

void VU1Interpreter::queueViWrite(uint8_t reg, int32_t value, uint32_t latency)
{
    if (reg == 0u)
        return;
    for (PendingViWrite &write : m_viWritePipeline)
    {
        if (!write.valid)
        {
            write.clear();
            write.valid = true;
            write.readyCycle = m_cycle + latency;
            noteReadyCycle(write.readyCycle);
            write.sequence = ++m_nextWriteSequence;
            write.reg = reg;
            write.value = value;
            m_viLatestWrite[reg] = write.sequence;
            return;
        }
    }
    reportReservedInstruction(false, 0xFFFFFFF6u);
}

void VU1Interpreter::queueAccWrite(uint8_t laneMask, const float value[4], uint32_t latency)
{
    if (laneMask == 0u)
        return;
    for (PendingAccWrite &write : m_accWritePipeline)
    {
        if (!write.valid)
        {
            write.clear();
            write.valid = true;
            write.readyCycle = m_cycle + latency;
            noteReadyCycle(write.readyCycle);
            write.sequence = ++m_nextWriteSequence;
            write.laneMask = laneMask;
            std::copy(value, value + 4, write.value.begin());
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if ((laneMask & laneForComponent(component)) != 0u)
                    m_accLatestWrite[component] = write.sequence;
            }
            return;
        }
    }
    reportReservedInstruction(true, 0xFFFFFFF5u);
}

// The first free slot (marked used): fast mode finds it in the used-slot mask, the original model scans as before.
template <typename Entry, size_t N>
Entry *VU1Interpreter::takeFreeSlot(std::array<Entry, N> &slots, uint8_t &used)
{
    uint32_t index = static_cast<uint32_t>(N);
    if (m_fastMode)
    {
        const uint32_t free = ~static_cast<uint32_t>(used) & ((1u << N) - 1u);
        if (free != 0u)
            index = static_cast<uint32_t>(std::countr_zero(free));
    }
    else
    {
        for (uint32_t slot = 0; slot < N; ++slot)
            if (!slots[slot].valid)
            {
                index = slot;
                break;
            }
    }
    if (index >= N)
        return nullptr;
    used = static_cast<uint8_t>(used | (1u << index));
    return &slots[index];
}

void VU1Interpreter::commitReadyPipelines()
{
    if (m_cycle < m_nextCommitCycle)
        return;
    if (!m_fastMode)
    {
        commitReadyPipelinesScan();
        return;
    }
    // Fast mode: the same commits in the same (slot) order, visiting only the slots in use. The register write
    // queues are empty in fast mode (see commitReadyPipelinesScan).
    uint64_t next = UINT64_MAX;
    for (uint32_t slots = m_flagSlotsUsed; slots != 0u; slots &= slots - 1u)
    {
        const uint32_t index = static_cast<uint32_t>(std::countr_zero(slots));
        FlagPipelineEntry &entry = m_flagPipeline[index];
        if (entry.readyCycle > m_cycle)
        {
            next = std::min(next, entry.readyCycle);
            continue;
        }
        if (entry.writesMac)
            m_state.mac = entry.mac;
        if (entry.writesStatus)
        {
            const uint32_t current = entry.status & 0xFu;
            m_state.status = (m_state.status & 0xFF0u) | current | ((current | entry.extraSticky) << 6);
        }
        if (entry.writesSticky)
        {
            m_state.status = (m_state.status & 0x03Fu) | (entry.status & 0xFC0u);
        }
        if (entry.writesClip)
            m_state.clip = entry.clip;
        entry.clear();
        m_flagSlotsUsed = static_cast<uint8_t>(m_flagSlotsUsed & ~(1u << index));
    }

    if (m_fdiv.valid)
    {
        if (m_fdiv.readyCycle > m_cycle)
            next = std::min(next, m_fdiv.readyCycle);
        else
        {
            m_state.q = m_fdiv.value;
            const uint32_t currentDi = m_fdiv.statusDi & 0x30u;
            m_state.status = (m_state.status & 0xFCFu) | currentDi | (currentDi << 6);
            m_fdiv.clear();
        }
    }

    for (ScalarPipelineEntry &entry : m_efu)
    {
        if (entry.valid && entry.readyCycle <= m_cycle)
        {
            m_state.p = entry.value;
            entry.clear();
        }
        else if (entry.valid)
            next = std::min(next, entry.readyCycle);
    }

    for (uint32_t slots = m_storeSlotsUsed; slots != 0u; slots &= slots - 1u)
    {
        const uint32_t index = static_cast<uint32_t>(std::countr_zero(slots));
        PendingStore &store = m_storePipeline[index];
        if (store.readyCycle > m_cycle)
        {
            next = std::min(next, store.readyCycle);
            continue;
        }
        if (m_activeVuData && store.address + 16u <= m_activeVuDataSize)
        {
            uint32_t oldWords[4]{};
            std::memcpy(oldWords, m_activeVuData + store.address, sizeof(oldWords));
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if ((store.laneMask & laneForComponent(component)) != 0u)
                    oldWords[component] = store.words[component];
            }
            std::memcpy(m_activeVuData + store.address, oldWords, sizeof(oldWords));
        }
        store.clear();
        m_storeSlotsUsed = static_cast<uint8_t>(m_storeSlotsUsed & ~(1u << index));
    }
    m_nextCommitCycle = next;
}

void VU1Interpreter::commitReadyPipelinesScan()
{
    uint64_t next = UINT64_MAX;
    for (FlagPipelineEntry &entry : m_flagPipeline)
    {
        if (!entry.valid)
            continue;
        if (entry.readyCycle > m_cycle)
        {
            next = std::min(next, entry.readyCycle);
            continue;
        }

        if (entry.writesMac)
            m_state.mac = entry.mac;
        if (entry.writesStatus)
        {
            const uint32_t current = entry.status & 0xFu;
            m_state.status = (m_state.status & 0xFF0u) | current | ((current | entry.extraSticky) << 6);
        }
        if (entry.writesSticky)
        {
            m_state.status = (m_state.status & 0x03Fu) | (entry.status & 0xFC0u);
        }
        if (entry.writesClip)
            m_state.clip = entry.clip;
        entry.clear();
        m_flagSlotsUsed = static_cast<uint8_t>(m_flagSlotsUsed & ~(1u << (&entry - m_flagPipeline.data())));
    }

    if (m_fdiv.valid && m_fdiv.readyCycle > m_cycle)
        next = std::min(next, m_fdiv.readyCycle);
    if (m_fdiv.valid && m_fdiv.readyCycle <= m_cycle)
    {
        m_state.q = m_fdiv.value;
        const uint32_t currentDi = m_fdiv.statusDi & 0x30u;
        m_state.status = (m_state.status & 0xFCFu) | currentDi | (currentDi << 6);
        m_fdiv.clear();
    }

    for (ScalarPipelineEntry &entry : m_efu)
    {
        if (entry.valid && entry.readyCycle <= m_cycle)
        {
            m_state.p = entry.value;
            entry.clear();
        }
        else if (entry.valid)
            next = std::min(next, entry.readyCycle);
    }

    for (PendingStore &store : m_storePipeline)
    {
        if (!store.valid)
            continue;
        if (store.readyCycle > m_cycle)
        {
            next = std::min(next, store.readyCycle);
            continue;
        }
        if (m_activeVuData && store.address + 16u <= m_activeVuDataSize)
        {
            uint32_t oldWords[4]{};
            std::memcpy(oldWords, m_activeVuData + store.address, sizeof(oldWords));
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if ((store.laneMask & laneForComponent(component)) != 0u)
                    oldWords[component] = store.words[component];
            }
            std::memcpy(m_activeVuData + store.address, oldWords, sizeof(oldWords));
        }
        store.clear();
        m_storeSlotsUsed = static_cast<uint8_t>(m_storeSlotsUsed & ~(1u << (&store - m_storePipeline.data())));
    }

    // The register write queues: only the original (queued) model uses them; fast mode writes registers at once
    // and drains the queues when a run starts, so they are empty there.
    if (!m_fastMode)
    {
        for (PendingVfWrite &write : m_vfWritePipeline)
        {
            if (!write.valid)
                continue;
            if (write.readyCycle > m_cycle)
            {
                next = std::min(next, write.readyCycle);
                continue;
            }
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if ((write.laneMask & laneForComponent(component)) != 0u &&
                    m_vfLatestWrite[write.reg][component] == write.sequence)
                {
                    m_state.vf[write.reg][component] = write.value[component];
                }
            }
            write.clear();
        }

        for (PendingViWrite &write : m_viWritePipeline)
        {
            if (!write.valid)
                continue;
            if (write.readyCycle > m_cycle)
            {
                next = std::min(next, write.readyCycle);
                continue;
            }
            if (m_viLatestWrite[write.reg] == write.sequence)
                m_state.vi[write.reg] = static_cast<int16_t>(write.value);
            write.clear();
        }

        for (PendingAccWrite &write : m_accWritePipeline)
        {
            if (!write.valid)
                continue;
            if (write.readyCycle > m_cycle)
            {
                next = std::min(next, write.readyCycle);
                continue;
            }
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if ((write.laneMask & laneForComponent(component)) != 0u &&
                    m_accLatestWrite[component] == write.sequence)
                {
                    m_state.acc[component] = write.value[component];
                }
            }
            write.clear();
        }
    }
    m_nextCommitCycle = next;
}

void VU1Interpreter::progressXgkick()
{
    if (!m_xgkick.active || !m_activeVuData || m_activeVuDataSize == 0u)
        return;

    ++m_xgkick.cycleCredit;
    while (m_xgkick.active && m_xgkick.cycleCredit >= 2u)
    {
        m_xgkick.cycleCredit -= 2u;
        if (m_xgkick.copiedBytes > XgkickPipeline::kBufferSize - 16u)
        {
            reportReservedInstruction(false, 0xFFFFFFFBu);
            m_xgkick.active = false;
            return;
        }

        const uint32_t qwordOffset = m_xgkick.copiedBytes;
        const uint32_t first = wrapVuDataAddress(m_xgkick.sourceAddress + m_xgkick.copiedBytes);
        if (first + 16u <= m_activeVuDataSize)
            std::memcpy(m_xgkick.packet.data() + m_xgkick.copiedBytes, m_activeVuData + first, 16u);
        else
        {
            for (uint32_t i = 0; i < 16u; ++i)
            {
                const uint32_t source = (m_xgkick.sourceAddress + m_xgkick.copiedBytes + i) % m_activeVuDataSize;
                m_xgkick.packet[m_xgkick.copiedBytes + i] = m_activeVuData[source];
            }
        }
        m_xgkick.copiedBytes += 16u;

        if (m_xgkick.currentTagEnd == 0u)
        {
            uint64_t tagLo = 0;
            std::memcpy(&tagLo, m_xgkick.packet.data() + qwordOffset, sizeof(tagLo));
            const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFFu);
            const uint32_t format = static_cast<uint32_t>((tagLo >> 58) & 0x3u);
            uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xFu);
            if (nreg == 0u)
                nreg = 16u;

            uint64_t tagBytes = 16u;
            if (format == 0u)
                tagBytes += static_cast<uint64_t>(nloop) * nreg * 16u;
            else if (format == 1u)
                tagBytes += ((static_cast<uint64_t>(nloop) * nreg + 1u) & ~1ull) * 8u;
            else if (format == 2u)
                tagBytes += static_cast<uint64_t>(nloop) * 16u;
            else
            {
                reportReservedInstruction(false, 0xFFFFFFF8u);
                m_xgkick.active = false;
                return;
            }

            if (tagBytes > XgkickPipeline::kBufferSize - qwordOffset)
            {
                reportReservedInstruction(false, 0xFFFFFFFBu);
                m_xgkick.active = false;
                return;
            }
            m_xgkick.currentTagEnd = qwordOffset + static_cast<uint32_t>(tagBytes);
            m_xgkick.currentTagEop = ((tagLo >> 15) & 1u) != 0u;
            if (m_xgkick.currentTagEop)
                m_xgkick.totalBytes = m_xgkick.currentTagEnd;
        }

        if (m_xgkick.copiedBytes >= m_xgkick.currentTagEnd)
        {
            if (m_xgkick.currentTagEop)
                finishXgkick();
            else
            {
                // The next transferred qword is another GIFtag.
                m_xgkick.currentTagEnd = 0u;
                m_xgkick.currentTagEop = false;
            }
        }
    }
}

void VU1Interpreter::finishXgkick()
{
    if (!m_xgkick.active)
        return;

    if (m_kickCapture)
    {
        m_kickCapture->emplace_back(m_xgkick.packet.begin(), m_xgkick.packet.begin() + m_xgkick.totalBytes);
        m_xgkick.active = false;
        return;
    }
    const int vuRoundingMode = std::fegetround();
    std::fesetround(t_callerRoundingMode);
    if (m_activeMemory)
        m_activeMemory->submitGifPacket(GifPathId::Path1, m_xgkick.packet.data(), m_xgkick.totalBytes);
    else if (m_activeGs)
        m_activeGs->processGIFPacket(m_xgkick.packet.data(), m_xgkick.totalBytes, 1u); // PATH1
    std::fesetround(vuRoundingMode);
    m_xgkick.active = false;
}

void VU1Interpreter::startXgkick(uint32_t qwordAddress)
{
    if (m_unit != Unit::VU1 || !m_activeVuData || m_activeVuDataSize < 16u)
        return;

    const uint32_t sourceAddress = wrapVuDataAddress(qwordAddress * 16u);
    m_xgkick.clear();
    m_xgkick.active = true;
    m_xgkick.sourceAddress = sourceAddress;
    m_xgkick.cycleCredit = 1u; // XGKICK's issue cycle counts toward PATH1.
    m_xgkick.issueCycle = m_cycle;
}

void VU1Interpreter::advanceOneCycle()
{
    ++m_cycle;
    m_state.cycles = m_cycle;
    // LSU commits become visible at the cycle boundary before PATH1 consumes
    // its next qword from VU memory.
    if (m_cycle >= m_nextCommitCycle)
        commitReadyPipelines();
    if (m_xgkick.active)
        progressXgkick();
}

void VU1Interpreter::advanceTo(uint64_t targetCycle)
{
    while (m_cycle < targetCycle)
        advanceOneCycle();
}

bool VU1Interpreter::pipelinesPending() const
{
    if (m_fdiv.valid || m_xgkick.active)
        return true;
    // fast mode writes registers at once; a program still ends when its last write would have landed
    if (m_fastMode && m_cycle < m_registerWritesSettleCycle)
        return true;
    for (const ScalarPipelineEntry &entry : m_efu)
        if (entry.valid)
            return true;
    for (const FlagPipelineEntry &entry : m_flagPipeline)
        if (entry.valid)
            return true;
    for (const PendingStore &store : m_storePipeline)
        if (store.valid)
            return true;
    for (const PendingVfWrite &write : m_vfWritePipeline)
        if (write.valid)
            return true;
    for (const PendingViWrite &write : m_viWritePipeline)
        if (write.valid)
            return true;
    for (const PendingAccWrite &write : m_accWritePipeline)
        if (write.valid)
            return true;
    return false;
}

void VU1Interpreter::flushPipelines()
{
    while (pipelinesPending())
        advanceOneCycle();
}

uint64_t VU1Interpreter::calculatePairReadyCycle(const DecodedInstructionPair &decoded) const
{
    uint64_t ready = m_cycle;
    for (uint32_t index = 0; index < decoded.stallReadCount; ++index)
    {
        const VfAccess &access = decoded.stallReads[index];
        const auto &laneReady = m_vfReady[access.reg];
        if (access.lanes & 0x8u)
            ready = std::max(ready, laneReady[0]);
        if (access.lanes & 0x4u)
            ready = std::max(ready, laneReady[1]);
        if (access.lanes & 0x2u)
            ready = std::max(ready, laneReady[2]);
        if (access.lanes & 0x1u)
            ready = std::max(ready, laneReady[3]);
    }
    for (uint32_t reads = decoded.stallViReads; reads != 0u; reads &= reads - 1u)
    {
        unsigned long reg = 0;
        _BitScanForward(&reg, reads);
        ready = std::max(ready, m_viReady[reg]);
    }
    if (decoded.stallAccReads != 0u)
    {
        for (uint32_t component = 0; component < 4u; ++component)
            if ((decoded.stallAccReads & laneForComponent(component)) != 0u)
                ready = std::max(ready, m_accReady[component]);
    }

    if (decoded.lowerUsage.pipeline == PipelineFdiv && m_fdiv.valid)
        ready = std::max(ready, m_fdiv.readyCycle);
    if (decoded.lowerUsage.pipeline == PipelineEfu)
        ready = std::max(ready, m_efuResourceReady);
    if (decoded.lowerUsage.waitQ && m_fdiv.valid)
        ready = std::max(ready, m_fdiv.readyCycle);
    if (decoded.lowerUsage.waitP)
    {
        for (const ScalarPipelineEntry &entry : m_efu)
            if (entry.valid)
                ready = std::max(ready, entry.readyCycle);
    }
    if (decoded.lowerUsage.pipeline == PipelineXgkick && m_xgkick.active)
        ready = std::max(ready, m_cycle + 1u);
    return ready;
}

void VU1Interpreter::markPairWrites(const DecodedInstructionPair &decoded)
{
    if (m_fastMode)
    {
        // the latest landing of this pair's register writes (VF / VI / ACC), for pipelinesPending()
        uint64_t settle = 0u;
        if (decoded.lowerUsage.vfWrite.reg != 0u && decoded.lowerUsage.vfWrite.lanes != 0u && decoded.suppressedLowerVf != decoded.lowerUsage.vfWrite.reg)
            settle = std::max<uint64_t>(settle, m_cycle + (decoded.lowerUsage.vfLatency != 0u ? decoded.lowerUsage.vfLatency : decoded.lowerUsage.latency));
        if (decoded.upperUsage.vfWrite.reg != 0u && decoded.upperUsage.vfWrite.lanes != 0u)
            settle = std::max<uint64_t>(settle, m_cycle + (decoded.upperUsage.vfLatency != 0u ? decoded.upperUsage.vfLatency : decoded.upperUsage.latency));
        if (decoded.lowerViWriteReg != 0u)
            settle = std::max<uint64_t>(settle, m_cycle + (decoded.lowerUsage.viLatency != 0u ? decoded.lowerUsage.viLatency : decoded.lowerUsage.latency));
        if (decoded.upperUsage.accWrite != 0u)
            settle = std::max<uint64_t>(settle, m_cycle + kAccForwardLatency);
        m_registerWritesSettleCycle = std::max(m_registerWritesSettleCycle, settle);
    }
    if (decoded.markLowerVf.reg != 0u)
    {
        const uint64_t ready = m_cycle + decoded.markLowerVfLatency;
        auto &laneReady = m_vfReady[decoded.markLowerVf.reg];
        for (uint32_t component = 0; component < 4u; ++component)
            if ((decoded.markLowerVf.lanes & laneForComponent(component)) != 0u)
                laneReady[component] = ready;
    }
    if (decoded.markUpperVf.reg != 0u)
    {
        const uint64_t ready = m_cycle + decoded.markUpperVfLatency;
        auto &laneReady = m_vfReady[decoded.markUpperVf.reg];
        for (uint32_t component = 0; component < 4u; ++component)
            if ((decoded.markUpperVf.lanes & laneForComponent(component)) != 0u)
                laneReady[component] = ready;
    }
    for (uint32_t writes = decoded.markViWrites; writes != 0u; writes &= writes - 1u)
    {
        unsigned long reg = 0;
        _BitScanForward(&reg, writes);
        m_viReady[reg] = m_cycle + decoded.markViLatency;
    }
    if (decoded.markAccWrites != 0u)
    {
        for (uint32_t component = 0; component < 4u; ++component)
            if ((decoded.markAccWrites & laneForComponent(component)) != 0u)
                m_accReady[component] = m_cycle + kAccForwardLatency;
    }
}

VU1Interpreter::InstructionUsage VU1Interpreter::decodeUpperUsage(uint32_t upper) const
{
    InstructionUsage usage;
    usage.pipeline = PipelineFmac;
    usage.latency = kFmacLatency;

    const uint8_t op = static_cast<uint8_t>(upper & 0x3Fu);
    const uint8_t dest = DEST(upper);
    const uint8_t fs = FS(upper);
    const uint8_t ft = FT(upper);
    const uint8_t fd = FD(upper);

    if (op <= 0x2Fu)
    {
        addVfRead(usage, fs, dest);
        addVfWrite(usage, fd, dest);
        if (op <= 0x1Bu)
            addVfRead(usage, ft, laneForComponent(op & 3u));
        else if (op >= 0x28u)
            addVfRead(usage, ft, op == 0x2Eu ? 0xEu : dest);
        if (op == 0x08u || op == 0x09u || op == 0x0Au || op == 0x0Bu ||
            op == 0x0Cu || op == 0x0Du || op == 0x0Eu || op == 0x0Fu ||
            op == 0x21u || op == 0x23u || op == 0x25u || op == 0x27u ||
            op == 0x29u || op == 0x2Du || op == 0x2Eu)
        {
            usage.accRead = dest;
        }
        return usage;
    }

    if (op >= 0x3Cu)
    {
        const uint8_t special = static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu));
        const bool writesAcc =
            special <= 0x0Fu ||
            (special >= 0x18u && special <= 0x1Cu) ||
            special == 0x1Eu ||
            (special >= 0x20u && special <= 0x2Au) ||
            (special >= 0x2Cu && special <= 0x2Eu);
        if (writesAcc)
        {
            addVfRead(usage, fs, dest);
            if (special <= 0x1Bu)
                addVfRead(usage, ft, laneForComponent(special & 3u));
            else if ((special >= 0x28u && special <= 0x2Eu))
                addVfRead(usage, ft, special == 0x2Eu ? 0xEu : dest);
            usage.accWrite = dest;
            if ((special >= 0x08u && special <= 0x0Fu) ||
                special == 0x21u || special == 0x23u || special == 0x25u ||
                special == 0x27u || special == 0x29u || special == 0x2Du)
            {
                usage.accRead = dest;
            }
        }
        else if (special >= 0x10u && special <= 0x17u)
        {
            addVfRead(usage, fs, dest);
            addVfWrite(usage, ft, dest);
        }
        else if (special == 0x1Du)
        {
            addVfRead(usage, fs, dest);
            addVfWrite(usage, ft, dest);
        }
        else if (special == 0x1Fu)
        {
            addVfRead(usage, fs, 0xEu);
            addVfRead(usage, ft, 0x1u);
            usage.writesClip = true;
        }
        else if (special != 0x2Fu && special != 0x30u)
        {
            usage.reserved = true;
        }
        return usage;
    }

    usage.reserved = true;
    return usage;
}

VU1Interpreter::InstructionUsage VU1Interpreter::decodeLowerUsage(uint32_t lower) const
{
    InstructionUsage usage;
    if (lower == 0u || lower == 0x8000033Cu)
        return usage;

    const uint8_t opHi = static_cast<uint8_t>((lower >> 25) & 0x7Fu);
    const uint8_t vfT = FT(lower);
    const uint8_t vfS = FS(lower);
    const uint8_t viT = VIT(lower);
    const uint8_t viS = VIS(lower);
    const uint8_t viD = VID(lower);
    const uint8_t dest = DEST(lower);
    auto readVi = [&](uint8_t reg)
    {
        if (reg != 0u)
            usage.viRead |= static_cast<uint16_t>(1u << reg);
    };
    auto writeVi = [&](uint8_t reg)
    {
        if (reg != 0u)
            usage.viWrite |= static_cast<uint16_t>(1u << reg);
    };

    switch (opHi)
    {
    case 0x00:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        readVi(viS);
        addVfWrite(usage, vfT, dest);
        return usage;
    case 0x01:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        readVi(viT);
        addVfRead(usage, vfS, dest);
        return usage;
    case 0x04:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x05:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        readVi(viS);
        readVi(viT);
        return usage;
    case 0x08:
    case 0x09:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x10:
    case 0x12:
    case 0x13:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.readsClip = true;
        writeVi(1u);
        return usage;
    case 0x11:
        usage.pipeline = PipelineFmac;
        usage.latency = kFmacLatency;
        usage.writesClip = true;
        return usage;
    case 0x14:
    case 0x16:
    case 0x17:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        writeVi(viT);
        return usage;
    case 0x15:
        usage.pipeline = PipelineFmac;
        usage.latency = kFmacLatency;
        return usage;
    case 0x18:
    case 0x1A:
    case 0x1B:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x1C:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.readsClip = true;
        writeVi(viT);
        return usage;
    case 0x20:
        usage.pipeline = PipelineBranch;
        return usage;
    case 0x21:
        usage.pipeline = PipelineBranch;
        usage.latency = 1u;
        writeVi(viT);
        return usage;
    case 0x24:
        usage.pipeline = PipelineBranch;
        readVi(viS);
        return usage;
    case 0x25:
        usage.pipeline = PipelineBranch;
        usage.latency = 1u;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x28:
    case 0x29:
        usage.pipeline = PipelineBranch;
        readVi(viS);
        readVi(viT);
        return usage;
    case 0x2C:
    case 0x2D:
    case 0x2E:
    case 0x2F:
        usage.pipeline = PipelineBranch;
        readVi(viS);
        return usage;
    case 0x40:
        break;
    default:
        usage.reserved = true;
        return usage;
    }

    const uint8_t direct = static_cast<uint8_t>(lower & 0x3Fu);
    if (direct == 0x30u || direct == 0x31u || direct == 0x34u || direct == 0x35u)
    {
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        readVi(viT);
        writeVi(viD);
        return usage;
    }
    if (direct == 0x32u)
    {
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        writeVi(viT);
        return usage;
    }
    if (direct < 0x3Cu)
    {
        usage.reserved = true;
        return usage;
    }

    const uint8_t special = static_cast<uint8_t>((lower & 3u) | ((lower >> 4) & 0x7Cu));
    switch (special)
    {
    case 0x30:
    case 0x31:
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        addVfRead(usage, vfS, special == 0x31u ? 0xFu : dest);
        addVfWrite(usage, vfT, dest);
        break;
    case 0x34:
    case 0x36:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        usage.viLatency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        writeVi(viS);
        addVfWrite(usage, vfT, dest);
        break;
    case 0x35:
    case 0x37:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viT);
        writeVi(viT);
        addVfRead(usage, vfS, dest);
        break;
    case 0x38:
        usage.pipeline = PipelineFdiv;
        usage.latency = 7u;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        addVfRead(usage, vfT, laneForComponent((lower >> 23) & 3u));
        break;
    case 0x39:
        usage.pipeline = PipelineFdiv;
        usage.latency = 7u;
        addVfRead(usage, vfT, laneForComponent((lower >> 23) & 3u));
        break;
    case 0x3A:
        usage.pipeline = PipelineFdiv;
        usage.latency = 13u;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        addVfRead(usage, vfT, laneForComponent((lower >> 23) & 3u));
        break;
    case 0x3B:
        usage.pipeline = PipelineFdiv;
        usage.waitQ = true;
        break;
    case 0x3C:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        writeVi(viT);
        break;
    case 0x3D:
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        readVi(viS);
        addVfWrite(usage, vfT, dest);
        break;
    case 0x3E:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        readVi(viS);
        writeVi(viT);
        break;
    case 0x3F:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        readVi(viS);
        readVi(viT);
        break;
    case 0x40:
    case 0x41:
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        addVfWrite(usage, vfT, dest);
        break;
    case 0x42:
    case 0x43:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        break;
    case 0x64:
        if (m_unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        addVfWrite(usage, vfT, dest);
        break;
    case 0x68:
    case 0x69:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        writeVi(viT);
        break;
    case 0x6C:
        if (m_unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineXgkick;
        usage.latency = 2u;
        readVi(viS);
        break;
    case 0x70:
    case 0x71:
    case 0x72:
    case 0x73:
    case 0x74:
    case 0x75:
    case 0x76:
    case 0x77:
    case 0x78:
    case 0x79:
    case 0x7A:
    case 0x7C:
    case 0x7D:
        if (m_unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineEfu;
        switch (special)
        {
        case 0x70:
            usage.latency = 11u;
            break;
        case 0x71:
        case 0x72:
        case 0x77:
            usage.latency = 18u;
            break;
        case 0x73:
            usage.latency = 24u;
            break;
        case 0x74:
        case 0x75:
        case 0x7C:
            usage.latency = 54u;
            break;
        case 0x76:
        case 0x78:
        case 0x7A:
            usage.latency = 12u;
            break;
        case 0x79:
            usage.latency = 29u;
            break;
        case 0x7D:
            usage.latency = 44u;
            break;
        default:
            break;
        }
        if (special >= 0x70u && special <= 0x73u)
            addVfRead(usage, vfS, 0xEu);
        else if (special == 0x74u)
            addVfRead(usage, vfS, 0xCu);
        else if (special == 0x75u)
            addVfRead(usage, vfS, 0xAu);
        else if (special == 0x76u)
            addVfRead(usage, vfS, 0xFu);
        else
            addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        break;
    case 0x7B:
        if (m_unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineEfu;
        usage.waitP = true;
        break;
    default:
        usage.reserved = true;
        break;
    }
    return usage;
}

VU1Interpreter::DecodedInstructionPair VU1Interpreter::decodeInstructionPair(const uint8_t *vuCode, uint32_t pc) const
{
    DecodedInstructionPair decoded;
    std::memcpy(&decoded.lower, vuCode + pc, sizeof(decoded.lower));
    std::memcpy(&decoded.upper, vuCode + pc + sizeof(decoded.lower), sizeof(decoded.upper));
    decoded.iBit = (decoded.upper & 0x80000000u) != 0u;
    decoded.eBit = (decoded.upper & 0x40000000u) != 0u;
    decoded.mBit = (decoded.upper & 0x20000000u) != 0u;
    decoded.dBit = (decoded.upper & 0x10000000u) != 0u;
    decoded.tBit = (decoded.upper & 0x08000000u) != 0u;
    decoded.upperUsage = decodeUpperUsage(decoded.upper);
    if (!decoded.iBit)
        decoded.lowerUsage = decodeLowerUsage(decoded.lower);

    for (uint32_t reg = 1; reg < 16u; ++reg)
    {
        if ((decoded.lowerUsage.viWrite & (1u << reg)) != 0u)
        {
            decoded.lowerViWriteReg = static_cast<uint8_t>(reg);
            break;
        }
    }

    const uint8_t upperWriteReg = decoded.upperUsage.vfWrite.reg;
    if (upperWriteReg != 0u && (vfReadLanes(decoded.lowerUsage, upperWriteReg) != 0u || decoded.lowerUsage.vfWrite.reg == upperWriteReg))
    {
        decoded.upperVfShadowReg = upperWriteReg;
        if (decoded.lowerUsage.vfWrite.reg == upperWriteReg)
            decoded.suppressedLowerVf = upperWriteReg;
    }

    for (const InstructionUsage *usage : {&decoded.upperUsage, &decoded.lowerUsage})
    {
        for (uint32_t index = 0; index < usage->vfReadCount; ++index)
        {
            const VfAccess &access = usage->vfRead[index];
            if (access.reg != 0u && access.lanes != 0u && decoded.stallReadCount < decoded.stallReads.size())
                decoded.stallReads[decoded.stallReadCount++] = access;
        }
        decoded.stallViReads |= static_cast<uint16_t>(usage->viRead & 0xFFFEu);
        decoded.stallAccReads |= usage->accRead;
    }
    const VfAccess lowerWrite = decoded.lowerUsage.vfWrite;
    if (lowerWrite.reg != 0u && decoded.suppressedLowerVf != lowerWrite.reg)
    {
        decoded.markLowerVf = lowerWrite;
        decoded.markLowerVfLatency = decoded.lowerUsage.vfLatency != 0u ? decoded.lowerUsage.vfLatency : decoded.lowerUsage.latency;
    }
    if (decoded.upperUsage.vfWrite.reg != 0u)
    {
        decoded.markUpperVf = decoded.upperUsage.vfWrite;
        decoded.markUpperVfLatency = decoded.upperUsage.vfLatency != 0u ? decoded.upperUsage.vfLatency : decoded.upperUsage.latency;
    }
    decoded.markViWrites = static_cast<uint16_t>(decoded.lowerUsage.viWrite & 0xFFFEu);
    decoded.markViLatency = decoded.lowerUsage.viLatency != 0u ? decoded.lowerUsage.viLatency : decoded.lowerUsage.latency;
    decoded.markAccWrites = decoded.upperUsage.accWrite;
    return decoded;
}

namespace
{
    // FSEQ / FSAND / FSOR / FMEQ / FMAND / FMOR in the lower slot (an I-bit pair's lower word is an immediate)
    bool pairReadsMacStatus(uint32_t lower, bool iBit)
    {
        if (iBit)
            return false;
        const uint32_t opHi = (lower >> 25) & 0x7Fu;
        return opHi == 0x14u || opHi == 0x16u || opHi == 0x17u || opHi == 0x18u || opHi == 0x1Au || opHi == 0x1Bu;
    }

    struct RegisteredVu1Piece
    {
        const Vu1CompiledPiece *piece = nullptr;
        int state = 0; // 0 = not checked yet, 1 = its decode matches the interpreter's, -1 = rejected
        bool announced = false;
    };

    std::vector<RegisteredVu1Piece> &registeredVu1Pieces()
    {
        static std::vector<RegisteredVu1Piece> pieces;
        return pieces;
    }

    // Does the piece's code stand in VU memory now (and does it fit)?
    bool pieceInPlace(const Vu1CompiledPiece &piece, const uint8_t *vuCode, uint32_t codeSize, uint32_t maxPairs)
    {
        const uint32_t bytes = piece.pairCount * 8u;
        return piece.vuAddress % 8u == 0u && piece.vuAddress + bytes <= codeSize &&
               piece.vuAddress / 8u + piece.pairCount <= maxPairs &&
               std::memcmp(vuCode + piece.vuAddress, piece.code, bytes) == 0;
    }

    // The compiled entries for the pairs [first, last) after a write there, as refreshCompiledEntries would set them:
    // the written pairs and all of every piece over them (it may have started or stopped matching) are cleared, then
    // every matching piece reaching into that range sets its entries there, in registration order. Elsewhere no piece's
    // match changed, so a full refresh would leave the same entries. accept(entry) = the piece is usable (checks a new
    // one's decode; prints as a full refresh does).
    template <size_t N, typename Accept>
    void refreshCompiledRange(std::array<Vu1CompiledFn, N> &fnAt, const uint8_t *vuCode, uint32_t codeSize,
                              uint32_t first, uint32_t last, Accept &&accept)
    {
        uint32_t low = first, high = last;
        for (const RegisteredVu1Piece &entry : registeredVu1Pieces())
        {
            const uint32_t start = entry.piece->vuAddress / 8u;
            const uint32_t stop = start + entry.piece->pairCount;
            if (start < last && stop > first)
            {
                low = std::min(low, start);
                high = std::max(high, std::min<uint32_t>(stop, static_cast<uint32_t>(N)));
            }
        }
        std::fill(fnAt.begin() + low, fnAt.begin() + high, nullptr);
        for (RegisteredVu1Piece &entry : registeredVu1Pieces())
        {
            const Vu1CompiledPiece &piece = *entry.piece;
            const uint32_t start = piece.vuAddress / 8u;
            if (start >= high || start + piece.pairCount <= low || entry.state < 0 ||
                !pieceInPlace(piece, vuCode, codeSize, static_cast<uint32_t>(N)) || !accept(entry))
                continue;
            for (uint32_t index = 0; index < piece.pairCount; ++index)
            {
                const uint32_t at = start + index;
                if (at >= low && at < high && piece.fns[index] != nullptr)
                    fnAt[at] = piece.fns[index];
            }
        }
    }
}

void VU1Interpreter::rebuildDecodedCodeCache(const uint8_t *vuCode, uint32_t codeSize,
                                             const PS2Memory *memory, uint64_t generation)
{
    ps2_host_profile::count(ps2_host_profile::Vu1DecodeFull);
    const uint32_t pairCount = std::min<uint32_t>(codeSize / 8u, kMaxDecodedPairs);
    bool readsMacStatus = false;
    for (uint32_t i = 0; i < pairCount; ++i)
    {
        m_decodedCodeCache[i] = decodeInstructionPair(vuCode, i * 8u);
        readsMacStatus = readsMacStatus || pairReadsMacStatus(m_decodedCodeCache[i].lower, m_decodedCodeCache[i].iBit);
    }
    m_codeReadsMacStatus = readsMacStatus;

    m_cachedVuCode = vuCode;
    m_cachedMemory = memory;
    m_cachedCodeSize = codeSize;
    m_cachedCodeGeneration = generation;
    m_decodedCodeCacheValid = true;
    refreshCompiledEntries(vuCode, codeSize);
}

bool VU1Interpreter::ensureDecodedCodeCache(const uint8_t *vuCode, uint32_t codeSize, PS2Memory *memory)
{
    const bool trackedCode = memory != nullptr &&
                             ((m_unit == Unit::VU1 && vuCode == memory->getVU1Code()) ||
                              (m_unit == Unit::VU0 && vuCode == memory->getVU0Code()));
    if (!trackedCode)
        return false;

    const uint64_t generation = m_unit == Unit::VU1 ? memory->getVU1CodeGeneration() : memory->getVU0CodeGeneration();
    if (!m_decodedCodeCacheValid ||
        m_cachedVuCode != vuCode ||
        m_cachedMemory != memory ||
        m_cachedCodeSize != codeSize)
    {
        rebuildDecodedCodeCache(vuCode, codeSize, memory, generation);
    }
    else if (m_cachedCodeGeneration != generation)
    {
        // VU1 code memory keeps a log of what was written (ps2_vu_code_log): re-decode only those pairs, all of them
        // only when the log no longer reaches back. The game uploads its programs many times a frame; decoding all
        // 2,048 pairs each time was ~10% of the chain worker in a 4-player match.
        uint32_t begin = 0u, end = 0u;
        if (m_unit == Unit::VU1 && ps2_vu_code_log::writtenSince(memory, m_cachedCodeGeneration, generation, begin, end))
        {
            ps2_host_profile::count(ps2_host_profile::Vu1DecodePartial);
            const uint32_t pairCount = std::min<uint32_t>(codeSize / 8u, kMaxDecodedPairs);
            const uint32_t first = std::min(begin / 8u, pairCount);
            const uint32_t last = std::min((end + 7u) / 8u, pairCount);
            for (uint32_t i = first; i < last; ++i)
                m_decodedCodeCache[i] = decodeInstructionPair(vuCode, i * 8u);
            bool readsMacStatus = false;
            for (uint32_t i = 0; i < pairCount && !readsMacStatus; ++i)
                readsMacStatus = pairReadsMacStatus(m_decodedCodeCache[i].lower, m_decodedCodeCache[i].iBit);
            m_codeReadsMacStatus = readsMacStatus;
            m_cachedCodeGeneration = generation;
            if (first < last && m_compiledEnabled && m_unit == Unit::VU1)
                refreshCompiledRange(m_compiledFnAt, vuCode, codeSize, first, last, [this](RegisteredVu1Piece &entry)
                                     {
                                         // as refreshCompiledEntries: a piece's decode is checked the first time it matches
                                         const Vu1CompiledPiece &piece = *entry.piece;
                                         if (entry.state == 0)
                                         {
                                             uint32_t badPair = 0u;
                                             entry.state = compiledShapesMatch(piece, badPair) ? 1 : -1;
                                             if (entry.state < 0)
                                             {
                                                 std::fprintf(stderr, "[vu1compiled] %s: the pair at VU 0x%04x decodes differently from the interpreter - not used%c",
                                                              piece.name, piece.vuAddress + badPair * 8u, 10);
                                                 return false;
                                             }
                                         }
                                         if (!entry.announced)
                                         {
                                             std::fprintf(stderr, "[vu1compiled] running %s compiled (%u pairs at VU 0x%04x)%c",
                                                          piece.name, piece.pairCount, piece.vuAddress, 10);
                                             entry.announced = true;
                                         }
                                         return true;
                                     });
        }
        else
        {
            rebuildDecodedCodeCache(vuCode, codeSize, memory, generation);
        }
        static const bool verify = []
        {
            const char *value = std::getenv("VENDETTA_VU1_DECODE_VERIFY");
            return value && *value && *value != '0';
        }();
        if (verify)
        {
            // against a full rebuild: decoding is a function of the pair's two words and its address, so equal words
            // mean equal entries; the MAC flag; the compiled entries a full refresh would set (computed aside)
            const uint32_t pairCount = std::min<uint32_t>(codeSize / 8u, kMaxDecodedPairs);
            uint32_t mismatches = 0u;
            bool readsMacStatus = false;
            for (uint32_t i = 0; i < pairCount; ++i)
            {
                uint32_t lower = 0u, upper = 0u;
                std::memcpy(&lower, vuCode + i * 8u, 4u);
                std::memcpy(&upper, vuCode + i * 8u + 4u, 4u);
                if (m_decodedCodeCache[i].lower != lower || m_decodedCodeCache[i].upper != upper)
                    ++mismatches;
                readsMacStatus = readsMacStatus || pairReadsMacStatus(lower, (upper & 0x80000000u) != 0u);
            }
            if (readsMacStatus != m_codeReadsMacStatus)
                ++mismatches;
            if (m_compiledEnabled && m_unit == Unit::VU1)
            {
                std::array<Vu1CompiledFn, kMaxDecodedPairs> expected{};
                refreshCompiledRange(expected, vuCode, codeSize, 0u, static_cast<uint32_t>(kMaxDecodedPairs),
                                     [this](RegisteredVu1Piece &entry)
                                     {
                                         uint32_t badPair = 0u;
                                         return entry.state > 0 || (entry.state == 0 && compiledShapesMatch(*entry.piece, badPair));
                                     });
                for (uint32_t i = 0; i < kMaxDecodedPairs; ++i)
                    if (expected[i] != m_compiledFnAt[i])
                        ++mismatches;
            }
            if (mismatches != 0u)
            {
                ps2_host_profile::count(ps2_host_profile::Vu1DecodeVerifyMismatch, mismatches);
                static uint32_t reports = 0u;
                if (reports++ < 16u)
                    std::fprintf(stderr, "[vu1-decode-verify] %u differences from a full rebuild%c", mismatches, 10);
            }
        }
    }
    return true;
}

const VU1Interpreter::DecodedInstructionPair &VU1Interpreter::getDecodedInstructionPairForPc(
    const uint8_t *vuCode, uint32_t codeSize, PS2Memory *memory, uint32_t pc)
{
    if ((pc & 7u) != 0u || !ensureDecodedCodeCache(vuCode, codeSize, memory))
        return m_decodeScratch = decodeInstructionPair(vuCode, pc);

    const uint32_t pairIndex = pc / 8u;
    if (pairIndex >= kMaxDecodedPairs)
        return m_decodeScratch = decodeInstructionPair(vuCode, pc);
    return m_decodedCodeCache[pairIndex];
}

void VU1Interpreter::reportReservedInstruction(bool upper, uint32_t instruction)
{
    RUNTIME_ERROR(
        "[VU" << (m_unit == Unit::VU1 ? "1" : "0")
              << " reserved " << (upper ? "upper" : "lower")
              << "] cycle=" << m_cycle
              << " pc=0x" << std::hex << m_state.pc
              << " instruction=0x" << instruction
              << std::dec << '\n');
    m_stopRequested = true;
}

void VU1Interpreter::execute(uint8_t *vuCode, uint32_t codeSize,
                             uint8_t *vuData, uint32_t dataSize,
                             GS &gs, PS2Memory *memory,
                             uint32_t startPC, uint32_t top, uint32_t itop,
                             uint32_t maxCycles)
{
    resetScheduler();
    m_state.pc = startPC & microAddressMask();
    m_state.ebit = false;
    m_state.haltAfterDelaySlot = false;
    m_state.stoppedByD = false;
    m_state.stoppedByT = false;
    m_state.top = top;
    m_state.itop = itop;
    m_state.branchPending = false;
    m_state.branchTarget = 0;
    m_state.branchDelay = 0;
    m_state.vf[0][0] = 0.0f;
    m_state.vf[0][1] = 0.0f;
    m_state.vf[0][2] = 0.0f;
    m_state.vf[0][3] = 1.0f;
    run(vuCode, codeSize, vuData, dataSize, gs, memory, maxCycles);
}

void VU1Interpreter::resume(uint8_t *vuCode, uint32_t codeSize,
                            uint8_t *vuData, uint32_t dataSize,
                            GS &gs, PS2Memory *memory,
                            uint32_t top, uint32_t itop, uint32_t maxCycles)
{
    m_state.top = top;
    m_state.itop = itop;
    m_state.stoppedByD = false;
    m_state.stoppedByT = false;
    run(vuCode, codeSize, vuData, dataSize, gs, memory, maxCycles);
}

void VU1Interpreter::run(uint8_t *vuCode, uint32_t codeSize,
                         uint8_t *vuData, uint32_t dataSize,
                         GS &gs, PS2Memory *memory, uint32_t maxCycles)
{
    static const bool verify = std::getenv("VENDETTA_VU1_VERIFY") != nullptr;
    if (verify && m_unit == Unit::VU1 && !m_verifying)
    {
        runVerified(vuCode, codeSize, vuData, dataSize, gs, memory, maxCycles);
        return;
    }
    runLoop(vuCode, codeSize, vuData, dataSize, gs, memory, maxCycles);
}

// Apply every queued VF / VI / ACC write now (the latest per lane wins, as when they commit). Their ready cycles
// stay in m_vfReady / m_viReady / m_accReady, so readers still stall until then.
void VU1Interpreter::drainRegisterWrites()
{
    for (PendingVfWrite &write : m_vfWritePipeline)
    {
        if (!write.valid)
            continue;
        for (uint32_t component = 0; component < 4u; ++component)
            if ((write.laneMask & laneForComponent(component)) != 0u &&
                m_vfLatestWrite[write.reg][component] == write.sequence)
                m_state.vf[write.reg][component] = write.value[component];
        write.clear();
    }
    for (PendingViWrite &write : m_viWritePipeline)
    {
        if (!write.valid)
            continue;
        if (m_viLatestWrite[write.reg] == write.sequence)
            m_state.vi[write.reg] = static_cast<int16_t>(write.value);
        write.clear();
    }
    for (PendingAccWrite &write : m_accWritePipeline)
    {
        if (!write.valid)
            continue;
        for (uint32_t component = 0; component < 4u; ++component)
            if ((write.laneMask & laneForComponent(component)) != 0u &&
                m_accLatestWrite[component] == write.sequence)
                m_state.acc[component] = write.value[component];
        write.clear();
    }
}

// VENDETTA_VU1_VERIFY=1: run the program on a copy with the original (queued) model and on this unit with the
// current mode, then compare registers, VU data memory and the XGKICK packets; report differences on stderr.
void VU1Interpreter::runVerified(uint8_t *vuCode, uint32_t codeSize,
                                 uint8_t *vuData, uint32_t dataSize,
                                 GS &gs, PS2Memory *memory, uint32_t maxCycles)
{
    static uint32_t programs = 0u, mismatches = 0u;
    const uint32_t startPc = m_state.pc;
    auto reference = std::make_unique<VU1Interpreter>(*this);
    reference->m_fastMode = false;
    reference->m_verifying = true;
    std::vector<uint8_t> referenceData(vuData, vuData + dataSize);
    std::vector<std::vector<uint8_t>> referenceKicks, kicks;
    reference->m_kickCapture = &referenceKicks;
    reference->runLoop(vuCode, codeSize, referenceData.data(), dataSize, gs, memory, maxCycles);

    m_verifying = true;
    m_kickCapture = &kicks;
    runLoop(vuCode, codeSize, vuData, dataSize, gs, memory, maxCycles);
    m_kickCapture = nullptr;
    m_verifying = false;
    for (const std::vector<uint8_t> &packet : kicks)
    {
        if (m_activeMemory)
            m_activeMemory->submitGifPacket(GifPathId::Path1, packet.data(), static_cast<uint32_t>(packet.size()));
        else if (m_activeGs)
            m_activeGs->processGIFPacket(packet.data(), static_cast<uint32_t>(packet.size()), 1u);
    }

    ++programs;
    std::string what;
    const VU1State &a = reference->m_state;
    const VU1State &b = m_state;
    for (uint32_t r = 0; r < 32u && what.empty(); ++r)
        if (std::memcmp(a.vf[r], b.vf[r], sizeof(a.vf[r])) != 0)
            what = "vf" + std::to_string(r);
    for (uint32_t r = 0; r < 16u && what.empty(); ++r)
        if (static_cast<int16_t>(a.vi[r]) != static_cast<int16_t>(b.vi[r]))
            what = "vi" + std::to_string(r);
    if (what.empty() && std::memcmp(a.acc, b.acc, sizeof(a.acc)) != 0)
        what = "acc";
    if (what.empty() && std::memcmp(&a.q, &b.q, sizeof(a.q)) != 0)
        what = "q";
    if (what.empty() && a.clip != b.clip)
        what = "clip";
    if (what.empty() && (a.pc != b.pc || a.ebit != b.ebit))
        what = "pc";
    if (what.empty() && reference->m_cycle != m_cycle)
        what = "cycles " + std::to_string(reference->m_cycle) + " vs " + std::to_string(m_cycle);
    if (what.empty() && std::memcmp(referenceData.data(), vuData, dataSize) != 0)
    {
        uint32_t at = 0u;
        while (at < dataSize && referenceData[at] == vuData[at])
            ++at;
        what = "data@0x" + std::to_string(at);
    }
    if (what.empty() && referenceKicks != kicks)
        what = "xgkick packets (" + std::to_string(referenceKicks.size()) + " vs " + std::to_string(kicks.size()) + ")";
    if (!what.empty() && ++mismatches <= 40u)
        std::fprintf(stderr, "[vu1verify] program %u (start pc 0x%x): differs in %s%c", programs, startPc, what.c_str(), 10);
    if ((programs & 0xFFFu) == 0u)
        std::fprintf(stderr, "[vu1verify] %u programs checked, %u differed%c", programs, mismatches, 10);
}

void VU1Interpreter::runLoop(uint8_t *vuCode, uint32_t codeSize,
                             uint8_t *vuData, uint32_t dataSize,
                             GS &gs, PS2Memory *memory, uint32_t maxCycles)
{
    ps2_host_profile::Scope profile(ps2_host_profile::Vu1);
    m_activeVuData = vuData;
    m_activeVuDataSize = dataSize;
    m_activeGs = &gs;
    m_activeMemory = memory;

    const int previousRoundingMode = std::fegetround();
    const int outerCallerRoundingMode = t_callerRoundingMode;
    if (previousRoundingMode != -1)
        t_callerRoundingMode = previousRoundingMode;
    const bool useVuRounding = std::fesetround(FE_TOWARDZERO) == 0;
    if (m_fastMode)
        drainRegisterWrites();
    const uint64_t budgetEnd = m_cycle + maxCycles;
    const uint64_t startCycle = m_cycle;
    uint64_t pairsRun = 0u;
    uint64_t compiledPairsRun = 0u;
    bool programEnded = false;
    // compiled VU1 code (ps2_vu1_compiled.h): fast mode only, on VU1's own (tracked) 16 KB code / data memory
    const bool compiledAllowed = m_compiledEnabled && m_fastMode && m_unit == Unit::VU1 && memory != nullptr &&
                                 vuCode == memory->getVU1Code() && codeSize == 0x4000u && dataSize == 0x4000u;
    while (m_cycle < budgetEnd && !m_stopRequested)
    {
        commitReadyPipelines();
        if (m_state.pc + 8u > codeSize)
            break;

        // Enter the compiled code at a compiled pair when no branch / E-bit / halt is pending; it runs until it
        // leaves its code, runs out of budget or ends the program, leaving the state as this loop would.
        if (compiledAllowed && !m_state.branchPending && !m_state.ebit && !m_state.haltAfterDelaySlot &&
            (m_state.pc & 7u) == 0u && ensureDecodedCodeCache(vuCode, codeSize, memory) && !m_codeReadsMacStatus)
        {
            if (const Vu1CompiledFn compiledFn = m_compiledFnAt[m_state.pc >> 3])
            {
                const uint32_t entryPc = m_state.pc;
                const uint64_t entryPairs = pairsRun;
                const Vu1CompiledResult result = compiledFn(*this, budgetEnd, pairsRun);
                compiledPairsRun += pairsRun - entryPairs;
                if (result == Vu1CompiledResult::ProgramEnded)
                {
                    programEnded = true;
                    break;
                }
                if (pairsRun != entryPairs || m_state.pc != entryPc || m_cycle >= budgetEnd || m_stopRequested)
                    continue;
                // nothing ran (not expected): the interpreter runs this pair
            }
        }

        const DecodedInstructionPair &decoded = getDecodedInstructionPairForPc(vuCode, codeSize, memory, m_state.pc);
        if (decoded.upperUsage.reserved || decoded.lowerUsage.reserved)
        {
            reportReservedInstruction(decoded.upperUsage.reserved, decoded.upperUsage.reserved ? decoded.upper : decoded.lower);
            break;
        }

        uint64_t readyCycle = calculatePairReadyCycle(decoded);
        while (readyCycle > m_cycle)
        {
            if (readyCycle >= budgetEnd)
            {
                advanceTo(budgetEnd);
                break;
            }
            advanceTo(readyCycle);
            readyCycle = calculatePairReadyCycle(decoded);
        }
        if (m_cycle >= budgetEnd)
            break;

        const uint8_t writtenVi = decoded.lowerViWriteReg;
        const int32_t oldVi = writtenVi != 0u ? m_state.vi[writtenVi] : 0;

        const VfAccess upperWrite = decoded.upperUsage.vfWrite;
        const VfAccess lowerWrite = decoded.lowerUsage.vfWrite;
        const bool hasUpperWrite = upperWrite.reg != 0u;
        const bool hasLowerWrite = lowerWrite.reg != 0u && decoded.suppressedLowerVf != lowerWrite.reg;
        const bool hasDistinctLowerWrite = hasLowerWrite && (!hasUpperWrite || lowerWrite.reg != upperWrite.reg);
        float oldUpperVf[4]{};
        float newUpperVf[4]{};
        float oldLowerVf[4]{};
        float newLowerVf[4]{};
        float oldAcc[4]{};
        float newAcc[4]{};
        if (!m_fastMode)
        {
            if (hasUpperWrite)
                std::memcpy(oldUpperVf, m_state.vf[upperWrite.reg], sizeof(oldUpperVf));
            if (hasDistinctLowerWrite)
                std::memcpy(oldLowerVf, m_state.vf[lowerWrite.reg], sizeof(oldLowerVf));
            if (decoded.upperUsage.accWrite != 0u)
                std::memcpy(oldAcc, m_state.acc, sizeof(oldAcc));
        }

        if (decoded.iBit)
        {
            runUpper(decoded.upper);
            float immediate = 0.0f;
            std::memcpy(&immediate, &decoded.lower, sizeof(immediate));
            m_state.i = normalizeOperand(immediate);
        }
        else if (decoded.upperVfShadowReg != 0u)
        {
            float oldVf[4]{};
            float upperVf[4]{};
            std::memcpy(oldVf,
                        m_state.vf[decoded.upperVfShadowReg],
                        sizeof(oldVf));
            runUpper(decoded.upper);
            std::memcpy(upperVf,
                        m_state.vf[decoded.upperVfShadowReg],
                        sizeof(upperVf));
            std::memcpy(m_state.vf[decoded.upperVfShadowReg],
                        oldVf,
                        sizeof(oldVf));
            execLower(decoded.lower, vuData, dataSize, gs, memory, decoded.upper);
            std::memcpy(m_state.vf[decoded.upperVfShadowReg],
                        upperVf,
                        sizeof(upperVf));
        }
        else
        {
            runUpper(decoded.upper);
            execLower(decoded.lower, vuData, dataSize, gs, memory, decoded.upper);
        }

        m_viBranchBackupValid = false;

        if (!m_fastMode)
        {
            if (hasUpperWrite)
            {
                std::memcpy(newUpperVf, m_state.vf[upperWrite.reg], sizeof(newUpperVf));
                std::memcpy(m_state.vf[upperWrite.reg], oldUpperVf, sizeof(oldUpperVf));
                const uint32_t latency =
                    decoded.upperUsage.vfLatency != 0u
                        ? decoded.upperUsage.vfLatency
                        : decoded.upperUsage.latency;
                queueVfWrite(upperWrite.reg, upperWrite.lanes, newUpperVf, latency);
            }
            if (hasDistinctLowerWrite)
            {
                std::memcpy(newLowerVf, m_state.vf[lowerWrite.reg], sizeof(newLowerVf));
                std::memcpy(m_state.vf[lowerWrite.reg], oldLowerVf, sizeof(oldLowerVf));
                const uint32_t latency = decoded.lowerUsage.vfLatency != 0u
                                             ? decoded.lowerUsage.vfLatency
                                             : decoded.lowerUsage.latency;
                queueVfWrite(lowerWrite.reg, lowerWrite.lanes, newLowerVf, latency);
            }
            if (decoded.upperUsage.accWrite != 0u)
            {
                std::memcpy(newAcc, m_state.acc, sizeof(newAcc));
                std::memcpy(m_state.acc, oldAcc, sizeof(oldAcc));
                // ACC is forwarded to the next upper instruction. Its arithmetic
                // flags still use the normal four-cycle FMAC timeline.
                queueAccWrite(decoded.upperUsage.accWrite, newAcc,
                              kAccForwardLatency);
            }
            if (writtenVi != 0u)
            {
                const int32_t newVi = m_state.vi[writtenVi];
                m_state.vi[writtenVi] = oldVi;
                const uint32_t latency =
                    decoded.lowerUsage.viLatency != 0u
                        ? decoded.lowerUsage.viLatency
                        : decoded.lowerUsage.latency;
                queueViWrite(writtenVi, newVi, latency);
            }
        }

        markPairWrites(decoded);
        if (writtenVi != 0u && decoded.lowerUsage.delaysNextBranchRead)
            recordViWriteForBranch(writtenVi, oldVi);

        m_state.vf[0][0] = 0.0f;
        m_state.vf[0][1] = 0.0f;
        m_state.vf[0][2] = 0.0f;
        m_state.vf[0][3] = 1.0f;
        m_state.vi[0] = 0;

        uint32_t nextPc = m_state.pc + 8u;
        if (nextPc >= codeSize)
            nextPc = 0u;
        m_state.pc = nextPc;

        if (m_state.branchPending)
        {
            if (m_state.branchDelay == 0u)
            {
                m_state.pc = m_state.branchTarget & microAddressMask();
                m_state.branchPending = false;
            }
            else
            {
                --m_state.branchDelay;
            }
        }

        const bool dHalt = decoded.dBit && m_state.dBitEnabled;
        const bool tHalt = decoded.tBit && m_state.tBitEnabled;
        const bool haltBit = dHalt || tHalt;
        const bool haltBranch = haltBit && decoded.lowerUsage.pipeline == PipelineBranch;

        if (m_state.haltAfterDelaySlot)
        {
            m_state.stoppedByD = m_pendingHaltD;
            m_state.stoppedByT = m_pendingHaltT;
            programEnded = true;
        }
        else if (m_state.ebit)
            programEnded = true;
        else if (haltBit && !haltBranch)
        {
            m_state.stoppedByD = dHalt;
            m_state.stoppedByT = tHalt;
            programEnded = true;
        }
        else if (decoded.eBit)
            m_state.ebit = true;
        else if (haltBranch)
        {
            m_state.haltAfterDelaySlot = true;
            m_pendingHaltD = dHalt;
            m_pendingHaltT = tHalt;
        }

        advanceOneCycle();
        ++pairsRun;
        if (programEnded)
            break;
    }

    if (programEnded)
    {
        flushPipelines();
        m_state.ebit = false;
        m_state.haltAfterDelaySlot = false;
        m_pendingHaltD = false;
        m_pendingHaltT = false;
    }
    m_state.cycles = m_cycle;
    ps2_host_profile::count(ps2_host_profile::Vu1Pairs, pairsRun);
    ps2_host_profile::count(ps2_host_profile::Vu1CompiledPairs, compiledPairsRun);
    ps2_host_profile::count(ps2_host_profile::Vu1Cycles, m_cycle - startCycle);
    if (useVuRounding && previousRoundingMode != -1)
        std::fesetround(previousRoundingMode);
    t_callerRoundingMode = outerCallerRoundingMode;
}

// ============================================================================
// Compiled VU1 code (runtime/ps2_vu1_compiled.h)
// ============================================================================

void vu1RegisterCompiledPieces(const Vu1CompiledPiece *pieces, size_t count)
{
    for (size_t index = 0; index < count; ++index)
        registeredVu1Pieces().push_back({&pieces[index]});
}

// Every pair of the piece as the interpreter decodes it: the stall reads, write marks, latencies and bits the
// generated code was specialised for.
bool VU1Interpreter::compiledShapesMatch(const Vu1CompiledPiece &piece, uint32_t &badPair) const
{
    for (uint32_t index = 0; index < piece.pairCount; ++index)
    {
        const DecodedInstructionPair &decoded = m_decodedCodeCache[piece.vuAddress / 8u + index];
        const Vu1PairShape &shape = piece.shapes[index];
        bool same = decoded.stallReadCount == shape.stallVfCount &&
                    decoded.stallViReads == shape.stallVi &&
                    decoded.stallAccReads == shape.stallAcc &&
                    static_cast<uint8_t>(decoded.lowerUsage.pipeline) == shape.lowerPipeline &&
                    decoded.markLowerVf.reg == shape.markLowerVf[0] && decoded.markLowerVf.lanes == shape.markLowerVf[1] &&
                    decoded.markLowerVfLatency == shape.markLowerVfLatency &&
                    decoded.markUpperVf.reg == shape.markUpperVf[0] && decoded.markUpperVf.lanes == shape.markUpperVf[1] &&
                    decoded.markUpperVfLatency == shape.markUpperVfLatency &&
                    decoded.markViWrites == shape.markVi &&
                    decoded.markViLatency == shape.markViLatency &&
                    decoded.markAccWrites == shape.markAcc &&
                    decoded.lowerViWriteReg == shape.lowerViWriteReg &&
                    decoded.upperVfShadowReg == shape.upperVfShadowReg &&
                    decoded.suppressedLowerVf == shape.suppressedLowerVf;
        const uint8_t flags = static_cast<uint8_t>((decoded.lowerUsage.waitQ ? kShapeWaitQ : 0u) |
                                                   (decoded.lowerUsage.waitP ? kShapeWaitP : 0u) |
                                                   (decoded.iBit ? kShapeIBit : 0u) |
                                                   (decoded.eBit ? kShapeEBit : 0u) |
                                                   (decoded.dBit ? kShapeDBit : 0u) |
                                                   (decoded.tBit ? kShapeTBit : 0u) |
                                                   (decoded.upperUsage.reserved || decoded.lowerUsage.reserved ? kShapeReserved : 0u) |
                                                   (decoded.lowerUsage.delaysNextBranchRead ? kShapeDelaysBranchRead : 0u));
        same = same && flags == shape.flags;
        for (uint32_t read = 0; same && read < decoded.stallReadCount; ++read)
            same = decoded.stallReads[read].reg == shape.stallVf[read][0] && decoded.stallReads[read].lanes == shape.stallVf[read][1];
        if (!same)
        {
            badPair = index;
            return false;
        }
    }
    return true;
}

void VU1Interpreter::refreshCompiledEntries(const uint8_t *vuCode, uint32_t codeSize)
{
    m_compiledFnAt.fill(nullptr);
    if (!m_compiledEnabled || m_unit != Unit::VU1)
        return;
    for (RegisteredVu1Piece &entry : registeredVu1Pieces())
    {
        const Vu1CompiledPiece &piece = *entry.piece;
        const uint32_t bytes = piece.pairCount * 8u;
        if (entry.state < 0 || piece.vuAddress % 8u != 0u || piece.vuAddress + bytes > codeSize ||
            piece.vuAddress / 8u + piece.pairCount > kMaxDecodedPairs ||
            std::memcmp(vuCode + piece.vuAddress, piece.code, bytes) != 0)
            continue;
        if (entry.state == 0)
        {
            uint32_t badPair = 0u;
            entry.state = compiledShapesMatch(piece, badPair) ? 1 : -1;
            if (entry.state < 0)
            {
                std::fprintf(stderr, "[vu1compiled] %s: the pair at VU 0x%04x decodes differently from the interpreter - not used%c",
                             piece.name, piece.vuAddress + badPair * 8u, 10);
                continue;
            }
        }
        if (!entry.announced)
        {
            std::fprintf(stderr, "[vu1compiled] running %s compiled (%u pairs at VU 0x%04x)%c", piece.name, piece.pairCount, piece.vuAddress, 10);
            entry.announced = true;
        }
        for (uint32_t index = 0; index < piece.pairCount; ++index)
            if (piece.fns[index] != nullptr)
                m_compiledFnAt[piece.vuAddress / 8u + index] = piece.fns[index];
    }
}

