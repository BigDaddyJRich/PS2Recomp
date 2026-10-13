#pragma once

#include <cstddef>
#include <cstdint>

// Native hooks: a routine of an IOP module run as C++ instead of interpreted, instruction by instruction — for hot
// leaf routines (a sound driver's mixing loop). A hook names the routine's address and length and a hash of its code
// (not the code itself); it is used only while the IOP's RAM holds exactly that code there, so another game, another
// module version or another load address keeps the interpreter. Hooks are registered before the IOP runs (a game's
// overrides) and stay for the process.
namespace ps2x::iop
{
    // What a hook sees of the IOP at the routine's first instruction: the calling thread's registers and the IOP's RAM.
    class NativeCall
    {
    public:
        virtual ~NativeCall() = default;
        [[nodiscard]] virtual uint32_t reg(uint32_t index) const = 0;
        virtual void setReg(uint32_t index, uint32_t value) = 0;
        [[nodiscard]] virtual bool readRam(uint32_t address, void *destination, size_t size) const = 0;
        virtual bool writeRam(uint32_t address, const void *source, size_t size) = 0;
    };

    struct NativeHook
    {
        const char *name = "";
        uint32_t address = 0;   // the routine's first instruction
        uint32_t codeBytes = 0; // its length; the hash covers [address, address + codeBytes)
        uint64_t codeHash = 0;  // nativeCodeHash of those bytes
        // Runs the routine: memory and registers as the routine leaves them (the IOP then continues at $ra), and the
        // number of IOP instructions the routine would have taken (charged as the IOP's time); 0 = not handled, the
        // IOP interprets the routine.
        uint32_t (*run)(NativeCall &call) = nullptr;
        // VENDETTA_IOP_NATIVE_VERIFY=1: the IOP interprets the routine; check(call, before) is called at its first
        // instruction (before = true: predict, change nothing) and when it has returned to $ra (before = false, with the
        // instructions this thread ran in it): compare. Returns the number of differences found.
        uint32_t (*check)(NativeCall &call, bool before, uint64_t instructions) = nullptr;
    };

    // FNV-1a over the bytes, the hash NativeHook::codeHash holds.
    [[nodiscard]] inline uint64_t nativeCodeHash(const uint8_t *bytes, size_t size) noexcept
    {
        uint64_t hash = 1469598103934665603ull;
        for (size_t i = 0; i < size; ++i)
            hash = (hash ^ bytes[i]) * 1099511628211ull;
        return hash;
    }

    void registerNativeHook(const NativeHook &hook);
}
