#ifndef PS2_PAD_H
#define PS2_PAD_H

#include <cstddef>
#include <cstdint>

class PSPadBackend
{
public:
    PSPadBackend() = default;
    ~PSPadBackend() = default;

    bool readState(int port, int slot, uint8_t *data, size_t size);
};

// Which PC controller is which PS2 pad. Two ports, and on a port with a multitap (ps2_host_settings: the controls'
// multitapPort) four slots, A-D. The PC's controllers fill the pads in a fixed order: the first goes in port 1, the
// next ones in port 2 — slot A, then B, C, D when port 2 has the multitap (as a PS2 with a multitap in port 2 and a
// pad in port 1). With the multitap in port 1 instead: 1A, 1B, 1C, 1D, then 2. The keyboard stands in for the first
// pad while no controller is plugged in.
namespace ps2_pad_layout
{
    constexpr int kPortCount = 2;
    constexpr int kSlotCount = 4; // a multitap's slots A-D

    // 4 on the multitap's port, else 1 (scePadGetSlotMax)
    int slotCount(int port);
    // where the Nth pad goes (0 = the first): false when there is no such place
    bool padPlace(int index, int &port, int &slot);
    // the Nth pad of (port, slot), or -1 when that place isn't used
    int padIndex(int port, int slot);
    // the PC controller (raylib's gamepad number) for (port, slot), or -1 when none is plugged in for it.
    // VENDETTA_PADS=<n> (tests): the first n places count as plugged in, with no controller behind them.
    int hostController(int port, int slot);
    // a controller (or a test's VENDETTA_PADS) is in this place: real pads in slots B-D appear only when plugged in
    bool connected(int port, int slot);
    // When (port, slot)'s controller is an XInput one (GLFW names those "Xbox Controller", "Wireless Xbox
    // Controller" or "XInput ..."): how many XInput controllers come before it (GLFW numbers them in XInput's
    // order), else -1. Its name in name (may be null).
    int xinputOrder(int port, int slot, const char **name = nullptr);
}

#endif
