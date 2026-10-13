#pragma once

#include <cstdint>

// The pads' motors on the PC's controllers. A DualShock 2 has a small motor (on / off) and a large one (0-255);
// libpad's scePadSetActDirect sends both. On Windows they go to the controller behind the pad (ps2_pad_layout) when it
// is an XInput one (Xbox controllers, and others through Steam Input or DS4Windows): the large motor to its left
// (low-frequency) motor, the small one to its right. Off for a player whose vibration setting is off.
namespace ps2_pad_rumble
{
    void set(int port, int slot, bool smallMotor, uint8_t largeMotor);
    // every motor off (the game closing)
    void stopAll();
}
