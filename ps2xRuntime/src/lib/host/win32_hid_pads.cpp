// PlayStation controllers on the native Win32 host, read straight over HID (no DS4Windows / Steam Input needed):
// DualShock 4 (054C:05C4, 09CC, 0BA0 dongle) and DualSense / DualSense Edge (054C:0CE6, 0DF2), USB or Bluetooth.
// Report layouts (notes\INPUT_MEMCARD_LAUNCHER_20261117.md sec 2, same as SDL's hidapi PS4/PS5 drivers):
//   DS4 USB 0x01 / DS4+DS5 BT "simple" 0x01: base 1 - sticks b..b+3, hat + face b+4, L1 R1 L2 R2 Share Options L3 R3
//     b+5, PS / touchpad b+6, L2 b+7, R2 b+8.  DS4 BT full 0x11: the same from base 3.
//   DS5 USB 0x01: base 1 - sticks b..b+3, L2 b+4, R2 b+5, hat + face b+7, buttons b+8, PS / touchpad / mute b+9.
//     DS5 BT full 0x31: the same from base 2.
//   Rumble: DS4 USB 0x05 [1]=0x01 [4]=small [5]=large (32 bytes); DS4 BT 0x11 [1]=0xC0 [3]=0x01 [6]=small [7]=large,
//     78 bytes, CRC32(0xA2 + bytes 0..73) at 74; DS5 USB 0x02 [1]=0x03 [3]=small [4]=large (48 bytes);
//     DS5 BT 0x31 [1]=0x02 [2]=0x03 [4]=small [5]=large, 78 bytes, CRC at 74.
// Each controller has a reader thread (overlapped reads on one handle, motor reports on a second handle); a
// background thread looks for newly plugged controllers every 2 s.
#include "host/win32_hid_pads.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <setupapi.h>
#include <hidsdi.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cwctype>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ps2host::hidpads
{
    namespace hidimpl
    {
        constexpr USHORT kSonyVid = 0x054C;

        enum class Model
        {
            None,
            DualShock4,
            DualSense
        };

        Model modelOf(USHORT vid, USHORT pid)
        {
            if (vid != kSonyVid)
                return Model::None;
            switch (pid)
            {
            case 0x05C4: // DualShock 4 (first model)
            case 0x09CC: // DualShock 4 (second model)
            case 0x0BA0: // DualShock 4 USB wireless adaptor
                return Model::DualShock4;
            case 0x0CE6: // DualSense
            case 0x0DF2: // DualSense Edge
                return Model::DualSense;
            default:
                return Model::None;
            }
        }

        uint32_t crc32Ps(const uint8_t *data, size_t len)
        {
            // zlib CRC-32 seeded with the Bluetooth HID output header byte 0xA2
            uint32_t crc = 0xFFFFFFFFu;
            auto feed = [&crc](uint8_t byte)
            {
                crc ^= byte;
                for (int k = 0; k < 8; ++k)
                    crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
            };
            feed(0xA2);
            for (size_t i = 0; i < len; ++i)
                feed(data[i]);
            return ~crc;
        }

        uint8_t deadzone(uint8_t v)
        {
            const int d = static_cast<int>(v) - 128;
            return (d > -24 && d < 24) ? uint8_t{0x80} : v;
        }

        struct Device
        {
            std::wstring path;
            Model model = Model::None;
            bool bluetooth = false;
            HANDLE readHandle = INVALID_HANDLE_VALUE;
            HANDLE writeHandle = INVALID_HANDLE_VALUE;
            USHORT inLen = 64, outLen = 0, featureLen = 0;
            std::thread thread;
            std::atomic<bool> alive{true};

            std::mutex m; // guards state / motors
            PadState state{};
            uint8_t wantSmall = 0, wantLarge = 0;
            bool motorsDirty = false;
        };

        std::mutex g_mutex; // guards g_devices
        std::array<std::unique_ptr<Device>, kHidPads> g_devices;
        std::thread g_scanThread;
        std::atomic<bool> g_stop{false};
        std::mutex g_wakeMutex;
        std::condition_variable g_wake;
        bool g_started = false;

        void parse(Device &dev, const uint8_t *buf, DWORD len)
        {
            if (len < 1)
                return;
            const uint8_t id = buf[0];
            int base = -1;
            bool ds5Layout = false;
            if (dev.model == Model::DualShock4)
            {
                if (id == 0x01)
                    base = 1;
                else if (id == 0x11)
                    base = 3;
            }
            else
            {
                if (id == 0x01)
                {
                    base = 1;
                    ds5Layout = !dev.bluetooth; // DS5 over BT before full mode sends the DS4-style simple report
                }
                else if (id == 0x31)
                {
                    base = 2;
                    ds5Layout = true;
                }
            }
            if (base < 0 || static_cast<DWORD>(base + 10) > len)
                return;
            const uint8_t *b = buf + base;
            const uint8_t faceHat = ds5Layout ? b[7] : b[4];
            const uint8_t btn1 = ds5Layout ? b[8] : b[5];
            const uint8_t btn2 = ds5Layout ? b[9] : b[6];
            const uint8_t l2 = ds5Layout ? b[4] : b[7];
            const uint8_t r2 = ds5Layout ? b[5] : b[8];

            PadState p{};
            p.connected = true;
            switch (faceHat & 0x0F)
            {
            case 0: p.buttons |= kBtnUp; break;
            case 1: p.buttons |= kBtnUp | kBtnRight; break;
            case 2: p.buttons |= kBtnRight; break;
            case 3: p.buttons |= kBtnDown | kBtnRight; break;
            case 4: p.buttons |= kBtnDown; break;
            case 5: p.buttons |= kBtnDown | kBtnLeft; break;
            case 6: p.buttons |= kBtnLeft; break;
            case 7: p.buttons |= kBtnUp | kBtnLeft; break;
            default: break; // 8 = released
            }
            if (faceHat & 0x10) p.buttons |= kBtnSquare;
            if (faceHat & 0x20) p.buttons |= kBtnCross;
            if (faceHat & 0x40) p.buttons |= kBtnCircle;
            if (faceHat & 0x80) p.buttons |= kBtnTriangle;
            if (btn1 & 0x01) p.buttons |= kBtnL1;
            if (btn1 & 0x02) p.buttons |= kBtnR1;
            if (btn1 & 0x04) p.buttons |= kBtnL2;
            if (btn1 & 0x08) p.buttons |= kBtnR2;
            if (btn1 & 0x10) p.buttons |= kBtnSelect; // Share / Create
            if (btn1 & 0x20) p.buttons |= kBtnStart;  // Options
            if (btn1 & 0x40) p.buttons |= kBtnL3;
            if (btn1 & 0x80) p.buttons |= kBtnR3;
            if (btn2 & 0x02) p.buttons |= kBtnSelect; // touchpad click (where SELECT sits on a DualShock 2)
            p.l2 = l2;
            p.r2 = r2;
            if (l2 > 40) p.buttons |= kBtnL2;
            if (r2 > 40) p.buttons |= kBtnR2;
            // Sony sticks are already 0 = up / left, as the PS2's
            p.lx = deadzone(b[0]);
            p.ly = deadzone(b[1]);
            p.rx = deadzone(b[2]);
            p.ry = deadzone(b[3]);

            std::lock_guard lock(dev.m);
            dev.state = p;
        }

        bool sendMotors(Device &dev, uint8_t smallMotor, uint8_t largeMotor)
        {
            std::vector<uint8_t> r;
            if (dev.model == Model::DualShock4)
            {
                if (dev.bluetooth)
                {
                    r.assign(78, 0);
                    r[0] = 0x11;
                    r[1] = 0xC0; // HID + CRC
                    r[3] = 0x01; // rumble only (the light bar stays as it is)
                    r[6] = smallMotor;
                    r[7] = largeMotor;
                }
                else
                {
                    r.assign(std::max<size_t>(32, dev.outLen), 0);
                    r[0] = 0x05;
                    r[1] = 0x01;
                    r[4] = smallMotor;
                    r[5] = largeMotor;
                }
            }
            else
            {
                if (dev.bluetooth)
                {
                    r.assign(78, 0);
                    r[0] = 0x31;
                    r[1] = 0x02;
                    r[2] = 0x03; // compatible vibration + haptics select
                    r[4] = smallMotor;
                    r[5] = largeMotor;
                }
                else
                {
                    r.assign(std::max<size_t>(48, dev.outLen), 0);
                    r[0] = 0x02;
                    r[1] = 0x03;
                    r[3] = smallMotor;
                    r[4] = largeMotor;
                }
            }
            if (dev.bluetooth)
            {
                const uint32_t crc = crc32Ps(r.data(), 74);
                r[74] = static_cast<uint8_t>(crc);
                r[75] = static_cast<uint8_t>(crc >> 8);
                r[76] = static_cast<uint8_t>(crc >> 16);
                r[77] = static_cast<uint8_t>(crc >> 24);
            }
            if (dev.outLen && r.size() > dev.outLen && !dev.bluetooth)
                r.resize(dev.outLen);
            DWORD written = 0;
            if (WriteFile(dev.writeHandle, r.data(), static_cast<DWORD>(r.size()), &written, nullptr))
                return true;
            // some drivers only take output reports through the control pipe
            return HidD_SetOutputReport(dev.writeHandle, r.data(), static_cast<ULONG>(r.size())) != FALSE;
        }

        void readerLoop(Device *dev)
        {
            std::vector<uint8_t> buf(std::max<USHORT>(dev->inLen, 64), 0);
            OVERLAPPED ov{};
            ov.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
            bool pending = false;
            while (!g_stop.load() && dev->alive.load())
            {
                if (!pending)
                {
                    ResetEvent(ov.hEvent);
                    if (!ReadFile(dev->readHandle, buf.data(), static_cast<DWORD>(buf.size()), nullptr, &ov) &&
                        GetLastError() != ERROR_IO_PENDING)
                        break; // unplugged
                    pending = true;
                }
                if (WaitForSingleObject(ov.hEvent, 16) == WAIT_OBJECT_0)
                {
                    DWORD got = 0;
                    if (!GetOverlappedResult(dev->readHandle, &ov, &got, FALSE))
                        break;
                    pending = false;
                    parse(*dev, buf.data(), got);
                }
                uint8_t smallMotor = 0, largeMotor = 0;
                bool dirty = false;
                {
                    std::lock_guard lock(dev->m);
                    if (dev->motorsDirty)
                    {
                        dirty = true;
                        dev->motorsDirty = false;
                        smallMotor = dev->wantSmall;
                        largeMotor = dev->wantLarge;
                    }
                }
                if (dirty)
                    sendMotors(*dev, smallMotor, largeMotor);
            }
            if (pending)
            {
                CancelIo(dev->readHandle);
                DWORD got = 0;
                GetOverlappedResult(dev->readHandle, &ov, &got, TRUE);
            }
            sendMotors(*dev, 0, 0); // motors off when the game stops reading it
            CloseHandle(ov.hEvent);
            {
                std::lock_guard lock(dev->m);
                dev->state = PadState{};
            }
            dev->alive.store(false);
        }

        void closeDevice(std::unique_ptr<Device> &slot)
        {
            if (!slot)
                return;
            slot->alive.store(false);
            if (slot->thread.joinable())
                slot->thread.join();
            if (slot->readHandle != INVALID_HANDLE_VALUE)
                CloseHandle(slot->readHandle);
            if (slot->writeHandle != INVALID_HANDLE_VALUE)
                CloseHandle(slot->writeHandle);
            slot.reset();
        }

        std::unique_ptr<Device> tryOpen(const std::wstring &path)
        {
            HANDLE probe = CreateFileW(path.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
            if (probe == INVALID_HANDLE_VALUE)
                return nullptr;
            HIDD_ATTRIBUTES attr{};
            attr.Size = sizeof(attr);
            Model model = Model::None;
            HIDP_CAPS caps{};
            if (HidD_GetAttributes(probe, &attr))
                model = modelOf(attr.VendorID, attr.ProductID);
            if (model != Model::None)
            {
                PHIDP_PREPARSED_DATA pre = nullptr;
                if (HidD_GetPreparsedData(probe, &pre))
                {
                    if (HidP_GetCaps(pre, &caps) != HIDP_STATUS_SUCCESS)
                        model = Model::None;
                    HidD_FreePreparsedData(pre);
                }
                else
                    model = Model::None;
            }
            CloseHandle(probe);
            // the gamepad collection (generic desktop / game pad)
            if (model == Model::None || caps.UsagePage != 0x01 || (caps.Usage != 0x05 && caps.Usage != 0x04))
                return nullptr;

            auto dev = std::make_unique<Device>();
            dev->path = path;
            dev->model = model;
            dev->inLen = caps.InputReportByteLength;
            dev->outLen = caps.OutputReportByteLength;
            dev->featureLen = caps.FeatureReportByteLength;
            dev->bluetooth = caps.InputReportByteLength > 64;
            dev->readHandle = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                          nullptr, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, nullptr);
            dev->writeHandle = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                           nullptr, OPEN_EXISTING, 0, nullptr);
            if (dev->readHandle == INVALID_HANDLE_VALUE || dev->writeHandle == INVALID_HANDLE_VALUE)
            {
                if (dev->readHandle != INVALID_HANDLE_VALUE)
                    CloseHandle(dev->readHandle);
                if (dev->writeHandle != INVALID_HANDLE_VALUE)
                    CloseHandle(dev->writeHandle);
                return nullptr; // held exclusively by another program (e.g. DS4Windows' hide mode)
            }
            if (dev->bluetooth)
            {
                // Over Bluetooth both send a short "simple" report until a calibration feature report is read; then
                // the full report (0x11 / 0x31) with the analog triggers and the motors' output report.
                // (DS4: feature 0x02, DualSense: feature 0x05)
                std::vector<uint8_t> feature(std::max<USHORT>(dev->featureLen, 64), 0);
                feature[0] = model == Model::DualShock4 ? 0x02 : 0x05;
                HidD_GetFeature(dev->writeHandle, feature.data(), static_cast<ULONG>(feature.size()));
            }
            HidD_SetNumInputBuffers(dev->readHandle, 8);
            return dev;
        }

        std::vector<std::wstring> listSonyPaths()
        {
            std::vector<std::wstring> paths;
            GUID hidGuid;
            HidD_GetHidGuid(&hidGuid);
            HDEVINFO set = SetupDiGetClassDevsW(&hidGuid, nullptr, nullptr, DIGCF_PRESENT | DIGCF_DEVICEINTERFACE);
            if (set == INVALID_HANDLE_VALUE)
                return paths;
            SP_DEVICE_INTERFACE_DATA ifData{};
            ifData.cbSize = sizeof(ifData);
            for (DWORD i = 0; SetupDiEnumDeviceInterfaces(set, nullptr, &hidGuid, i, &ifData); ++i)
            {
                DWORD needed = 0;
                SetupDiGetDeviceInterfaceDetailW(set, &ifData, nullptr, 0, &needed, nullptr);
                if (needed == 0)
                    continue;
                std::vector<uint8_t> raw(needed, 0);
                auto *detail = reinterpret_cast<SP_DEVICE_INTERFACE_DETAIL_DATA_W *>(raw.data());
                detail->cbSize = sizeof(SP_DEVICE_INTERFACE_DETAIL_DATA_W);
                if (!SetupDiGetDeviceInterfaceDetailW(set, &ifData, detail, needed, nullptr, nullptr))
                    continue;
                std::wstring path = detail->DevicePath;
                std::wstring lower = path;
                for (auto &c : lower)
                    c = static_cast<wchar_t>(towlower(c));
                // quick filter before opening: Sony's vendor id in the path (USB "vid_054c", Bluetooth "_vid&0002054c")
                if (lower.find(L"054c") == std::wstring::npos)
                    continue;
                paths.push_back(std::move(path));
            }
            SetupDiDestroyDeviceInfoList(set);
            return paths;
        }

        void scanOnce()
        {
            // drop controllers whose reader stopped (unplugged)
            {
                std::lock_guard lock(g_mutex);
                for (auto &slot : g_devices)
                    if (slot && !slot->alive.load())
                        closeDevice(slot);
            }
            const std::vector<std::wstring> paths = listSonyPaths();
            for (const std::wstring &path : paths)
            {
                {
                    std::lock_guard lock(g_mutex);
                    bool known = false, room = false;
                    for (auto &slot : g_devices)
                    {
                        if (slot && slot->path == path)
                            known = true;
                        if (!slot)
                            room = true;
                    }
                    if (known || !room)
                        continue;
                }
                std::unique_ptr<Device> dev = tryOpen(path);
                if (!dev)
                    continue;
                std::lock_guard lock(g_mutex);
                for (int i = 0; i < kHidPads; ++i)
                {
                    if (g_devices[static_cast<size_t>(i)])
                        continue;
                    std::fprintf(stderr, "[pads] %s over %s -> controller %d%c",
                                 dev->model == Model::DualShock4 ? "DualShock 4" : "DualSense",
                                 dev->bluetooth ? "Bluetooth" : "USB", kHidFirstIndex + i + 1, 10);
                    Device *raw = dev.get();
                    raw->thread = std::thread(readerLoop, raw);
                    g_devices[static_cast<size_t>(i)] = std::move(dev);
                    break;
                }
            }
        }

        void scanLoop()
        {
            while (!g_stop.load())
            {
                scanOnce();
                std::unique_lock lock(g_wakeMutex);
                g_wake.wait_for(lock, std::chrono::seconds(2), [] { return g_stop.load(); });
            }
        }
    }

    void start()
    {
        using namespace hidimpl;
        if (g_started)
            return;
        if (std::getenv("PS2X_BACKGROUND") || std::getenv("PS2X_NO_XINPUT") || std::getenv("PS2X_NO_HIDPADS"))
            return; // hidden test runs ignore real controllers (as XInput)
        g_started = true;
        g_stop.store(false);
        g_scanThread = std::thread(scanLoop);
    }

    void stop()
    {
        using namespace hidimpl;
        if (!g_started)
            return;
        g_stop.store(true);
        g_wake.notify_all();
        if (g_scanThread.joinable())
            g_scanThread.join();
        std::lock_guard lock(g_mutex);
        for (auto &slot : g_devices)
            closeDevice(slot);
        g_started = false;
    }

    bool read(int index, PadState &out, const char *&name)
    {
        using namespace hidimpl;
        out = PadState{};
        name = nullptr;
        if (index < 0 || index >= kHidPads)
            return false;
        std::lock_guard lock(g_mutex);
        Device *dev = g_devices[static_cast<size_t>(index)].get();
        if (!dev || !dev->alive.load())
            return false;
        std::lock_guard devLock(dev->m);
        out = dev->state;
        out.connected = true; // plugged in, even before its first report
        name = dev->model == Model::DualShock4 ? "DualShock 4" : "DualSense";
        return true;
    }

    void setVibration(int index, uint8_t smallMotor, uint8_t largeMotor)
    {
        using namespace hidimpl;
        if (index < 0 || index >= kHidPads)
            return;
        std::lock_guard lock(g_mutex);
        Device *dev = g_devices[static_cast<size_t>(index)].get();
        if (!dev)
            return;
        std::lock_guard devLock(dev->m);
        if (dev->wantSmall == smallMotor && dev->wantLarge == largeMotor)
            return; // already asked for (sent by the reader thread)
        dev->wantSmall = smallMotor;
        dev->wantLarge = largeMotor;
        dev->motorsDirty = true;
    }
}
