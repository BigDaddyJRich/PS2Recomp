#pragma once

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

// The player's settings: settings.toml, written by a settings program or by hand. Read once at start, before anything
// else reads a switch. A VENDETTA_* switch set in the environment always wins over the file, so test runs and debugging
// work as before; a setting only fills in a switch that isn't set.
//
// Where the file is: VENDETTA_SETTINGS=<path> (or =none: the defaults, no file), else settings.toml next to the
// program, else in the current folder. When there is none, a commented default is written to the current folder.
namespace ps2_host_settings
{
    enum class WindowMode
    {
        Window,
        Borderless, // a frameless window the size of the screen
        Fullscreen  // the screen switched to the game's resolution and refresh rate
    };

    // refresh_rate = "smoothest": the highest rate the screen has that is a multiple of 60 Hz (each of the game's ~60
    // pictures a second then stays on screen the same time); a screen with none: its highest rate
    constexpr int kRefreshSmoothest = -1;
    // refresh_rate = "automatic" (the default): the screen's own rate when it already fits the game's 60 pictures a
    // second (60, 120, 240, or 59 / 119 for 59.94 / 119.88), so nothing switches; else the smoothest rate the screen
    // has (a switch, e.g. 165 -> 120 or 60); a screen with no fitting rate keeps its own. Worked out at every start.
    constexpr int kRefreshAutomatic = -2;

    // Does a rate show each of the game's ~60 pictures for the same time?
    inline bool fitsSixtyPictures(int hz) { return hz > 0 && (hz % 60 == 0 || (hz + 1) % 60 == 0); }

    // The rate to ask the screen for. setting: refreshRate; screenRate: the screen's rate now (0 = not known); rates:
    // the rates the screen has at the fullscreen size, ascending. A fixed rate the screen hasn't got is asked for as it
    // is (the window library takes the closest one). 0 back = the screen's own, whatever it is.
    inline int chooseRefreshRate(int setting, int screenRate, const std::vector<int> &rates)
    {
        if (setting > 0)
            return setting;
        if (setting == 0)
            return screenRate;
        int smoothest = 0;
        for (int r : rates)
            if (fitsSixtyPictures(r))
                smoothest = std::max(smoothest, r);
        if (setting == kRefreshSmoothest)
            return smoothest ? smoothest : (rates.empty() ? screenRate : rates.back());
        const bool screenHasIt = rates.empty() || std::find(rates.begin(), rates.end(), screenRate) != rates.end();
        if (fitsSixtyPictures(screenRate) && screenHasIt)
            return screenRate;
        return smoothest ? smoothest : screenRate;
    }

    struct Settings
    {
        // [display]
        WindowMode mode = WindowMode::Window;
        int screen = 1;            // 1 = the main screen
        bool maximised = true;     // Window only
        int fullscreenWidth = 0;   // Fullscreen only; 0 = the screen's own resolution
        int fullscreenHeight = 0;
        int refreshRate = kRefreshAutomatic; // Fullscreen only: Hz, 0 = the screen's own, kRefreshSmoothest, kRefreshAutomatic
        bool vsync = true;         // wait for the screen before showing each picture
        bool stretch = false;      // the picture fills the window (false: 4:3, bars at the sides)
        // [sound]
        int volume = 100;          // 0..100
        // [controls]
        int multitapPort = 1;      // the port with a multitap (4 pads there): 0 = port 1, 1 = port 2, -1 = none.
                                   // Def Jam Vendetta wants it in port 2: with one in port 1 it stops reading pad 1A.
        bool vibration[4] = {true, true, true, true}; // per player (the Nth pad)
        // [graphics] renderer / scale / filter / sharpening and [sound] reverb set their VENDETTA_* switches

        std::filesystem::path file; // the file they came from (empty: the defaults)
    };

    // programPath: argv[0] (the program's own path is looked up on Windows)
    void load(const char *programPath);
    const Settings &get();

    const char *windowModeName(WindowMode mode);
}
