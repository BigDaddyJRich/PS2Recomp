#include "runtime/ps2_host_settings.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <system_error>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace ps2_host_settings
{
    namespace
    {
        Settings g_settings;

        // What a missing file is replaced by: every setting, at its default, with what it does.
        const char *kDefaultFile =
            "# The game's settings. A settings program writes this file; you can also edit it by hand.\n"
            "# A VENDETTA_* switch set in the environment wins over a line here.\n"
            "\n"
            "[display]\n"
            "mode = \"fullscreen\"      # \"window\", \"borderless\" (a frameless window the size of the screen) or \"fullscreen\"\n"
            "screen = 1               # which screen: 1 = the main one\n"
            "maximised = true         # start maximised (window only)\n"
            "fullscreen_width = 0     # fullscreen resolution; 0 = the screen's own\n"
            "fullscreen_height = 0\n"
            "refresh_rate = \"automatic\" # fullscreen only: \"automatic\" (the screen's rate when it suits the game's 60 frames a\n"
            "                         # second, else the smoothest rate it has), \"smoothest\" (the highest multiple of 60 Hz),\n"
            "                         # 0 = the screen's own (no switch), or a number such as 60 or 120\n"
            "vsync = true             # wait for the screen before showing each picture (no tearing)\n"
            "shape = \"widescreen\"     # \"widescreen\" (matches at 16:9 with more of the arena; menus and movies 4:3),\n"
            "                         # \"4:3\" (as on the PS2, bars at the sides) or \"stretch\" (fill the window)\n"
            "\n"
            "[graphics]\n"
            "renderer = \"gpu\"         # \"gpu\" (the graphics card) or \"cpu\"\n"
            "scale = 16               # how much sharper than the PS2: 1, 2, 4, 8 or 16 (16: four times its detail)\n"
            "filter = \"fsr\"           # scaling to the window: \"fsr\", \"smooth\" or \"blocky\"\n"
            "sharpening = 75          # 0-100, with \"fsr\"\n"
            "sharp_2d = true          # the HUD, text and menus in the same detail as the 3-D (as PCSX2 upscales them); false = as on the PS2\n"
            "high_poly = true         # every wrestler and arena at full detail in every match; false = as on the PS2\n"
            "                         # (lower-detail wrestlers and arenas in 3- and 4-fighter matches)\n"
            "camera = \"vendetta\"      # the camera in matches: \"vendetta\" (the game's own), \"nomercy\" (WWF No Mercy's\n"
            "                         # style) or \"wk2\" (Wrestle Kingdom 2's style)\n"
            "\n"
            "[sound]\n"
            "volume = 100             # 0-100\n"
            "reverb = true            # the echo the PS2's sound chip adds in matches\n"
            "\n"
            "[controls]\n"
            "multitap = true          # up to 4 players (a multitap in port 2, as on the PS2); false = 2 controllers\n"
            "vibration_1 = true       # each player's controller vibrates (Xbox-style / XInput controllers)\n"
            "vibration_2 = true\n"
            "vibration_3 = true\n"
            "vibration_4 = true\n"
            "keyboard = 0             # which player the keyboard plays as (with their controller): 1-4, 0 = nobody\n"
            "\n"
            "[gameplay]\n"
            "unlock_all = false       # every wrestler, attire, arena and picture for every player, with or without a save.\n"
            "                         # The save itself is not changed (false = what the save has unlocked)\n"
            "\n"
            "[updates]\n"
            "check_on_open = true     # check for a new version when the game or the settings program starts\n";

        std::string trim(const std::string &text)
        {
            size_t begin = 0;
            size_t end = text.size();
            while (begin < end && std::isspace(static_cast<unsigned char>(text[begin])))
                ++begin;
            while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])))
                --end;
            return text.substr(begin, end - begin);
        }

        std::string lower(std::string text)
        {
            std::transform(text.begin(), text.end(), text.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return text;
        }

        // The TOML this file needs: [section], key = "string" | number | true / false, # comments. Values come back
        // unquoted, keyed "section.key".
        std::map<std::string, std::string> parse(std::istream &in, const std::filesystem::path &file)
        {
            std::map<std::string, std::string> values;
            std::string section;
            std::string line;
            int lineNumber = 0;
            while (std::getline(in, line))
            {
                ++lineNumber;
                bool inString = false;
                for (size_t i = 0; i < line.size(); ++i)
                {
                    if (line[i] == '"')
                        inString = !inString;
                    else if (line[i] == '#' && !inString)
                    {
                        line.resize(i);
                        break;
                    }
                }
                line = trim(line);
                if (line.empty())
                    continue;
                if (line.front() == '[' && line.back() == ']')
                {
                    section = lower(trim(line.substr(1, line.size() - 2)));
                    continue;
                }
                const size_t equals = line.find('=');
                if (equals == std::string::npos)
                {
                    std::cerr << "[settings] " << file.string() << ":" << lineNumber << ": not \"key = value\": " << line
                              << std::endl;
                    continue;
                }
                std::string key = lower(trim(line.substr(0, equals)));
                std::string value = trim(line.substr(equals + 1));
                if (value.size() >= 2 && value.front() == '"' && value.back() == '"')
                    value = value.substr(1, value.size() - 2);
                values[section.empty() ? key : section + "." + key] = value;
            }
            return values;
        }

        std::filesystem::path programDirectory(const char *programPath)
        {
#if defined(_WIN32)
            wchar_t buffer[MAX_PATH];
            const DWORD length = GetModuleFileNameW(nullptr, buffer, MAX_PATH);
            if (length > 0 && length < MAX_PATH)
                return std::filesystem::path(buffer).parent_path();
#endif
            if (programPath && *programPath)
            {
                std::error_code error;
                const std::filesystem::path path = std::filesystem::absolute(programPath, error);
                if (!error)
                    return path.parent_path();
            }
            return {};
        }

        bool environmentSet(const char *name)
        {
            const char *value = std::getenv(name);
            return value && *value;
        }

        // A setting fills in a switch only when the environment doesn't set it.
        void fillSwitch(const char *name, const std::string &value)
        {
            if (environmentSet(name))
                return;
#if defined(_WIN32)
            _putenv_s(name, value.c_str());
#else
            setenv(name, value.c_str(), 0);
#endif
        }

        class Reader
        {
        public:
            Reader(std::map<std::string, std::string> values, std::filesystem::path file)
                : m_values(std::move(values)), m_file(std::move(file)) {}

            bool peek(const std::string &key, std::string &out) const
            {
                const auto it = m_values.find(key);
                if (it == m_values.end())
                    return false;
                out = it->second;
                return true;
            }

            // the raw value (removed from the map, so what is left at the end was not understood)
            bool take(const std::string &key, std::string &out)
            {
                const auto it = m_values.find(key);
                if (it == m_values.end())
                    return false;
                out = it->second;
                m_values.erase(it);
                return true;
            }

            void integer(const std::string &key, int &out, int minimum, int maximum)
            {
                std::string text;
                if (!take(key, text))
                    return;
                char *end = nullptr;
                const long value = std::strtol(text.c_str(), &end, 10);
                if (end == text.c_str() || *end != '\0' || value < minimum || value > maximum)
                {
                    warn(key, text, "a whole number from " + std::to_string(minimum) + " to " + std::to_string(maximum));
                    return;
                }
                out = static_cast<int>(value);
            }

            void boolean(const std::string &key, bool &out)
            {
                std::string text;
                if (!take(key, text))
                    return;
                const std::string value = lower(text);
                if (value == "true")
                    out = true;
                else if (value == "false")
                    out = false;
                else
                    warn(key, text, "true or false");
            }

            // one of the given words; returns its index or -1 (absent or not understood)
            int choice(const std::string &key, std::initializer_list<const char *> words)
            {
                std::string text;
                if (!take(key, text))
                    return -1;
                const std::string value = lower(text);
                int index = 0;
                std::string list;
                for (const char *word : words)
                {
                    if (value == word)
                        return index;
                    list += (index ? ", " : "") + std::string("\"") + word + "\"";
                    ++index;
                }
                warn(key, text, list);
                return -1;
            }

            void reportLeftovers() const
            {
                for (const auto &[key, value] : m_values)
                {
                    if (key.rfind("updates.", 0) == 0)
                        continue; // [updates]: the settings program may keep more of its own there
                    std::cerr << "[settings] " << m_file.string() << ": " << key << " is not a setting (ignored)" << std::endl;
                }
            }

        private:
            void warn(const std::string &key, const std::string &text, const std::string &expected) const
            {
                std::cerr << "[settings] " << m_file.string() << ": " << key << " = " << text << " is not " << expected
                          << "; the default is used" << std::endl;
            }

            std::map<std::string, std::string> m_values;
            std::filesystem::path m_file;
        };

        void apply(Reader &reader)
        {
            Settings &s = g_settings;
            switch (reader.choice("display.mode", {"window", "borderless", "fullscreen"}))
            {
            case 1:
                s.mode = WindowMode::Borderless;
                break;
            case 2:
                s.mode = WindowMode::Fullscreen;
                break;
            case 0:
                s.mode = WindowMode::Window;
                break;
            default: // no mode line: fullscreen, the default (no settings file at all, as in test runs: a window)
                s.mode = WindowMode::Fullscreen;
                break;
            }
            reader.integer("display.screen", s.screen, 1, 16);
            reader.boolean("display.maximised", s.maximised);
            reader.integer("display.fullscreen_width", s.fullscreenWidth, 0, 16384);
            reader.integer("display.fullscreen_height", s.fullscreenHeight, 0, 16384);
            // "automatic" (the default, also when the line is missing: the screen's rate when it suits 60 pictures a
            // second, else the smoothest it has, worked out at each start), "smoothest", 0 = the screen's own, or Hz
            std::string refresh;
            s.refreshRate = kRefreshAutomatic;
            if (reader.peek("display.refresh_rate", refresh) && (lower(refresh) == "smoothest" || lower(refresh) == "automatic"))
            {
                reader.take("display.refresh_rate", refresh);
                s.refreshRate = lower(refresh) == "smoothest" ? kRefreshSmoothest : kRefreshAutomatic;
            }
            else
                reader.integer("display.refresh_rate", s.refreshRate, 0, 1000);
            reader.boolean("display.vsync", s.vsync);
            // "widescreen" (the default, also when the line is missing): matches at 16:9 with more of the arena (menus
            // and movies stay 4:3) — the game's widescreen switch, VENDETTA_WIDESCREEN (runtime/ps2_widescreen.h),
            // filled in like the other switches. No settings file at all (VENDETTA_SETTINGS=none, test runs): off.
            const int shape = reader.choice("display.shape", {"4:3", "stretch", "widescreen"});
            s.stretch = shape == 1;
            if (shape == 2 || shape < 0)
                fillSwitch("VENDETTA_WIDESCREEN", "16:9");
            reader.integer("sound.volume", s.volume, 0, 100);
            bool multitap = s.multitapPort >= 0;
            reader.boolean("controls.multitap", multitap);
            s.multitapPort = multitap ? 1 : -1;
            for (int player = 0; player < 4; ++player)
                reader.boolean("controls.vibration_" + std::to_string(player + 1), s.vibration[player]);
            // which player the keyboard plays as (ps2_pad.cpp, VENDETTA_KEYBOARD): 1-4, 0 = nobody (the default, also
            // when the line is missing; with no settings file at all the keyboard is player 1)
            int keyboard = 0;
            reader.integer("controls.keyboard", keyboard, 0, 4);
            fillSwitch("VENDETTA_KEYBOARD", std::to_string(keyboard));
            // unlock everything (vendetta_unlocks.cpp, VENDETTA_UNLOCKALL): off by default, also when the line is missing
            bool unlockAll = false;
            reader.boolean("gameplay.unlock_all", unlockAll);
            fillSwitch("VENDETTA_UNLOCKALL", unlockAll ? "1" : "0");
            // the update check when the game starts (ps2_host_update.cpp, VENDETTA_UPDATE_CHECK): the same switch as
            // the settings program's own check, on by default, also when the line is missing
            bool checkUpdates = true;
            reader.boolean("updates.check_on_open", checkUpdates);
            fillSwitch("VENDETTA_UPDATE_CHECK", checkUpdates ? "1" : "0");

            // the ones the runtime already reads as switches
            if (const int renderer = reader.choice("graphics.renderer", {"gpu", "cpu"}); renderer >= 0)
                fillSwitch("VENDETTA_GS", renderer == 0 ? "gpu" : "cpu");
            // 16 (the default, also when the line is missing): sharp 2-D and a 4x picture; no settings file at all
            // (test runs): the runtime's own 8
            int scale = 16;
            reader.integer("graphics.scale", scale, 1, 16);
            if (scale != 1 && scale != 2 && scale != 4 && scale != 8 && scale != 16)
            {
                std::cerr << "[settings] graphics.scale = " << scale << " is not 1, 2, 4, 8 or 16; 16 is used" << std::endl;
                scale = 16;
            }
            fillSwitch("VENDETTA_SSAA", std::to_string(scale));
            if (const int filter = reader.choice("graphics.filter", {"fsr", "smooth", "blocky"}); filter >= 0)
                fillSwitch("VENDETTA_FILTER", filter == 0 ? "fsr" : (filter == 1 ? "bilinear" : "nearest"));
            int sharpening = -1;
            reader.integer("graphics.sharpening", sharpening, 0, 100);
            if (sharpening >= 0)
            {
                // FSR's sharpening is in stops, 0 = the most: 100% = 0 stops, 0% = 2 stops
                std::ostringstream stops;
                stops << 2.0 * (100 - sharpening) / 100.0;
                fillSwitch("VENDETTA_SHARPEN", stops.str());
            }
            // true (the default, also when the line is missing): the game's High Poly switch, VENDETTA_HIGHPOLY. No
            // settings file at all (test runs): off, as on the PS2.
            bool highPoly = true;
            reader.boolean("graphics.high_poly", highPoly);
            fillSwitch("VENDETTA_HIGHPOLY", highPoly ? "1" : "0");
            // the fight camera (vendetta_camera.cpp): "vendetta" (the game's own, also when the line is missing),
            // "nomercy" (in the style of WWF No Mercy) or "wk2" (in the style of Wrestle Kingdom 2)
            if (const int camera = reader.choice("graphics.camera", {"vendetta", "nomercy", "wk2"}); camera >= 0)
                fillSwitch("VENDETTA_CAMERA", camera == 0 ? "vendetta" : (camera == 1 ? "nomercy" : "wk2"));
            // the 2-D (sprites: HUD, text, menus) super-sampled like the 3-D (gs_gpu_parallel.cpp, paraLLEl-GS's
            // Hacks::upscale_sprites); true also when the line is missing; no settings file at all: off
            bool sharp2d = true;
            reader.boolean("graphics.sharp_2d", sharp2d);
            fillSwitch("VENDETTA_UPSCALE_SPRITES", sharp2d ? "1" : "0");
            std::string reverbText;
            if (reader.take("sound.reverb", reverbText))
            {
                const std::string value = lower(reverbText);
                if (value == "false")
                    fillSwitch("VENDETTA_REVERB", "0");
                else if (value != "true")
                    std::cerr << "[settings] sound.reverb = " << reverbText << " is not true or false; the default is used"
                              << std::endl;
            }
        }

        // The switches for the new settings, for tests and debugging (they win over the file, like the others)
        void applyEnvironment()
        {
            Settings &s = g_settings;
            if (const char *mode = std::getenv("VENDETTA_WINDOW"); mode && *mode)
            {
                const std::string value = lower(mode);
                if (value == "window")
                    s.mode = WindowMode::Window;
                else if (value == "borderless")
                    s.mode = WindowMode::Borderless;
                else if (value == "fullscreen")
                    s.mode = WindowMode::Fullscreen;
                else
                    std::cerr << "[settings] VENDETTA_WINDOW=" << mode << " not understood (window, borderless or fullscreen)"
                              << std::endl;
            }
            if (const char *value = std::getenv("VENDETTA_SCREEN"); value && *value)
                s.screen = std::clamp(std::atoi(value), 1, 16);
            if (const char *value = std::getenv("VENDETTA_MAXIMISED"); value && *value)
                s.maximised = std::string(value) != "0";
            if (const char *value = std::getenv("VENDETTA_FULLSCREEN_SIZE"); value && *value) // e.g. 1920x1080
            {
                int width = 0;
                int height = 0;
                if (std::sscanf(value, "%dx%d", &width, &height) == 2 && width > 0 && height > 0)
                {
                    s.fullscreenWidth = width;
                    s.fullscreenHeight = height;
                }
            }
            if (const char *value = std::getenv("VENDETTA_REFRESH"); value && *value)
                s.refreshRate = lower(value) == "smoothest"   ? kRefreshSmoothest
                                : lower(value) == "automatic" ? kRefreshAutomatic
                                                              : std::clamp(std::atoi(value), 0, 1000);
            if (const char *value = std::getenv("VENDETTA_VSYNC"); value && *value)
                s.vsync = std::string(value) != "0";
            if (const char *value = std::getenv("VENDETTA_VOLUME"); value && *value)
                s.volume = std::clamp(std::atoi(value), 0, 100);
            if (const char *value = std::getenv("VENDETTA_ASPECT"); value && *value)
                s.stretch = std::string(value) == "fill";
            // VENDETTA_MULTITAP=0 (none) / 1 (port 1) / 2 (port 2); VENDETTA_VIBRATION=0 / 1 (every player)
            if (const char *value = std::getenv("VENDETTA_MULTITAP"); value && *value)
                s.multitapPort = std::clamp(std::atoi(value), 0, 2) - 1;
            if (const char *value = std::getenv("VENDETTA_VIBRATION"); value && *value)
                for (bool &on : s.vibration)
                    on = std::string(value) != "0";
        }
    }

    void load(const char *programPath)
    {
#if defined(PS2X_NATIVE_HOST)
        // HCTP native build defaults (what run_hctp_native.bat used to set): the D3D11 renderer, the native VU1
        // programs (p01-p04 stage 1, p08) and native ADX music. The environment still wins; HCTP_NATIVE_DEFAULTS=0
        // turns them all off (VU1 emulation, CRI IOP audio).
        if (!(std::getenv("HCTP_NATIVE_DEFAULTS") && std::getenv("HCTP_NATIVE_DEFAULTS")[0] == '0'))
        {
            fillSwitch("HCTP_RENDERER", "d3d11");
            fillSwitch("HCTP_MOVIES", "1");
            fillSwitch("HCTP_VU1_FAST", "1");
            fillSwitch("HCTP_NATIVE_P08", "1");
            fillSwitch("HCTP_NATIVE_P18", "1");
            fillSwitch("HCTP_NATIVE_P10", "1");
            fillSwitch("HCTP_NATIVE_P17", "1"); // crowd, verified 100% (Oct 19)
            fillSwitch("HCTP_NATIVE_P16", "1"); // particles, verified 100% where handled
            // written, not yet seen in a test match (Jordan: everything native that exists); =0 turns one off
            fillSwitch("HCTP_NATIVE_P19", "1");
            fillSwitch("HCTP_NATIVE_P11", "1");
            fillSwitch("HCTP_NATIVE_P09", "1");
            fillSwitch("HCTP_NATIVE_P15", "1");
            fillSwitch("HCTP_NATIVE_LINES", "1");
            fillSwitch("HCTP_NATIVE_P01", "1");
            fillSwitch("HCTP_NATIVE_S1", "15");
            fillSwitch("HCTP_S1_ANYHASH", "1");
            fillSwitch("HCTP_NATIVE_ADX", "1");
            fillSwitch("VENDETTA_EE_CYCLE_PCT", "100");
        }
#endif
        g_settings = Settings{};
        std::filesystem::path file;
        std::error_code error;
        const char *chosen = std::getenv("VENDETTA_SETTINGS");
        if (chosen && *chosen)
        {
            if (lower(chosen) != "none")
                file = chosen;
        }
        else
        {
            const std::filesystem::path besideProgram = programDirectory(programPath) / "settings.toml";
            if (std::filesystem::exists(besideProgram, error))
                file = besideProgram;
            else
            {
                const std::filesystem::path folder = std::filesystem::current_path(error);
                if (!error)
                    file = folder / "settings.toml";
            }
        }
        const bool writeDefault = !file.empty() && !std::filesystem::exists(file, error);

        if (writeDefault)
        {
            std::ofstream out(file, std::ios::binary);
            if (out && (out << kDefaultFile))
                std::cerr << "[settings] wrote the default settings to " << file.string() << std::endl;
            else
                file.clear(); // a folder we can't write to: the defaults
        }

        if (!file.empty())
        {
            std::ifstream in(file, std::ios::binary);
            if (in)
            {
                Reader reader(parse(in, file), file);
                apply(reader);
                reader.reportLeftovers();
                g_settings.file = file;
            }
            else
                std::cerr << "[settings] can't read " << file.string() << "; the defaults are used" << std::endl;
        }
        applyEnvironment();

        const Settings &s = g_settings;
        std::cerr << "[settings] " << (s.file.empty() ? std::string("defaults") : s.file.string()) << ": "
                  << windowModeName(s.mode) << ", screen " << s.screen
                  << (s.mode == WindowMode::Window ? (s.maximised ? ", maximised" : "") : "")
                  << (s.mode == WindowMode::Fullscreen
                          ? ", " + (s.fullscreenWidth ? std::to_string(s.fullscreenWidth) + "x" + std::to_string(s.fullscreenHeight)
                                                      : std::string("screen's resolution")) +
                                (s.refreshRate > 0 ? " at " + std::to_string(s.refreshRate) + " Hz"
                                                   : std::string(s.refreshRate == kRefreshSmoothest   ? " at the smoothest rate"
                                                                 : s.refreshRate == kRefreshAutomatic ? " at the automatic rate"
                                                                                                      : " at the screen's rate"))
                          : std::string(""))
                  << ", vsync " << (s.vsync ? "on" : "off") << ", "
                  << (s.stretch ? std::string("stretched") : std::getenv("VENDETTA_WIDESCREEN") && *std::getenv("VENDETTA_WIDESCREEN") && std::string(std::getenv("VENDETTA_WIDESCREEN")) != "0" ? "widescreen " + std::string(std::getenv("VENDETTA_WIDESCREEN")) + " in matches" : std::string("4:3"))
                  << ", volume " << s.volume << "%"
                  << (s.multitapPort >= 0 ? ", multitap in port " + std::to_string(s.multitapPort + 1) : std::string(", no multitap"))
                  << ", vibration " << s.vibration[0] << s.vibration[1] << s.vibration[2] << s.vibration[3] << std::endl;
    }

    const Settings &get()
    {
        return g_settings;
    }

    const char *windowModeName(WindowMode mode)
    {
        switch (mode)
        {
        case WindowMode::Borderless:
            return "borderless";
        case WindowMode::Fullscreen:
            return "fullscreen";
        default:
            return "window";
        }
    }
}
