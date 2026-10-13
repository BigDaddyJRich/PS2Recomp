#pragma once

#include <string>

// An update check when the game starts, before its window opens (Windows). A game build names its GitHub repository and
// the program that installs updates (its launcher); the installed version comes from release-manifest.txt beside the
// game program, so a developer's build (no manifest) never checks. When a newer release is out, a popup offers Update
// (start the launcher with --update and end) or Play.
namespace ps2_host_update
{
    struct Config
    {
        std::string repo;     // "owner/name" on GitHub
        std::string zipName;  // the release's download: a release without it doesn't count
        std::string launcher; // the program that updates, in the folder above the game program's
        std::string title;    // the popup's title, and the name before the version in release-manifest.txt
    };

    // Called by a game build's code before main runs (a static initialiser). Returns true.
    bool configure(const Config &config);

    // After the settings are read, before the window. VENDETTA_UPDATE_CHECK=0 (settings.toml [updates] check_on_open =
    // false): no check. VENDETTA_UPDATE_URL: ask another address that answers as GitHub does (tests). Asking GitHub
    // gives up after 2 seconds; the popup waits for the player. Returns false when the player chose Update: the game
    // should end.
    bool checkAtStart();
}
