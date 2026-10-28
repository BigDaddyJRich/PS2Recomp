#include "runtime/ps2_host_update.h"

#include <iostream>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <winhttp.h>

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#endif

namespace ps2_host_update
{
    namespace
    {
        Config &config()
        {
            static Config value;
            return value;
        }
    }

    bool configure(const Config &value)
    {
        config() = value;
        return true;
    }

#if !defined(_WIN32)
    bool checkAtStart() { return true; }
#else
    namespace
    {
        constexpr auto kAskFor = std::chrono::milliseconds(2000); // how long GitHub has to answer
        constexpr DWORD kNetworkTimeout = 2000;                    // each WinHTTP step, so a late request ends soon
        const char *kAskedFile = "update-asked.txt";              // the newest version the player chose Play for

        using Version = std::array<int, 3>;

        bool parseVersion(std::string text, Version &out)
        {
            while (!text.empty() && (text.front() == 'v' || text.front() == 'V'))
                text.erase(text.begin());
            int a = 0, b = 0, c = 0;
            const int n = std::sscanf(text.c_str(), "%d.%d.%d", &a, &b, &c);
            if (n < 2)
                return false;
            out = {a, b, n == 3 ? c : 0};
            return true;
        }

        std::string versionText(const Version &v)
        {
            return std::to_string(v[0]) + "." + std::to_string(v[1]) + "." + std::to_string(v[2]);
        }

        std::wstring wide(const std::string &text)
        {
            if (text.empty())
                return {};
            const int n = MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0);
            std::wstring out(static_cast<size_t>(n), L'\0');
            MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), out.data(), n);
            return out;
        }

        std::filesystem::path programFolder()
        {
            std::wstring path(32768, L'\0');
            const DWORD n = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
            path.resize(n);
            return std::filesystem::path(path).parent_path();
        }

        // release-manifest.txt's first line: "# <title> X.Y.Z"
        bool installedVersion(const std::filesystem::path &folder, Version &out)
        {
            std::ifstream file(folder / "release-manifest.txt");
            std::string line;
            if (!file || !std::getline(file, line))
                return false;
            const std::string start = "# " + config().title + " ";
            if (line.rfind(start, 0) != 0)
                return false;
            return parseVersion(line.substr(start.size()), out);
        }

        // The answer to one GET: the status code and the body (0 = no answer).
        struct Answer
        {
            std::mutex mutex;
            std::condition_variable ready;
            bool done = false;
            DWORD status = 0;
            std::string body;
            std::string error;
        };

        void fetch(const std::wstring &url, const std::wstring &agent, std::shared_ptr<Answer> answer)
        {
            DWORD status = 0;
            std::string body;
            std::string error;
            URL_COMPONENTS parts{};
            parts.dwStructSize = sizeof(parts);
            wchar_t host[256] = {};
            wchar_t path[2048] = {};
            parts.lpszHostName = host;
            parts.dwHostNameLength = 256;
            parts.lpszUrlPath = path;
            parts.dwUrlPathLength = 2048;
            HINTERNET session = nullptr, connection = nullptr, request = nullptr;
            if (!WinHttpCrackUrl(url.c_str(), 0, 0, &parts))
                error = "the address can't be read";
            else if (!(session = WinHttpOpen(agent.c_str(), WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME,
                                             WINHTTP_NO_PROXY_BYPASS, 0)))
                error = "WinHttpOpen failed";
            else
            {
                WinHttpSetTimeouts(session, kNetworkTimeout, kNetworkTimeout, kNetworkTimeout, kNetworkTimeout);
                const bool secure = parts.nScheme == INTERNET_SCHEME_HTTPS;
                connection = WinHttpConnect(session, host, parts.nPort, 0);
                if (connection)
                    request = WinHttpOpenRequest(connection, L"GET", path, nullptr, WINHTTP_NO_REFERER,
                                                 WINHTTP_DEFAULT_ACCEPT_TYPES, secure ? WINHTTP_FLAG_SECURE : 0);
                if (!request)
                    error = "no connection";
                else if (!WinHttpSendRequest(request, L"Accept: application/vnd.github+json\r\n", static_cast<DWORD>(-1),
                                             WINHTTP_NO_REQUEST_DATA, 0, 0, 0) ||
                         !WinHttpReceiveResponse(request, nullptr))
                    error = "no answer (error " + std::to_string(GetLastError()) + ")";
                else
                {
                    DWORD size = sizeof(status);
                    WinHttpQueryHeaders(request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                                        WINHTTP_HEADER_NAME_BY_INDEX, &status, &size, WINHTTP_NO_HEADER_INDEX);
                    for (;;)
                    {
                        DWORD available = 0;
                        if (!WinHttpQueryDataAvailable(request, &available) || available == 0)
                            break;
                        std::string chunk(available, '\0');
                        DWORD read = 0;
                        if (!WinHttpReadData(request, chunk.data(), available, &read) || read == 0)
                            break;
                        body.append(chunk.data(), read);
                        if (body.size() > (1u << 20))
                            break; // a release's answer is a few kB
                    }
                }
            }
            if (request)
                WinHttpCloseHandle(request);
            if (connection)
                WinHttpCloseHandle(connection);
            if (session)
                WinHttpCloseHandle(session);
            std::lock_guard<std::mutex> lock(answer->mutex);
            answer->status = status;
            answer->body = std::move(body);
            answer->error = std::move(error);
            answer->done = true;
            answer->ready.notify_all();
        }

        // "key": "value" in GitHub's JSON (the first one)
        std::string jsonText(const std::string &json, const std::string &key)
        {
            const std::string quoted = "\"" + key + "\"";
            size_t at = json.find(quoted);
            if (at == std::string::npos)
                return {};
            at = json.find(':', at + quoted.size());
            if (at == std::string::npos)
                return {};
            at = json.find('"', at);
            if (at == std::string::npos)
                return {};
            const size_t end = json.find('"', at + 1);
            return end == std::string::npos ? std::string() : json.substr(at + 1, end - at - 1);
        }

        // The popup's two buttons say Update and Play (a message box's Yes and No, renamed as it opens).
        HHOOK g_hook = nullptr;

        LRESULT CALLBACK renameButtons(int code, WPARAM wparam, LPARAM lparam)
        {
            if (code == HCBT_ACTIVATE)
            {
                const HWND box = reinterpret_cast<HWND>(wparam);
                SetDlgItemTextW(box, IDYES, L"Update");
                SetDlgItemTextW(box, IDNO, L"Play");
                UnhookWindowsHookEx(g_hook);
                g_hook = nullptr;
                return 0;
            }
            return CallNextHookEx(g_hook, code, wparam, lparam);
        }

        bool askUpdate(const std::string &latest, const std::string &installed)
        {
            const std::wstring text =
                L"Version " + wide(latest) + L" is available. You have version " + wide(installed) +
                L".\n\nUpdate opens the settings program, where you can see what's new and update in a few minutes. "
                L"Your saves and settings stay as they are.\n\nPlay starts the game now. You won't be asked about "
                L"version " + wide(latest) + L" again.";
            g_hook = SetWindowsHookExW(WH_CBT, renameButtons, nullptr, GetCurrentThreadId());
            const int choice = MessageBoxW(nullptr, text.c_str(), wide(config().title).c_str(),
                                           MB_YESNO | MB_ICONINFORMATION | MB_DEFBUTTON1 | MB_SETFOREGROUND | MB_TOPMOST);
            if (g_hook)
            {
                UnhookWindowsHookEx(g_hook);
                g_hook = nullptr;
            }
            return choice == IDYES;
        }

        bool startLauncher(const std::filesystem::path &program)
        {
            std::wstring command = L"\"" + program.wstring() + L"\" --update";
            STARTUPINFOW startup{};
            startup.cb = sizeof(startup);
            PROCESS_INFORMATION process{};
            // the player just pressed Update here: the settings program may come to the front (Windows only lets a
            // program take the foreground when the one in front allows it)
            AllowSetForegroundWindow(ASFW_ANY);
            if (!CreateProcessW(program.c_str(), command.data(), nullptr, nullptr, FALSE, 0, nullptr,
                                program.parent_path().c_str(), &startup, &process))
                return false;
            CloseHandle(process.hThread);
            CloseHandle(process.hProcess);
            return true;
        }
    }

    bool checkAtStart()
    {
        const Config &cfg = config();
        if (cfg.repo.empty())
            return true;
        if (const char *check = std::getenv("VENDETTA_UPDATE_CHECK"); check && check[0] == '0')
        {
            std::cerr << "[update] checking for updates is off (settings: check_on_open)" << std::endl;
            return true;
        }
        const std::filesystem::path folder = programFolder();
        Version installed{};
        if (!installedVersion(folder, installed))
            return true; // not a player's install (no release-manifest.txt): a developer's build never checks

        std::string url = "https://api.github.com/repos/" + cfg.repo + "/releases/latest";
        if (const char *test = std::getenv("VENDETTA_UPDATE_URL"); test && *test)
            url = test;
        auto answer = std::make_shared<Answer>();
        const auto started = std::chrono::steady_clock::now();
        std::thread(fetch, wide(url), wide("HCTPRecompiled/" + versionText(installed)), answer).detach();
        {
            std::unique_lock<std::mutex> lock(answer->mutex);
            if (!answer->ready.wait_for(lock, kAskFor, [&] { return answer->done; }))
            {
                std::cerr << "[update] no answer from GitHub within 2 s: playing" << std::endl;
                return true;
            }
        }
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started).count();
        if (!answer->error.empty())
        {
            std::cerr << "[update] couldn't check (" << answer->error << ", " << ms << " ms): playing" << std::endl;
            return true;
        }
        if (answer->status != 200)
        {
            // 404: no release yet, or a private repository
            std::cerr << "[update] no release to offer (GitHub answered " << answer->status << ", " << ms << " ms)" << std::endl;
            return true;
        }
        Version latest{};
        const std::string tag = jsonText(answer->body, "tag_name");
        if (!parseVersion(tag, latest) || answer->body.find("\"" + cfg.zipName + "\"") == std::string::npos)
        {
            std::cerr << "[update] the newest release (" << (tag.empty() ? "no tag" : tag) << ") has no "
                      << cfg.zipName << " to offer" << std::endl;
            return true;
        }
        const std::string latestText = versionText(latest), installedText = versionText(installed);
        if (latest <= installed)
        {
            std::cerr << "[update] up to date: " << installedText << " (" << ms << " ms)" << std::endl;
            return true;
        }
        std::string asked;
        std::getline(std::ifstream(folder / kAskedFile), asked);
        if (asked == latestText)
        {
            std::cerr << "[update] " << latestText << " is out; the player chose Play for it before: not asking again"
                      << std::endl;
            return true;
        }
        std::cerr << "[update] " << latestText << " is out (installed " << installedText << "): asking" << std::endl;
        if (!askUpdate(latestText, installedText))
        {
            std::ofstream(folder / kAskedFile) << latestText << "\n";
            std::cerr << "[update] the player chose Play" << std::endl;
            return true;
        }
        const std::filesystem::path launcher = folder.parent_path() / std::filesystem::path(wide(cfg.launcher));
        if (!startLauncher(launcher))
        {
            std::cerr << "[update] couldn't start " << launcher.string() << ": playing" << std::endl;
            return true;
        }
        std::cerr << "[update] the player chose Update: " << cfg.launcher << " --update started, the game ends" << std::endl;
        return false;
    }
#endif
}
