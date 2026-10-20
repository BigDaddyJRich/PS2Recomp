#include "hctp_commentary.h"
#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace
{
    constexpr uint16_t kNone = 0xFFFFu;
    // wrestler keys: roster id in the low byte, attire variant (cast code % 100) above it - 0 = any attire
    constexpr uint32_t kRosterMask = 0xFFu, kVariantShift = 8u;
    // entrance key only: belt held + 1 (CTitleHistoryData index 0..8) above the roster id + variant
    constexpr uint32_t kEntrantMask = 0xFFFFu, kBeltShift = 16u;

    struct Row
    {
        uint16_t w[8];
        uint16_t owner = 0xFFFFu; // move lines only: plays only when this roster id did the move (shared moves)
        uint32_t part = 13; // AFS partition its clips live in: 13 = PBPSVR.afs (retail), 15 = PBPBETA.afs (public beta)
    };

    // SvR LIST table numbers per commentary team (from the transcripts). A row's w1 = first line, w2 = partner reply.
    struct TeamTables
    {
        int welcome, prediction, title, banter, wrestlerBanter, outside, matchEnd, tag, interfere, finisherA, finisherB,
            moveReaction; // rows (move id, clip, chance %) - "What a shot!", "That's so dirty" (ids 1200-1203)
    };
    constexpr TeamTables kColeTazz = {0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 7};
    constexpr TeamTables kJrKing = {18, 19, 20, 21, 22, 5 /*no Raw table found yet*/, 23, 25, 26, 28, 29, 24};

    using Clock = std::chrono::steady_clock;

    class Engine
    {
    public:
        static Engine &get()
        {
            static Engine e;
            return e;
        }

        void event(uint32_t ev)
        {
            std::lock_guard<std::mutex> lock(m_);
            if (!loaded_)
                return;
            const auto now = Clock::now();
            const TeamTables &t = team();
            switch (ev)
            {
            case kPbpBell:
                if (now - lastBell_ < std::chrono::seconds(10))
                    return; // the end-of-match bell rings several times
                lastBell_ = now;
                if (!inMatch_ || now - matchStart_ > std::chrono::minutes(60))
                {
                    inMatch_ = true;
                    matchStart_ = now;
                    lastActivity_ = now;
                    lastMove_.clear();
                    queue_.clear();
                    std::fprintf(stderr, "[hctp-pbp] match start (%s, arena %d)%c", raw_ ? "JR/King" : "Cole/Tazz", arena_, 10);
                    // pay-per-views (stage >= 10): the welcome lines name Raw / SmackDown and a city - skip them
                    if (arena_ < 10)
                        enqueueRow(team().welcome, 0.0);
                    enqueueRow(team().prediction, arena_ < 10 ? 1.5 : 0.0);
                    nextBanter_ = now + std::chrono::seconds(25);
                    banterCount_ = 0; // first banter of the match is about a wrestler in it
                    heard_.clear();
                }
                else
                {
                    inMatch_ = false;
                    queue_.clear();
                    entrants_.clear();
                    std::fprintf(stderr, "[hctp-pbp] match end%c", 10);
                    enqueueRow(t.matchEnd, 1.0);
                }
                break;
            case kPbpStop:
                if (inMatch_ || !queue_.empty())
                    std::fprintf(stderr, "[hctp-pbp] stop: match torn down%c", 10);
                inMatch_ = false;
                queue_.clear();
                entrants_.clear();
                busyUntil_ = now;
                return;
            case kPbpTag:
                if (inMatch_ && queue_.empty())
                    enqueueRow(t.tag, 0.3);
                break;
            case kPbpInterfere:
                if (inMatch_ && queue_.empty())
                    enqueueRow(t.interfere, 0.3);
                break;
            case kPbpOutside:
                if (inMatch_ && queue_.empty())
                    enqueueRow(t.outside, 0.3);
                break;
            default:
                return;
            }
            cv_.notify_all();
        }

        // the arena picks the team (HCTP stage order, SLUS_207.87 arena name table): 0-3 SmackDown, 4-7 Raw, 8 Velocity,
        // 9 Heat, 10 SummerSlam, 11 Unforgiven, 12 No Mercy, 13 Survivor Series, 14 Armageddon, 15 No Way Out,
        // 16 WrestleMania XIX, 17 Backlash, 18 Judgment Day, 19 Vengeance, 20 Rebellion, 21 Insurrextion, 22 Bad Blood
        void setArena(int stage)
        {
            std::lock_guard<std::mutex> lock(m_);
            arena_ = stage;
            if (teamForced_)
                return;
            switch (stage)
            {
            case 4: case 5: case 6: case 7: case 9: case 11: case 14: case 21: case 22: raw_ = true; break;
            case 0: case 1: case 2: case 3: case 8: case 12: case 18: case 19: case 20: raw_ = false; break;
            default: raw_ = std::uniform_int_distribution<int>(0, 1)(rng_) == 1; // shared pay-per-views
            }
        }

        void setCharaRoster(uint32_t chara, uint32_t rosterId)
        {
            std::lock_guard<std::mutex> lock(m_);
            charaRoster_[chara] = rosterId & kRosterMask;
        }

        void setParticipants(const uint32_t *ids, uint32_t n)
        {
            std::lock_guard<std::mutex> lock(m_);
            participants_.assign(ids, ids + n);
            std::fprintf(stderr, "[hctp-pbp] participants:");
            for (uint32_t id : participants_)
                std::fprintf(stderr, " 0x%02X", id);
            std::fprintf(stderr, "%c", 10);
        }

        // one of the team's lines about this wrestler (rows with w0 == roster id; w1 = clip). id may carry an attire
        // variant (kVariantShift): that attire's own lines when it has any (nWo Hogan), else the wrestler's lines.
        void enqueueWrestlerLine(uint32_t id)
        {
            const int t = team().wrestlerBanter;
            if (t < 0 || size_t(t) >= tables_.size())
                return;
            std::vector<const Row *> rows;
            // Undertaker (Legend) 0x48 speaks with the Undertaker's lines (Jordan)
            const uint32_t alias = (id & kRosterMask) == 0x48u ? 0x03u : (id & kRosterMask);
            for (uint32_t key : {id, id & kRosterMask, alias})
            {
                for (const Row &r : tables_[t])
                    if (r.w[0] == key && r.w[1] < clipCount(r.part) && !heard_.count({r.part, r.w[1]}))
                        rows.push_back(&r);
                if (!rows.empty())
                    break;
            }
            if (rows.empty())
            {
                enqueueRow(team().banter, 0.0);
                return;
            }
            const Row &r = *rows[std::uniform_int_distribution<size_t>(0, rows.size() - 1)(rng_)];
            heard_.insert({r.part, r.w[1]});
            queue_.push_back({r.w[1], 0.0, r.part});
            if (r.w[2] != kNone && r.w[2] < clipCount(r.part)) // the partner's reply (pairBanter)
                queue_.push_back({r.w[2], 0.4, r.part});
        }

        // finisher / signature move calls: rows (w0 wrestler, w1 move id, w2 clip, w3 delay) of the team's two tables
        void move(uint32_t chara, uint32_t moveId)
        {
            std::lock_guard<std::mutex> lock(m_);
            lastActivity_ = Clock::now(); // wrestlers making move sounds = a match is really on screen
            bool always = false;
            const auto pit = charaRoster_.find(chara); // who did it (CChar -> roster id, set by the hook)
            const uint32_t performer = pit != charaRoster_.end() ? pit->second : 0xFFFFFFFFu;
            if (const auto it = moveMap_.find(moveId); it != moveMap_.end())
            {
                always = it->second.second;
                moveId = it->second.first; // HCTP move -> the SvR id whose lines name it (tools/map_moves.py)
            }
            if (!loaded_ || !inMatch_)
                return;
            uint32_t &last = lastMove_[chara];
            if (last == moveId)
                return; // same move still playing
            last = moveId;
            if (loggedMoves_.insert(moveId).second && loggedMoves_.size() < 2000)
                std::fprintf(stderr, "[hctp-pbp] move %u (chara %08x)%c", moveId, chara, 10);
            // timeline for mapping moves by hand ("the Spear was at 1:32"): every new move, seconds since the bell
            std::fprintf(stderr, "[hctp-pbp] mv %4.0fs %u chara %08x%c",
                         std::chrono::duration<double>(Clock::now() - matchStart_).count(), moveId, chara, 10);
            std::vector<const Row *> rows;
            for (int t : {team().finisherA, team().finisherB})
                if (t >= 0 && size_t(t) < tables_.size())
                    for (const Row &r : tables_[t])
                        if (r.w[1] == moveId && r.w[2] < clipCount(r.part) &&
                            (r.owner == 0xFFFFu || r.owner == performer))
                            rows.push_back(&r);
            if (rows.empty())
            {
                moveReaction(moveId, always, performer);
                return;
            }
            // attacker and receiver both carry the move id while it happens: one call per move
            const auto now = Clock::now();
            if (moveId == lastCalledMove_ && now - lastCalledAt_ < std::chrono::seconds(8))
                return;
            lastCalledMove_ = moveId;
            lastCalledAt_ = now;
            const Row &r = *rows[std::uniform_int_distribution<size_t>(0, rows.size() - 1)(rng_)];
            std::fprintf(stderr, "[hctp-pbp] finisher move %u -> line %u/%u%c", moveId, r.part, r.w[2], 10);
            queue_.clear(); // a finisher call beats whatever banter was waiting
            queue_.push_back({r.w[2], 0.2, r.part});
            cv_.notify_all();
        }

        // an ordinary move: the team's reaction line for it, by the game's own chance (w2 %), only when nobody is
        // talking and at most every 10 s (banter and finisher calls come first). Called with m_ held.
        void moveReaction(uint32_t moveId, bool always, uint32_t performer)
        {
            const int t = team().moveReaction;
            const auto now = Clock::now();
            if (t < 0 || size_t(t) >= tables_.size())
                return;
            // signature moves (Stunner, STO, Jackhammer...) are called every time, like finishers; others only
            // when nobody is talking, at the game's chance, at most every 10 s
            if (always)
            {
                if (moveId == lastCalledMove_ && now - lastCalledAt_ < std::chrono::seconds(8))
                    return;
            }
            else if (!queue_.empty() || now < busyUntil_ || now < nextReaction_)
                return;
            std::vector<const Row *> rows;
            for (const Row &r : tables_[t])
                if (r.w[0] == moveId && r.w[1] < clipCount(r.part) && (r.owner == 0xFFFFu || r.owner == performer))
                    rows.push_back(&r);
            if (rows.empty())
                return;
            const Row &r = *rows[std::uniform_int_distribution<size_t>(0, rows.size() - 1)(rng_)];
            const int chance = always ? 100 : r.w[2] <= 100u ? int(r.w[2]) : 25;
            if (std::uniform_int_distribution<int>(0, 99)(rng_) >= chance)
                return;
            nextReaction_ = now + std::chrono::seconds(10);
            if (always)
            {
                lastCalledMove_ = moveId;
                lastCalledAt_ = now;
                queue_.clear();
            }
            std::fprintf(stderr, "[hctp-pbp] move %u reaction -> line %u/%u%c", moveId, r.part, r.w[1], 10);
            queue_.push_back({r.w[1], 0.2, r.part});
            cv_.notify_all();
        }

        // Howard Finkel: the full "From ..., weighing ..., NAME" intro (non-title version), else from + weight + name,
        // else the name alone, else a generic line. Entrances happen before the opening bell.
        void entrance(uint32_t rosterId)
        {
            std::lock_guard<std::mutex> lock(m_);
            if (!loaded_ || inMatch_)
                return;
            auto pick = [&](uint32_t who, const std::string &kind, bool noTitle) -> int {
                std::vector<int> c;
                const auto it = finkel_.find(who);
                if (it != finkel_.end())
                    for (const auto &l : it->second)
                        if (l.kind == kind && (!noTitle || !isTitleLine(l.text)))
                            c.push_back(l.clip);
                return c.empty() ? -1 : c[std::uniform_int_distribution<size_t>(0, c.size() - 1)(rng_)];
            };
            // the longest line of a kind = the one with the nickname ("the people's champion the rock"); SYM clips
            // (pack 0..528) first: they are recorded as separate hometown / weight / name pieces for every wrestler
            auto best = [&](uint32_t who, const std::string &kind, bool symOnly) -> int {
                int clip = -1;
                size_t len = 0;
                const auto it = finkel_.find(who);
                if (it != finkel_.end())
                    for (const auto &l : it->second)
                        if (l.kind == kind && (!symOnly || l.clip < kSymClips) && l.text.size() > len &&
                            !isTitleLine(l.text) && l.text.rfind("this is", 0) != 0)
                            clip = l.clip, len = l.text.size();
                return clip;
            };
            // an attire with its own Finkel lines (nWo Hogan "0x20@N") uses them, else the wrestler's own
            const uint32_t entrant = rosterId & kEntrantMask;
            const uint32_t who = finkel_.count(entrant) ? entrant : rosterId & kRosterMask;
            // a belt holder (hook: CTitleHistoryData holders, belt + 1 above kBeltShift): Finkel's separate title
            // call ("the WWE Champion", 2K14; FINKEL_LINES connector kinds title_b<belt> / title_half_b<belt>) goes
            // right before the name, so the pieced intro is used whenever it exists
            const uint32_t belt = rosterId >> kBeltShift;
            int title = -1;
            if (belt != 0u)
            {
                const std::string b = std::to_string(belt - 1u);
                title = pick(kConnectors, "title_half_b" + b, false); // tag belts: "one half of the ... Champions"
                if (title < 0)
                    title = pick(kConnectors, "title_b" + b, false);
            }
            std::vector<int> intro;
            const int from = best(who, "from", true), weight = best(who, "weight", true),
                      name = best(who, "name", true);
            const int full = pick(who, "full", true);
            // both a SYM pieced intro and a LoW one-line intro: alternate at random so intros vary match to match
            const bool pieced = from >= 0 && weight >= 0 && name >= 0;
            const bool useFull = full >= 0 && (!pieced || (title < 0 && std::uniform_int_distribution<int>(0, 1)(rng_) == 1));
            if (int special = pick(who, "special", false); special >= 0)
                intro = {special}; // people who break the format ("the CEO of World Wrestling Entertainment ...")
            else if (useFull)
                intro = {full}; // Legends of WrestleMania: "From ..., weighing ... pounds, NAME" in one line
            else if (from >= 0 && weight >= 0 && name >= 0)
                intro = {from, weight, name};
            else if (int fw = pick(who, "from_weight", false), nm = best(who, "name", false); fw >= 0 && nm >= 0)
                intro = {fw, nm}; // 2K14: "From ..., weighing ... pounds" then the name
            else
            {
                if (from >= 0) intro.push_back(from);
                if (weight >= 0) intro.push_back(weight);
                if (int n = name >= 0 ? name : best(who, "name", false); n >= 0) intro.push_back(n);
            }
            if (intro.empty())
            {
                std::fprintf(stderr, "[hctp-pbp] entrance roster 0x%02X: no Finkel lines, silent%c", rosterId & kRosterMask, 10);
                return; // Jordan: no lines at all rather than generic ones
            }
            if (title >= 0) // "... weighing 295 pounds, | the WWE Champion, | Brock Lesnar" (one-line intros: title first)
                intro.insert(intro.size() > 1 ? intro.end() - 1 : intro.begin(), title);
            // a new show once the last entrance was long ago (or a match ended since)
            const auto now = Clock::now();
            if (now - lastEntrance_ > std::chrono::minutes(4))
                entrants_.clear();
            lastEntrance_ = now;
            if (std::find(entrants_.begin(), entrants_.end(), rosterId) != entrants_.end())
                return; // the same entrance event set more than once
            const bool female = isFemale(rosterId & kRosterMask);
            std::vector<int> seq;
            if (entrants_.empty())
            {
                if (int c = pick(kConnectors, "open", false); c >= 0) seq.push_back(c);
            }
            else
            {
                const bool prevFemale = isFemale(entrants_.front() & kRosterMask);
                if (int c = pick(kConnectors, prevFemale ? "and_her_opponent" : "and_his_opponent", false); c >= 0)
                    seq.push_back(c);
            }
            if (int c = pick(kConnectors, female ? "making_her" : "making_his", false); c >= 0) seq.push_back(c);
            seq.insert(seq.end(), intro.begin(), intro.end());
            entrants_.push_back(rosterId);
            std::fprintf(stderr, "[hctp-pbp] entrance roster 0x%02X -> %zu Finkel line(s)%c", rosterId & kRosterMask, seq.size(), 10);
            for (size_t i = 0; i < seq.size(); ++i)
                queue_.push_back({uint16_t(seq[i]), i ? 0.25 : 1.0, 14u});
            cv_.notify_all();
        }

        bool load()
        {
            std::ifstream f("BGM/PBPSVR_LIST.pac", std::ios::binary);
            std::vector<uint8_t> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            if (b.size() < 8 || std::memcmp(b.data(), "PAC ", 4) != 0)
            {
                std::cerr << "[hctp-pbp] BGM/PBPSVR_LIST.pac missing or bad - commentary off" << std::endl;
                return false;
            }
            // u32 count, count x (u16 id, u24 offset, u24 size); offsets relative to the end of the directory. Each
            // table: u32 category, u32 rowCount, 8 pad bytes, then rowCount rows of 8 u16 words (0xFFFF = unused).
            uint32_t count = 0;
            std::memcpy(&count, &b[4], 4);
            const size_t base = 8 + size_t(count) * 8;
            auto u24 = [&b](size_t p) { return uint32_t(b[p]) | uint32_t(b[p + 1]) << 8 | uint32_t(b[p + 2]) << 16; };
            tables_.assign(count, {});
            for (uint32_t i = 0; i < count && base <= b.size(); ++i)
            {
                const size_t off = base + u24(8 + i * 8 + 2), size = u24(8 + i * 8 + 5);
                if (size < 16 || off + size > b.size())
                    continue;
                uint32_t rows = 0;
                std::memcpy(&rows, &b[off + 4], 4);
                for (uint32_t r = 0; r < rows && 16 + (r + 1) * 16 <= size; ++r)
                {
                    Row row;
                    std::memcpy(row.w, &b[off + 16 + r * 16], 16);
                    tables_[i].push_back(row);
                }
            }
            // clip lengths from the ADX headers (total samples / rate) so lines never talk over each other
            std::ifstream a("BGM/PBPSVR.afs", std::ios::binary);
            char head[8] = {};
            if (!a.read(head, 8) || std::memcmp(head, "AFS", 3) != 0)
            {
                std::cerr << "[hctp-pbp] BGM/PBPSVR.afs missing or bad - commentary off" << std::endl;
                return false;
            }
            uint32_t n = 0;
            std::memcpy(&n, head + 4, 4);
            std::vector<uint32_t> dir(size_t(n) * 2);
            a.read(reinterpret_cast<char *>(dir.data()), std::streamsize(dir.size() * 4));
            seconds_.assign(n, 3.0f);
            samples_.assign(n, 0u);
            for (uint32_t i = 0; i < n && a; ++i)
            {
                uint8_t h[16] = {};
                a.seekg(dir[i * 2]);
                if (!a.read(reinterpret_cast<char *>(h), 16))
                    break;
                const uint32_t rate = uint32_t(h[8]) << 24 | h[9] << 16 | h[10] << 8 | h[11];
                const uint32_t total = uint32_t(h[12]) << 24 | h[13] << 16 | h[14] << 8 | h[15];
                if (h[0] == 0x80 && rate)
                    seconds_[i] = float(total) / float(rate), samples_[i] = total;
            }
            loadBeta();
            loadExtra();
            loadMoveMap();
            loadLineOwners();
            dropMisfiledBanter();
            pairBanter();
            loadFinkel();
            const char *teamEnv = std::getenv("HCTP_PBP_TEAM");
            raw_ = teamEnv && (teamEnv[0] == 'r' || teamEnv[0] == 'R');
            teamForced_ = teamEnv && teamEnv[0]; // HCTP_PBP_TEAM=raw|cole overrides the arena
            std::cerr << "[hctp-pbp] loaded " << tables_.size() << " tables, " << n << " clips" << std::endl;
            loaded_ = true;
            std::thread([this] { run(); }).detach();
            return true;
        }

    private:
        static constexpr int kSymClips = 529;            // FINKEL.afs: SYM clips first, then Legends of WrestleMania
        static constexpr uint32_t kConnectors = 0x10000u; // FINKEL_LINES.csv rows with roster_id "C"

        // "0x20" = roster id; "0x20@2" = that wrestler in attire variant 2 (cast code % 100) -> id | 2 << 8
        static uint32_t parseKey(const std::string &s)
        {
            const size_t at = s.find('@');
            const uint32_t id = uint32_t(std::strtoul(s.substr(0, at).c_str(), nullptr, 16)) & kRosterMask;
            return at == std::string::npos ? id : id | uint32_t(std::strtoul(s.c_str() + at + 1, nullptr, 10)) << kVariantShift;
        }

        // HCTP roster ids of the women (notes/hctp_roster_ids.csv): his/her in "and his opponent", "making her way"
        static bool isFemale(uint32_t id)
        {
            switch (id)
            {
            case 0x0E: case 0x14: case 0x1B: case 0x1F: case 0x26: case 0x29: case 0x2A: case 0x30: case 0x37: case 0x54:
                return true;
            default:
                return false;
            }
        }

        // title-holder versions ("the WWE Champion ...") wait until belts are tracked; nicknames like "the People's
        // Champion" are kept
        static bool isTitleLine(std::string t)
        {
            for (char &c : t)
                c = char(std::tolower(static_cast<unsigned char>(c)));
            for (const char *w : {"wwe champion", "intercontinental", "million dollar champ", "million-dollar champ", "tag team champ",
                                  "co-holder", "world heavyweight", "united states champion", "cruiserweight"})
                if (t.find(w) != std::string::npos)
                    return true;
            return false;
        }

        struct FinkelLine
        {
            std::string kind, text;
            int clip;
        };

        uint32_t clipCount(uint32_t part) const
        {
            return uint32_t(part == 15 ? betaSeconds_.size() : part == 16 ? extraSeconds_.size() : seconds_.size());
        }

        // SvR records banter as exchanges: Cole (even clip) then Tazz's reply (the next, odd clip) - JR / King the same.
        // Join each even line with the odd line right after it (same table, same wrestler) into one row that plays
        // both in order, and drop the reply as a row of its own so it never plays without the line it answers.
        void pairBanter()
        {
            size_t n = 0;
            std::vector<std::pair<uint16_t, uint16_t>> pairs; // (clip, reply), partition 13
            {
                std::ifstream f("BGM/PBP_BANTER_PAIRS.csv");
                std::string line;
                std::getline(f, line); // header
                unsigned part = 0, clip = 0, reply = 0;
                while (std::getline(f, line))
                    if (std::sscanf(line.c_str(), "%u,%u,%u", &part, &clip, &reply) == 3 && part == 13u)
                        pairs.push_back({uint16_t(clip), uint16_t(reply)});
            }
            for (const TeamTables *t : {&kColeTazz, &kJrKing})
                for (int table : {t->banter, t->wrestlerBanter})
                {
                    if (table < 0 || size_t(table) >= tables_.size())
                        continue;
                    auto &v = tables_[table];
                    std::map<std::tuple<uint32_t, uint16_t, uint16_t>, size_t> at; // (part, w0, clip) -> row
                    for (size_t i = 0; i < v.size(); ++i)
                        if (v[i].w[2] == kNone)
                            at[{v[i].part, v[i].w[0], v[i].w[1]}] = i;
                    std::vector<bool> gone(v.size(), false);
                    if (table == kJrKing.wrestlerBanter) // no parity there: BGM/PBP_BANTER_PAIRS.csv (tools/build_banter_pairs.py)
                    {
                        std::map<uint16_t, size_t> clipAt;
                        for (size_t i = 0; i < v.size(); ++i)
                            if (v[i].part == 13 && v[i].w[2] == kNone)
                                clipAt[v[i].w[1]] = i;
                        for (const auto &[clip, reply] : pairs)
                        {
                            const auto a = clipAt.find(clip), b = clipAt.find(reply);
                            if (a == clipAt.end() || b == clipAt.end() || gone[a->second] || gone[b->second])
                                continue;
                            v[a->second].w[2] = reply;
                            gone[b->second] = true;
                            ++n;
                        }
                    }
                    for (size_t i = 0; i < v.size(); ++i)
                    {
                        Row &r = v[i];
                        if (table == kJrKing.wrestlerBanter) // even/odd there is mostly one man talking twice
                            break;
                        if (gone[i] || r.w[2] != kNone || (r.w[1] & 1u) != 0u)
                            continue;
                        const auto it = at.find({r.part, r.w[0], uint16_t(r.w[1] + 1u)});
                        if (it == at.end() || gone[it->second])
                            continue;
                        r.w[2] = uint16_t(r.w[1] + 1u);
                        gone[it->second] = true;
                        ++n;
                    }
                    std::vector<Row> kept;
                    for (size_t i = 0; i < v.size(); ++i)
                        if (!gone[i])
                            kept.push_back(v[i]);
                    v = std::move(kept);
                }
            std::cerr << "[hctp-pbp] banter exchanges paired: " << n << std::endl;
        }

        // BGM/PBP_BANTER_DROP.csv (tools/build_banter_drop.py): wrestler-banter lines SvR files under a number that is a
        // different wrestler in HCTP (Beefcake under Hogan (Legend), Mankind under DiBiase...) - removed
        void dropMisfiledBanter()
        {
            std::ifstream f("BGM/PBP_BANTER_DROP.csv");
            std::string row;
            std::getline(f, row);
            std::set<std::pair<uint32_t, uint32_t>> drop;
            while (std::getline(f, row))
            {
                std::stringstream ss(row);
                std::string a, b;
                if (std::getline(ss, a, ',') && std::getline(ss, b, ','))
                    drop.insert({uint32_t(std::strtoul(a.c_str(), nullptr, 10)), uint32_t(std::strtoul(b.c_str(), nullptr, 10))});
            }
            size_t n = 0;
            for (const TeamTables *t : {&kColeTazz, &kJrKing})
                if (t->wrestlerBanter >= 0 && size_t(t->wrestlerBanter) < tables_.size())
                {
                    auto &v = tables_[t->wrestlerBanter];
                    const size_t before = v.size();
                    v.erase(std::remove_if(v.begin(), v.end(), [&](const Row &r) { return drop.count({r.part, r.w[1]}) != 0; }), v.end());
                    n += before - v.size();
                }
            std::cerr << "[hctp-pbp] misfiled banter dropped: " << n << std::endl;
        }

        // BGM/PBP_LINE_OWNERS.csv (tools/map_moves.py): partition, clip, roster id - retail move lines that name a
        // wrestler ("Kane with the chokeslam", Edge's spear) only play when that wrestler did the move
        void loadLineOwners()
        {
            std::ifstream f("BGM/PBP_LINE_OWNERS.csv");
            std::string row;
            std::getline(f, row);
            std::map<std::pair<uint32_t, uint32_t>, uint16_t> own;
            while (std::getline(f, row))
            {
                std::stringstream ss(row);
                std::string a, b, c;
                if (std::getline(ss, a, ',') && std::getline(ss, b, ',') && std::getline(ss, c, ','))
                    own[{uint32_t(std::strtoul(a.c_str(), nullptr, 10)), uint32_t(std::strtoul(b.c_str(), nullptr, 10))}] =
                        uint16_t(std::strtoul(c.c_str(), nullptr, 16));
            }
            size_t n = 0;
            for (const TeamTables *t : {&kColeTazz, &kJrKing})
                for (const auto &[table, col] : {std::pair{t->finisherA, 2}, std::pair{t->finisherB, 2}, std::pair{t->moveReaction, 1}})
                    if (table >= 0 && size_t(table) < tables_.size())
                        for (Row &r : tables_[table])
                            if (r.owner == 0xFFFFu)
                                if (const auto it = own.find({r.part, r.w[col]}); it != own.end())
                                    r.owner = it->second, ++n;
            std::cerr << "[hctp-pbp] owner-only move lines: " << n << " rows" << std::endl;
        }

        // BGM/PBP_MOVEMAP.csv (tools/map_moves.py): hctp_move, svr_move, always
        void loadMoveMap()
        {
            std::ifstream f("BGM/PBP_MOVEMAP.csv");
            std::string row;
            std::getline(f, row);
            while (std::getline(f, row))
            {
                std::stringstream ss(row);
                std::string a, b, c;
                if (std::getline(ss, a, ',') && std::getline(ss, b, ','))
                {
                    std::getline(ss, c, ',');
                    moveMap_[uint32_t(std::strtoul(a.c_str(), nullptr, 10))] = {uint32_t(std::strtoul(b.c_str(), nullptr, 10)),
                                                                                !c.empty() && c[0] == '1'};
                }
            }
            std::cerr << "[hctp-pbp] move map: " << moveMap_.size() << " HCTP moves" << std::endl;
        }

        // SvR public beta (Jul 2004): its Cole/Tazz set is the HCTP-era recording with every HCTP wrestler (legends
        // included) - the main Cole/Tazz source. Each beta table is merged into the retail table with the same layout:
        // all beta rows, plus the retail rows whose line is not in the beta (same recording = ADX length +-2 samples).
        void loadBeta()
        {
            std::ifstream f("BGM/PBPBETA_LIST.pac", std::ios::binary);
            std::vector<uint8_t> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            std::vector<uint32_t> samples;
            betaSeconds_ = adxSeconds("BGM/PBPBETA.afs", &samples);
            if (b.size() < 8 || std::memcmp(b.data(), "PAC ", 4) != 0 || betaSeconds_.empty())
            {
                std::cerr << "[hctp-pbp] BGM/PBPBETA.afs / PBPBETA_LIST.pac missing - retail Cole/Tazz only" << std::endl;
                betaSeconds_.clear();
                return;
            }
            uint32_t count = 0;
            std::memcpy(&count, &b[4], 4);
            const size_t base = 8 + size_t(count) * 8;
            auto u24 = [&b](size_t p) { return uint32_t(b[p]) | uint32_t(b[p + 1]) << 8 | uint32_t(b[p + 2]) << 16; };
            std::vector<std::vector<Row>> beta(count);
            for (uint32_t i = 0; i < count && base <= b.size(); ++i)
            {
                const size_t off = base + u24(8 + i * 8 + 2), size = u24(8 + i * 8 + 5);
                if (size < 16 || off + size > b.size())
                    continue;
                uint32_t rows = 0;
                std::memcpy(&rows, &b[off + 4], 4);
                for (uint32_t r = 0; r < rows && 16 + (r + 1) * 16 <= size; ++r)
                {
                    Row row;
                    std::memcpy(row.w, &b[off + 16 + r * 16], 16);
                    row.part = 15;
                    beta[i].push_back(row);
                }
            }
            // the beta's own data files two lines under the wrong wrestler (checked against the transcripts)
            for (Row &r : beta.size() > 4 ? beta[4] : beta[0])
            {
                if (beta.size() <= 4)
                    break;
                if (r.w[1] == 218u && r.w[0] == 0x03u)
                    r.w[0] = 0x04u; // "...RVD's gonna smoke this guy" was filed under the Undertaker
                else if (r.w[1] == 394u && r.w[0] == 0x53u)
                    r.w[0] = 0x55u; // the Ultimo Dragon line was filed under Rhyno
                else if (r.w[1] == 306u && r.w[0] == 0x09u)
                    r.w[0] = 0x53u; // "the man beast from Detroit" (Rhyno) was filed under Benoit
            }
            std::set<uint32_t> inBeta;
            for (uint32_t s : samples)
                for (uint32_t d = 0; d <= 4; ++d)
                    inBeta.insert(s + d - 2);
            // beta table, retail table (same layout), clip column checked for duplicates
            static const int kMap[][3] = {{0, 0, 1}, {1, 1, 1}, {2, 2, 1}, {3, 3, 1}, {4, 4, 1},
                                          {5, 5, 1}, {6, 6, 2}, {8, 11, 2}, {9, 12, 2}};
            size_t added = 0, dropped = 0;
            for (const auto &m : kMap)
            {
                if (size_t(m[0]) >= beta.size() || size_t(m[1]) >= tables_.size())
                    continue;
                std::vector<Row> merged = beta[m[0]];
                if (m[2] == 2) // beta finisher rows: w0 = the HCTP roster id the line is about -> owner-only
                    for (Row &r : merged)
                        if (r.w[0] < 0x56u)
                            r.owner = r.w[0];
                added += merged.size();
                for (const Row &r : tables_[m[1]])
                {
                    const uint16_t c = r.w[m[2]];
                    if (c != kNone && c < samples_.size() && inBeta.count(samples_[c]))
                        ++dropped;
                    else
                        merged.push_back(r);
                }
                tables_[m[1]] = std::move(merged);
            }
            std::cerr << "[hctp-pbp] public beta: " << betaSeconds_.size() << " clips, " << added << " rows, "
                      << dropped << " retail duplicates dropped" << std::endl;
        }

        // Lines no table points at, keyed by hand (tools/build_pbp_extra.py): retail clips left on the disc for cut
        // wrestlers, and other games' lines in BGM/PBPEXTRA.afs (partition 16: SYM, SvR 2006, 2K14).
        // BGM/PBPEXTRA_LIST.csv: team (cole|raw), kind (wrestler|finisher), key (roster id hex | move id), partition, clip
        void loadExtra()
        {
            extraSeconds_ = adxSeconds("BGM/PBPEXTRA.afs");
            std::ifstream f("BGM/PBPEXTRA_LIST.csv");
            std::string row;
            std::getline(f, row); // header
            size_t n = 0;
            while (std::getline(f, row))
            {
                std::stringstream ss(row);
                std::string team, kind, key, part, clip;
                if (!std::getline(ss, team, ',') || !std::getline(ss, kind, ',') || !std::getline(ss, key, ',') ||
                    !std::getline(ss, part, ',') || !std::getline(ss, clip, ','))
                    continue;
                const TeamTables &t = (team == "raw") ? kJrKing : kColeTazz;
                Row r;
                std::fill(std::begin(r.w), std::end(r.w), kNone);
                r.part = uint32_t(std::strtoul(part.c_str(), nullptr, 10));
                const uint16_t c = uint16_t(std::strtoul(clip.c_str(), nullptr, 10));
                const uint16_t k = uint16_t(kind == "wrestler" ? parseKey(key) : std::strtoul(key.c_str(), nullptr, 10));
                if (const size_t at = key.find('@'); kind == "finisher" && at != std::string::npos)
                    r.owner = uint16_t(std::strtoul(key.c_str() + at + 1, nullptr, 16)); // "4595@0x2F"
                int table = -1;
                if (kind == "wrestler")
                    table = t.wrestlerBanter, r.w[0] = k, r.w[1] = c;
                else if (kind == "finisher")
                    table = t.finisherA, r.w[1] = k, r.w[2] = c;
                if (table < 0 || size_t(table) >= tables_.size() || c >= clipCount(r.part))
                    continue;
                tables_[table].push_back(r);
                ++n;
            }
            std::cerr << "[hctp-pbp] extra lines: " << n << " (" << extraSeconds_.size() << " clips in PBPEXTRA.afs)"
                      << std::endl;
        }

        static std::vector<float> adxSeconds(const char *path, std::vector<uint32_t> *samplesOut = nullptr)
        {
            std::vector<float> out;
            std::ifstream a(path, std::ios::binary);
            char head[8] = {};
            if (!a.read(head, 8) || std::memcmp(head, "AFS", 3) != 0)
                return out;
            uint32_t n = 0;
            std::memcpy(&n, head + 4, 4);
            std::vector<uint32_t> dir(size_t(n) * 2);
            a.read(reinterpret_cast<char *>(dir.data()), std::streamsize(dir.size() * 4));
            out.assign(n, 3.0f);
            for (uint32_t i = 0; i < n && a; ++i)
            {
                uint8_t h[16] = {};
                a.seekg(dir[i * 2]);
                if (!a.read(reinterpret_cast<char *>(h), 16))
                    break;
                const uint32_t rate = uint32_t(h[8]) << 24 | h[9] << 16 | h[10] << 8 | h[11];
                const uint32_t total = uint32_t(h[12]) << 24 | h[13] << 16 | h[14] << 8 | h[15];
                if (h[0] == 0x80 && rate)
                    out[i] = float(total) / float(rate);
                if (samplesOut)
                    samplesOut->push_back(total);
            }
            return out;
        }

        void loadFinkel()
        {
            finkelSeconds_ = adxSeconds("BGM/FINKEL.afs");
            std::ifstream f("BGM/FINKEL_LINES.csv");
            std::string row;
            std::getline(f, row); // header
            size_t n = 0;
            while (std::getline(f, row))
            {
                std::stringstream ss(row);
                std::string id, kind, clip, text;
                if (!std::getline(ss, id, ',') || !std::getline(ss, kind, ',') || !std::getline(ss, clip, ','))
                    continue;
                std::getline(ss, text);
                if (id.empty() || id[0] == '-')
                    continue;
                const uint32_t key = id == "C" ? kConnectors : parseKey(id);
                finkel_[key].push_back({kind, text, std::stoi(clip)});
                ++n;
            }
            std::cerr << "[hctp-pbp] Finkel: " << finkelSeconds_.size() << " clips, " << n << " lines for "
                      << finkel_.size() << " wrestlers" << std::endl;
        }

        struct Line
        {
            uint16_t clip;
            double gapBefore;       // seconds of silence before this line
            uint32_t partition = 13; // 13 = PBPSVR.afs, 14 = FINKEL.afs
        };

        const TeamTables &team() const { return raw_ ? kJrKing : kColeTazz; }

        // queue one random row of a table: its first line, then the partner's reply (0 / 0xFFFF / keys skipped)
        void enqueueRow(int table, double gap)
        {
            if (table < 0 || size_t(table) >= tables_.size() || tables_[table].empty())
                return;
            // no repeats within a match: pick among rows not heard yet (all heard: start over)
            std::vector<const Row *> fresh;
            for (const Row &r : tables_[table])
                if (!heard_.count({r.part, r.w[1]}))
                    fresh.push_back(&r);
            if (fresh.empty())
                for (const Row &r : tables_[table])
                    fresh.push_back(&r);
            const Row &row = *fresh[std::uniform_int_distribution<size_t>(0, fresh.size() - 1)(rng_)];
            heard_.insert({row.part, row.w[1]});
            bool first = true;
            for (int k = 1; k <= 2; ++k)
            {
                const uint16_t c = row.w[k];
                if (c == kNone || c >= clipCount(row.part) || (c == 0 && k == 1 && row.w[2] != kNone))
                    continue;
                queue_.push_back({c, first ? gap : 0.4, row.part});
                first = false;
            }
        }

        // player thread: plays queued lines back to back, and general banter every 25-45 s during a match
        void run()
        {
            std::unique_lock<std::mutex> lock(m_);
            for (;;)
            {
                cv_.wait_for(lock, std::chrono::milliseconds(250));
                const auto now = Clock::now();
                if (now < busyUntil_)
                    continue;
                // left the match without a closing bell (quit to the menu): no move sounds for 90 s = match over;
                // a shorter lull (replays, menus, pauses) only holds the banter back
                if (inMatch_ && now - lastActivity_ > std::chrono::seconds(90))
                {
                    inMatch_ = false;
                    queue_.clear();
                    std::fprintf(stderr, "[hctp-pbp] no match activity for 90 s - match over%c", 10);
                }
                if (queue_.empty() && inMatch_ && now >= nextBanter_ && now - lastActivity_ < std::chrono::seconds(20))
                {
                    // every second banter is about someone in the match - only those with lines for this team
                    // (the match list also holds the referee / managers)
                    std::vector<uint32_t> withLines;
                    if (const int t = team().wrestlerBanter; t >= 0 && size_t(t) < tables_.size())
                        for (uint32_t id : participants_)
                            for (const Row &r : tables_[t])
                                if ((r.w[0] == id || r.w[0] == (id & kRosterMask) || ((id & kRosterMask) == 0x48u && r.w[0] == 0x03u)) &&
                                    !heard_.count({r.part, r.w[1]})) // only wrestlers with lines not heard yet
                                {
                                    withLines.push_back(id);
                                    break;
                                }
                    // the first banter is about someone in the match, then one of every two (Cole/Tazz) or every three
                    // (JR/King, whose wrestler lines are many) - general banter dominates (Jordan, Oct 20)
                    const uint32_t every = team().wrestlerBanter == kJrKing.wrestlerBanter ? 3u : 2u;
                    if (!withLines.empty() && (banterCount_++ % every) == 0)
                        enqueueWrestlerLine(withLines[std::uniform_int_distribution<size_t>(0, withLines.size() - 1)(rng_)]);
                    else
                        enqueueRow(team().banter, 0.0);
                    nextBanter_ = now + std::chrono::seconds(20 + std::uniform_int_distribution<int>(0, 15)(rng_));
                }
                if (queue_.empty())
                {
                    // quiet: give the audio player back so the commentary never holds one between lines
                    if (playerHeld_)
                    {
                        playerHeld_ = false;
                        lock.unlock();
                        hctpNativeStopCommentary();
                        lock.lock();
                    }
                    continue;
                }
                Line line = queue_.front();
                if (now < busyUntil_ + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(line.gapBefore)))
                    continue;
                queue_.pop_front();
                const std::vector<float> &secs =
                    line.partition == 14 ? finkelSeconds_ : line.partition == 15 ? betaSeconds_
                                                          : line.partition == 16 ? extraSeconds_ : seconds_;
                const float len = line.clip < secs.size() ? secs[line.clip] : 3.0f;
                busyUntil_ = now + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(len));
                lock.unlock();
                const bool ok = hctpNativePlayAfsClip(line.partition, line.clip);
                playerHeld_ = ok;
                std::fprintf(stderr, "[hctp-pbp] line %u/%u (%.1f s)%s%c", line.partition, line.clip, len, ok ? "" : " REJECTED", 10);
                lock.lock();
            }
        }

        std::mutex m_;
        std::condition_variable cv_;
        std::vector<std::vector<Row>> tables_;
        std::vector<float> seconds_, finkelSeconds_, betaSeconds_, extraSeconds_;
        std::vector<uint32_t> samples_; // retail clip lengths in samples (beta duplicate check)
        std::map<uint32_t, std::vector<FinkelLine>> finkel_;
        std::vector<uint32_t> entrants_;
        std::map<uint32_t, uint32_t> lastMove_;
        std::map<uint32_t, std::pair<uint32_t, bool>> moveMap_;
        std::set<std::pair<uint32_t, uint32_t>> heard_; // (partition, clip) said this match - no repeats
    public:
        std::map<uint32_t, uint32_t> charaRoster_; // CChar guest address -> roster id
    private:
        std::vector<uint32_t> participants_;
        uint32_t banterCount_ = 0;
        std::set<uint32_t> loggedMoves_;
        Clock::time_point lastEntrance_{};
        std::deque<Line> queue_;
        std::mt19937 rng_{std::random_device{}()};
        bool loaded_ = false, raw_ = false, inMatch_ = false, teamForced_ = false, playerHeld_ = false;
        int arena_ = 0;
        Clock::time_point lastBell_{}, matchStart_{}, nextBanter_{}, busyUntil_{}, lastActivity_{}, lastCalledAt_{}, nextReaction_{};
        uint32_t lastCalledMove_ = 0;
    };
}

namespace
{
    bool engineReady()
    {
        static const bool s_loaded = Engine::get().load(); // one load, one player thread
        return s_loaded;
    }
}

extern "C" void hctpCommentaryEvent(uint32_t event)
{
    if (engineReady())
        Engine::get().event(event);
}

extern "C" void hctpCommentaryEntrance(uint32_t rosterId)
{
    if ((rosterId & kRosterMask) < 0x56u && engineReady())
        Engine::get().entrance(rosterId);
}

extern "C" void hctpCommentaryMove(uint32_t chara, uint32_t moveId)
{
    if (engineReady())
        Engine::get().move(chara, moveId);
}

extern "C" void hctpCommentaryParticipants(const uint32_t *rosterIds, uint32_t count)
{
    if (engineReady())
        Engine::get().setParticipants(rosterIds, count);
}

extern "C" void hctpCommentaryCharaRoster(uint32_t chara, uint32_t rosterId)
{
    if (engineReady())
        Engine::get().setCharaRoster(chara, rosterId);
}

extern "C" void hctpCommentaryArena(int stage)
{
    if (engineReady())
        Engine::get().setArena(stage);
}
