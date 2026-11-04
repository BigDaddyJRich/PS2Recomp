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
            moveReaction, // rows (move id, clip, chance %) - "What a shot!", "That's so dirty" (ids 1200-1203)
            women,        // women's-match banter (SvR cat 9: Cole id 9, JR id 27)
            braPanties;   // bra & panties banter (SvR cat 11: Cole id 11, JR id 30)
    };
    constexpr TeamTables kColeTazz = {0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 7, 10, 13};
    // JR/King outside = id 21 (SvR maps cat 4 punishment to the JR banter table, 0x34c6b0)
    constexpr TeamTables kJrKing = {18, 19, 20, 21, 22, 21, 23, 25, 26, 28, 29, 24, 27, 30};

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
                    spoken_.clear(); // SvR: one superstar line per wrestler per match
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
                // ReleaseGameLinkObj also fires on the scene change between two entrances: keep the entrants then,
                // or Finkel opens the 2nd intro with "scheduled for one fall" again instead of "and his opponent"
                if (inMatch_)
                    entrants_.clear();
                inMatch_ = false;
                queue_.clear();
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
        // 16 WrestleMania XIX, 17 Backlash, 18 Judgment Day, 19 Vengeance, 20 Rebellion, 21 Insurrextion, 22 Bad Blood,
        // 23-29 backstage / outside areas, 30 Royal Rumble (verified against the ELF list at 0x358740)
        void setArena(int stage)
        {
            std::lock_guard<std::mutex> lock(m_);
            arena_ = stage;
            if (teamForced_)
                return;
            switch (stage)
            {
            case 4: case 5: case 6: case 7: case 9: case 11: case 14: case 21: case 22: raw_ = true; break;
            case 0: case 1: case 2: case 3: case 8: case 12: case 19: case 20: raw_ = false; break;
            // 2003 brand split (Jordan): Royal Rumble 30, No Way Out, WrestleMania, Backlash, Judgment Day, SummerSlam,
            // Survivor Series and the backstage areas 23-29 are shared -> the wrestlers' brand (SvR: a shared show
            // keeps the announce team of the brand), random only when they split evenly / have none
            default:
                raw_ = brand_ >= 0 ? brand_ == 0 : std::uniform_int_distribution<int>(0, 1)(rng_) == 1;
                std::fprintf(stderr, "[hctp-pbp] shared arena %d -> %s (%s)%c", stage, raw_ ? "JR/King" : "Cole/Tazz",
                             brand_ >= 0 ? "wrestlers' brand" : "random", 10);
            }
        }

        // majority Raw/SmackDown brand of the match's wrestlers: 0 Raw, 1 SmackDown, -1 none / tie
        void setBrand(int brand)
        {
            std::lock_guard<std::mutex> lock(m_);
            brand_ = brand;
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

        // at the opening bell (cage / HIAC / Chamber / Rumble have no entrances): rule byte 0x4920C0 and the B&P flag
        // (rule object +0x27: MakeKindBit2 0x1d37b0 only allows the Brapan moves 0x1130-0x113D when it is set)
        void setBellRule(uint32_t rule, bool braPanties)
        {
            std::lock_guard<std::mutex> lock(m_);
            matchRule_ = rule & 0xFFu;
            braPanties_ = braPanties;
            std::fprintf(stderr, "[hctp-pbp] bell rule 0x%02X%s%c", matchRule_, braPanties ? " (bra & panties)" : "", 10);
            // Royal Rumble (0x0D debug "ROYAL RUMBLE" / 0x52): no entrance events. Opening bell (inMatch_ still false;
            // the match-end bell comes with it true): LoW Finkel 1369 "it is now time for the Royal Rumble ... the
            // individual who drew number one" + the first starter, 1418 "here is the individual who drew number two"
            // + the second; later entrants: rumbleEntrant (CGameSync::Draw spawn hook)
            if ((matchRule_ == 0x0Du || matchRule_ == 0x52u) && !inMatch_)
            {
                rumbleCount_ = 0;
                std::vector<uint32_t> starters;
                for (uint32_t id : participants_)
                    if ((id & kRosterMask) != 0x31u && (id & kRosterMask) != 0x33u && starters.size() < 2u)
                        starters.push_back(id & kRosterMask);
                for (size_t k = 0; k < starters.size(); ++k)
                {
                    queue_.push_back({uint16_t(k ? 1418 : 1369), k ? 0.25 : 1.0, 14u});
                    if (int nm = nameClip(starters[k]); nm >= 0)
                        queue_.push_back({uint16_t(nm), 0.25, 14u});
                    ++rumbleCount_;
                    std::fprintf(stderr, "[hctp-pbp] rumble starter #%zu roster 0x%02X%c", k + 1, starters[k], 10);
                }
                cv_.notify_all();
            }
        }

        // every wrestler in the match is a woman (referees 0x31 / 0x33 and empty lists do not count)
        bool womenMatch() const
        {
            uint32_t n = 0;
            for (uint32_t id : participants_)
            {
                const uint32_t r = id & kRosterMask;
                if (r == 0x31u || r == 0x33u)
                    continue;
                if (!isFemale(r))
                    return false;
                ++n;
            }
            if (n)
                return true;
            return !entrants_.empty() &&
                   std::all_of(entrants_.begin(), entrants_.end(), [](uint32_t id) { return isFemale(id & kRosterMask); });
        }

        // a row whose clips (w[from]..) talk about a man (BGM/PBP_MALE_LINES.csv, tools/build_male_lines.py)
        bool maleRow(const Row &r, int from) const
        {
            for (int k = from; k < 8 && r.w[k] != kNone; ++k)
                if (maleLines_.count({r.part, r.w[k]}))
                    return true;
            return false;
        }

        void loadMaleLines()
        {
            std::ifstream f("BGM/PBP_MALE_LINES.csv");
            std::string row;
            std::getline(f, row);
            while (std::getline(f, row))
            {
                std::stringstream ss(row);
                std::string a, b;
                if (std::getline(ss, a, ',') && std::getline(ss, b, ','))
                    maleLines_.insert({uint32_t(std::strtoul(a.c_str(), nullptr, 10)), uint32_t(std::strtoul(b.c_str(), nullptr, 10))});
            }
            std::cerr << "[hctp-pbp] male-pronoun lines (skipped for women): " << maleLines_.size() << std::endl;
        }

        void setTagMatch(bool tag)
        {
            std::lock_guard<std::mutex> lock(m_);
            tagMatch_ = tag;
        }

        void setTeamSize(uint32_t size)
        {
            std::lock_guard<std::mutex> lock(m_);
            teamSize_ = size == 3u ? 3u : 2u;
        }

        void setMatchRule(uint32_t rule)
        {
            std::lock_guard<std::mutex> lock(m_);
            matchRule_ = rule & 0xFFu;
            openerClip_ = openerFor(matchRule_);
            std::fprintf(stderr, "[hctp-pbp] entrance match rule 0x%02X -> Finkel opener %d%c", matchRule_, openerClip_, 10);
        }

        // Finkel's "The following contest is ..." per HCTP eRuleType (names: misc.pac debug rule table; team layout:
        // ELF rule table 0x45AF60; notes/COMMENTARY_MATCHTYPE_AUDIT_20261027.md). FINKEL.afs 1370-1405 = LoW Finkel.
        static int openerFor(uint32_t rule)
        {
            switch (rule)
            {
            case 0x01: case 0x02: case 0x05: case 0x0A: case 0x0B: case 0x0C: return 1381; // handicap
            case 0x03: case 0x04: case 0x07: return 1375;                         // tag team
            case 0x06: case 0x08: case 0x09: return 1376;                         // six-man tag
            case 0x10: case 0x11: case 0x12: case 0x13: case 0x14: case 0x15: case 0x16: return 1389; // hardcore
            case 0x0E: return 1398;                                               // King of the Ring tournament
            case 0x17: return 1379;                                               // triple threat
            case 0x18: return 1380;                                               // fatal four-way
            case 0x19: return 1385;                                               // 4/5/6-man over the top battle royal
            case 0x1A: return 1386;
            case 0x1B: return 1387;
            case 0x1C: case 0x1D: case 0x1E: case 0x1F: case 0x20: case 0x21: case 0x22: return 1394; // steel cage
            case 0x23: case 0x24: case 0x25: case 0x26: case 0x27: return 1390;  // hell in a cell
            case 0x2B: case 0x2E: case 0x39: return 1391;                         // iron man
            case 0x2F: case 0x30: case 0x31: case 0x32: case 0x33: return 1393;  // ladder
            case 0x34: case 0x36: case 0x37: return 1396;                         // tables
            case 0x35: case 0x38: return 1397;                                    // tag team tables
            case 0x46: return 1392;                                               // last man standing
            case 0x48: return 1405;                                               // street fight = no holds barred
            case 0x49: return 1395;                                               // submission
            case 0x4A: return 442;                                                // ultimate submission (SYM)
            case 0x4B: case 0x4C: return 1377;                                    // elimination tag
            case 0x4D: case 0x4E: case 0x4F: case 0x50: return 1378;              // six-man elimination tag
            default: return -1;                                                   // one fall (connector "open")
            }
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
            std::fprintf(stderr, "[hctp-pbp] wrestler line for 0x%02X (row key 0x%02X) -> %u/%u%c", id, r.w[0], r.part,
                         r.w[1], 10);
            pushSequence(r, 1, 0.0);
        }

        // finisher / signature move calls: rows (w0 wrestler, w1 move id, w2 clip, w3 delay) of the team's two tables
        void move(uint32_t chara, uint32_t moveId)
        {
            std::lock_guard<std::mutex> lock(m_);
            lastActivity_ = Clock::now(); // wrestlers making move sounds = a match is really on screen
            bool always = false;
            // retail SvR rows (partition 13) key on SvR's move numbering; the public beta (15) and the extra lines (16)
            // key on HCTP's own ids (beta moves.pac == HCTP), so only retail rows use the translated id
            const uint32_t hctpMove = moveId;
            const auto pit = charaRoster_.find(chara); // who did it (CChar -> roster id, set by the hook)
            uint32_t performer = pit != charaRoster_.end() ? pit->second : 0xFFFFFFFFu;
            if (const auto it = moveMap_.find(moveId); it != moveMap_.end())
            {
                always = it->second.second;
                moveId = it->second.first; // HCTP move -> the SvR id whose lines name it (tools/map_moves.py)
            }
            if (!loaded_ || !inMatch_)
                return;
            uint32_t &last = lastMove_[chara];
            if (last == hctpMove)
                return; // same move still playing
            last = hctpMove;
            // attacker and receiver both report the move: the chara that reported it first is the attacker (Jordan:
            // Torrie's "she can slap that hard" row, w0 0x2A, played whenever Torrie RECEIVED the women's slap 4740)
            {
                const auto now = Clock::now();
                auto st = moveStarter_.find(hctpMove);
                if (st == moveStarter_.end() || now - st->second.second > std::chrono::seconds(3))
                    moveStarter_[hctpMove] = {chara, now};
                else if (st->second.first != chara)
                {
                    const auto ait = charaRoster_.find(st->second.first);
                    const uint32_t attacker = ait != charaRoster_.end() ? ait->second : 0xFFFFFFFFu;
                    if (attacker != performer)
                        std::fprintf(stderr, "[hctp-pbp] move %u: receiver 0x%X -> attacker 0x%X%c", hctpMove, performer, attacker, 10);
                    performer = attacker;
                }
            }
            const bool femalePerformer = performer < 0x100u && isFemale(performer);
            if (loggedMoves_.insert(hctpMove).second && loggedMoves_.size() < 2000)
                std::fprintf(stderr, "[hctp-pbp] move %u -> svr %u (chara %08x)%c", hctpMove, moveId, chara, 10);
            // timeline for mapping moves by hand ("the Spear was at 1:32"): every new move, seconds since the bell
            std::fprintf(stderr, "[hctp-pbp] mv %4.0fs %u chara %08x%c",
                         std::chrono::duration<double>(Clock::now() - matchStart_).count(), moveId, chara, 10);
            // SvR 0x213870: ONE of the two finisher tables at random, a row with w0 == the performer's roster id and
            // w1 == the move (retail rows; w0 255 = anyone); beta / extra rows carry their owner instead
            (void)always;
            std::vector<const Row *> rows;
            const uint32_t who = performer == 0x48u ? 0x03u : performer; // Undertaker (Legend) uses Taker's lines
            const int coin = std::uniform_int_distribution<int>(0, 1)(rng_) ? team().finisherB : team().finisherA;
            for (int t : {team().finisherA, team().finisherB})
                if (t >= 0 && size_t(t) < tables_.size())
                for (const Row &r : tables_[t])
                    if ((t == coin || r.part != 13) && // the coin is SvR's rule for its own rows; added packs always
                        r.w[1] == (r.part == 13 ? moveId : hctpMove) && r.w[2] < clipCount(r.part) &&
                        (r.part == 13 && r.owner == 0xFFFFu ? (r.w[0] == 0xFFu || r.w[0] == who)
                                                             : (r.owner == 0xFFFFu || r.owner == performer)) &&
                        !(femalePerformer && (r.part != 13 || r.w[0] == 0xFFu) && maleRow(r, 2)))
                        rows.push_back(&r);
            if (rows.empty())
            {
                moveReaction(moveId, false, performer, hctpMove, femalePerformer);
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
            // w3 = the row's delay in frames (SvR counts it down before speaking)
            queue_.push_back({r.w[2], r.part == 13 && r.w[3] != kNone && r.w[3] < 600u ? r.w[3] / 60.0 : 0.2, r.part});
            cv_.notify_all();
        }

        // an ordinary move: the team's reaction line for it, by the game's own chance (w2 %), only when nobody is
        // talking and at most every 10 s (banter and finisher calls come first). Called with m_ held.
        void moveReaction(uint32_t moveId, bool always, uint32_t performer, uint32_t hctpMove, bool femalePerformer)
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
            else if (!queue_.empty() || now < busyUntil_) // SvR 0x213ae0: only while nobody is talking
                return;
            std::vector<const Row *> rows;
            for (const Row &r : tables_[t])
                if (r.w[0] == (r.part == 13 ? moveId : hctpMove) && r.w[1] < clipCount(r.part) && (r.owner == 0xFFFFu || r.owner == performer) &&
                    !(femalePerformer && maleRow(r, 1)))
                    rows.push_back(&r);
            if (rows.empty())
                return;
            const Row &r = *rows[std::uniform_int_distribution<size_t>(0, rows.size() - 1)(rng_)];
            const int chance = always ? 100 : r.w[2] <= 100u ? int(r.w[2]) : 25;
            if (std::uniform_int_distribution<int>(0, 99)(rng_) >= chance)
                return;
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
            entranceLocked(rosterId);
        }

        // SvR ring announcer (SLUS_210.60 0x211c10): everyone walking out in ONE entrance is one group. A known team
        // gets one team call instead of the members' intros; any other group = each member's intro back to back.
        // Tag champions in one entrance (runner bit 24): the team call follows the whole group.
        void entranceGroup(const uint32_t *ids, uint32_t n)
        {
            std::lock_guard<std::mutex> lock(m_);
            if (n < 2u)
            {
                if (n)
                    entranceLocked(ids[0]);
                return;
            }
            // a new show first (entranceLocked clears it too, but 'before' below must not count the old show)
            if (Clock::now() - lastEntrance_ > std::chrono::minutes(4))
                entrants_.clear();
            std::vector<uint32_t> g;
            for (uint32_t k = 0; k < n; ++k)
                g.push_back(ids[k] & kRosterMask);
            std::sort(g.begin(), g.end());
            int teamClip = -1; // Finkel only: SYM StableCall 0x3a3ef0 team lines, LoW DX
            if (n == 2u && g[0] == 0x0F && g[1] == 0x10) teamClip = 232;            // "D-Von, Bubba Ray, the Dudley Boyz"
            else if (n == 2u && g[0] == 0x0C && g[1] == 0x0D) teamClip = 236;       // "Matt, Jeff, the Hardy Boyz"
            else if (n == 2u && g[0] == 0x21 && g[1] == 0x23) teamClip = 1587;      // "Jamal and Rosey. 3-Minute Warning!"
            // other known teams walking out together: the members, then Finkel's team call (FINKEL_LINES team_<key>)
            const std::string key = stableKey(g);
            const size_t before = entrants_.size();
            inGroup_ = true;
            if (teamClip >= 0)
            {
                std::fprintf(stderr, "[hctp-pbp] shared entrance: known team -> Finkel team call %d%c", teamClip, 10);
                introOverride_ = teamClip;
                entranceLocked(ids[0]);
                introOverride_ = -1;
                if (entrants_.size() > before) // leader announced -> the partners are done too
                    for (uint32_t k = 1; k < n; ++k)
                        entrants_.push_back(ids[k]);
            }
            else
                for (uint32_t k = 0; k < n; ++k)
                    entranceLocked(ids[k]);
            inGroup_ = false;
            if (teamClip < 0 && !key.empty() && entrants_.size() > before)
                if (int c = pickConnector("team_" + key); c >= 0)
                {
                    queue_.push_back({uint16_t(c), 0.25, 14u});
                    std::fprintf(stderr, "[hctp-pbp] %s together -> Finkel %d%c", key.c_str(), c, 10);
                    cv_.notify_all();
                }
            // tag champions together: "They are the WWE Tag Team Champions!" / "The World Tag Team Champions!"
            uint32_t belt = 0;
            for (uint32_t k = 0; k < n && !belt; ++k)
                if ((ids[k] >> 24) & 1u)
                    belt = (ids[k] >> kBeltShift) & 0xFFu;
            if (entrants_.size() > before && (belt == 3u || belt == 4u) &&
                !(teamTitleSaid_ & (1u << belt)))
                if (int c = pickConnector("team_title_b" + std::to_string(belt - 1u)); c >= 0)
                {
                    queue_.push_back({uint16_t(c), 0.25, 14u});
                    teamTitleSaid_ |= 1u << belt;
                    std::fprintf(stderr, "[hctp-pbp] tag champions together (belt %u) -> Finkel team call %d%c", belt - 1u, c, 10);
                    cv_.notify_all();
                }
        }

        // 2K14 Finkel team calls (Jordan-verified Nov 2): FINKEL_LINES team_<key> ("Degeneration X!") after the members,
        // only when exactly these wrestlers walk out in ONE entrance (pairings, not stables; Jordan Nov 2). g = sorted ids.
        static std::string stableKey(const std::vector<uint32_t> &g)
        {
            // one rule for every team: 2+ different wrestlers walking out together, all members of the same team list.
            // Not the game's stables (people overlap). Several lists fit -> the smallest one (Matt + Jeff = Hardys,
            // Matt + Lita = Team Extreme). FINKEL_LINES team_<key> = the call; a team with no clip stays silent.
            struct Team { const char *key; std::initializer_list<uint32_t> ids; };
            static const Team kTeams[] = {
                {"hardys", {0x0C, 0x0D}},
                {"teamextreme", {0x0C, 0x0D, 0x0E}},
                {"dudleys", {0x0F, 0x10, 0x2B}},
                {"dx", {0x02, 0x2D, 0x24}},
                {"evolution", {0x02, 0x1E, 0x3C, 0x3D}},
                {"powertrip", {0x01, 0x02}},
                {"bod", {0x03, 0x48, 0x07}},
                {"nwo", {0x20, 0x42}},
                {"edgechristian", {0x0A, 0x0B}},
                {"3mw", {0x21, 0x22, 0x23}},
            };
            if (g.size() < 2u || std::adjacent_find(g.begin(), g.end()) != g.end())
                return {};
            const Team *best = nullptr;
            for (const Team &t : kTeams)
                if (std::all_of(g.begin(), g.end(), [&](uint32_t r) { return std::find(t.ids.begin(), t.ids.end(), r) != t.ids.end(); }) &&
                    (!best || t.ids.size() < best->ids.size()))
                    best = &t;
            return best ? best->key : std::string();
        }

        int pickConnector(const std::string &kind)
        {
            std::vector<int> c;
            if (const auto it = finkel_.find(kConnectors); it != finkel_.end())
                for (const auto &l : it->second)
                    if (l.kind == kind)
                        c.push_back(l.clip);
            return c.empty() ? -1 : c[std::uniform_int_distribution<size_t>(0, c.size() - 1)(rng_)];
        }

        void entranceLocked(uint32_t rosterId)
        {
            if (!loaded_ || inMatch_)
                return;
            // HCTP eRuleType 0x52 is Royal Rumble (ConvSelectPosition special-cases the same value). Only the first two entrants
            // receive the pre-match entrance sequence; later entrants are
            // handled by the Rumble scene itself rather than Finkel's normal
            // opponent/team connectors.
            // 0x52 = ConvSelectPosition's Rumble, 0x0D = the debug table's "ROYAL RUMBLE" (which one a real Rumble uses
            // is unlogged - accept both). Only the first two entrants have entrances: 1369 on #1, 1418 on #2.
            const bool royalRumble = matchRule_ == 0x52u || matchRule_ == 0x0Du;
            if (royalRumble)
                std::fprintf(stderr, "[hctp-pbp] Royal Rumble entrance #%zu (rule 0x%02X)%c", entrants_.size() + 1u, matchRule_, 10);
            if (royalRumble && entrants_.size() >= 2u)
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
            const bool withPartner = (rosterId >> 24) & 1u; // tag-belt partner on the same team (runner hook)
            const uint32_t belt = (rosterId >> kBeltShift) & 0xFFu;
            int title = -1, titleFull = -1;
            if (belt != 0u)
            {
                const std::string b = std::to_string(belt - 1u);
                const bool tagBelt = belt - 1u == 2u || belt - 1u == 3u;
                // the wrestler's own whole intro with the belt (LoW "From ..., the co-holder of the WWE Tag Team
                // Championship, NAME"; FINKEL_LINES kind title_full_b<belt>) beats a pieced title call
                titleFull = pick(who, "title_full_b" + b, false);
                // tag belts: "one half of the ... Champions" only when his partner is not with him; together the team
                // call after both names covers it
                if (!(tagBelt && withPartner))
                    title = pick(kConnectors, "title_half_b" + b, false);
                // tag belts (2 = WWE Tag / SmackDown, 3 = World Tag / Raw): each partner gets only "one half of the ...
                // Champions"; the plural team call (team_title_b<belt>) follows the second partner (Jordan Nov 2)
                if (title < 0 && !tagBelt)
                    title = pick(kConnectors, "title_b" + b, false);
                // The shipped HCTP FINKEL_LINES.csv labels these clips as
                // generic rows, not title_bN keys. Keep the holder lookup
                // dynamic, but provide the verified nine-belt fallback.
                if (title < 0)
                {
                    static constexpr int kTitleClips[9][2] = {
                        {461, 462}, // undisputed WWE title
                        {463, 464}, // WWE/world heavyweight title
                        {477, 478}, // WWE Tag Team title / one half
                        {477, 478}, // World Tag Team title / one half (overridden by title_b3 when imported)
                        {467, 468}, // Intercontinental title
                        {-1, -1},   // United States title (provided by imported title_b5 clips)
                        {479, 480}, // Cruiserweight title
                        {471, 472}, // Hardcore title
                        {473, 474}, // Women's title
                    };
                    const uint32_t beltIndex = belt - 1u;
                    if (beltIndex < 9u && !tagBelt) // tag fallbacks 477/478 are the plural / no "one half of" call
                    {
                        title = kTitleClips[beltIndex][std::uniform_int_distribution<int>(0, 1)(rng_)];
                        if (title >= 0)
                            std::fprintf(stderr, "[hctp-pbp] dynamic title belt %u -> Finkel clip %d%c",
                                         beltIndex, title, 10);
                    }
                }
            }
            std::vector<int> intro;
            const int from = best(who, "from", true), weight = best(who, "weight", true),
                      name = best(who, "name", true);
            const int full = pick(who, "full", true);
            // both a SYM pieced intro and a LoW one-line intro: alternate at random so intros vary match to match
            const bool pieced = from >= 0 && weight >= 0 && name >= 0;
            const bool useFull = full >= 0 && (!pieced || (title < 0 && std::uniform_int_distribution<int>(0, 1)(rng_) == 1));
            if (titleFull >= 0)
            {
                intro = {titleFull};
                title = -1;
                std::fprintf(stderr, "[hctp-pbp] belt %u: wrestler's own title intro -> Finkel %d%c", belt - 1u, titleFull, 10);
            }
            else if (int special = pick(who, "special", false); special >= 0)
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
            // SYM FinkelCall time tables (notes/COMMENTARY_BEAT_FOR_BEAT_20261102.md): tag / six-man / handicap teams
            // are called by NAME only (tables 4-11: connector, name; table 28: "and his partner", short name).
            // HCTP_PBP_TAG_FULL=1 keeps the from / weight intros for team members.
            // Jordan Nov 2: everyone gets the full intro (hometown, weight, name); HCTP_PBP_TAG_NAMES=1 = SYM names only
            static const bool tagFull = [] { const char *e = std::getenv("HCTP_PBP_TAG_NAMES"); return !(e && *e == '1'); }();
            if (tagMatch_ && !tagFull && titleFull < 0)
            {
                const bool newShow = Clock::now() - lastEntrance_ > std::chrono::minutes(4); // entrants_ cleared below
                const bool partner = !newShow && entrants_.size() % teamSize_ != 0u;
                int nm = -1;
                size_t len = partner ? SIZE_MAX : 0; // leader: longest name (SYM field 15), partner: shortest (field 17)
                if (const auto it = finkel_.find(who); it != finkel_.end())
                    for (const auto &l : it->second)
                        if (l.kind == "name" && !isTitleLine(l.text) && l.text.rfind("this is", 0) != 0 &&
                            (partner ? l.text.size() < len : l.text.size() > len))
                            nm = l.clip, len = l.text.size();
                if (nm >= 0 && partner) // each team's first man keeps his full intro (Jordan Nov 2)
                    intro = {nm};
            }
            if (introOverride_ >= 0) // SvR: a known team in one entrance = ONE team call, no titles / members
                intro = {introOverride_}, title = -1;
            // the champion's title right before his name, in every match (Jordan Nov 2): "From Greenwich, Connecticut,
            // weighing 255 pounds, the World Heavyweight Champion, Triple H" (one-line intros: title first)
            if (title >= 0)
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
                teamTitleSaid_ = 0;
                // title match: LoW "The following contest is for the WWE / Intercontinental / WWE Tag Team Championship"
                // (1399-1401) for a one-fall match; a match-type opener is followed by LoW "And is for the ..."
                // (1407-1409); other belts: SYM championship pieces after the opener (SYM time table 12/20 code 4)
                static constexpr int kFollowingFor[9] = {1399, -1, 1401, -1, 1400, -1, -1, -1, -1};
                static constexpr int kAndIsFor[9] = {1407, -1, 1409, -1, 1408, -1, -1, -1, -1};
                static constexpr int kSymFor[9] = {481, 483, 489, -1, 484, -1, 490, 486, 487};
                const int tb = titleMatchBelt_;
                if (royalRumble)
                    seq.push_back(1369); // "Ladies and gentlemen, it is now time for the Royal Rumble..."
                else if (tb >= 0 && openerClip_ < 0 && kFollowingFor[tb] >= 0)
                    seq.push_back(kFollowingFor[tb]);
                else
                {
                    if (openerClip_ >= 0)
                        seq.push_back(openerClip_);
                    else if (int c = pick(kConnectors, tagMatch_ ? "open_tag" : "open", false); c >= 0)
                        seq.push_back(c);
                    if (tb >= 0)
                        if (int c = kAndIsFor[tb] >= 0 && openerClip_ >= 0 ? kAndIsFor[tb] : kSymFor[tb]; c >= 0)
                            seq.push_back(c);
                }
                if (tb >= 0)
                    std::fprintf(stderr, "[hctp-pbp] title match belt %d -> opener %d..%c", tb, seq.empty() ? -1 : seq.back(), 10);
            }
            else if (royalRumble)
            {
                seq.push_back(1418); // "And now, here is the individual who drew number two."
            }
            else
            {
                const bool prevFemale = isFemale(entrants_.back() & kRosterMask); // the wrestler just announced (his / her partner)
                // Team boundaries come from the match cast count: 2-v-2 groups by two,
                // 3-v-3 groups by three. The first entrant of the next group is an
                // opponent; the remaining entrants in that group are partners.
                const bool opposingTeamFirst = tagMatch_ && entrants_.size() % teamSize_ == 0u; // triangle tag: 3 teams
                const char *kind = tagMatch_ && !opposingTeamFirst
                                       ? (prevFemale ? "and_her_tag_partner" : "and_his_tag_partner")
                                       : (prevFemale ? "and_her_opponent" : "and_his_opponent");
                // SYM table 5/9/17: the next team opens with "and the opponents" (SYM 0x1B4 = FINKEL 433)
                // title match, the side that does not hold the belt: SYM "and the challenger(s)" (434 / 435)
                const bool challenger = titleMatchBelt_ >= 0 && int(belt) - 1 != titleMatchBelt_ &&
                                        (!tagMatch_ || opposingTeamFirst);
                // third member of a six-man team: "and their tag team partner" does not exist anywhere; FINKEL_LINES
                // and_their_partner (2K14, once Jordan OKs one) else LoW 1346 "And the tag team partner" (Jordan Nov 2)
                const bool thirdMember = tagMatch_ && !opposingTeamFirst && entrants_.size() % teamSize_ == 2u;
                if (thirdMember)
                {
                    const int c = pick(kConnectors, "and_their_partner", false);
                    seq.push_back(c >= 0 ? c : 1346);
                }
                else if (challenger) // LoW Finkel "And introducing the challenger(s)"
                    seq.push_back(tagMatch_ && teamSize_ > 1u ? 1345 : 1344);
                else if (int c = tagMatch_ && opposingTeamFirst && teamSize_ > 1u ? pick(kConnectors, "and_the_opponents", false) : -1; c >= 0)
                    seq.push_back(c);
                else if (int c2 = pick(kConnectors, kind, false); c2 >= 0)
                    seq.push_back(c2);
            }
            // SYM: only the first entrant is "making his/her way to the ring"; every later entrant is just the
            // connector + intro ("and the opponent, from ..., NAME"; table 1/3/5/28)
            // first out (Jordan Nov 2): "Introducing first, making his way to the ring, from ..." (LoW 1415); in a title
            // match the side without the belt: "Introducing the challenger(s)" (LoW 1416 / 1423)
            if (entrants_.empty() && !royalRumble)
            {
                const bool challengerFirst = titleMatchBelt_ >= 0 && int(belt) - 1 != titleMatchBelt_;
                seq.push_back(challengerFirst ? (tagMatch_ && teamSize_ > 1u ? 1423 : 1416) : 1415);
            }
            if (entrants_.empty())
                if (int c = pick(kConnectors, female ? "making_her" : "making_his", false); c >= 0) seq.push_back(c);
            seq.insert(seq.end(), intro.begin(), intro.end());
            // tag champions coming out TOGETHER (one shared stable entrance, runner bit 24): "They are the WWE Tag Team
            // Champions!" / "The World Tag Team Champions!" (team_title_b2/b3), once per match; separate entrances get
            // "one half of" each instead (Jordan Nov 2)
            if (!inGroup_ && withPartner && (belt == 3u || belt == 4u) && !(teamTitleSaid_ & (1u << belt)))
                if (int c = pick(kConnectors, "team_title_b" + std::to_string(belt - 1u), false); c >= 0)
                {
                    seq.push_back(c);
                    teamTitleSaid_ |= 1u << belt;
                    std::fprintf(stderr, "[hctp-pbp] tag champions together (belt %u) -> Finkel team call %d%c", belt - 1u, c, 10);
                }
            // team names: entranceGroup only (wrestlers walking out together; Jordan Nov 2)
            entrants_.push_back(rosterId);
            std::fprintf(stderr, "[hctp-pbp] entrance roster 0x%02X -> %zu Finkel line(s)%c", rosterId & kRosterMask, seq.size(), 10);
            for (size_t i = 0; i < seq.size(); ++i)
                queue_.push_back({uint16_t(seq[i]), i ? 0.25 : 1.0, 14u});
            cv_.notify_all();
        }

        // Season title match (runner: match setup byte 0x4922C5, written by CSeasonSystem::nvGameInit from the card's
        // CheckTitleMatch belt; 0xFF = no title)
        void setTitleMatch(uint32_t belt)
        {
            std::lock_guard<std::mutex> lock(m_);
            titleMatchBelt_ = belt < 9u ? int(belt) : -1;
        }

        // a wrestler's longest plain name clip ("the people's champion the rock"), -1 if none
        int nameClip(uint32_t roster) const
        {
            int clip = -1;
            size_t len = 0;
            if (const auto it = finkel_.find(roster); it != finkel_.end())
                for (const auto &l : it->second)
                    if (l.kind == "name" && !isTitleLine(l.text) && l.text.rfind("this is", 0) != 0 && l.text.size() > len)
                        clip = l.clip, len = l.text.size();
            return clip;
        }

        // Royal Rumble entrant #3+ (runner: CGameSync::Draw pops the entry queue 0x527A90 -> 0x527AB8). No Finkel clip
        // for "number three" and up exists (LoW / SYM / 2K14 scanned) -> the name alone.
        void rumbleEntrant(uint32_t roster)
        {
            std::lock_guard<std::mutex> lock(m_);
            if (!loaded_)
                return;
            ++rumbleCount_;
            const int nm = nameClip(roster & kRosterMask);
            std::fprintf(stderr, "[hctp-pbp] rumble entrant #%u roster 0x%02X -> Finkel %d%c", rumbleCount_, roster & kRosterMask, nm, 10);
            if (nm >= 0)
            {
                queue_.push_back({uint16_t(nm), 1.0, 14u});
                cv_.notify_all();
            }
        }

        void resetEntrances()
        {
            std::lock_guard<std::mutex> lock(m_);
            entrants_.clear();
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
            loadMaleLines();
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
                    for (size_t i = 0; i < v.size(); ++i)
                    {
                        Row &r = v[i];
                        if (t == &kJrKing) // SvR's JR/King rows already list their exchanges (w1, w2...); no guessing
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
            // two beta superstar rows are filed under the wrong HCTP id (notes/SVR_BETA_COMMENTARY_DEEP_DIVE_20261028.md):
            // clip 218 is about RVD (0x04, filed 0x03), clip 394 about Ultimo Dragon (0x55, filed 0x53 Rhyno)
            if (const int t = kColeTazz.wrestlerBanter; size_t(t) < tables_.size())
                for (Row &r : tables_[t])
                    if (r.part == 15 && (r.w[1] == 218 || r.w[1] == 394))
                        r.w[0] = r.w[1] == 218 ? 0x04 : 0x55;
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
            const bool women = womenMatch();
            std::vector<const Row *> fresh;
            for (const Row &r : tables_[table])
                if (!heard_.count({r.part, r.w[1]}) && !(women && maleRow(r, 1)))
                    fresh.push_back(&r);
            if (fresh.empty())
                for (const Row &r : tables_[table])
                    if (!(women && maleRow(r, 1)))
                        fresh.push_back(&r);
            if (fresh.empty())
                return;
            const Row &row = *fresh[std::uniform_int_distribution<size_t>(0, fresh.size() - 1)(rng_)];
            heard_.insert({row.part, row.w[1]});
            pushSequence(row, row.w[1] == 0 && row.w[2] != kNone ? 2 : 1, gap);
        }

        // SvR plays a row as a sequence: clips w[from]..w7 in order until -1 (0x212930 / 0x212df0) - that is how its
        // exchanges work (JR line, King's reply...)
        void pushSequence(const Row &row, int from, double gap)
        {
            bool first = true;
            for (int k = from; k < 8; ++k)
            {
                const uint16_t c = row.w[k];
                if (c == kNone)
                    break;
                if (c >= clipCount(row.part))
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
                                    !spoken_.count(id & kRosterMask) && !heard_.count({r.part, r.w[1]}))
                                {
                                    withLines.push_back(id);
                                    break;
                                }
                    // SvR (0x212f30, list 0x34c700): a topic is a superstar line or general banter 50/50; each wrestler
                    // in the match gets one superstar line (queued once at match setup, removed when used), then the
                    // topics are general banter; the next topic starts 30 s (0x708 frames) after this one started
                    // all-women match (SvR 0x34c740): 8 superstar : 5 general : 6 women's banter
                    // bra & panties (SvR PBPMatchBraPanRandomIndex): 8 superstar : 4 B&P : 5 general : 6 women's banter
                    const bool women = womenMatch();
                    const int roll = std::uniform_int_distribution<int>(0, braPanties_ ? 22 : women ? 18 : 1)(rng_);
                    if (braPanties_ && roll >= 8 && roll < 12)
                        enqueueRow(team().braPanties, 0.0);
                    else if (braPanties_ && roll >= 17)
                        enqueueRow(team().women, 0.0);
                    else if (braPanties_ && roll >= 12)
                        enqueueRow(team().banter, 0.0);
                    else if (!braPanties_ && women && roll >= 13)
                        enqueueRow(team().women, 0.0);
                    else if (!withLines.empty() && (women ? roll < 8 : roll == 0))
                    {
                        const uint32_t id = withLines[std::uniform_int_distribution<size_t>(0, withLines.size() - 1)(rng_)];
                        spoken_.insert(id & kRosterMask);
                        enqueueWrestlerLine(id);
                    }
                    else
                        enqueueRow(team().banter, 0.0);
                    nextBanter_ = now + std::chrono::seconds(30);
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
        std::set<uint32_t> spoken_; // wrestlers whose one superstar line this match was used
        std::set<uint32_t> loggedMoves_;
        Clock::time_point lastEntrance_{};
        std::deque<Line> queue_;
        std::mt19937 rng_{std::random_device{}()};
        bool loaded_ = false, raw_ = false, inMatch_ = false, teamForced_ = false, playerHeld_ = false, tagMatch_ = false;
        uint32_t matchRule_ = 0, teamSize_ = 2;
        bool braPanties_ = false;
        uint32_t rumbleCount_ = 0; // Royal Rumble wrestlers announced this match
        int titleMatchBelt_ = -1; // Season title match: belt at stake (0..8, CTitleHistoryData order), -1 = none
        bool inGroup_ = false;    // entranceGroup: team call after the whole group
        int introOverride_ = -1;  // entranceGroup: known team call replaces the leader's intro
        uint32_t teamTitleSaid_ = 0; // belts whose team call played this show (bit = belt + 1)
        std::set<std::pair<uint32_t, uint32_t>> maleLines_;
        std::map<uint32_t, std::pair<uint32_t, Clock::time_point>> moveStarter_; // move -> first chara + when
        int openerClip_ = -1;
        int arena_ = 0, brand_ = -1;
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

extern "C" void hctpCommentaryRumbleEntrant(uint32_t rosterId)
{
    if ((rosterId & kRosterMask) < 0x56u && engineReady())
        Engine::get().rumbleEntrant(rosterId);
}

extern "C" void hctpCommentarySetTitleMatch(uint32_t belt)
{
    if (engineReady())
        Engine::get().setTitleMatch(belt);
}

extern "C" void hctpCommentaryEntranceGroup(const uint32_t *rosterIds, uint32_t count)
{
    uint32_t ids[8];
    uint32_t n = 0;
    for (uint32_t k = 0; k < count && n < 8u; ++k)
        if ((rosterIds[k] & kRosterMask) < 0x56u)
            ids[n++] = rosterIds[k];
    if (n && engineReady())
        Engine::get().entranceGroup(ids, n);
}

extern "C" void hctpCommentaryEntrance(uint32_t rosterId)
{
    if ((rosterId & kRosterMask) < 0x56u && engineReady())
        Engine::get().entrance(rosterId);
}

extern "C" void hctpCommentaryResetEntrances()
{
    if (engineReady())
        Engine::get().resetEntrances();
}

extern "C" void hctpCommentarySetTagMatch(uint32_t tag)
{
    if (engineReady())
        Engine::get().setTagMatch(tag != 0u);
}

extern "C" void hctpCommentarySetTeamSize(uint32_t size)
{
    if (engineReady())
        Engine::get().setTeamSize(size);
}

extern "C" void hctpCommentarySetMatchRule(uint32_t rule)
{
    if (engineReady())
        Engine::get().setMatchRule(rule);
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

extern "C" void hctpCommentaryBrand(int brand)
{
    if (engineReady())
        Engine::get().setBrand(brand);
}

extern "C" void hctpCommentaryArena(int stage)
{
    if (engineReady())
        Engine::get().setArena(stage);
}

extern "C" void hctpCommentaryBellRule(uint32_t rule, uint32_t braPanties)
{
    if (engineReady())
        Engine::get().setBellRule(rule, braPanties != 0u);
}
