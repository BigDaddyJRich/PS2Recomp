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
                    lastCalledWho_ = 0xFFFFFFFFu; // [COMMFIX Nov 6]
                    queue_.clear();
                    std::fprintf(stderr, "[hctp-pbp] match start (%s, arena %d)%c", raw_ ? "JR/King" : "Cole/Tazz", arena_, 10);
                    // [COMMFIX Nov 6] Royal Rumble opening (setBellRule, just before this): Finkel first, after the clear
                    if (!rumbleOpen_.empty())
                    {
                        std::fprintf(stderr, "[hctp-pbp] rumble opening: %zu Finkel clips queued%c", rumbleOpen_.size(), 10);
                        queue_.insert(queue_.end(), rumbleOpen_.begin(), rumbleOpen_.end());
                        rumbleOpen_.clear();
                    }
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
                rumbleOpen_.clear(); // [COMMFIX Nov 6]
                std::vector<uint32_t> starters;
                for (uint32_t id : participants_)
                    if ((id & kRosterMask) != 0x31u && (id & kRosterMask) != 0x33u && starters.size() < 2u)
                        starters.push_back(id & kRosterMask);
                for (size_t k = 0; k < starters.size(); ++k)
                {
                    // [COMMFIX Nov 6] into rumbleOpen_: event(kPbpBell) runs right after this and its match start did
                    // queue_.clear() -> 1369/1418 + names were wiped every time (never heard, Jordan Nov 2 + Nov 6)
                    rumbleOpen_.push_back({uint16_t(k ? 1418 : 1369), k ? 0.25 : 1.0, 14u});
                    if (int nm = nameClip(starters[k]); nm >= 0)
                        rumbleOpen_.push_back({uint16_t(nm), 0.25, 14u});
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

        // non-SYM openers (Jordan Nov 6): LoW Finkel match-type lines for HCTP rules that have entrances and that SYM
        // only called "one fall" (triple threat, fatal 4-way) or nothing (ladder). Every other rule = SYM FNKL fields
        // 1-3 ("scheduled for one fall" / tag / six-man) via GetCategory. Royal Rumble: setBellRule (LoW 1369 / 1418).
        static int openerFor(uint32_t rule)
        {
            switch (rule)
            {
            case 0x17: return 1379;                                               // triple threat
            case 0x18: return 1380;                                               // fatal four-way
            case 0x2F: case 0x30: case 0x31: case 0x32: case 0x33: return 1393;  // ladder
            default: return -1;
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
            // Undertaker (Legend) 0x48 has his own lines now (Jordan Nov 6: Deadman lines = Legend, Big Evil = default)
            for (uint32_t key : {id, id & kRosterMask})
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
                // nobody's own line (a CAW, a restored slot, a borrowed finisher): a name-free call for this move
                // (BGM/PBP_GENERIC_FINISHERS.csv, tools/build_generic_finishers.py), else the ordinary reaction
                if (!genericMove(hctpMove, femalePerformer))
                    moveReaction(moveId, false, performer, hctpMove, femalePerformer);
                return;
            }
            // attacker and receiver both carry the move id while it happens: one call per move
            const auto now = Clock::now();
            if (moveId == lastCalledMove_ && now - lastCalledAt_ < std::chrono::seconds(8))
                return;
            // [COMMFIX Nov 6] SvR 0x213870 (notes/SVR_PBP_SYSTEM.md): ignored when it is the same (wrestler, move) as the
            // last call -> the same superkick twice in a row gets ONE line (Jordan: SCM line on every SCM)
            if (performer == lastCalledWho_ && moveId == lastCalledMove_)
                return;
            lastCalledWho_ = performer;
            // [/COMMFIX]
            lastCalledMove_ = moveId;
            lastCalledAt_ = now;
            const Row &r = *rows[std::uniform_int_distribution<size_t>(0, rows.size() - 1)(rng_)];
            std::fprintf(stderr, "[hctp-pbp] finisher move %u -> line %u/%u%c", moveId, r.part, r.w[2], 10);
            queue_.clear(); // a finisher call beats whatever banter was waiting
            // w3 = the row's delay in frames (SvR counts it down before speaking)
            queue_.push_back({r.w[2], r.part == 13 && r.w[3] != kNone && r.w[3] < 600u ? r.w[3] / 60.0 : 0.2, r.part});
            cv_.notify_all();
        }

        // finisher without a line for this performer: one of the move's name-free lines (never "Goldberg with the
        // spear" for a CAW), else a name-free finisher exclamation. Called with m_ held; true = a call was queued.
        bool genericMove(uint32_t hctpMove, bool femalePerformer)
        {
            const auto it = generic_[raw_ ? 1 : 0].find(hctpMove);
            if (it == generic_[raw_ ? 1 : 0].end())
                return false;
            std::vector<const GenericLine *> lines;
            for (const GenericLine &g : it->second)
                if (g.clip < clipCount(g.part) && !(femalePerformer && g.male) &&
                    !(g.part == lastGeneric_.first && g.clip == lastGeneric_.second && it->second.size() > 1))
                    lines.push_back(&g);
            if (lines.empty())
                return false;
            const auto now = Clock::now();
            if (hctpMove == lastCalledMove_ && now - lastCalledAt_ < std::chrono::seconds(8))
                return true;
            lastCalledMove_ = hctpMove;
            lastCalledAt_ = now;
            const GenericLine &g = *lines[std::uniform_int_distribution<size_t>(0, lines.size() - 1)(rng_)];
            lastGeneric_ = {g.part, g.clip};
            std::fprintf(stderr, "[hctp-pbp] generic finisher move %u -> line %u/%u%c", hctpMove, g.part, g.clip, 10);
            queue_.clear();
            queue_.push_back({g.clip, 0.2, g.part});
            cv_.notify_all();
            return true;
        }

        // BGM/PBP_GENERIC_FINISHERS.csv: team (cole|raw), hctp_move, partition, clip, male (he/his line), text
        void loadGenericFinishers()
        {
            std::ifstream f("BGM/PBP_GENERIC_FINISHERS.csv");
            std::string row;
            std::getline(f, row); // header
            size_t n = 0;
            while (std::getline(f, row))
            {
                std::stringstream ss(row);
                std::string team, move, part, clip, male;
                if (!std::getline(ss, team, ',') || !std::getline(ss, move, ',') || !std::getline(ss, part, ',') ||
                    !std::getline(ss, clip, ',') || !std::getline(ss, male, ','))
                    continue;
                generic_[team == "raw" ? 1 : 0][uint32_t(std::strtoul(move.c_str(), nullptr, 10))].push_back(
                    {uint32_t(std::strtoul(part.c_str(), nullptr, 10)), uint16_t(std::strtoul(clip.c_str(), nullptr, 10)),
                     male == "1"});
                ++n;
            }
            std::cerr << "[hctp-pbp] generic finisher lines: " << n << " rows" << std::endl;
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

        // ===== Howard Finkel = SYM FinkelCall port (Jordan Nov 6; SYM slus_204.83, notes/FINKEL_SYM_PORT_PLAN_20261106.md) =====
        // Data: BGM/FINKEL_DAT.csv = SYM misc.pac /FNKL/DAT layout, 17 fields per wrestler as FINKEL.afs clips
        // (tools/build_finkel_dat.py; -1 = no clip, said as silence exactly like SYM's 0xFFFF).
        // Everything not marked "non-SYM" is SYM code: GetCategory 0x36ca20, MakeTimeTable 0x36ce90 (table 0x3c2970),
        // MakeCallListNormal 0x36bfe0, StableCall 0x3a3ef0 / stableCheckTable 0x3a3e70, colorCheckTable 0x3a4010.
        // MakeCallListSpecial 0x36c6b0 (SDAT custom intros) has no HCTP trigger (flag 0x42c2a9 / special table
        // 0x4f17f0 are SYM story data) -> not ported.
        struct FnklRow
        {
            int f[17];
            int full = -1;                 // non-SYM: LoW one-line intro, only when from/weight/name are incomplete
            std::map<uint32_t, int> titleFull; // non-SYM: LoW one-line intro with a belt (HCTP belt index -> clip)
        };

        // MakeTimeTable: time table 0x3c2970 + cat*20 (codes = 1-based FNKL fields)
        static const std::vector<int> &timeTable(int cat)
        {
            static const std::vector<int> k[29] = {
                {1, 5, 13, 14, 15}, {7, 13, 14, 15}, {6, 13, 14, 15}, {7, 13, 14, 15}, {2, 5, 15}, {8, 15}, {6, 15}, {8, 15},
                {3, 5, 15}, {8, 15}, {6, 15}, {8, 15}, {1, 4, 5, 11, 13, 14, 16}, {9, 13, 14, 15}, {4, 6, 11, 13, 14, 16},
                {9, 13, 14, 15}, {2, 4, 5, 11, 16}, {10, 15}, {4, 6, 11, 16}, {10, 15}, {1, 4, 9, 13, 14, 15},
                {7, 11, 13, 14, 16}, {4, 9, 13, 14, 15}, {7, 11, 13, 14, 16}, {2, 4, 10, 15}, {8, 11, 16}, {4, 10, 15},
                {8, 11, 16}, {12, 17}};
            return k[cat >= 0 && cat < 29 ? cat : 0];
        }

        // GetCategory__10FinkelCallFiiib 0x36ca20, branch for branch. idx = entrant count so far (this+0x9c),
        // title = a belt is at stake (!= 0xFF), champ = this group holds it. Rule lists 0x3c28f0..0x3c2968; HCTP's
        // eRuleType ids 0x00-0x50 are SYM's (same debug rule names: 0x0A-0x0C handicap, 0x17 triple threat ...).
        static int getCategory(uint32_t rule, uint32_t idx, bool title, bool champ)
        {
            static const uint8_t kHandicap[] = {0x0A, 0x0B, 0x0C};
            static const uint8_t kSingleOp[] = {0x00, 0x01, 0x02, 0x03, 0x0E, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x29, 0x2C, 0x3D, 0x3F, 0x40, 0x45, 0x48};
            static const uint8_t kSingle[] = {0x0D, 0x0F, 0x10, 0x11, 0x14, 0x15, 0x16, 0x1C, 0x1E, 0x1F, 0x20, 0x21, 0x22, 0x23, 0x25, 0x26, 0x27, 0x28,
                                              0x2B, 0x2E, 0x2F, 0x32, 0x33, 0x34, 0x36, 0x37, 0x39, 0x3A, 0x3B, 0x3C, 0x41, 0x43, 0x44, 0x46, 0x47, 0x49, 0x4A};
            static const uint8_t kTagOp[] = {0x04, 0x05, 0x06, 0x07, 0x2A, 0x2D, 0x3E};
            static const uint8_t kTag[] = {0x11, 0x13, 0x1D, 0x24, 0x30, 0x31, 0x35, 0x3E, 0x42, 0x4B, 0x4C};
            static const uint8_t kSixOp[] = {0x08, 0x09};
            static const uint8_t kSix[] = {0x12, 0x4D, 0x4E, 0x4F, 0x50, 0x38};
            auto in = [rule](const uint8_t *b, const uint8_t *e) { return std::find(b, e, uint8_t(rule)) != e; };
#define FK_IN(a) in(std::begin(a), std::end(a))
            if (FK_IN(kHandicap))
            {
                if (idx == 0) return 0;
                if (idx == 1) return 5;
                if (idx == 2 || idx == 3) return 28;
            }
            if (FK_IN(kSingleOp))
            {
                if (!title) return idx ? 1 : 0;
                return idx == 0 ? (champ ? 12 : 20) : (champ ? 21 : 13);
            }
            if (FK_IN(kSingle))
            {
                if (!title) return idx ? 3 : 2;
                return idx == 0 ? (champ ? 14 : 22) : (champ ? 23 : 15);
            }
            struct T { const uint8_t *b, *e; int a, c, ta, tc, ca, cc; };
            const T tags[] = {{std::begin(kTagOp), std::end(kTagOp), 4, 5, 16, 25, 24, 17},
                              {std::begin(kTag), std::end(kTag), 6, 7, 18, 27, 26, 19}};
            for (const T &t : tags)
                if (in(t.b, t.e))
                {
                    if (idx == 1 || idx == 3) return 28;
                    if (idx != 0 && idx != 2) continue; // SYM falls through to the next list
                    if (!title) return idx == 0 ? t.a : t.c;
                    if (champ) return idx == 0 ? t.ta : t.tc;
                    return idx == 0 ? t.ca : t.cc;
                }
            if (FK_IN(kSixOp)) return idx == 0 ? 8 : idx == 3 ? 9 : 28;
            if (FK_IN(kSix)) return idx == 0 ? 10 : idx == 3 ? 11 : 28;
#undef FK_IN
            return 0;
        }

        // code 4 "for the <belt> championship" (SYM callTypeB 0x3a3e30) / code 11 "the <belt> champion(s)" (callTypeG
        // 0x3a3e50) per HCTP belt (0 WWE, 1 World Hvt, 2 WWE Tag, 3 World Tag, 4 IC, 5 US, 6 Cruiser, 7 Hardcore,
        // 8 Women's). SYM has 6 belts; HCTP belts SYM lacks use SYM's other title pieces (heavyweight 483/463,
        // women's 487/473). A FINKEL_LINES connector title_b<belt> (2K14, Jordan-verified) replaces code 11.
        int beltClip(int code, int belt)
        {
            static constexpr int kB[9] = {481, 483, 489, 489, 484, -1, 490, 486, 487};
            static constexpr int kG[9] = {461, 463, 477, 477, 467, -1, 479, 471, 473};
            if (belt < 0 || belt > 8)
                return -1;
            if (code == 11)
                if (int c = pickConnector("title_b" + std::to_string(belt)); c >= 0)
                    return c;
            return code == 4 ? kB[belt] : kG[belt];
        }

        // the FNKL row: attire variant row first (SYM colorCheckTable -> otherColorCall: "0x20@1" nWo Hogan), then the
        // wrestler's; unknown ids = SYM's created-wrestler rows 67/68 (connectors only, no name pieces)
        const FnklRow &fnklRow(uint32_t id)
        {
            if (auto it = fnkl_.find(id & kEntrantMask & ~(0xFFu << kBeltShift)); it != fnkl_.end())
                return it->second;
            if (auto it = fnkl_.find(id & kRosterMask); it != fnkl_.end())
                return it->second;
            static FnklRow caw[2];
            static const bool init = [] {
                const int c[12] = {427, 436, 443, 0, 428, 429, 432, 433, 434, 435, 0, 449};
                for (int g = 0; g < 2; ++g)
                {
                    std::fill(std::begin(caw[g].f), std::end(caw[g].f), -1);
                    std::copy(std::begin(c), std::end(c), caw[g].f);
                }
                caw[1].f[5] = 430;
                return true;
            }();
            (void)init;
            return caw[isFemale(id & kRosterMask) ? 1 : 0];
        }

        void loadFinkelDat()
        {
            std::ifstream f("BGM/FINKEL_DAT.csv");
            std::string row;
            std::getline(f, row);
            while (std::getline(f, row))
            {
                std::vector<std::string> c;
                std::stringstream ss(row);
                for (std::string v; std::getline(ss, v, ',');)
                    c.push_back(v);
                if (c.size() < 20 || c[0].empty())
                    continue;
                FnklRow r;
                for (int k = 0; k < 17; ++k)
                    r.f[k] = std::atoi(c[1 + k].c_str());
                r.full = std::atoi(c[18].c_str());
                std::stringstream tf(c[19]);
                for (std::string p; std::getline(tf, p, ';');)
                    if (const size_t colon = p.find(':'); colon != std::string::npos)
                        r.titleFull[uint32_t(std::atoi(p.c_str()))] = std::atoi(p.c_str() + colon + 1);
                fnkl_[parseKey(c[0])] = r;
            }
            std::cerr << "[hctp-pbp] Finkel FNKL rows (BGM/FINKEL_DAT.csv): " << fnkl_.size() << std::endl;
        }

        // StableCall 0x3a3ef0 rows for stableCheckTable 0x3a3e70 (a group that is exactly this stable = ONE call with the
        // stable's own 17 fields, partners silent). SYM: Dudleys, Hardys, Billy & Chuck (+ Rico); HCTP has no Chuck slot.
        // 3-Minute Warning = non-SYM data (Jordan-made clip 1587), same row shape.
        static const int *stableRow(std::vector<uint32_t> g)
        {
            struct S { std::initializer_list<uint32_t> ids; int f[17]; };
            static const S kStables[] = {
                {{0x0F, 0x10}, {-1, 436, 443, 0, 428, 431, -1, 433, -1, 435, 0, 449, 230, 231, 232, 232, 232}},
                {{0x0C, 0x0D}, {-1, 436, 443, 0, 428, 431, -1, 433, -1, 435, 0, 449, 234, 235, 236, 236, 236}},
                {{0x21, 0x23}, {-1, 436, 443, 0, 428, 431, -1, 433, -1, 435, 0, 449, -1, -1, 1587, 1587, 1587}}, // non-SYM
            };
            std::sort(g.begin(), g.end());
            for (const S &s : kStables)
                if (std::vector<uint32_t>(s.ids) == g)
                    return s.f;
            return nullptr;
        }

        void entrance(uint32_t rosterId)
        {
            std::lock_guard<std::mutex> lock(m_);
            if (loaded_ && !inMatch_ && beginEntrance(rosterId))
                finkelCall(&rosterId, 1u);
        }

        // runner: everyone walking out in ONE entrance (SYM MakeCallList's up-to-3 ids)
        void entranceGroup(const uint32_t *ids, uint32_t n)
        {
            std::lock_guard<std::mutex> lock(m_);
            if (n && loaded_ && !inMatch_ && beginEntrance(ids[0]))
                finkelCall(ids, std::min(n, 3u));
        }

        // MakeCallListNormal 0x36bfe0 for one entrance group (m_ held)
        void finkelCall(const uint32_t *ids, uint32_t n)
        {
            const auto now = Clock::now();
            if (now - lastEntrance_ > std::chrono::minutes(4))
                entrants_.clear(); // a new show (SYM ResetCall)
            lastEntrance_ = now;
            // Royal Rumble: no entrance events; setBellRule / rumbleEntrant call it (non-SYM, LoW 1369 / 1418)
            if (matchRule_ == 0x52u || matchRule_ == 0x0Du)
            {
                std::fprintf(stderr, "[finkel] Royal Rumble rule 0x%02X entrance -> bell / spawn calls only%c", matchRule_, 10);
                return;
            }
            std::vector<uint32_t> g;
            for (uint32_t k = 0; k < n; ++k)
            {
                const uint32_t r = ids[k] & kRosterMask;
                if (std::any_of(entrants_.begin(), entrants_.end(), [&](uint32_t e) { return (e & kRosterMask) == r; }))
                {
                    std::fprintf(stderr, "[finkel] roster 0x%02X already introduced this show -> skipped%c", r, 10);
                    continue; // safety net (repeated entrance #0 setup), not SYM
                }
                g.push_back(ids[k]);
            }
            if (g.empty())
                return;
            const uint32_t idx = uint32_t(entrants_.size());
            const int stake = titleMatchBelt_; // belt at stake (Season title match), -1 = none (SYM 0xFF)
            bool champ = false;                // SYM t1 & (1 << belt): this group holds the belt at stake
            for (uint32_t id : g)
                champ |= stake >= 0 && int((id >> kBeltShift) & 0xFFu) - 1 == stake;
            const int cat = getCategory(matchRule_, idx, stake >= 0, champ);
            std::vector<uint32_t> members;
            for (uint32_t id : g)
                members.push_back(id & kRosterMask);
            const int *stable = g.size() > 1u ? stableRow(members) : nullptr;
            std::vector<int> seq;
            auto say = [&](const int *f, const std::vector<int> &table, uint32_t id, bool member0) {
                std::vector<int> t = table;
                const size_t first = seq.size();
                int titleBelt = stake;
                // non-SYM (SvR SLUS_210.60 0x211c10, Jordan Nov 6): a belt holder gets his title call in a non-title
                // match. SvR categories 2-5 = US / IC / WWE / World Heavyweight only (tag belts never called). Placed
                // like SYM's title tables: "the <belt> champion" before the hometown, the title name (code 16) last.
                const int held = int((id >> kBeltShift) & 0xFFu) - 1;
                // [COMMFIX Nov 6] was 'f &&' = stable rows only, so no singles/partner holder was ever called (Jordan:
                // Christian IC / Eddie US silent in Exhibition). Every non-stable entrant now; partners (table 28
                // {12,17}) too: SvR calls each entrant's belt -> "and his partner, the <belt> champion, <full name>".
                // [COMMFIX Nov 6b] Jordan: ALL belts (tag, women's, cruiserweight, hardcore...), not SvR's 4. Tag belts
                // (2/3, plural "Tag Team Champions" clips): once per entrance group, on the first holder only.
                const bool tagBelt = held == 2 || held == 3;
                bool tagFirst = true;
                if (tagBelt)
                    for (uint32_t o : g)
                    {
                        if ((o & kRosterMask) == (id & kRosterMask))
                            break;
                        if (int((o >> kBeltShift) & 0xFFu) - 1 == held)
                            tagFirst = false;
                    }
                if (stake < 0 && !f && held >= 0 && held <= 8 && tagFirst && beltClip(11, held) > 0 &&
                    (std::find(t.begin(), t.end(), 15) != t.end() || std::find(t.begin(), t.end(), 17) != t.end()))
                {
                    auto at = std::find_if(t.begin(), t.end(), [](int c) { return c >= 13; });
                    t.insert(at, 11);
                    std::replace(t.begin(), t.end(), 15, 16);
                    std::replace(t.begin(), t.end(), 17, 16);
                    titleBelt = held;
                    std::fprintf(stderr, "[finkel] roster 0x%02X holds belt %d -> title call (non-SYM, SvR)%c", id & kRosterMask, held, 10);
                }
                // non-SYM data: wrestlers with no SYM pieces (from/weight/name incomplete) use their LoW one-line intro
                // for the 13/14/15|16 run (title version when the title piece is in the table and one exists)
                const FnklRow *row = f ? nullptr : &fnklRow(id);
                const int *ff = f ? f : row->f;
                bool fullDone = false;
                for (int code : t)
                {
                    if (code < 0) // non-SYM opener clip (below)
                    {
                        seq.push_back(-code);
                        continue;
                    }
                    int clip = ff[code - 1];
                    if (code == 4 || code == 11)
                        clip = beltClip(code, titleBelt);
                    if (row && !(row->f[12] >= 0 && row->f[13] >= 0 && row->f[14] >= 0) && (code == 11 || code >= 13) && code != 17)
                    {
                        const auto tf = row->titleFull.find(uint32_t(titleBelt));
                        const bool hasTitle = std::find(t.begin(), t.end(), 11) != t.end();
                        const int one = hasTitle && tf != row->titleFull.end() ? tf->second : row->full;
                        if (one >= 0 && (code >= 13 || tf != row->titleFull.end()))
                        {
                            if (!fullDone)
                                seq.push_back(one);
                            fullDone = true;
                            continue;
                        }
                    }
                    if (clip > 0 || (clip == 0 && code >= 13)) // 0 = empty field in SYM (codes 4/11 resolved above)
                        seq.push_back(clip);
                }
                std::string tl, cl;
                for (int c : t) tl += (tl.empty() ? "" : ",") + std::to_string(c);
                for (size_t i = first; i < seq.size(); ++i) cl += (cl.empty() ? "" : ",") + std::to_string(seq[i]);
                std::fprintf(stderr, "[finkel] roster 0x%02X%s idx %u cat %d table [%s] clips [%s]%c", id & kRosterMask,
                             f ? " (stable)" : "", idx, member0 ? cat : 28, tl.c_str(), cl.c_str(), 10);
            };
            // non-SYM openers (Jordan Nov 6): LoW Finkel match-type lines SYM never had, for HCTP rules with entrances
            std::vector<int> table = timeTable(cat);
            if (idx == 0 && openerClip_ >= 0)
            {
                if (table.front() == 1)
                    table.front() = -openerClip_; // replaces "scheduled for one fall" (triple threat / fatal 4-way)
                else
                    table.insert(table.begin(), -openerClip_); // ladder: before "making his way"
            }
            if (stable)
            {
                say(stable, table, g[0], true); // StableCall: ONE call for the whole group, partners silent
            }
            else
                for (size_t k = 0; k < g.size(); ++k)
                    say(nullptr, k ? timeTable(28) : table, g[k], k == 0);
            // non-SYM (Jordan Nov 2, 2K14 / LoW team clips): a known team walking out together that SYM has no
            // StableCall row for: the members, then Finkel's team call (FINKEL_LINES team_<key>)
            if (!stable && g.size() > 1u)
            {
                std::vector<uint32_t> s = members;
                std::sort(s.begin(), s.end());
                const std::string key = stableKey(s);
                if (!key.empty())
                    if (int c = pickConnector("team_" + key); c >= 0)
                    {
                        seq.push_back(c);
                        std::fprintf(stderr, "[finkel] %s together -> team call %d (non-SYM)%c", key.c_str(), c, 10);
                    }
            }
            // MakeCallListNormal: this+0x9c += every member of a stable group (non-stable groups also count each
            // member here: HCTP shared entrances are stable/partner entrances)
            for (uint32_t id : g)
                entrants_.push_back(id);
            for (size_t i = 0; i < seq.size(); ++i)
                queue_.push_back({uint16_t(seq[i]), i ? 0.25 : 1.0, 14u});
            cv_.notify_all();
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

        // [COMMFIX Nov 6] entrance preview with no match setup (Create-a-Moveset): one wrestler, its own show (the last
        // match's entrants no longer silence him), no match rule / opener / title at stake -> SYM cat 0 singles intro.
        void previewEntrance(uint32_t roster)
        {
            std::lock_guard<std::mutex> lock(m_);
            if (!loaded_ || inMatch_)
                return;
            entrants_.clear();
            resetPending_ = false;
            matchRule_ = 0u;
            openerClip_ = -1;
            titleMatchBelt_ = -1;
            std::fprintf(stderr, "[finkel] preview entrance roster 0x%02X (CAM)%c", roster & kRosterMask, 10);
            finkelCall(&roster, 1u);
        }
        // [/COMMFIX]

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

        // runner: entrance #0 was set up (CSd5EventData type 2, order 0). Some entrances set #0 up again part way
        // (a second scene) -> clearing right away let the same wrestler be introduced twice (tester report Nov 4,
        // John Cena in a singles match). Clear lazily: the next entrance decides whether this is a new show.
        void resetEntrances()
        {
            std::lock_guard<std::mutex> lock(m_);
            resetPending_ = true;
        }

        // called with m_ held before an entrance group is announced; false = this entrant was already introduced in
        // this show (a repeated #0 setup / Play), so say nothing
        bool beginEntrance(uint32_t firstId)
        {
            const uint32_t key = firstId & kRosterMask;
            const bool recent = Clock::now() - lastEntrance_ <= std::chrono::minutes(4);
            const bool known = std::any_of(entrants_.begin(), entrants_.end(),
                                           [&](uint32_t e) { return (e & kRosterMask) == key; });
            if (recent && known)
            {
                std::fprintf(stderr, "[hctp-pbp] entrance roster 0x%02X already introduced this show%s -> silent%c",
                             firstId & kRosterMask, resetPending_ ? " (entrance #0 set up again)" : "", 10);
                return false;
            }
            if (resetPending_)
                entrants_.clear();
            resetPending_ = false;
            return true;
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
            loadGenericFinishers();
            dropMisfiledBanter();
            // SvR plays each table row as written (FUN_00212f30 picks a row, FUN_00212df0 plays w1..w7); it has no
            // even/odd pairing (notes/SVR_BANTER_EXCHANGES_20261106.md). HCTP_PBP_PAIR_BANTER=1 = old invented joins.
            if (const char *p = std::getenv("HCTP_PBP_PAIR_BANTER"); p && *p == '1')
                pairBanter();
            loadMaleLines();
            loadFinkel();
            loadFinkelDat();
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
            std::set<std::tuple<uint32_t, uint32_t, uint32_t>> drop;
            while (std::getline(f, row))
            {
                std::stringstream ss(row);
                std::string a, b, c;
                if (std::getline(ss, a, ',') && std::getline(ss, b, ','))
                {
                    // optional 3rd column = only that roster id's row (a clip moved to another wrestler, e.g. JR's "dead
                    // man" line from Undertaker 0x03 to Legend 0x48 via PBPEXTRA_LIST)
                    const uint32_t key = std::getline(ss, c, ',') && !c.empty() ? uint32_t(std::strtoul(c.c_str(), nullptr, 16)) : 0xFFFFu;
                    drop.insert({uint32_t(std::strtoul(a.c_str(), nullptr, 10)), uint32_t(std::strtoul(b.c_str(), nullptr, 10)), key});
                }
            }
            size_t n = 0;
            for (const TeamTables *t : {&kColeTazz, &kJrKing})
                if (t->wrestlerBanter >= 0 && size_t(t->wrestlerBanter) < tables_.size())
                {
                    auto &v = tables_[t->wrestlerBanter];
                    const size_t before = v.size();
                    v.erase(std::remove_if(v.begin(), v.end(), [&](const Row &r) { return drop.count({r.part, r.w[1], 0xFFFFu}) != 0 || drop.count({r.part, r.w[1], uint32_t(r.w[0])}) != 0; }), v.end());
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
                                if ((r.w[0] == id || r.w[0] == (id & kRosterMask)) &&
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
        bool resetPending_ = false;
        bool loaded_ = false, raw_ = false, inMatch_ = false, teamForced_ = false, playerHeld_ = false, tagMatch_ = false;
        uint32_t matchRule_ = 0, teamSize_ = 2;
        bool braPanties_ = false;
        uint32_t rumbleCount_ = 0; // Royal Rumble wrestlers announced this match
        uint32_t lastCalledWho_ = 0xFFFFFFFFu; // [COMMFIX Nov 6] SvR: last (wrestler, move) call
        std::vector<decltype(queue_)::value_type> rumbleOpen_; // [COMMFIX Nov 6] Rumble opening, queued at match start
        int titleMatchBelt_ = -1; // Season title match: belt at stake (0..8, CTitleHistoryData order), -1 = none
        std::map<uint32_t, FnklRow> fnkl_; // BGM/FINKEL_DAT.csv (SYM /FNKL/DAT layout)
        std::set<std::pair<uint32_t, uint32_t>> maleLines_;
        struct GenericLine
        {
            uint32_t part;
            uint16_t clip;
            bool male;
        };
        std::map<uint32_t, std::vector<GenericLine>> generic_[2]; // [Cole/Tazz, JR/King] HCTP move id -> lines
        std::pair<uint32_t, uint32_t> lastGeneric_{0, 0};
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

extern "C" void hctpCommentaryPreviewEntrance(uint32_t rosterId) // [COMMFIX Nov 6]
{
    if (engineReady())
        Engine::get().previewEntrance(rosterId);
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
