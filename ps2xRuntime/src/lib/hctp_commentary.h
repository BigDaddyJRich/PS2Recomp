#pragma once
// HCTP commentary restoration: HCTP's own play-by-play engine (SvR's PBPSystem) was stripped from the ELF, so this
// host-side engine plays SvR retail commentary from a host-only pack next to the game:
//   BGM/PBPSVR.afs       = SvR retail BGM/pbppac.afs (2,472 plain ADX clips)
//   BGM/PBPSVR_LIST.pac  = SvR's nested PBP LIST ('PAC ', 33 tables; layout in tools/decode_svr_pbplist.py)
// Table meanings come from transcribing every clip (notes/svr_pbp_transcripts.csv, svr_pbp_tagged.csv).
#include <cstdint>

// Implemented in the game overrides (native ADX player, AFS partition 13 = BGM/PBPSVR.afs).
extern "C" bool hctpNativePlayPbpClip(uint32_t clipIndex);
// Any host AFS partition (14 = BGM/FINKEL.afs).
extern "C" bool hctpNativePlayAfsClip(uint32_t partition, uint32_t clipIndex);
// Stops the commentary line and frees its player (teardown, and whenever the engine goes quiet).
extern "C" void hctpNativeStopCommentary();

enum HctpPbpEvent : uint32_t
{
    kPbpBell = 1,      // referee bell: first one = match start, the next one = match end
    kPbpTag = 2,       // tag made
    kPbpInterfere = 3, // someone not in the match gets involved
    kPbpOutside = 4,   // action outside the ring
    kPbpStop = 5,      // match / demo torn down (skip, quit, restart): stop at once
};

// Called by the overrides on game events (any thread).
extern "C" void hctpCommentaryEvent(uint32_t event);

// An entrance starts (hooks in the overrides): Howard Finkel introduces roster id `rosterId`
// from BGM/FINKEL.afs + BGM/FINKEL_LINES.csv (roster_id,kind,clip,text; tools/build_finkel_pack.py).
extern "C" void hctpCommentaryEntrance(uint32_t rosterId);

// A wrestler's move made a sound (GameSE hook): chara = CChar* (guest address), moveId = CChar+0xA4.
// New moves that are in SvR's finisher tables (11-12 Cole/Tazz, 28-29 JR/King: wrestler, move, clip, delay) get called.
extern "C" void hctpCommentaryMove(uint32_t chara, uint32_t moveId);

// Which wrestler a CChar is (shared moves: owner-only lines). Called by the GameSE hook.
extern "C" void hctpCommentaryCharaRoster(uint32_t chara, uint32_t rosterId);

// Arena / show at the bell (GetNowStage): picks Cole/Tazz or JR/King; PPVs skip the city welcome.
extern "C" void hctpCommentaryArena(int stage);

// The wrestlers' majority Raw/SmackDown brand (0 Raw, 1 SmackDown, -1 none): picks the team in shared arenas.
// Send before hctpCommentaryArena.
extern "C" void hctpCommentaryBrand(int brand);

// Who is in the match (roster ids), sent at the opening bell -> wrestler-specific banter (SvR tables 4 / 22, w0 = id).
extern "C" void hctpCommentaryParticipants(const uint32_t *rosterIds, uint32_t count);
