#pragma once
// HCTP native movie player: the game's Sofdec movies (.sfd = MPEG-1 video + ADX audio, alone or inside an .afs) are
// demuxed and decoded on the host with FFmpeg instead of running Sofdec on the emulated EE/IPU/IOP. The game keeps
// its own movie control flow (CADX_CTRL's SFD channels); the overrides hand each channel's start / stop / pause /
// status / "load the current frame" here. Audio plays on its own host stream and is the clock the video follows.
#include <cstdint>

// Opens entry `fid` of an AFS archive (fid < 0: the whole file is one .sfd) on channel ch and starts playing.
extern "C" bool hctpMovieOpen(int ch, const char *path, int fid, bool loop);
extern "C" void hctpMovieStop(int ch);
extern "C" void hctpMoviePause(int ch, bool pause);
extern "C" bool hctpMovieActive(int ch);
// 0 = stopped, 2 = playing, 3 = ended (Sofdec's MWE_PLY_STAT values)
extern "C" int hctpMovieStatus(int ch);
// Copies the frame due now into rgba (w x h, RGBA8888, alpha 0x80 like Sofdec's output); returns true when it is a
// frame not handed out before.
extern "C" bool hctpMovieFrame(int ch, uint8_t *rgba, int w, int h);
// Output volume (0..1) of a channel's audio.
extern "C" void hctpMovieSetVolume(int ch, float volume);
