# HCTP recomp - Vulkan base (Claude Code instructions)

Native PC port of WWE SmackDown! Here Comes the Pain (PS2 NTSC, SLUS_207.87), static recompilation (PS2Recomp family).
THIS TREE (`F:\hctp-newbase-vulcan`, git worktree of `F:\hctp-newbase\PS2Recomp`, branch `codex/new-base-vulkan`)
IS THE ACTIVE BASE as of Oct 26 2026. Renderer = `ps2-native-gs` (Vulkan, GPL-3.0, third_party\ps2-native-gs).

## Read first (in this order)
1. `F:\hctp-project\notes\CLAUDE_REVIEW_20261026.md` - tree map, decisions, Jordan's corrections.
2. `F:\hctp-project\notes\CLAUDE_AUDIT_COPY_20261026.md` - what Codex's DX11 work really did (true vs false).
3. Bottom of `F:\hctp-project\notes\PROGRESS_NOTES.md` (newest at the bottom; 630 KB - read only the tail).
4. Codex's own handoff (conservative, useful repo catalog): ask Jordan for CLAUDE_HANDOFF_FULL_2026-10-26.md if needed.

## Goal (Jordan)
Phase 1 NOW: finish/perfect the game on THIS Vulkan tree (full playthrough clean: boot, menus, CAW, season,
entrances, matches, saves, audio, movies). Phase 2: native rendering, starting with one wrestler: hook
CSkinModel::DrawChar (0x155dc0), draw the exported model (Copy tree game\PC\assets\hctp_native) in Vulkan with the
live pose/camera, skip the PS2 draw, compare side by side. Long-term: no PS2 limits (any model/arena), then remove
emulation layer by layer. Jordan wants native, not emulation - but the emulated GS stays as scaffold/reference
until each piece is replaced.

## Build / run (this tree)
- Build: `F:\hctp-newbase-vulcan\build_hctp_vulkan.bat` -> `out\native-gs\ps2xRuntime\ps2EntryRunner.exe`
  (MSVC VS 18 Community vcvars64, Ninja, PS2X_SKIP_IPO_PROBE=ON for the local C1902 PDB issue).
- Run: `run_hctp_native_gs.bat` (game folder = `F:\hctp-newbase-vulcan\WWE SmackDown! Here Comes the Pain`, vanilla).
  Diagnostics: `run_native_gs_stats.bat` (PS2X_GS_NATIVE_STATS skip counters, readback log) -> logs\native-gs-stats.err.
- Run in background, kill after the needed time, never two instances. stderr is huge: grep tags, never read whole.
- Useful native-gs env: PS2X_GS_NATIVE_STATS, _READBACK_LOG, _SCANOUT_LOG, _NO_FIFO, _ALIAS_SKIP; ours:
  HCTP_NATIVE_GS_DIRECT (0 = CPU-readback/D3D11 fallback), HCTP_NATIVE_GS_FULL_FIELDS=1 (undo field fix),
  HCTP_NATIVE_GS_READBACK_DIAG.

## State on Oct 26 (UNTESTED changes first)
- gs_gpu_native_module.cpp: field-mode scanout fix written by Claude, NOT YET BUILT. HCTP uses SMODE2 INT+FFMD
  (smode2=3), 640x224 field buffers at FBP 0/70; the module read DH+1=448 lines -> lower band on every screen.
  Fix halves DH (applies MAGV) before calling the module. First job: build, run, confirm band gone.
- Earlier fix kept: scanout row stride = width*4.
- Direct Vulkan window presentation (attachWindow) added by Codex late Oct 25; verify it in the log.
- Native GS skips draws whose FRAME is a depth format (SkipDepthTarget) - watch for missing effects.
- Last run reached menus, entrances #0/#1, arena 0 at ~56-60 game FPS. Commentary off (vanilla folder lacks BGM\PBPSVR*).

## Phase-1 TODO
1. Build + test the field fix (caution, title, movie, menu, entrance, match). Commit.
2. COMMIT the Copy tree too (`F:\hctp-newbase - Copy`, and its PS2Recomp submodule): everything after Oct 24 is uncommitted.
3. Port proven fixes from the Copy one at a time (diff vs F:\hctp-newbase): ADX native-handle guards (overrides),
   native movies, p07 (Jordan must verify leg shake), settings/launcher. NOT the host-resource/RT-alias experiments.
4. Game heap: Kernel/Syscalls/System.cpp SetupHeap only extends with HCTP_EXTENDED_GUEST_HEAP set; Copy runtime
   defaults ON -> make consistent so the game actually gets the extra RAM.
5. Native VU1 defaults: Codex turned p16/p17/p08/p18/p01-p04 OFF in the Copy; re-verify by output and re-enable what matches.
6. Real FPS: report the GAME frame counter (`game=`), not the window rate.
7. Then playthrough bug list with Jordan.

## Rules from Jordan
- USAGE: weekly limit is tight (he burns a week in 2-3 days). Batch changes, ONE build, one test. No polling loops.
  Targeted greps. Look at few frames.
- PCSX2 source (`F:\pcsx2-master\pcsx2-master`) is the behaviour reference. Our CPU backend (gs_cpu_backend.cpp) also
  has proven HCTP display rules (field mode, DBY stabilisation, interlace).
- Don't regress caution/memory-card screen, menu flicker, nameplates/HUD text (Native GS renders them right).
- Test yourself, including reaching a match. Never change system/security settings.
- Keep `F:\hctp-newbase` untouched; `F:\hctp-newbase - Copy` = frozen DX11 parts bin; `F:\hctp-project` = notes + old fork.
- Don't hand-edit generated runner code unless unavoidable. Keep GPL/Apache credits.
- Update PROGRESS_NOTES.md after every batch (exact tree, build result, flags, test path, what was seen). Commit often.
- Be honest about what is verified vs assumed; Jordan was burned by unverified claims.

## Other material Jordan has (ask for paths when needed)
SvR06 PS2 betas x2 (symbol maps, ~2,166 name overlaps with HCTP), SvR07 PS2 Sep 2006 proto (stripped, .sndata),
SvR07 360 beta, WM21 + 2 WM21 betas (Xbox XACT commentary = HCTP+SvR lines + extras; check betas for more lines),
Rumble Roses XX recomp, ps2-port-engine, DevilKings PS2Recomp fork, AC5/BT3 recomps. HCTP's own ELF has 24,475 symbols.
