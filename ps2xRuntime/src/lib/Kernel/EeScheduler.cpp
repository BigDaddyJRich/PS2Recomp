#include <set>
#include <tuple>
#include <thread>
#include <cstdio>
#include "runtime/ps2_host_profile.h"
#include "runtime/ee_scheduler.h"

#include "ps2_log.h"
#include "ps2_runtime_macros.h"

#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <variant>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace
{
    constexpr int KE_OK = 0;
    constexpr int KE_ERROR = -1;
    constexpr int KE_ILLEGAL_PRIORITY = -403;
    constexpr int KE_ILLEGAL_THID = -406;
    constexpr int KE_UNKNOWN_THID = -407;
    constexpr int KE_UNKNOWN_SEMID = -408;
    constexpr int KE_UNKNOWN_EVFID = -409;
    constexpr int KE_DORMANT = -413;
    constexpr int KE_NOT_DORMANT = -414;
    constexpr int KE_NOT_SUSPEND = -415;
    constexpr int KE_NOT_WAIT = -416;
    constexpr int KE_RELEASE_WAIT = -418;
    constexpr int KE_SEMA_ZERO = -419;
    constexpr int KE_SEMA_OVF = -420;
    constexpr int KE_EVF_COND = -421;
    constexpr int KE_WAIT_DELETE = -425;

    constexpr uint32_t WEF_OR = 0x01u;
    constexpr uint32_t WEF_CLEAR = 0x10u;
    constexpr uint32_t WEF_CLEAR_ALL = 0x20u;
    constexpr auto kVBlankPeriod = std::chrono::microseconds(16667);
    constexpr auto kVBlankDuration = std::chrono::microseconds(500);
    // how far the VBlank schedule may fall behind real time before it restarts from now (see processDueDeadlines)
    constexpr auto kVBlankResync = std::chrono::milliseconds(100);
    constexpr uint64_t kAlarmTickMicroseconds = 64u;
    constexpr uint32_t kDebugPublishDispatchInterval = 4096u;

    constexpr uint64_t microsecondsToEeCycles(uint64_t microseconds)
    {
        return (microseconds * EeScheduler::kEeClockHz + 999999ull) / 1000000ull;
    }

    // Waits on cv until pred() or the deadline. A short wait (under 2 ms) is timed by yielding instead of sleeping: the
    // OS rounds a sleep up to its timer tick (1 ms or more), and an event due in a quarter of a millisecond (a disc
    // read's completion alarm, 64 us ticks) then costs a whole tick — a game that loads one 2 KB sector per alarm
    // loaded several times slower than its alarms ask for. Longer waits (the next vblank) sleep as before.
    template <typename Pred>
    bool waitUntilPrecise(std::condition_variable &cv, std::unique_lock<std::mutex> &lock,
                          std::chrono::steady_clock::time_point deadline, Pred pred)
    {
        constexpr auto kPreciseWait = std::chrono::milliseconds(2);
        if (deadline == std::chrono::steady_clock::time_point::max() ||
            deadline - std::chrono::steady_clock::now() > kPreciseWait)
            return cv.wait_until(lock, deadline, pred);
        bool done = false;
        while (!(done = pred()) && std::chrono::steady_clock::now() < deadline)
        {
            lock.unlock();
            std::this_thread::yield();
            lock.lock();
        }
        return done;
    }

    std::chrono::nanoseconds eeCyclesToHostDuration(uint64_t cycles)
    {
        constexpr uint64_t kNanosecondsPerSecond = 1000000000ull;
        const uint64_t wholeSeconds = cycles / EeScheduler::kEeClockHz;
        const uint64_t remainingCycles = cycles % EeScheduler::kEeClockHz;
        const uint64_t remainingNanoseconds = (remainingCycles * kNanosecondsPerSecond + EeScheduler::kEeClockHz - 1u) / EeScheduler::kEeClockHz;
        return std::chrono::seconds(wholeSeconds) + std::chrono::nanoseconds(remainingNanoseconds);
    }

    constexpr uint64_t kVBlankPeriodCycles = microsecondsToEeCycles(16667u);

    // VENDETTA_HOSTPROF: when the scheduler is about to sleep 30 ms or more, print why — every pending deadline with how
    // far away it is in EE time and in real time ("[sleep]"); a sleep that long holds a vblank up (a hitch on screen)
    bool sleepLogEnabled()
    {
        static const bool enabled = std::getenv("VENDETTA_HOSTPROF") != nullptr;
        return enabled;
    }

    const char *eventTypeName(EeEventType type)
    {
        switch (type)
        {
        case EeEventType::Stop: return "stop";
        case EeEventType::VBlankStart: return "vblank";
        case EeEventType::VBlankEnd: return "vblank-end";
        case EeEventType::Dmac: return "dmac";
        case EeEventType::ExternalWake: return "wake";
        case EeEventType::Alarm: return "alarm";
        }
        return "?";
    }

    struct SleepDiary
    {
        std::vector<std::string> entries;
        std::chrono::steady_clock::time_point lastVBlank{};
    };

    SleepDiary &sleepDiary()
    {
        static SleepDiary diary;
        return diary;
    }

    template <typename Deadlines>
    std::string describeDeadlines(const Deadlines &deadlines, uint64_t eeCycle, std::chrono::steady_clock::time_point now)
    {
        std::string text;
        char item[128];
        for (const auto &scheduled : deadlines)
        {
            const double eeMs = static_cast<double>(static_cast<int64_t>(scheduled.deadlineCycle - eeCycle)) /
                                static_cast<double>(EeScheduler::kEeClockHz / 1000u);
            const double realMs = std::chrono::duration<double, std::milli>(scheduled.hostDeadline - now).count();
            std::snprintf(item, sizeof(item), " | %s %u: EE %+.1f ms, real %+.1f ms", eventTypeName(scheduled.event.type),
                          static_cast<unsigned>(scheduled.event.id), eeMs, realMs);
            text += item;
        }
        return text;
    }
    constexpr uint64_t kVBlankDurationCycles = microsecondsToEeCycles(500u);
    constexpr uint64_t kAlarmTickCycles = microsecondsToEeCycles(kAlarmTickMicroseconds);

    template <typename Map>
    int allocatePositiveId(int &nextId, const Map &objects)
    {
        const int first = std::max(1, nextId);
        int candidate = first;
        do
        {
            if (!objects.contains(candidate))
            {
                nextId = (candidate == std::numeric_limits<int>::max()) ? 1 : candidate + 1;
                return candidate;
            }
            candidate = (candidate == std::numeric_limits<int>::max()) ? 1 : candidate + 1;
        } while (candidate != first);
        return 0;
    }
}

EeScheduler::EeScheduler(PS2Runtime &runtime)
    : m_runtime(runtime)
{
}

EeScheduler::~EeScheduler()
{
    requestStop();
}

void EeScheduler::reset(uint8_t *rdram, const R5900Context &mainContext)
{
    m_executorThread = std::this_thread::get_id();
    m_rdram = rdram;
    m_readyQueues = {};
    m_threads.clear();
    m_semaphores.clear();
    m_eventFlags.clear();
    m_alarms.clear();
    m_intcHandlers.clear();
    m_dmacHandlers.clear();
    m_nextThreadId = kFirstThreadId;
    m_nextInvocationThreadId = -1;
    m_nextSemaphoreId = 1;
    m_nextEventFlagId = 1;
    m_nextAlarmId = 1;
    m_nextIntcHandlerId = 1;
    m_nextDmacHandlerId = 1;
    m_intcHeadOrder = 0;
    m_intcTailOrder = 1000;
    m_dmacHeadOrder = 0;
    m_dmacTailOrder = 1000;
    m_enabledIntcMask = 0xFFFFFFFFu;
    m_enabledDmacMask = 0xFFFFFFFFu;
    m_currentThreadId = 0;
    m_rescheduleRequested = false;
    m_timeSliceExpired = false;
    m_insideInterrupt = false;
    m_pendingEeTimerInterrupts = 0u;
    m_eeCycle = 0u;
    m_iopPendingCycles = 0u;
    m_sliceEndCycle = kDefaultTimeSliceCycles;
    m_stopRequested.store(false, std::memory_order_release);
    m_checkpointPending.store(false, std::memory_order_release);
    m_debugPublishCountdown = 0u;
    {
        std::lock_guard lock(m_eventMutex);
        m_events.clear();
        m_deadlines.clear();
        m_pendingInvocations.clear();
    }
    m_eventSequence = 0;
    m_invocationSequence = 0;
    m_presentDue = false;
    m_presentTick = 0;
    m_presentDueVblanks = 0;
    m_vsyncHandlerSeqs.clear();
    {
        const char *mode = std::getenv("VENDETTA_GS_PRESENT");
        m_presentMode = (mode && std::strcmp(mode, "vblank") == 0) ? PresentMode::VBlank : PresentMode::Settle;
    }
    m_vsyncTick = 0;
    m_vsyncFlagAddress = 0;
    m_vsyncTickAddress = 0;
    m_gsVSyncCallback = 0;
    m_gsVSyncCallbackGp = 0;
    m_gsVSyncCallbackSp = 0;
    m_runtime.memory().gs().vsyncTick.store(0u, std::memory_order_release);
    m_runtime.memory().resetEeTimers();

    GuestThread main{};
    main.id = kMainThreadId;
    main.context = mainContext;
    main.entry = mainContext.pc;
    // $sp is live execution state, not the stable initial stack descriptor
    // returned by ReferThreadStatus. SetupThread records that metadata.
    main.stack = 0u;
    main.gp = getRegU32(&mainContext, 28);
    main.initialPriority = 0;
    main.currentPriority = 0;
    main.status = EeThreadStatus::Ready;
    m_threads.emplace(main.id, std::move(main));
    m_readyQueues[0].push_back(kMainThreadId);
    scheduleEvent(m_eeCycle + kVBlankPeriodCycles,
                  std::chrono::steady_clock::now() + kVBlankPeriod,
                  EeEvent{EeEventType::VBlankStart, 0, 0});
    publishSnapshot();
}

void ps2RecordDispatch(char what, uint32_t pc, uint32_t sp, uint32_t ra, int thread); // ps2_runtime.cpp
std::string ps2FormatDispatchRecords();
bool ps2DispatchRecording() noexcept; // ps2_runtime.cpp: VENDETTA_DISPATCHLOG

void EeScheduler::run()
{
    assertExecutor();
    m_running.store(true, std::memory_order_release);

    while (!m_stopRequested.load(std::memory_order_acquire))
    {
        processPendingEvents();
        maybePresent();
        if (m_stopRequested.load(std::memory_order_acquire))
        {
            break;
        }

        if (m_currentThreadId == 0)
        {
            GuestThread *next = selectReady();
            if (!next && m_pendingInvocations.empty())
            {
                copyMainContextToRuntime();
                publishIdleDebugContext();
                publishSnapshot();
                waitForEvent();
                continue;
            }
            if (next)
            {
                makeRunning(*next);
            }
            else
            {
                GuestThread *owner = &acquireInvocationThread();
                GuestInvocation invocation = std::move(m_pendingInvocations.front());
                m_pendingInvocations.pop_front();
                owner->status = EeThreadStatus::Running;
                m_currentThreadId = owner->id;
                renewTimeSlice();
                if (getRegU32(&invocation.context, 29) == 0u)
                {
                    SET_GPR_U32(&invocation.context, 29, invocationStackTop());
                }
                owner->invocations.push_back(std::move(invocation));
            }
        }

        GuestThread *running = currentThread();
        assert(running != nullptr);
        if (running->resumeCompletion)
        {
            auto completion = std::move(running->resumeCompletion);
            running->resumeCompletion = {};
            try
            {
                completion(running->activeContext());
            }
            catch (const EeDispatcherTransfer &)
            {
            }
            if (m_currentThreadId == 0)
            {
                continue;
            }
        }
        R5900Context &context = running->activeContext();
        if (m_debugPublishCountdown == 0u)
        {
            copyMainContextToRuntime();
            publishSnapshot();
            m_debugPublishCountdown = kDebugPublishDispatchInterval - 1u;
        }
        else
        {
            --m_debugPublishCountdown;
        }

        publishDebugContext(context);

        if (context.pc == 0u)
        {
            if (!running->invocations.empty())
            {
                GuestInvocation completed = std::move(running->invocations.back());
                running->invocations.pop_back();
                if (completed.onComplete)
                {
                    try
                    {
                        completed.onComplete(completed.context, running->activeContext());
                    }
                    catch (const EeDispatcherTransfer &)
                    {
                    }
                }
                if (!m_vsyncHandlerSeqs.empty())
                {
                    const auto handler = std::find(m_vsyncHandlerSeqs.begin(), m_vsyncHandlerSeqs.end(), completed.sequence);
                    if (handler != m_vsyncHandlerSeqs.end())
                    {
                        m_vsyncHandlerSeqs.erase(handler);
                        maybePresent();
                    }
                }
                continue;
            }
            // pc 0 with no callback = the thread left its entry function (or jumped through a null pointer): it
            // ends. Say so — a thread that loops forever only gets here by a bug.
            if (running->id > 0) // not the runtime's own invocation threads (negative ids), which end here normally
                std::cerr << "[ee] thread " << running->id << " (entry 0x" << std::hex << running->entry << ") ended at pc 0: ra=0x"
                      << getRegU32(&context, 31) << " sp=0x" << getRegU32(&context, 29) << " trace=" << m_runtime.dispatchTrace()
                      << " records:" << ps2FormatDispatchRecords() << std::dec << std::endl;
            if (running->id > 0 && m_rdram)
            {
                // the thread's stack around sp (zeroed save slots show which frame was overwritten)
                const uint32_t sp = getRegU32(&context, 29) & 0x01FFFFF0u;
                for (uint32_t row = (sp >= 0x100u ? sp - 0x100u : 0u); row < sp + 0x400u && row + 16u <= PS2_RAM_SIZE; row += 16u)
                {
                    uint32_t w[4];
                    std::memcpy(w, m_rdram + row, sizeof(w));
                    std::fprintf(stderr, "[ee]   %08x: %08x %08x %08x %08x%c", row, w[0], w[1], w[2], w[3], 10);
                }
            }
            makeDormant(*running);
            m_currentThreadId = 0;
            copyMainContextToRuntime();
            publishIdleDebugContext();
            publishSnapshot();
            continue;
        }

        if (!m_pendingInvocations.empty() &&
            (running->invocations.empty() ||
             running->invocations.back().kind == GuestInvocationKind::ThreadCallback))
        {
            GuestInvocation invocation = std::move(m_pendingInvocations.front());
            m_pendingInvocations.pop_front();
            if (getRegU32(&invocation.context, 29) == 0u)
            {
                SET_GPR_U32(&invocation.context, 29, invocationStackTop());
            }
            running->invocations.push_back(std::move(invocation));
            continue;
        }

        if (!m_runtime.hasFunction(context.pc))
        {
            if (!running->invocations.empty())
            {
                context.pc = 0u;
            }
            else
            {
                m_runtime.reportMissingFunction(m_rdram,
                                                &context,
                                                context.pc,
                                                context.pc,
                                                PS2Runtime::GuestBranchKind::DirectJump,
                                                "EE scheduler");
                makeDormant(*running);
                m_currentThreadId = 0;
            }
            continue;
        }
        PS2Runtime::RecompiledFunction function = m_runtime.lookupFunction(context.pc);

        if (checkpointDue(kGuestDispatchCycles))
        {
            continue;
        }

        try
        {
            m_insideInterrupt = !running->invocations.empty() && running->invocations.back().kind == GuestInvocationKind::Interrupt;
            m_guestExecuting.store(true, std::memory_order_release);
            if (ps2DispatchRecording())
                ps2RecordDispatch('D', context.pc, getRegU32(&context, 29), getRegU32(&context, 31), running->id);
            function(m_rdram, &context, &m_runtime);
            m_guestExecuting.store(false, std::memory_order_release);
            m_insideInterrupt = false;
        }
        catch (const EeDispatcherTransfer &)
        {
            m_guestExecuting.store(false, std::memory_order_release);
            m_insideInterrupt = false;
        }
        catch (...)
        {
            m_guestExecuting.store(false, std::memory_order_release);
            m_running.store(false, std::memory_order_release);
            publishSnapshot();
            throw;
        }

        processPendingEvents();
        if (m_rescheduleRequested && m_currentThreadId != 0)
        {
            GuestThread *preempted = currentThread();
            assert(preempted != nullptr);
            enqueueReady(*preempted, !m_timeSliceExpired);
            m_currentThreadId = 0;
            m_rescheduleRequested = false;
            m_timeSliceExpired = false;
        }
    }

    m_guestExecuting.store(false, std::memory_order_release);
    m_running.store(false, std::memory_order_release);
    copyMainContextToRuntime();
    publishSnapshot();
}

void EeScheduler::requestStop()
{
    m_stopRequested.store(true, std::memory_order_release);
    m_checkpointPending.store(true, std::memory_order_release);
    m_eventCv.notify_all();
}

void EeScheduler::postEvent(EeEvent event)
{
    if (event.type == EeEventType::Stop)
    {
        requestStop();
        return;
    }

    {
        std::lock_guard lock(m_eventMutex);
        m_events.push_back(event);
        m_checkpointPending.store(true, std::memory_order_release);
    }
    m_eventCv.notify_one();
}

// The EE accounts a small fixed number of cycles per dispatch / checkpoint, far less than the real time a busy
// guest loop takes on the host, so while the guest is busy EE time (timers, alarms, the IOP — a sound driver's
// 48 kHz stream) fell behind real time: Def Jam Vendetta's movie player polled at ~half speed and its sound
// stuttered. Every 1024 checkpoints, catch EE time up with the host clock — at most 1 ms at a time, so a timer
// compare is never jumped by a whole period (interrupts are taken at the next checkpoint); re-anchor after a long
// stall (> 100 ms) instead of racing through it. When EE time is ahead, the VBlank / timer pacing waits hold it.
void EeScheduler::catchUpWithHostClock() noexcept
{
    // Every 16 checkpoints, up to 2 ms (was 1024 / 1 ms): HCTP's CRender::Synch -> y_sceGsSyncPath busy-polls DMA / GS
    // status, each poll a register access + chain fence (~0.1 ms of real time) charged a few EE cycles, so 1024 polls
    // (~100 ms) caught EE time up by only 1 ms: vblanks came 100-120 ms apart, matches ran under 10 fps on an idle PC.
    // A clock read every 16 checkpoints costs nothing next to them; 2 ms stays under one period of a 100 Hz timer.
    constexpr uint32_t kCheckInterval = 16u;
    constexpr uint64_t kMaxStepCycles = EeScheduler::kEeClockHz / 500u;     // 2 ms
    constexpr uint64_t kResyncCycles = EeScheduler::kEeClockHz / 10u;       // 100 ms
    if (m_hostClockCheckCountdown != 0u)
    {
        --m_hostClockCheckCountdown;
        return;
    }
    m_hostClockCheckCountdown = kCheckInterval;

    const auto now = std::chrono::steady_clock::now();
    if (m_hostClockOrigin == std::chrono::steady_clock::time_point{})
    {
        m_hostClockOrigin = now;
        m_hostClockOriginCycle = m_eeCycle;
        return;
    }
    const uint64_t elapsedNs = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now - m_hostClockOrigin).count());
    const uint64_t hostCycle = m_hostClockOriginCycle + elapsedNs / 1000u * (EeScheduler::kEeClockHz / 1000000u);
    if (hostCycle <= m_eeCycle)
    {
        return;
    }
    const uint64_t behind = hostCycle - m_eeCycle;
    if (behind > kResyncCycles)
    {
        m_hostClockOrigin = now;
        m_hostClockOriginCycle = m_eeCycle;
        return;
    }
    // Never onto an event that is not due yet in real time: its pace hold would stop the guest until that moment.
    // The anchor and the VBlank schedule can disagree by up to the resync limit (the schedule restarts from now after a
    // stall, the anchor only past 100 ms behind); then every step here pushed EE time straight to the next VBlank and
    // the guest waited out the rest of the frame in the hold: between VBlanks it ran only the few checkpoints until the
    // next step, and Def Jam Vendetta's main thread finished no frame for seconds while the PC slept (a 4-player match
    // after a slow patch). Stop one cycle short: the guest keeps running, and once the event is due in real time the next
    // check here (or the guest's own cycles) reaches it.
    uint64_t step = std::min(behind, kMaxStepCycles);
    {
        std::lock_guard lock(m_eventMutex);
        for (const ScheduledEvent &scheduled : m_deadlines)
        {
            if (scheduled.deadlineCycle > m_eeCycle && scheduled.hostDeadline > now &&
                scheduled.deadlineCycle - m_eeCycle <= step)
            {
                step = scheduled.deadlineCycle - m_eeCycle - 1u;
            }
        }
    }
    if (step == 0u)
    {
        return;
    }
    accountCycles(static_cast<uint32_t>(step));
}

// Before a new deadline is set from EE time (an alarm): if EE time has fallen behind the real clock — the guest kept
// waiting on host work its cycle count does not show — catch it up. Behind = a pending event whose real-time moment has
// passed while its EE deadline has not (or EE time behind the host-clock anchor of catchUpWithHostClock). EE time moves
// up to the latest such deadline, but never past an event not yet due in real time, nor past a timer compare (they keep
// their order, and a handler that re-arms with "target += period - count" its one period), and not across a stall of
// over 100 ms (re-anchor instead, as catchUpWithHostClock). catchUpWithHostClock alone needs 1024 checkpoints per
// millisecond, and a guest that mostly waits makes too few. Def Jam Vendetta reads its loads one 2 KB sector at a time,
// each read opening the disc image on the PC (~1 ms of real time) against the 0.25 ms its completion alarm adds to EE
// time: EE time crawled at a quarter of real speed through a load, and the next vblank, due in EE time, came 60-90 ms
// late — a hitch on screen as a match loads and starts (and, with the old vblank catch-up, a fast-forward after it).
void EeScheduler::catchUpBeforeDeadline() noexcept
{
    constexpr uint64_t kResyncCycles = EeScheduler::kEeClockHz / 10u; // 100 ms
    const auto now = std::chrono::steady_clock::now();
    uint64_t target = m_eeCycle;
    if (m_hostClockOrigin != std::chrono::steady_clock::time_point{})
    {
        const uint64_t elapsedNs = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now - m_hostClockOrigin).count());
        const uint64_t hostCycle = m_hostClockOriginCycle + elapsedNs / 1000u * (EeScheduler::kEeClockHz / 1000000u);
        if (hostCycle > m_eeCycle)
        {
            if (hostCycle - m_eeCycle > kResyncCycles)
            {
                m_hostClockOrigin = now;
                m_hostClockOriginCycle = m_eeCycle;
            }
            else
            {
                target = hostCycle;
            }
        }
    }
    uint64_t barrier = m_eeCycle + m_runtime.memory().cyclesUntilNextEeTimerInterrupt();
    if (barrier < m_eeCycle)
    {
        barrier = std::numeric_limits<uint64_t>::max(); // no timer compare pending (the sum wrapped)
    }
    {
        std::lock_guard lock(m_eventMutex);
        for (const ScheduledEvent &scheduled : m_deadlines)
        {
            if (scheduled.deadlineCycle <= m_eeCycle)
            {
                continue;
            }
            if (scheduled.hostDeadline <= now)
            {
                target = std::max(target, scheduled.deadlineCycle); // overdue in real time: EE time is behind
            }
            else
            {
                // not due yet: stop short of it (reaching it would hold the guest until its moment — catchUpWithHostClock)
                barrier = std::min(barrier, scheduled.deadlineCycle - 1u);
            }
        }
    }
    target = std::min({target, barrier, m_eeCycle + kResyncCycles});
    while (target > m_eeCycle)
    {
        const uint32_t chunk = static_cast<uint32_t>(std::min<uint64_t>(target - m_eeCycle, std::numeric_limits<uint32_t>::max()));
        accountCycles(chunk);
    }
}

// VENDETTA_EE_CYCLE_PCT (default 100): scales the cycles guest code is charged at its checkpoints (32 per loop back
// edge, 8 per call - flat guesses, not real instruction costs). HCTP's matches are tight EE loops (skinning, crowd,
// matrices) that cost a few real cycles per iteration: charged 32, EE time ran several times ahead of the work done,
// the game ran out of frame budget and the host slept ("pace" ~90%) - under 10 fps on an idle PC, audio starved.
static uint32_t guestCyclePercent() noexcept
{
    static const uint32_t percent = []
    {
        const char *value = std::getenv("VENDETTA_EE_CYCLE_PCT");
        const int parsed = (value && *value) ? std::atoi(value) : 100;
        const uint32_t clamped = static_cast<uint32_t>(std::clamp(parsed, 5, 400));
        if (clamped != 100u)
            std::fprintf(stderr, "[ee] guest cycle charge %u%% (VENDETTA_EE_CYCLE_PCT)\n", clamped);
        return clamped;
    }();
    return percent;
}

bool EeScheduler::checkpointDue(uint32_t cycles) noexcept
{
    static const uint32_t percent = guestCyclePercent();
    if (percent != 100u)
        cycles = std::max<uint32_t>(1u, static_cast<uint32_t>((uint64_t(cycles) * percent) / 100u));
    accountCycles(cycles);
    catchUpWithHostClock();

    if (m_checkpointPending.load(std::memory_order_acquire) ||
        m_stopRequested.load(std::memory_order_acquire))
    {
        return true;
    }

    const uint64_t nextEventCycle = m_nextDeadlineCycle.load(std::memory_order_acquire);
    if (nextEventCycle != 0u && m_eeCycle >= nextEventCycle)
    {
        m_checkpointPending.store(true, std::memory_order_release);
        return true;
    }

    if (m_eeCycle < m_sliceEndCycle)
    {
        return false;
    }

    const GuestThread *running = currentThread();
    if (running != nullptr && hasReadyAtOrAbovePriority(running->currentPriority))
    {
        m_rescheduleRequested = true;
        m_timeSliceExpired = true;
        return true;
    }

    renewTimeSlice();
    return false;
}

void EeScheduler::accountCycles(uint32_t cycles) noexcept
{
    const uint64_t elapsed = std::max<uint64_t>(1u, cycles);
    m_eeCycle += elapsed;
    m_pendingEeTimerInterrupts |= m_runtime.memory().advanceEeTimers(elapsed);
    // The IOP gets the EE's time in batches of 4096 EE cycles (512 IOP cycles, ~14 us): handing it over at every
    // checkpoint (8-32 cycles) was a call chain and a profiler mark millions of times a second for an IOP that runs
    // its own work in such batches anyway (IopEmulator::runEeCycles). Nothing the EE can see is that fine.
    constexpr uint64_t kIopHandOverCycles = 4096u;
    m_iopPendingCycles += elapsed;
    if (m_iopPendingCycles >= kIopHandOverCycles)
    {
        m_runtime.advanceIopEeCycles(m_iopPendingCycles);
        m_iopPendingCycles = 0u;
    }
    if (m_pendingEeTimerInterrupts != 0u)
    {
        m_checkpointPending.store(true, std::memory_order_release);
    }
}

bool EeScheduler::isExecutingGuest() const noexcept
{
    return m_guestExecuting.load(std::memory_order_acquire);
}

void EeScheduler::setupCurrentThread(uint32_t stack, uint32_t stackSize, uint32_t gp)
{
    assertExecutor();
    GuestThread *target = currentThread();
    if (!target)
    {
        return;
    }

    target->stack = stack;
    target->stackSize = stackSize;
    target->gp = gp;
    publishSnapshot();
}

int EeScheduler::createThread(const EeThreadCreateParams &params)
{
    assertExecutor();
    // EE kernel: priorities 0..127 (0 = highest). HCTP's CRI ADX threads ask for 0; refusing it left them uncreated
    // and the main thread asleep forever in ADXM_WaitVsync (first loading screen).
    if (params.priority < 0 || params.priority >= kPriorityCount)
    {
        std::fprintf(stderr, "[thread] CreateThread refused: priority %d%c", params.priority, 10);
        return KE_ILLEGAL_PRIORITY;
    }

    const int id = allocateThreadId();
    if (id == 0)
    {
        return KE_ERROR;
    }

    GuestThread thread{};
    thread.id = id;
    thread.entry = params.entry;
    thread.stack = params.stack;
    thread.stackSize = params.stackSize;
    thread.gp = params.gp;
    thread.attr = params.attr;
    thread.option = params.option;
    thread.initialPriority = params.priority;
    thread.currentPriority = params.priority;
    thread.status = EeThreadStatus::Dormant;
    m_threads.emplace(id, std::move(thread));
    publishSnapshot();
    return id;
}

int EeScheduler::deleteThread(int id, uint32_t &ownedStack)
{
    assertExecutor();
    ownedStack = 0;
    if (id <= kMainThreadId)
    {
        return KE_ILLEGAL_THID;
    }
    auto it = m_threads.find(id);
    if (it == m_threads.end())
    {
        return KE_UNKNOWN_THID;
    }
    if (it->second.status != EeThreadStatus::Dormant)
    {
        return KE_NOT_DORMANT;
    }
    if (it->second.ownsStack)
    {
        ownedStack = it->second.stack;
    }
    m_threads.erase(it);
    publishSnapshot();
    return KE_OK;
}

int EeScheduler::startThread(int id, uint32_t arg, const R5900Context &caller, bool interruptSafe)
{
    assertExecutor();
    GuestThread *target = thread(id);
    if (!target)
    {
        return KE_UNKNOWN_THID;
    }
    if (target->status != EeThreadStatus::Dormant)
    {
        return KE_NOT_DORMANT;
    }

    target->context = R5900Context{};
    target->context.pc = target->entry;
    target->arg = arg;
    target->suspendCount = 0;
    target->wakeupCount = 0;
    target->wait = {};
    SET_GPR_U32(&target->context, 4, arg);
    SET_GPR_U32(&target->context, 28, target->gp != 0u ? target->gp : getRegU32(&caller, 28));
    const uint32_t stackTop = target->stack != 0u
                                  ? (target->stack + target->stackSize) & ~0xFu
                                  : getRegU32(&caller, 29);
    SET_GPR_U32(&target->context, 29, stackTop);
    SET_GPR_U32(&target->context, 31, 0u);
    enqueueReady(*target);
    requestPreemptionIfHigher(*target, interruptSafe);
    publishSnapshot();
    return KE_OK;
}

[[noreturn]] void EeScheduler::exitCurrent(bool deleteThreadRecord)
{
    assertExecutor();
    GuestThread *exiting = currentThread();
    assert(exiting != nullptr);
    const int id = exiting->id;
    const uint32_t ownedStack = deleteThreadRecord && exiting->ownsStack ? exiting->stack : 0u;
    makeDormant(*exiting);
    m_currentThreadId = 0;
    if (deleteThreadRecord && id != kMainThreadId)
    {
        m_threads.erase(id);
    }
    if (ownedStack != 0u)
    {
        m_runtime.guestFree(ownedStack);
    }
    publishSnapshot();
    throw EeDispatcherTransfer{};
}

int EeScheduler::terminateThread(int id, uint32_t &ownedStack, bool interruptSafe)
{
    assertExecutor();
    ownedStack = 0;
    if (id == 0 || id == m_currentThreadId)
    {
        return KE_ILLEGAL_THID;
    }
    GuestThread *target = thread(id);
    if (!target)
    {
        return KE_UNKNOWN_THID;
    }
    if (target->status == EeThreadStatus::Dormant)
    {
        return KE_DORMANT;
    }
    if (target->ownsStack)
    {
        ownedStack = target->stack;
        target->ownsStack = false;
    }
    makeDormant(*target);
    (void)interruptSafe;
    publishSnapshot();
    return KE_OK;
}

int EeScheduler::suspendThread(int id, bool interruptSafe)
{
    assertExecutor();
    if (id == 0)
    {
        id = m_currentThreadId;
    }
    GuestThread *target = thread(id);
    if (!target)
    {
        return KE_UNKNOWN_THID;
    }
    if (target->status == EeThreadStatus::Dormant)
    {
        return KE_DORMANT;
    }

    ++target->suspendCount;
    switch (target->status)
    {
    case EeThreadStatus::Running:
        target->status = EeThreadStatus::Suspended;
        m_currentThreadId = 0;
        m_rescheduleRequested = true;
        break;
    case EeThreadStatus::Ready:
        removeReady(*target);
        target->status = EeThreadStatus::Suspended;
        break;
    case EeThreadStatus::Waiting:
        target->status = EeThreadStatus::WaitingSuspended;
        break;
    case EeThreadStatus::WaitingSuspended:
    case EeThreadStatus::Suspended:
        break;
    case EeThreadStatus::Dormant:
        break;
    }
    if (interruptSafe && m_insideInterrupt)
    {
        m_rescheduleRequested = true;
    }
    publishSnapshot();
    return KE_OK;
}

int EeScheduler::resumeThread(int id, bool interruptSafe)
{
    assertExecutor();
    if (id == 0)
    {
        return KE_ILLEGAL_THID;
    }
    GuestThread *target = thread(id);
    if (!target)
    {
        return KE_UNKNOWN_THID;
    }
    if (target->suspendCount == 0)
    {
        return KE_NOT_SUSPEND;
    }
    --target->suspendCount;
    if (target->suspendCount != 0)
    {
        return KE_OK;
    }
    if (target->status == EeThreadStatus::WaitingSuspended)
    {
        target->status = EeThreadStatus::Waiting;
    }
    else if (target->status == EeThreadStatus::Suspended)
    {
        enqueueReady(*target);
        requestPreemptionIfHigher(*target, interruptSafe);
    }
    publishSnapshot();
    return KE_OK;
}

void EeScheduler::sleepCurrent()
{
    assertExecutor();
    GuestThread *self = currentThread();
    assert(self != nullptr);
    if (self->wakeupCount != 0u)
    {
        --self->wakeupCount;
        setReturnS32(&self->activeContext(), KE_OK);
        return;
    }
    blockCurrent(EeWaitState{EeWaitReason::Sleep, std::monostate{}});
}

int EeScheduler::wakeupThread(int id, bool interruptSafe)
{
    assertExecutor();
    if (id == 0 || id == m_currentThreadId)
    {
        return KE_ILLEGAL_THID;
    }
    GuestThread *target = thread(id);
    if (!target)
    {
        return KE_UNKNOWN_THID;
    }
    if (target->status == EeThreadStatus::Dormant)
    {
        return KE_DORMANT;
    }
    if ((target->status == EeThreadStatus::Waiting || target->status == EeThreadStatus::WaitingSuspended) &&
        target->wait.reason == EeWaitReason::Sleep)
    {
        makeReady(*target, KE_OK, interruptSafe);
    }
    else
    {
        ++target->wakeupCount;
    }
    publishSnapshot();
    return KE_OK;
}

int EeScheduler::cancelWakeup(int id)
{
    assertExecutor();
    if (id == 0)
    {
        id = m_currentThreadId;
    }
    GuestThread *target = thread(id);
    if (!target)
    {
        return KE_UNKNOWN_THID;
    }
    const int old = static_cast<int>(target->wakeupCount);
    target->wakeupCount = 0;
    publishSnapshot();
    return old;
}

int EeScheduler::changePriority(int id, int priority, bool interruptSafe, int &oldPriority)
{
    assertExecutor();
    if (priority < 0 || priority >= kPriorityCount)
    {
        return KE_ILLEGAL_PRIORITY;
    }
    if (id == 0)
    {
        id = m_currentThreadId;
    }
    GuestThread *target = thread(id);
    if (!target)
    {
        return KE_UNKNOWN_THID;
    }
    oldPriority = target->currentPriority;
    if (target->status == EeThreadStatus::Ready)
    {
        removeReady(*target);
        target->currentPriority = priority;
        enqueueReady(*target);
        requestPreemptionIfHigher(*target, interruptSafe);
    }
    else
    {
        target->currentPriority = priority;
        if (target->status == EeThreadStatus::Running)
        {
            for (int p = 0; p < target->currentPriority; ++p)
            {
                if (!m_readyQueues[p].empty())
                {
                    m_rescheduleRequested = true;
                    break;
                }
            }
        }
    }
    publishSnapshot();
    return KE_OK;
}

int EeScheduler::rotateReadyQueue(int priority, bool interruptSafe)
{
    assertExecutor();
    if (priority == 0)
    {
        const GuestThread *self = currentThread();
        priority = self ? self->currentPriority : 0;
    }
    if (priority < 0 || priority >= kPriorityCount)
    {
        return KE_ILLEGAL_PRIORITY;
    }

    GuestThread *self = currentThread();
    if (self && self->currentPriority == priority)
    {
        enqueueReady(*self);
        m_currentThreadId = 0;
        m_rescheduleRequested = true;
    }
    else
    {
        auto &queue = m_readyQueues[priority];
        if (queue.size() > 1u)
        {
            const int head = queue.front();
            queue.pop_front();
            queue.push_back(head);
        }
    }
    (void)interruptSafe;
    publishSnapshot();
    return KE_OK;
}

int EeScheduler::releaseWait(int id, bool interruptSafe)
{
    assertExecutor();
    if (id == 0)
    {
        return KE_ILLEGAL_THID;
    }
    GuestThread *target = thread(id);
    if (!target)
    {
        return KE_UNKNOWN_THID;
    }
    if (target->status != EeThreadStatus::Waiting && target->status != EeThreadStatus::WaitingSuspended)
    {
        return KE_NOT_WAIT;
    }
    removeFromWaitObject(*target);
    makeReady(*target, KE_RELEASE_WAIT, interruptSafe);
    publishSnapshot();
    return KE_OK;
}

void EeScheduler::transferIfRequested(bool interruptSafe)
{
    assertExecutor();
    if (interruptSafe || m_insideInterrupt || !m_rescheduleRequested)
    {
        return;
    }
    if (m_currentThreadId != 0)
    {
        GuestThread *self = currentThread();
        assert(self != nullptr);
        enqueueReady(*self, true);
        m_currentThreadId = 0;
    }
    m_rescheduleRequested = false;
    m_timeSliceExpired = false;
    publishSnapshot();
    throw EeDispatcherTransfer{};
}

int EeScheduler::createSemaphore(int initCount, int maxCount, uint32_t attr, uint32_t option)
{
    assertExecutor();
    if (maxCount <= 0 || initCount < 0 || initCount > maxCount)
    {
        return KE_ERROR;
    }
    const int id = allocatePositiveId(m_nextSemaphoreId, m_semaphores);
    if (id == 0)
    {
        return KE_ERROR;
    }
    EeSemaphore semaphore{};
    semaphore.id = id;
    semaphore.count = initCount;
    semaphore.maxCount = maxCount;
    semaphore.initCount = initCount;
    semaphore.attr = attr;
    semaphore.option = option;
    m_semaphores.emplace(id, std::move(semaphore));
    publishSnapshot();
    return id;
}

int EeScheduler::deleteSemaphore(int id, bool interruptSafe)
{
    assertExecutor();
    auto it = m_semaphores.find(id);
    if (it == m_semaphores.end())
    {
        return KE_UNKNOWN_SEMID;
    }
    std::deque<int> waiters = std::move(it->second.waiters);
    m_semaphores.erase(it);
    for (const int threadId : waiters)
    {
        if (GuestThread *waiter = thread(threadId))
        {
            makeReady(*waiter, KE_WAIT_DELETE, interruptSafe);
        }
    }
    publishSnapshot();
    return id;
}

int EeScheduler::signalSemaphore(int id, bool interruptSafe)
{
    assertExecutor();
    EeSemaphore *object = semaphore(id);
    if (!object)
    {
        return KE_UNKNOWN_SEMID;
    }
    if (!object->waiters.empty())
    {
        const int waiterId = object->waiters.front();
        object->waiters.pop_front();
        GuestThread *waiter = thread(waiterId);
        assert(waiter != nullptr);
        makeReady(*waiter, id, interruptSafe);
        publishSnapshot();
        return id;
    }
    if (object->count == object->maxCount)
    {
        return KE_SEMA_OVF;
    }
    ++object->count;
    publishSnapshot();
    return id;
}

int EeScheduler::pollSemaphore(int id)
{
    assertExecutor();
    EeSemaphore *object = semaphore(id);
    if (!object)
    {
        return KE_UNKNOWN_SEMID;
    }
    if (object->count == 0)
    {
        return KE_SEMA_ZERO;
    }
    --object->count;
    publishSnapshot();
    return id;
}

void EeScheduler::waitSemaphore(int id)
{
    assertExecutor();
    EeSemaphore *object = semaphore(id);
    if (!object)
    {
        GuestThread *self = currentThread();
        assert(self != nullptr);
        setReturnS32(&self->activeContext(), KE_UNKNOWN_SEMID);
        return;
    }
    if (object->count != 0)
    {
        --object->count;
        GuestThread *self = currentThread();
        assert(self != nullptr);
        setReturnS32(&self->activeContext(), id);
        publishSnapshot();
        return;
    }
    GuestThread *self = currentThread();
    assert(self != nullptr);
    object->waiters.push_back(self->id);
    blockCurrent(EeWaitState{EeWaitReason::Semaphore, EeSemaphoreWait{id}});
}

int EeScheduler::createEventFlag(uint32_t initialBits, uint32_t attr, uint32_t option)
{
    assertExecutor();
    const int id = allocatePositiveId(m_nextEventFlagId, m_eventFlags);
    if (id == 0)
    {
        return KE_ERROR;
    }
    EeEventFlag flag{};
    flag.id = id;
    flag.attr = attr;
    flag.option = option;
    flag.initBits = initialBits;
    flag.bits = initialBits;
    m_eventFlags.emplace(id, std::move(flag));
    publishSnapshot();
    return id;
}

int EeScheduler::deleteEventFlag(int id, bool interruptSafe)
{
    assertExecutor();
    auto it = m_eventFlags.find(id);
    if (it == m_eventFlags.end())
    {
        return KE_UNKNOWN_EVFID;
    }
    std::deque<int> waiters = std::move(it->second.waiters);
    m_eventFlags.erase(it);
    for (const int threadId : waiters)
    {
        if (GuestThread *waiter = thread(threadId))
        {
            makeReady(*waiter, KE_WAIT_DELETE, interruptSafe);
        }
    }
    publishSnapshot();
    return KE_OK;
}

int EeScheduler::setEventFlag(int id, uint32_t bits, bool interruptSafe)
{
    assertExecutor();
    EeEventFlag *flag = eventFlag(id);
    if (!flag)
    {
        return KE_UNKNOWN_EVFID;
    }
    flag->bits |= bits;
    finishEventWaiters(*flag, interruptSafe);
    publishSnapshot();
    return KE_OK;
}

int EeScheduler::clearEventFlag(int id, uint32_t mask)
{
    assertExecutor();
    EeEventFlag *flag = eventFlag(id);
    if (!flag)
    {
        return KE_UNKNOWN_EVFID;
    }
    flag->bits &= mask;
    publishSnapshot();
    return KE_OK;
}

int EeScheduler::pollEventFlag(int id, uint32_t bits, uint32_t mode, uint32_t &observedBits)
{
    assertExecutor();
    EeEventFlag *flag = eventFlag(id);
    if (!flag)
    {
        return KE_UNKNOWN_EVFID;
    }
    if (!eventCondition(flag->bits, bits, mode))
    {
        return KE_EVF_COND;
    }
    observedBits = flag->bits;
    if ((mode & WEF_CLEAR_ALL) != 0u)
    {
        flag->bits = 0;
    }
    else if ((mode & WEF_CLEAR) != 0u)
    {
        flag->bits &= ~bits;
    }
    publishSnapshot();
    return KE_OK;
}

void EeScheduler::waitEventFlag(int id, uint32_t bits, uint32_t mode, uint32_t resultAddress)
{
    assertExecutor();
    EeEventFlag *flag = eventFlag(id);
    GuestThread *self = currentThread();
    assert(self != nullptr);
    if (!flag)
    {
        setReturnS32(&self->activeContext(), KE_UNKNOWN_EVFID);
        return;
    }
    if (eventCondition(flag->bits, bits, mode))
    {
        const uint32_t observed = flag->bits;
        writeGuestU32(resultAddress, observed);
        if ((mode & WEF_CLEAR_ALL) != 0u)
        {
            flag->bits = 0;
        }
        else if ((mode & WEF_CLEAR) != 0u)
        {
            flag->bits &= ~bits;
        }
        setReturnS32(&self->activeContext(), KE_OK);
        publishSnapshot();
        return;
    }
    flag->waiters.push_back(self->id);
    blockCurrent(EeWaitState{EeWaitReason::EventFlag,
                             EeEventFlagWait{id, bits, mode, resultAddress}});
}

int EeScheduler::setAlarm(uint16_t ticks,
                          uint32_t handler,
                          uint32_t argument,
                          uint32_t gp,
                          uint32_t sp,
                          bool argumentInA0)
{
    assertExecutor();
    if (handler == 0u || !m_runtime.hasFunction(handler))
    {
        return KE_ERROR;
    }
    const int id = allocatePositiveId(m_nextAlarmId, m_alarms);
    if (id == 0)
    {
        return KE_ERROR;
    }
    m_alarms.emplace(id, EeAlarm{id, ticks, handler, argument, gp, sp, argumentInA0});
    catchUpBeforeDeadline();
    const uint64_t tickCount = ticks == 0u ? 1u : static_cast<uint64_t>(ticks);
    scheduleEvent(m_eeCycle + tickCount * kAlarmTickCycles,
                  std::chrono::steady_clock::now() + std::chrono::microseconds(tickCount * kAlarmTickMicroseconds),
                  EeEvent{EeEventType::Alarm, static_cast<uint32_t>(id), 0});
    return id;
}

int EeScheduler::cancelAlarm(int id)
{
    assertExecutor();
    if (m_alarms.erase(id) == 0u)
    {
        return KE_ERROR;
    }
    {
        std::lock_guard lock(m_eventMutex);
        std::erase_if(m_deadlines, [id](const ScheduledEvent &scheduled)
                      { return scheduled.event.type == EeEventType::Alarm &&
                               scheduled.event.id == static_cast<uint32_t>(id); });
        updateNextDeadline();
    }
    return KE_OK;
}

void EeScheduler::queueInvocation(GuestInvocation invocation)
{
    assertExecutor();
    invocation.sequence = ++m_invocationSequence;
    m_pendingInvocations.push_back(std::move(invocation));
    m_checkpointPending.store(true, std::memory_order_release);
}

[[noreturn]] void EeScheduler::invokeCurrent(GuestInvocation invocation)
{
    assertExecutor();
    GuestThread *owner = currentThread();
    assert(owner != nullptr);
    if (getRegU32(&invocation.context, 29) == 0u)
    {
        SET_GPR_U32(&invocation.context, 29, invocationStackTop());
    }
    invocation.sequence = ++m_invocationSequence;
    owner->invocations.push_back(std::move(invocation));
    publishSnapshot();
    throw EeDispatcherTransfer{};
}

[[noreturn]] void EeScheduler::invokeCurrentSequence(std::vector<GuestInvocation> invocations)
{
    assertExecutor();
    GuestThread *owner = currentThread();
    assert(owner != nullptr);
    assert(!invocations.empty());
    for (auto it = invocations.rbegin(); it != invocations.rend(); ++it)
    {
        if (getRegU32(&it->context, 29) == 0u)
        {
            SET_GPR_U32(&it->context, 29, invocationStackTop());
        }
        it->sequence = ++m_invocationSequence;
        owner->invocations.push_back(std::move(*it));
    }
    publishSnapshot();
    throw EeDispatcherTransfer{};
}

bool EeScheduler::hasInvocation(GuestInvocationKind kind, uint64_t tag) const
{
    const GuestThread *owner = currentThread();
    if (!owner)
    {
        return false;
    }
    return std::any_of(owner->invocations.begin(), owner->invocations.end(),
                       [kind, tag](const GuestInvocation &invocation)
                       {
                           return invocation.kind == kind && invocation.tag == tag;
                       });
}

uint32_t EeScheduler::invocationStackTop()
{
    assertExecutor();
    const GuestThread *owner = currentThread();
    if (!owner)
    {
        throw std::logic_error("EE invocation stack requested without a current guest context");
    }
    const size_t depth = owner ? owner->invocations.size() : 0u;
    const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(owner->id)) << 32u) |
                         static_cast<uint32_t>(depth);
    const auto existing = m_invocationStackTops.find(key);
    if (existing != m_invocationStackTops.end())
    {
        return existing->second;
    }
    constexpr uint32_t kInvocationStackSize = 0x4000u; // in kernel RAM (0x48000-0x80000), 14 of them
    const uint32_t top = m_runtime.reserveAsyncCallbackStack(kInvocationStackSize, 16u);
    if (top == 0u)
    {
        throw std::runtime_error("EE invocation stack space exhausted");
    }
    m_invocationStackTops.emplace(key, top);
    std::cerr << "[ee] callback stack " << m_invocationStackTops.size() << " for thread " << owner->id << " depth " << depth
              << " at 0x" << std::hex << top << std::dec << std::endl;
    return top;
}

int EeScheduler::addIrqHandler(bool dmac,
                               uint32_t cause,
                               uint32_t handler,
                               bool append,
                               uint32_t argument,
                               uint32_t gp,
                               uint32_t sp)
{
    assertExecutor();
    auto &handlers = dmac ? m_dmacHandlers : m_intcHandlers;
    int &nextId = dmac ? m_nextDmacHandlerId : m_nextIntcHandlerId;
    const int id = allocatePositiveId(nextId, handlers);
    if (id == 0)
    {
        return KE_ERROR;
    }
    int &head = dmac ? m_dmacHeadOrder : m_intcHeadOrder;
    int &tail = dmac ? m_dmacTailOrder : m_intcTailOrder;
    handlers.emplace(id,
                     EeIrqHandler{id,
                                  cause,
                                  handler,
                                  argument,
                                  gp,
                                  sp,
                                  true,
                                  append ? ++tail : --head});
    return id;
}

int EeScheduler::removeIrqHandler(bool dmac, uint32_t cause, int id)
{
    assertExecutor();
    auto &handlers = dmac ? m_dmacHandlers : m_intcHandlers;
    auto it = handlers.find(id);
    if (it != handlers.end() && it->second.cause == cause)
    {
        handlers.erase(it);
    }
    return KE_OK;
}

int EeScheduler::setIrqHandlerEnabled(bool dmac, int id, bool enabled)
{
    assertExecutor();
    auto &handlers = dmac ? m_dmacHandlers : m_intcHandlers;
    auto it = handlers.find(id);
    if (it != handlers.end())
    {
        it->second.enabled = enabled;
    }
    return KE_OK;
}

int EeScheduler::setIrqCauseEnabled(bool dmac, uint32_t cause, bool enabled)
{
    assertExecutor();
    if (cause < 32u)
    {
        uint32_t &mask = dmac ? m_enabledDmacMask : m_enabledIntcMask;
        if (enabled)
        {
            mask |= 1u << cause;
        }
        else
        {
            mask &= ~(1u << cause);
        }
    }
    return KE_OK;
}

void EeScheduler::dispatchIrq(bool dmac, uint32_t cause)
{
    assertExecutor();
    const uint32_t mask = dmac ? m_enabledDmacMask : m_enabledIntcMask;
    if (cause < 32u && (mask & (1u << cause)) == 0u)
    {
        if (std::getenv("VENDETTA_IRQLOG"))
            std::cerr << "[irqlog] dropped " << (dmac ? "DMAC" : "INTC") << " cause " << cause << " (masked, mask=0x"
                      << std::hex << mask << std::dec << ")" << std::endl;
        return;
    }
    const auto &handlers = dmac ? m_dmacHandlers : m_intcHandlers;
    std::vector<EeIrqHandler> matching;
    for (const auto &[id, handler] : handlers)
    {
        (void)id;
        if (handler.enabled && handler.cause == cause && handler.handler != 0u &&
            m_runtime.hasFunction(handler.handler))
        {
            matching.push_back(handler);
        }
    }
    std::sort(matching.begin(), matching.end(), [](const EeIrqHandler &left, const EeIrqHandler &right)
              { return left.order < right.order; });
    for (const EeIrqHandler &handler : matching)
    {
        GuestInvocation invocation{};
        invocation.kind = GuestInvocationKind::Interrupt;
        invocation.context.pc = handler.handler;
        SET_GPR_U32(&invocation.context, 4, cause);
        SET_GPR_U32(&invocation.context, 5, handler.argument);
        SET_GPR_U32(&invocation.context, 28, handler.gp);
        SET_GPR_U32(&invocation.context, 29, 0u);
        SET_GPR_U32(&invocation.context, 31, 0u);
        queueInvocation(std::move(invocation));
    }
}

void EeScheduler::setVSyncFlag(uint32_t flagAddress, uint32_t tickAddress)
{
    assertExecutor();
    m_vsyncFlagAddress = flagAddress;
    m_vsyncTickAddress = tickAddress;
    writeGuestU32(flagAddress, 0u);
    if (tickAddress != 0u)
    {
        const uint32_t physical = tickAddress & 0x1FFFFFFFu;
        if (m_rdram && physical <= PS2_RAM_BACKING_SIZE - sizeof(uint64_t))
        {
            const uint64_t zero = 0u;
            std::memcpy(m_rdram + physical, &zero, sizeof(zero));
        }
    }
}

uint64_t EeScheduler::currentVSyncTick() const noexcept
{
    return m_vsyncTick;
}

uint32_t EeScheduler::setGsVSyncCallback(uint32_t callback, uint32_t gp, uint32_t sp)
{
    assertExecutor();
    (void)sp;
    const uint32_t previous = m_gsVSyncCallback;
    m_gsVSyncCallback = callback;
    m_gsVSyncCallbackGp = gp;
    m_gsVSyncCallbackSp = 0u;
    return previous;
}

[[noreturn]] void EeScheduler::waitVSync(uint64_t afterTick, int fixedResult, std::function<void(R5900Context &)> completion)
{
    blockCurrent(EeWaitState{
        EeWaitReason::VSync,
        EeVSyncWait{afterTick, fixedResult},
        std::move(completion)});
}

void EeScheduler::completeVSync(uint64_t tick)
{
    assertExecutor();
    std::vector<int> completed;
    for (const auto &[id, candidate] : m_threads)
    {
        if ((candidate.status == EeThreadStatus::Waiting || candidate.status == EeThreadStatus::WaitingSuspended) &&
            candidate.wait.reason == EeWaitReason::VSync &&
            std::get<EeVSyncWait>(candidate.wait.payload).afterTick < tick)
        {
            completed.push_back(id);
        }
    }
    // HCTP diag (VENDETTA_WAITLOG): every 300 vsyncs, every thread that is not running/ready and why
    static const bool dumpBlocked = std::getenv("VENDETTA_WAITLOG") != nullptr;
    if (dumpBlocked && tick % 300u == 0u)
    {
        for (auto &[id, candidate] : m_threads)
        {
            int waitId = 0;
            if (const auto *sema = std::get_if<EeSemaphoreWait>(&candidate.wait.payload))
                waitId = sema->id;
            else if (const auto *flag = std::get_if<EeEventFlagWait>(&candidate.wait.payload))
                waitId = flag->id;
            else if (const auto *external = std::get_if<EeExternalWait>(&candidate.wait.payload))
                waitId = static_cast<int>(external->type);
            std::fprintf(stderr, "[thdump] tick %llu thread %d status %d reason %d id %d ra 0x%x pc 0x%x prio %d inv %d%c",
                         static_cast<unsigned long long>(tick), id, static_cast<int>(candidate.status),
                         static_cast<int>(candidate.wait.reason), waitId,
                         getRegU32(&candidate.activeContext(), 31), candidate.activeContext().pc,
                         candidate.currentPriority, static_cast<int>(candidate.invocations.size()), 10);
        }
    }
    std::sort(completed.begin(), completed.end());
    for (const int id : completed)
    {
        GuestThread *waiter = thread(id);
        assert(waiter != nullptr);
        const EeVSyncWait wait = std::get<EeVSyncWait>(waiter->wait.payload);
        const int result = wait.fixedResult >= 0
                               ? wait.fixedResult
                               : static_cast<int>((tick - 1u) & 1u);
        makeReady(*waiter, result, false);
    }
    publishSnapshot();
}

void EeScheduler::completeExternalWait(uint32_t type, uint64_t token, int result)
{
    assertExecutor();
    std::vector<int> completed;
    for (const auto &[id, candidate] : m_threads)
    {
        if ((candidate.status != EeThreadStatus::Waiting && candidate.status != EeThreadStatus::WaitingSuspended) ||
            (candidate.wait.reason != EeWaitReason::External &&
             candidate.wait.reason != EeWaitReason::Mpeg))
        {
            continue;
        }
        const auto &external = std::get<EeExternalWait>(candidate.wait.payload);
        if (external.type == type && external.token == token)
        {
            completed.push_back(id);
        }
    }
    std::sort(completed.begin(), completed.end());
    for (const int id : completed)
    {
        GuestThread *waiter = thread(id);
        assert(waiter != nullptr);
        makeReady(*waiter, result, false);
    }
    publishSnapshot();
}

[[noreturn]] void EeScheduler::waitExternal(EeWaitReason reason,
                                            uint32_t type,
                                            uint64_t token,
                                            std::function<void(R5900Context &)> completion)
{
    EeWaitState wait{reason, EeExternalWait{type, token}, std::move(completion)};
    blockCurrent(std::move(wait));
}

GuestThread *EeScheduler::thread(int id)
{
    auto it = m_threads.find(id);
    return it == m_threads.end() ? nullptr : &it->second;
}

const GuestThread *EeScheduler::thread(int id) const
{
    auto it = m_threads.find(id);
    return it == m_threads.end() ? nullptr : &it->second;
}

EeSemaphore *EeScheduler::semaphore(int id)
{
    auto it = m_semaphores.find(id);
    return it == m_semaphores.end() ? nullptr : &it->second;
}

const EeSemaphore *EeScheduler::semaphore(int id) const
{
    auto it = m_semaphores.find(id);
    return it == m_semaphores.end() ? nullptr : &it->second;
}

EeEventFlag *EeScheduler::eventFlag(int id)
{
    auto it = m_eventFlags.find(id);
    return it == m_eventFlags.end() ? nullptr : &it->second;
}

const EeEventFlag *EeScheduler::eventFlag(int id) const
{
    auto it = m_eventFlags.find(id);
    return it == m_eventFlags.end() ? nullptr : &it->second;
}

GuestThread *EeScheduler::currentThread()
{
    return thread(m_currentThreadId);
}

const GuestThread *EeScheduler::currentThread() const
{
    return thread(m_currentThreadId);
}

int EeScheduler::currentThreadId() const noexcept
{
    return m_currentThreadId;
}

R5900Context *EeScheduler::currentContext()
{
    GuestThread *self = currentThread();
    return self ? &self->activeContext() : nullptr;
}

uint8_t *EeScheduler::rdram() const noexcept
{
    return m_rdram;
}

void EeScheduler::bindMainContextForSyscall(R5900Context &ctx, uint8_t *rdram)
{
    if (m_executorThread == std::thread::id{})
    {
        reset(rdram, ctx);
        GuestThread *main = selectReady();
        assert(main != nullptr);
        makeRunning(*main);
        return;
    }
    assertExecutor();
    m_rdram = rdram;
    if (m_currentThreadId == 0)
    {
        GuestThread *main = thread(kMainThreadId);
        assert(main != nullptr);
        assert(main->status == EeThreadStatus::Ready);
        removeReady(*main);
        makeRunning(*main);
    }
}

EeKernelSnapshot EeScheduler::snapshot() const
{
    std::lock_guard lock(m_snapshotMutex);
    return m_snapshot;
}

void EeScheduler::publishSnapshot()
{
    EeKernelSnapshot next{};
    next.sequence = ++m_snapshotSequence;
    next.eeCycle = m_eeCycle;
    next.sliceEndCycle = m_sliceEndCycle;
    next.nextEventCycle = m_nextDeadlineCycle.load(std::memory_order_acquire);
    next.runningThreadId = m_currentThreadId;
    next.threads.reserve(m_threads.size());
    for (const auto &[id, item] : m_threads)
    {
        if (id < 0)
        {
            continue;
        }
        EeThreadSnapshot snapshot{};
        snapshot.id = id;
        const R5900Context &context = item.activeContext();
        snapshot.pc = context.pc;
        snapshot.ra = getRegU32(&context, 31);
        snapshot.sp = getRegU32(&context, 29);
        snapshot.contextGp = getRegU32(&context, 28);
        snapshot.entry = item.entry;
        snapshot.stack = item.stack;
        snapshot.stackSize = item.stackSize;
        snapshot.gp = item.gp;
        snapshot.initialPriority = item.initialPriority;
        snapshot.currentPriority = item.currentPriority;
        snapshot.status = item.status;
        snapshot.waitReason = item.wait.reason;
        snapshot.waitId = waitObjectId(item.wait);
        snapshot.suspendCount = item.suspendCount;
        snapshot.wakeupCount = item.wakeupCount;
        snapshot.invocationDepth = static_cast<uint32_t>(item.invocations.size());
        next.threads.push_back(snapshot);
    }
    std::sort(next.threads.begin(), next.threads.end(), [](const auto &left, const auto &right)
              { return left.id < right.id; });
    next.semaphores.reserve(m_semaphores.size());
    for (const auto &[id, item] : m_semaphores)
    {
        next.semaphores.push_back(EeSemaphoreSnapshot{id,
                                                      item.count,
                                                      item.maxCount,
                                                      static_cast<uint32_t>(item.waiters.size())});
    }
    std::sort(next.semaphores.begin(), next.semaphores.end(), [](const auto &left, const auto &right)
              { return left.id < right.id; });
    next.eventFlags.reserve(m_eventFlags.size());
    for (const auto &[id, item] : m_eventFlags)
    {
        next.eventFlags.push_back(EeEventFlagSnapshot{id,
                                                      item.bits,
                                                      item.initBits,
                                                      item.attr,
                                                      static_cast<uint32_t>(item.waiters.size())});
    }
    std::sort(next.eventFlags.begin(), next.eventFlags.end(), [](const auto &left, const auto &right)
              { return left.id < right.id; });
    {
        std::lock_guard lock(m_snapshotMutex);
        m_snapshot = std::move(next);
    }
}

void EeScheduler::assertExecutor() const
{
    assert(m_executorThread == std::this_thread::get_id());
}

int EeScheduler::allocateThreadId()
{
    for (int attempts = 0; attempts <= kLastThreadId - kFirstThreadId; ++attempts)
    {
        const int candidate = m_nextThreadId;
        m_nextThreadId = candidate == kLastThreadId ? kFirstThreadId : candidate + 1;
        if (!m_threads.contains(candidate))
        {
            return candidate;
        }
    }
    return 0;
}

GuestThread &EeScheduler::acquireInvocationThread()
{
    for (auto &[id, candidate] : m_threads)
    {
        if (id < 0 && candidate.status == EeThreadStatus::Dormant && candidate.invocations.empty())
        {
            return candidate;
        }
    }

    GuestThread dispatcher{};
    dispatcher.id = m_nextInvocationThreadId--;
    dispatcher.initialPriority = 0;
    dispatcher.currentPriority = 0;
    dispatcher.status = EeThreadStatus::Dormant;
    return m_threads.emplace(dispatcher.id, std::move(dispatcher)).first->second;
}

void EeScheduler::enqueueReady(GuestThread &item, bool front)
{
    assert(item.currentPriority >= 0 && item.currentPriority < kPriorityCount);
    item.status = EeThreadStatus::Ready;
    auto &queue = m_readyQueues[item.currentPriority];
    if (front)
    {
        queue.push_front(item.id);
    }
    else
    {
        queue.push_back(item.id);
    }
}

void EeScheduler::removeReady(GuestThread &item)
{
    if (item.status != EeThreadStatus::Ready)
    {
        return;
    }
    auto &queue = m_readyQueues[item.currentPriority];
    auto it = std::find(queue.begin(), queue.end(), item.id);
    assert(it != queue.end());
    queue.erase(it);
}

GuestThread *EeScheduler::selectReady()
{
    for (auto &queue : m_readyQueues)
    {
        if (queue.empty())
        {
            continue;
        }
        const int id = queue.front();
        queue.pop_front();
        GuestThread *selected = thread(id);
        assert(selected != nullptr);
        assert(selected->status == EeThreadStatus::Ready);
        return selected;
    }
    return nullptr;
}

void EeScheduler::makeRunning(GuestThread &item)
{
    assert(m_currentThreadId == 0);
    assert(item.status == EeThreadStatus::Ready);
    item.status = EeThreadStatus::Running;
    m_currentThreadId = item.id;
    renewTimeSlice();
}

void EeScheduler::makeDormant(GuestThread &item)
{
    removeReady(item);
    removeFromWaitObject(item);
    item.status = EeThreadStatus::Dormant;
    item.wait = {};
    item.resumeCompletion = {};
    item.suspendCount = 0;
    item.wakeupCount = 0;
    item.invocations.clear();
}

void EeScheduler::removeFromWaitObject(GuestThread &item)
{
    const int id = item.id;
    if (item.wait.reason == EeWaitReason::Semaphore)
    {
        const int objectId = std::get<EeSemaphoreWait>(item.wait.payload).id;
        if (EeSemaphore *object = semaphore(objectId))
        {
            auto it = std::find(object->waiters.begin(), object->waiters.end(), id);
            if (it != object->waiters.end())
            {
                object->waiters.erase(it);
            }
        }
    }
    else if (item.wait.reason == EeWaitReason::EventFlag)
    {
        const int objectId = std::get<EeEventFlagWait>(item.wait.payload).id;
        if (EeEventFlag *object = eventFlag(objectId))
        {
            auto it = std::find(object->waiters.begin(), object->waiters.end(), id);
            if (it != object->waiters.end())
            {
                object->waiters.erase(it);
            }
        }
    }
    item.wait = {};
}

namespace
{
    // VENDETTA_WAITLOG=<ms> (1 = 50): every guest wait that lasts longer (host time) — what it waited on, from where,
    // and what woke it (finding what a pause on screen is waiting for). Executor thread only.
    struct BlockedSince
    {
        std::chrono::steady_clock::time_point at;
        EeWaitReason reason;
        int id;
        uint32_t ra;
    };
    long long waitLogThresholdMs()
    {
        static const long long threshold = []() -> long long
        {
            const char *text = std::getenv("VENDETTA_WAITLOG");
            if (!text)
                return -1;
            const long long value = std::atoll(text);
            return value > 1 ? value : 50;
        }();
        return threshold;
    }
    bool waitLogEnabled()
    {
        return waitLogThresholdMs() >= 0;
    }
    std::unordered_map<int, BlockedSince> &blockedSince()
    {
        static std::unordered_map<int, BlockedSince> map;
        return map;
    }
    const char *waitReasonName(EeWaitReason reason)
    {
        switch (reason)
        {
        case EeWaitReason::Sleep: return "sleep";
        case EeWaitReason::Semaphore: return "semaphore";
        case EeWaitReason::EventFlag: return "event flag";
        case EeWaitReason::VSync: return "vsync";
        case EeWaitReason::External: return "external";
        case EeWaitReason::Mpeg: return "mpeg";
        default: return "none";
        }
    }
}

void EeScheduler::blockCurrent(EeWaitState wait)
{
    GuestThread *self = currentThread();
    assert(self != nullptr);
    if (waitLogEnabled())
    {
        int id = 0;
        if (const auto *sema = std::get_if<EeSemaphoreWait>(&wait.payload))
            id = sema->id;
        else if (const auto *flag = std::get_if<EeEventFlagWait>(&wait.payload))
            id = flag->id;
        else if (const auto *external = std::get_if<EeExternalWait>(&wait.payload))
            id = static_cast<int>(external->type);
        blockedSince()[self->id] = BlockedSince{std::chrono::steady_clock::now(), wait.reason, id,
                                                getRegU32(&self->activeContext(), 31)};
        {
            // HCTP diag: each new (thread, reason, id, ra) the first time - a wait that never ends is the last new line
            static std::set<std::tuple<int, int, int, uint32_t>> seen;
            const uint32_t ra = getRegU32(&self->activeContext(), 31);
            if (seen.insert({self->id, static_cast<int>(wait.reason), id, ra}).second)
                std::fprintf(stderr, "[waitnew] thread %d blocks on %s %d (ra 0x%x)%c", self->id, waitReasonName(wait.reason), id, ra, 10);
        }
    }
    self->wait = std::move(wait);
    self->status = self->suspendCount == 0 ? EeThreadStatus::Waiting : EeThreadStatus::WaitingSuspended;
    m_currentThreadId = 0;
    publishSnapshot();
    throw EeDispatcherTransfer{};
}

void EeScheduler::makeReady(GuestThread &item, int result, bool interruptSafe)
{
    if (waitLogEnabled())
    {
        auto &map = blockedSince();
        if (auto found = map.find(item.id); found != map.end())
        {
            const auto waited = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - found->second.at).count();
            if (waited > waitLogThresholdMs())
            {
                const GuestThread *waker = currentThread();
                std::fprintf(stderr, "[waitlog] thread %d waited %lld ms on %s %d (ra 0x%x); woken by %s%d ra 0x%x tick %llu%c",
                             item.id, static_cast<long long>(waited), waitReasonName(found->second.reason), found->second.id,
                             found->second.ra, m_insideInterrupt ? "interrupt / thread " : "thread ",
                             waker ? waker->id : 0, waker ? getRegU32(&const_cast<GuestThread *>(waker)->activeContext(), 31) : 0u,
                             static_cast<unsigned long long>(currentVSyncTick()), 10);
            }
            map.erase(found);
        }
    }
    auto completion = std::move(item.wait.completion);
    item.wait = {};
    setReturnS32(&item.activeContext(), result);
    item.resumeCompletion = std::move(completion);
    if (item.suspendCount != 0)
    {
        item.status = EeThreadStatus::Suspended;
        return;
    }
    enqueueReady(item);
    requestPreemptionIfHigher(item, interruptSafe);
}

void EeScheduler::requestPreemptionIfHigher(const GuestThread &readyThread, bool interruptSafe)
{
    const GuestThread *running = currentThread();
    if (!running || readyThread.currentPriority >= running->currentPriority)
    {
        return;
    }
    m_rescheduleRequested = true;
    if (interruptSafe || m_insideInterrupt)
    {
        m_checkpointPending.store(true, std::memory_order_release);
    }
}

void EeScheduler::applyPendingPreemption()
{
    if (!m_rescheduleRequested)
    {
        return;
    }
    if (m_currentThreadId == 0)
    {
        m_rescheduleRequested = false;
        m_timeSliceExpired = false;
        return;
    }
    GuestThread *self = currentThread();
    assert(self != nullptr);
    enqueueReady(*self, !m_timeSliceExpired);
    m_currentThreadId = 0;
    m_rescheduleRequested = false;
    m_timeSliceExpired = false;
}

void EeScheduler::processPendingEvents()
{
    assertExecutor();
    processDueDeadlines();
    const uint32_t timerInterrupts = m_pendingEeTimerInterrupts;
    m_pendingEeTimerInterrupts = 0u;
    for (uint32_t timer = 0u; timer < 4u; ++timer)
    {
        if ((timerInterrupts & (1u << timer)) != 0u)
        {
            dispatchIrq(false, 9u + timer);
        }
    }
    std::deque<EeEvent> pending;
    {
        std::lock_guard lock(m_eventMutex);
        pending.swap(m_events);
    }
    for (const EeEvent &event : pending)
    {
        processEvent(event);
    }

    {
        std::lock_guard lock(m_eventMutex);
        const uint64_t nextEventCycle = m_nextDeadlineCycle.load(std::memory_order_acquire);
        const bool cycleEventDue = nextEventCycle != 0u && m_eeCycle >= nextEventCycle;
        const bool pendingWork = !m_events.empty() || cycleEventDue || m_stopRequested.load(std::memory_order_acquire);
        m_checkpointPending.store(pendingWork, std::memory_order_release);
    }
    applyPendingPreemption();
}

void EeScheduler::processDueDeadlines()
{
    for (;;)
    {
        std::vector<ScheduledEvent> due;
        std::chrono::steady_clock::time_point pacingDeadline{};
        {
            std::unique_lock lock(m_eventMutex);
            const auto now = std::chrono::steady_clock::now();
            for (const ScheduledEvent &item : m_deadlines)
            {
                if (item.deadlineCycle <= m_eeCycle &&
                    (pacingDeadline == std::chrono::steady_clock::time_point{} ||
                     item.hostDeadline < pacingDeadline))
                {
                    pacingDeadline = item.hostDeadline;
                }
            }

            if (pacingDeadline == std::chrono::steady_clock::time_point{})
            {
                updateNextDeadline();
                return;
            }

            if (now < pacingDeadline)
            {
                if (sleepLogEnabled() && sleepDiary().entries.size() < 64u)
                {
                    char head[64];
                    std::snprintf(head, sizeof(head), "pace hold %.1f ms:", std::chrono::duration<double, std::milli>(pacingDeadline - now).count());
                    sleepDiary().entries.push_back(head + describeDeadlines(m_deadlines, m_eeCycle, now));
                }
                ps2_host_profile::Scope pace(ps2_host_profile::Pace);
                waitUntilPrecise(m_eventCv, lock, pacingDeadline, [this]()
                                 { return !m_events.empty() ||
                                          m_stopRequested.load(std::memory_order_acquire); });
                if (!m_events.empty() || m_stopRequested.load(std::memory_order_acquire))
                {
                    updateNextDeadline();
                    return;
                }
            }

            const auto pacedNow = std::chrono::steady_clock::now();
            auto firstFuture = std::partition(m_deadlines.begin(), m_deadlines.end(),
                                              [this, pacedNow](const ScheduledEvent &item)
                                              { return item.deadlineCycle <= m_eeCycle &&
                                                       item.hostDeadline <= pacedNow; });
            due.insert(due.end(),
                       std::make_move_iterator(m_deadlines.begin()),
                       std::make_move_iterator(firstFuture));
            m_deadlines.erase(m_deadlines.begin(), firstFuture);
            updateNextDeadline();
        }

        std::sort(due.begin(), due.end(), [](const ScheduledEvent &left, const ScheduledEvent &right)
                  {
                      if (left.deadlineCycle != right.deadlineCycle)
                      {
                          return left.deadlineCycle < right.deadlineCycle;
                      }
                      if (left.event.type != right.event.type)
                      {
                          return left.event.type < right.event.type;
                      }
                      if (left.event.id != right.event.id)
                      {
                          return left.event.id < right.event.id;
                      }
                      return left.sequence < right.sequence; });

        if (due.empty())
        {
            return;
        }

        for (ScheduledEvent &scheduled : due)
        {
            if (scheduled.event.type == EeEventType::VBlankStart)
            {
                // A VBlank comes every 16.7 ms of real time whatever the game is doing. When the runtime has fallen
                // far behind (a stall of over 100 ms: a long load step, the drawing thread busy with a big upload),
                // the missed VBlanks must not be delivered afterwards at full speed: the game ran fast-forward until
                // the schedule caught up with the clock (Def Jam Vendetta: about a second at up to double speed after
                // loading a match, and again as the match starts). The schedule restarts from now instead, as the EE
                // clock does (catchUpWithHostClock); a smaller lag still catches up, so the rate stays 60 Hz.
                // VENDETTA_VBLANK_CATCHUP=1 restores the old catch-up.
                if (sleepLogEnabled())
                {
                    SleepDiary &diary = sleepDiary();
                    const auto at = std::chrono::steady_clock::now();
                    if (diary.lastVBlank != std::chrono::steady_clock::time_point{} && at - diary.lastVBlank > std::chrono::milliseconds(50))
                    {
                        std::string text;
                        for (const std::string &entry : diary.entries)
                            text += "\n[sleep]   " + entry;
                        std::fprintf(stderr, "[sleep] %.1f ms between vblanks %llu and %llu (%zu sleeps):%s\n",
                                     std::chrono::duration<double, std::milli>(at - diary.lastVBlank).count(),
                                     static_cast<unsigned long long>(m_vsyncTick), static_cast<unsigned long long>(m_vsyncTick + 1u),
                                     diary.entries.size(), text.c_str());
                    }
                    diary.lastVBlank = at;
                    diary.entries.clear();
                }
                static const bool catchUp = std::getenv("VENDETTA_VBLANK_CATCHUP") != nullptr;
                auto hostDeadline = scheduled.hostDeadline;
                const auto now = std::chrono::steady_clock::now();
                // HCTP_VBLANK_RESYNC_MS (default 100): the lag that restarts the schedule. The restart also re-anchors
                // the host clock (catchUpBeforeDeadline's origin): without it HCTP's matches froze 5-9 s after each
                // restart (every ~6 s at 30 fps). HCTP_VBLANK_NOREANCHOR=1 = old behaviour.
                static const auto resync = std::chrono::milliseconds(std::getenv("HCTP_VBLANK_RESYNC_MS") ? std::atoi(std::getenv("HCTP_VBLANK_RESYNC_MS")) : 100);
                static const bool reanchor = std::getenv("HCTP_VBLANK_NOREANCHOR") == nullptr;
                if (!catchUp && now - hostDeadline > resync)
                {
                    if (reanchor)
                    {
                        m_hostClockOrigin = now;
                        m_hostClockOriginCycle = m_eeCycle;
                    }
                    static const bool log = std::getenv("VENDETTA_HOSTPROF") != nullptr;
                    if (log)
                    {
                        std::fprintf(stderr, "[vblank] %lld ms behind real time at tick %llu: the schedule restarts from now\n",
                                     static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(now - hostDeadline).count()),
                                     static_cast<unsigned long long>(m_vsyncTick));
                    }
                    hostDeadline = now;
                }
                scheduleEvent(scheduled.deadlineCycle + kVBlankDurationCycles,
                              hostDeadline + kVBlankDuration,
                              EeEvent{EeEventType::VBlankEnd, 0, m_vsyncTick + 1u});
                scheduleEvent(scheduled.deadlineCycle + kVBlankPeriodCycles,
                              hostDeadline + kVBlankPeriod,
                              EeEvent{EeEventType::VBlankStart, 0, 0});
            }
            processEvent(scheduled.event);
        }
    }
}

void EeScheduler::processEvent(const EeEvent &event)
{
    switch (event.type)
    {
    case EeEventType::Stop:
        requestStop();
        break;
    case EeEventType::VBlankStart:
    {
        ++m_vsyncTick;
        m_runtime.memory().gs().vsyncTick.store(m_vsyncTick, std::memory_order_release);
        if ((m_vsyncTick & 1u) != 0u)
        {
            m_runtime.memory().gs().csr.fetch_or(0x2000ull, std::memory_order_acq_rel);
        }
        else
        {
            m_runtime.memory().gs().csr.fetch_and(~0x2000ull, std::memory_order_acq_rel);
        }
        writeGuestU32(m_vsyncFlagAddress, 1u);
        if (m_vsyncTickAddress != 0u)
        {
            const uint32_t physical = m_vsyncTickAddress & 0x1FFFFFFFu;
            if (m_rdram && physical <= PS2_RAM_BACKING_SIZE - sizeof(uint64_t))
            {
                std::memcpy(m_rdram + physical, &m_vsyncTick, sizeof(m_vsyncTick));
            }
        }
        m_vsyncFlagAddress = 0u;
        m_vsyncTickAddress = 0u;
        completeVSync(m_vsyncTick);
        const uint64_t firstHandlerSeq = m_invocationSequence + 1u;
        if (m_gsVSyncCallback != 0u && m_runtime.hasFunction(m_gsVSyncCallback))
        {
            GuestInvocation invocation{};
            invocation.kind = GuestInvocationKind::GsCallback;
            invocation.context.pc = m_gsVSyncCallback;
            SET_GPR_U32(&invocation.context, 4, static_cast<uint32_t>(m_vsyncTick));
            SET_GPR_U32(&invocation.context, 28, m_gsVSyncCallbackGp);
            SET_GPR_U32(&invocation.context, 29, m_gsVSyncCallbackSp);
            SET_GPR_U32(&invocation.context, 31, 0u);
            queueInvocation(std::move(invocation));
        }
        dispatchIrq(false, 2u);
        for (uint64_t seq = firstHandlerSeq; seq <= m_invocationSequence; ++seq)
            m_vsyncHandlerSeqs.push_back(seq);
        if (m_presentDue)
            ++m_presentDueVblanks;
        else
        {
            m_presentDue = true;
            m_presentTick = m_vsyncTick;
            m_presentDueVblanks = 1u;
        }
        break;
    }
    case EeEventType::ExternalWake:
        completeExternalWait(event.id, event.value, KE_OK);
        break;
    case EeEventType::VBlankEnd:
        dispatchIrq(false, 3u);
        break;
    case EeEventType::Dmac:
        break;
    case EeEventType::Alarm:
    {
        auto it = m_alarms.find(static_cast<int>(event.id));
        if (it == m_alarms.end())
        {
            break;
        }
        const EeAlarm alarm = it->second;
        m_alarms.erase(it);
        GuestInvocation invocation{};
        invocation.kind = GuestInvocationKind::Alarm;
        invocation.context.pc = alarm.handler;
        SET_GPR_U32(&invocation.context, 4, alarm.argumentInA0 ? alarm.argument : static_cast<uint32_t>(alarm.id));
        SET_GPR_U32(&invocation.context, 5, alarm.argumentInA0 ? 0u : static_cast<uint32_t>(alarm.ticks));
        SET_GPR_U32(&invocation.context, 6, alarm.argumentInA0 ? 0u : alarm.argument);
        SET_GPR_U32(&invocation.context, 28, alarm.gp);
        SET_GPR_U32(&invocation.context, 29, 0u);
        SET_GPR_U32(&invocation.context, 31, 0u);
        queueInvocation(std::move(invocation));
        break;
    }
    }
}

void EeScheduler::finishEventWaiters(EeEventFlag &flag, bool interruptSafe)
{
    for (auto it = flag.waiters.begin(); it != flag.waiters.end();)
    {
        GuestThread *waiter = thread(*it);
        assert(waiter != nullptr);
        const EeEventFlagWait wait = std::get<EeEventFlagWait>(waiter->wait.payload);
        if (!eventCondition(flag.bits, wait.bits, wait.mode))
        {
            ++it;
            continue;
        }
        const uint32_t observed = flag.bits;
        writeGuestU32(wait.resultAddress, observed);
        if ((wait.mode & WEF_CLEAR_ALL) != 0u)
        {
            flag.bits = 0;
        }
        else if ((wait.mode & WEF_CLEAR) != 0u)
        {
            flag.bits &= ~wait.bits;
        }
        it = flag.waiters.erase(it);
        makeReady(*waiter, KE_OK, interruptSafe);
    }
}

bool EeScheduler::eventCondition(uint32_t current, uint32_t requested, uint32_t mode)
{
    return (mode & WEF_OR) != 0u ? (current & requested) != 0u
                                 : (current & requested) == requested;
}

int EeScheduler::waitObjectId(const EeWaitState &wait)
{
    switch (wait.reason)
    {
    case EeWaitReason::Semaphore:
        return std::get<EeSemaphoreWait>(wait.payload).id;
    case EeWaitReason::EventFlag:
        return std::get<EeEventFlagWait>(wait.payload).id;
    default:
        return 0;
    }
}

void EeScheduler::writeGuestU32(uint32_t address, uint32_t value)
{
    if (address == 0u)
    {
        return;
    }
    const uint32_t physical = address & 0x1FFFFFFFu;
    if (!m_rdram || physical > PS2_RAM_BACKING_SIZE - sizeof(value))
    {
        return;
    }
    std::memcpy(m_rdram + physical, &value, sizeof(value));
}

void EeScheduler::maybePresent()
{
    if (!m_presentDue)
        return;
    const bool forced = m_presentDueVblanks >= 2u;
    if (m_presentMode != PresentMode::VBlank && !m_vsyncHandlerSeqs.empty() && !forced)
        return;
    m_presentDue = false;
    if (forced && !m_vsyncHandlerSeqs.empty())
    {
        ps2_host_profile::count(ps2_host_profile::GsPresentForced);
        m_vsyncHandlerSeqs.clear();
    }
    // the display registers as they are now; the chain presents in order with the drawing it was given
    m_runtime.chain().present(GS::captureDisplayRegs(m_runtime.memory().gs()), m_presentTick);
}

void EeScheduler::waitForEvent()
{
    {
        // The EE is idle: if the chain worker is more than 2 vblanks behind, then if the GS thread is more than 2 flips
        // behind, let them catch up first (no EE time is lost: an overdue host deadline below returns at once and EE
        // time jumps to the event).
        static const uint32_t chainLag = []
        {
            const char *lag = std::getenv("VENDETTA_CHAIN_LAG");
            return lag && *lag ? static_cast<uint32_t>(std::strtoul(lag, nullptr, 10)) : 2u;
        }();
        m_runtime.chain().waitBacklog(chainLag, m_stopRequested);
        ps2_host_profile::Scope throttle(ps2_host_profile::GsWait);
        m_runtime.gs().waitFlipBacklog(2u, m_stopRequested);
    }
    ps2_host_profile::Scope profile(ps2_host_profile::Wait);
    std::unique_lock lock(m_eventMutex);
    if (!m_events.empty() || m_stopRequested.load(std::memory_order_acquire))
    {
        return;
    }
    const uint64_t timerCycles = m_runtime.memory().cyclesUntilNextEeTimerInterrupt();
    const bool hasTimerDeadline = timerCycles != std::numeric_limits<uint64_t>::max();
    if (m_deadlines.empty() && !hasTimerDeadline)
    {
        m_eventCv.wait(lock, [this]()
                       { return !m_events.empty() || m_stopRequested.load(std::memory_order_acquire); });
        return;
    }

    uint64_t deadlineCycle = 0u;
    auto hostDeadline = std::chrono::steady_clock::time_point::max();
    if (!m_deadlines.empty())
    {
        const auto next = std::min_element(m_deadlines.begin(), m_deadlines.end(),
                                           [](const ScheduledEvent &left, const ScheduledEvent &right)
                                           {
                                               if (left.deadlineCycle != right.deadlineCycle)
                                               {
                                                   return left.deadlineCycle < right.deadlineCycle;
                                               }
                                               return left.sequence < right.sequence;
                                           });
        deadlineCycle = next->deadlineCycle;
        hostDeadline = next->hostDeadline;
    }
    if (hasTimerDeadline)
    {
        // Advance EE time only to whichever event comes first in EE time. Choosing by host deadline let an
        // overdue VBlank (its host deadline already passed while the runtime runs behind real time) jump EE time
        // past a timer compare: the timer interrupt then arrived over a period late, and a handler that re-arms
        // with "target += period - count" (Def Jam Vendetta's 100 Hz tick) wrapped its 16-bit target and fell to
        // ~9 Hz. Sleep until the earlier host deadline, so a runtime that is behind does not wait.
        const uint64_t timerDeadlineCycle = m_eeCycle + timerCycles;
        const auto timerHostDeadline = std::chrono::steady_clock::now() + eeCyclesToHostDuration(timerCycles);
        if (m_deadlines.empty() || timerDeadlineCycle < deadlineCycle)
        {
            deadlineCycle = timerDeadlineCycle;
        }
        hostDeadline = std::min(hostDeadline, timerHostDeadline);
    }

    if (sleepLogEnabled())
    {
        const auto now = std::chrono::steady_clock::now();
        if (sleepDiary().entries.size() < 64u)
        {
            char head[96];
            std::snprintf(head, sizeof(head), "idle %.1f ms (to EE %+.1f ms; timer in EE %+.1f ms):",
                          hostDeadline == std::chrono::steady_clock::time_point::max()
                              ? -1.0
                              : std::chrono::duration<double, std::milli>(hostDeadline - now).count(),
                          static_cast<double>(static_cast<int64_t>(deadlineCycle - m_eeCycle)) / static_cast<double>(kEeClockHz / 1000u),
                          hasTimerDeadline ? static_cast<double>(timerCycles) / static_cast<double>(kEeClockHz / 1000u) : -1.0);
            sleepDiary().entries.push_back(head + describeDeadlines(m_deadlines, m_eeCycle, now));
        }
    }
    const auto woken = [this]()
    { return !m_events.empty() || m_stopRequested.load(std::memory_order_acquire); };
    const bool signaled = waitUntilPrecise(m_eventCv, lock, hostDeadline, woken);
    if (!signaled)
    {
        const uint64_t elapsed = deadlineCycle > m_eeCycle ? deadlineCycle - m_eeCycle : 0u;
        lock.unlock();
        uint64_t remaining = elapsed;
        while (remaining > 0u)
        {
            const uint32_t step = static_cast<uint32_t>(std::min<uint64_t>(remaining, std::numeric_limits<uint32_t>::max()));
            accountCycles(step);
            remaining -= step;
        }
        m_checkpointPending.store(true, std::memory_order_release);
    }
}

void EeScheduler::scheduleEvent(uint64_t deadlineCycle,
                                std::chrono::steady_clock::time_point hostDeadline,
                                EeEvent event)
{
    {
        std::lock_guard lock(m_eventMutex);
        m_deadlines.push_back(ScheduledEvent{deadlineCycle, hostDeadline, event, ++m_eventSequence});
        updateNextDeadline();
    }
    m_eventCv.notify_one();
}

void EeScheduler::updateNextDeadline()
{
    if (m_deadlines.empty())
    {
        m_nextDeadlineCycle.store(0u, std::memory_order_release);
        return;
    }
    const auto it = std::min_element(m_deadlines.begin(), m_deadlines.end(),
                                     [](const ScheduledEvent &left, const ScheduledEvent &right)
                                     {
                                         if (left.deadlineCycle != right.deadlineCycle)
                                         {
                                             return left.deadlineCycle < right.deadlineCycle;
                                         }
                                         return left.sequence < right.sequence;
                                     });
    m_nextDeadlineCycle.store(it->deadlineCycle, std::memory_order_release);
}

bool EeScheduler::hasReadyAtOrAbovePriority(int priority) const
{
    const int last = std::clamp(priority, 0, kPriorityCount - 1);
    for (int p = 0; p <= last; ++p)
    {
        if (!m_readyQueues[static_cast<size_t>(p)].empty())
        {
            return true;
        }
    }
    return false;
}

void EeScheduler::renewTimeSlice()
{
    m_sliceEndCycle = m_eeCycle + kDefaultTimeSliceCycles;
    m_timeSliceExpired = false;
}

void EeScheduler::copyMainContextToRuntime()
{
    const GuestThread *main = thread(kMainThreadId);
    if (main)
    {
        m_runtime.m_cpuContext = main->context;
    }
}

void EeScheduler::publishDebugContext(const R5900Context &context)
{
    m_runtime.m_debugPc.store(context.pc, std::memory_order_relaxed);
    m_runtime.m_debugRa.store(getRegU32(&context, 31), std::memory_order_relaxed);
    m_runtime.m_debugSp.store(getRegU32(&context, 29), std::memory_order_relaxed);
    m_runtime.m_debugGp.store(getRegU32(&context, 28), std::memory_order_relaxed);
}

void EeScheduler::publishIdleDebugContext()
{
    // Temporary IRQ/RPC/alarm invocations deliberately return to PC=0. Once
    // the scheduler is idle, show a real EE thread context instead of leaving
    // the debugger pinned to that completed dispatcher frame.
    const GuestThread *selected = thread(kMainThreadId);
    if (!selected)
    {
        for (const auto &[id, candidate] : m_threads)
        {
            if (id > 0 && candidate.status != EeThreadStatus::Dormant)
            {
                selected = &candidate;
                break;
            }
        }
    }

    if (selected)
    {
        publishDebugContext(selected->activeContext());
    }
}
