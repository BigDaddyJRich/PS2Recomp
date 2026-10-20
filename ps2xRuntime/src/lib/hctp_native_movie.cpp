// HCTP native movie player (see hctp_native_movie.h). One decoder thread per channel: FFmpeg demuxes the .sfd (MPEG
// program stream: MPEG-1 video + CRI ADX audio), decodes both, queues the pictures (kept as decoded YUV, converted to
// the size the game asks for when it asks) and resamples the audio to 48 kHz s16 stereo for a host stream. The audio
// stream's played samples are the clock; a movie without audio runs on the wall clock.
#include "hctp_native_movie.h"
#include "ps2_host_backend.h"

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace
{
    constexpr int kChannels = 4;
    constexpr int kRate = 48000;
    constexpr size_t kMaxPictures = 60;          // decoded pictures held ahead of the clock (~2 s)
    constexpr size_t kMaxAudioFrames = kRate;    // 1 s of audio held ahead
    constexpr size_t kAudioPrimeFrames = kRate / 20;

    struct Picture
    {
        AVFrame *frame = nullptr;
        double pts = 0.0;
    };

    struct Movie
    {
        std::mutex m;
        std::condition_variable cv;
        std::thread thread;
        std::atomic<bool> quit{false};
        bool active = false, ended = false, paused = false, loop = false, hasAudio = false, eof = false;
        bool started = false; // the clock (and sound) runs from the first frame the game draws, as Sofdec's
        std::string path;
        int64_t base = 0, size = 0; // byte range of the .sfd inside the file
        std::deque<Picture> pictures;
        std::vector<int16_t> audio; // interleaved stereo ring
        size_t audioRead = 0, audioQueued = 0;
        bool audioPrimed = false;
        uint64_t audioPlayed = 0; // frames the stream has played (the clock)
        double loopOffset = 0.0;  // seconds added per completed loop
        std::chrono::steady_clock::time_point wallStart;
        double pausedAt = 0.0;
        float volume = 1.0f;
        AudioStream stream{};
        bool streamOpen = false;
        uint64_t servedSerial = 0, serial = 0;
        SwsContext *sws = nullptr;
        int swsW = 0, swsH = 0, swsSrcW = 0, swsSrcH = 0, swsFmt = -1;
    };
    Movie g_movies[kChannels];

    // the host audio stream callbacks (no user pointer): one per channel
    template <int Ch>
    void audioCallback(void *buffer, unsigned int frames)
    {
        Movie &mv = g_movies[Ch];
        int16_t *out = static_cast<int16_t *>(buffer);
        std::lock_guard lock(mv.m);
        if (!mv.audioPrimed && (mv.audioQueued >= kAudioPrimeFrames || mv.eof))
            mv.audioPrimed = true;
        unsigned done = 0;
        if (mv.active && mv.started && !mv.paused && mv.audioPrimed)
        {
            const size_t ringFrames = mv.audio.size() / 2u;
            const unsigned take = static_cast<unsigned>(std::min<size_t>(frames, mv.audioQueued));
            for (; done < take; ++done)
            {
                const size_t f = (mv.audioRead + done) % ringFrames;
                out[done * 2u] = static_cast<int16_t>(std::clamp(mv.audio[f * 2u] * mv.volume, -32768.0f, 32767.0f));
                out[done * 2u + 1u] = static_cast<int16_t>(std::clamp(mv.audio[f * 2u + 1u] * mv.volume, -32768.0f, 32767.0f));
            }
            mv.audioRead = (mv.audioRead + take) % ringFrames;
            mv.audioQueued -= take;
            mv.audioPlayed += take;
            if (take < frames && !mv.eof)
                mv.audioPrimed = false; // ran dry: rebuild a little reserve
        }
        std::memset(out + done * 2u, 0, (frames - done) * 2u * sizeof(int16_t));
        mv.cv.notify_all();
    }
    AudioCallback g_audioCallbacks[kChannels] = {audioCallback<0>, audioCallback<1>, audioCallback<2>, audioCallback<3>};

    // the clock in seconds (caller holds mv.m)
    double clockLocked(Movie &mv)
    {
        if (!mv.started)
            return 0.0;
        if (mv.hasAudio)
            return static_cast<double>(mv.audioPlayed) / kRate;
        if (mv.paused)
            return mv.pausedAt;
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - mv.wallStart).count();
    }

    // AVIO over a byte range of a file (an .afs entry)
    struct RangeIo
    {
        std::FILE *f = nullptr;
        int64_t base = 0, size = 0, pos = 0;
    };
    int ioRead(void *opaque, uint8_t *buf, int n)
    {
        auto *io = static_cast<RangeIo *>(opaque);
        const int64_t left = io->size - io->pos;
        if (left <= 0)
            return AVERROR_EOF;
        n = static_cast<int>(std::min<int64_t>(n, left));
        _fseeki64(io->f, io->base + io->pos, SEEK_SET);
        const size_t got = std::fread(buf, 1, static_cast<size_t>(n), io->f);
        io->pos += static_cast<int64_t>(got);
        return got ? static_cast<int>(got) : AVERROR_EOF;
    }
    int64_t ioSeek(void *opaque, int64_t offset, int whence)
    {
        auto *io = static_cast<RangeIo *>(opaque);
        if (whence == AVSEEK_SIZE)
            return io->size;
        whence &= ~AVSEEK_FORCE;
        int64_t p = whence == SEEK_SET ? offset : whence == SEEK_CUR ? io->pos + offset : io->size + offset;
        if (p < 0 || p > io->size)
            return -1;
        io->pos = p;
        return p;
    }

    void decodeThread(int ch)
    {
        Movie &mv = g_movies[ch];
        RangeIo io;
        io.f = std::fopen(mv.path.c_str(), "rb");
        io.base = mv.base;
        io.size = mv.size;
        AVFormatContext *fmt = nullptr;
        AVIOContext *avio = nullptr;
        AVCodecContext *vdec = nullptr, *adec = nullptr;
        SwrContext *swr = nullptr;
        AVPacket *pkt = av_packet_alloc();
        AVFrame *frame = av_frame_alloc();
        int vIndex = -1, aIndex = -1;
        bool ok = io.f != nullptr;
        if (ok)
        {
            uint8_t *ioBuf = static_cast<uint8_t *>(av_malloc(64 * 1024));
            avio = avio_alloc_context(ioBuf, 64 * 1024, 0, &io, ioRead, nullptr, ioSeek);
            fmt = avformat_alloc_context();
            fmt->pb = avio;
            ok = avformat_open_input(&fmt, nullptr, nullptr, nullptr) == 0 && avformat_find_stream_info(fmt, nullptr) >= 0;
        }
        if (ok)
        {
            vIndex = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
            aIndex = av_find_best_stream(fmt, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
            const auto openDec = [&](int index) -> AVCodecContext * {
                if (index < 0)
                    return nullptr;
                const AVCodec *codec = avcodec_find_decoder(fmt->streams[index]->codecpar->codec_id);
                AVCodecContext *c = codec ? avcodec_alloc_context3(codec) : nullptr;
                if (c && (avcodec_parameters_to_context(c, fmt->streams[index]->codecpar) < 0 || avcodec_open2(c, codec, nullptr) < 0))
                    avcodec_free_context(&c);
                return c;
            };
            vdec = openDec(vIndex);
            adec = openDec(aIndex);
            ok = vdec != nullptr;
        }
        std::fprintf(stderr, "[nmovie] ch %d open %s @0x%llx size 0x%llx: %s (video %s, audio %s)%c", ch, mv.path.c_str(),
                     static_cast<unsigned long long>(mv.base), static_cast<unsigned long long>(mv.size), ok ? "ok" : "FAILED",
                     vdec ? vdec->codec->name : "-", adec ? adec->codec->name : "-", 10);
        {
            std::lock_guard lock(mv.m);
            mv.hasAudio = adec != nullptr;
            if (!ok)
                mv.ended = true;
        }
        if (adec)
        {
            AVChannelLayout stereo = AV_CHANNEL_LAYOUT_STEREO;
            if (swr_alloc_set_opts2(&swr, &stereo, AV_SAMPLE_FMT_S16, kRate, &adec->ch_layout, adec->sample_fmt,
                                    adec->sample_rate, 0, nullptr) < 0 || swr_init(swr) < 0)
                swr_free(&swr);
        }
        std::vector<int16_t> pcm;
        const auto takeAudio = [&](AVFrame *f) {
            if (!swr)
                return;
            const int outMax = swr_get_out_samples(swr, f->nb_samples);
            pcm.resize(static_cast<size_t>(std::max(outMax, 0)) * 2u + 2u);
            uint8_t *outPtr = reinterpret_cast<uint8_t *>(pcm.data());
            const int n = swr_convert(swr, &outPtr, outMax, const_cast<const uint8_t **>(f->extended_data), f->nb_samples);
            std::unique_lock lock(mv.m);
            for (int i = 0; i < n && !mv.quit; ++i)
            {
                mv.cv.wait(lock, [&] { return mv.quit || mv.audioQueued < kMaxAudioFrames; });
                if (mv.quit)
                    break;
                const size_t ringFrames = mv.audio.size() / 2u;
                const size_t w = (mv.audioRead + mv.audioQueued) % ringFrames;
                mv.audio[w * 2u] = pcm[static_cast<size_t>(i) * 2u];
                mv.audio[w * 2u + 1u] = pcm[static_cast<size_t>(i) * 2u + 1u];
                ++mv.audioQueued;
            }
        };
        double loopBase = 0.0, lastPts = 0.0;
        const auto takePicture = [&](AVFrame *f) {
            const AVRational tb = fmt->streams[vIndex]->time_base;
            const int64_t ts = f->best_effort_timestamp;
            double pts = ts == AV_NOPTS_VALUE ? lastPts + 1.0 / 30.0 : ts * av_q2d(tb);
            if (fmt->start_time != AV_NOPTS_VALUE)
                pts -= static_cast<double>(fmt->start_time) / AV_TIME_BASE;
            lastPts = pts;
            Picture p;
            p.frame = av_frame_clone(f);
            p.pts = loopBase + pts;
            std::unique_lock lock(mv.m);
            // never block the video while the audio (the clock) is starving: the audio of a later picture may still be
            // further on in the stream, and waiting here would stop the clock for good
            mv.cv.wait(lock, [&] {
                return mv.quit || mv.pictures.size() < kMaxPictures || (mv.hasAudio && mv.audioQueued < kAudioPrimeFrames * 2u);
            });
            if (mv.quit)
            {
                av_frame_free(&p.frame);
                return;
            }
            mv.pictures.push_back(p);
        };
        const auto drain = [&](AVCodecContext *c, bool video) {
            while (avcodec_receive_frame(c, frame) == 0)
            {
                if (video)
                    takePicture(frame);
                else
                    takeAudio(frame);
                av_frame_unref(frame);
            }
        };
        while (ok && !mv.quit)
        {
            const int r = av_read_frame(fmt, pkt);
            if (r < 0)
            {
                // end of the stream: flush the decoders; loop or finish
                for (AVCodecContext *c : {vdec, adec})
                    if (c)
                    {
                        avcodec_send_packet(c, nullptr);
                        drain(c, c == vdec);
                        avcodec_flush_buffers(c);
                    }
                bool loop;
                {
                    std::lock_guard lock(mv.m);
                    loop = mv.loop;
                }
                if (!loop)
                    break;
                loopBase = lastPts + 1.0 / 30.0;
                avformat_seek_file(fmt, -1, INT64_MIN, 0, INT64_MAX, AVSEEK_FLAG_BYTE);
                avio_seek(fmt->pb, 0, SEEK_SET);
                continue;
            }
            if (pkt->stream_index == vIndex && avcodec_send_packet(vdec, pkt) == 0)
                drain(vdec, true);
            else if (adec && pkt->stream_index == aIndex && avcodec_send_packet(adec, pkt) == 0)
                drain(adec, false);
            av_packet_unref(pkt);
        }
        {
            std::lock_guard lock(mv.m);
            mv.eof = true;
        }
        swr_free(&swr);
        avcodec_free_context(&vdec);
        avcodec_free_context(&adec);
        av_frame_free(&frame);
        av_packet_free(&pkt);
        if (fmt)
            avformat_close_input(&fmt);
        if (avio)
        {
            av_freep(&avio->buffer);
            avio_context_free(&avio);
        }
        if (io.f)
            std::fclose(io.f);
    }

    void stopLocked(int ch, std::unique_lock<std::mutex> &lock)
    {
        Movie &mv = g_movies[ch];
        mv.quit = true;
        mv.cv.notify_all();
        if (mv.thread.joinable())
        {
            lock.unlock();
            mv.thread.join();
            lock.lock();
        }
        for (Picture &p : mv.pictures)
            av_frame_free(&p.frame);
        mv.pictures.clear();
        mv.active = false;
        mv.quit = false;
    }
}

extern "C" bool hctpMovieOpen(int ch, const char *path, int fid, bool loop)
{
    if (ch < 0 || ch >= kChannels || !path)
        return false;
    int64_t base = 0, size = 0;
    {
        std::FILE *f = std::fopen(path, "rb");
        if (!f)
        {
            std::fprintf(stderr, "[nmovie] ch %d cannot open %s%c", ch, path, 10);
            return false;
        }
        _fseeki64(f, 0, SEEK_END);
        size = _ftelli64(f);
        if (fid >= 0)
        {
            uint8_t head[8];
            uint32_t entry[2] = {};
            _fseeki64(f, 0, SEEK_SET);
            const bool okHead = std::fread(head, 1, 8, f) == 8 && std::memcmp(head, "AFS", 3) == 0;
            uint32_t count = 0;
            std::memcpy(&count, head + 4, 4);
            if (!okHead || static_cast<uint32_t>(fid) >= count || _fseeki64(f, 8 + int64_t(fid) * 8, SEEK_SET) != 0 ||
                std::fread(entry, 4, 2, f) != 2)
            {
                std::fclose(f);
                std::fprintf(stderr, "[nmovie] ch %d %s: no AFS entry %d%c", ch, path, fid, 10);
                return false;
            }
            base = entry[0];
            size = entry[1];
        }
        std::fclose(f);
    }
    Movie &mv = g_movies[ch];
    std::unique_lock lock(mv.m);
    stopLocked(ch, lock);
    mv.path = path;
    mv.base = base;
    mv.size = size;
    mv.loop = loop;
    mv.ended = mv.eof = mv.paused = mv.audioPrimed = mv.started = false;
    mv.audio.assign(kMaxAudioFrames * 2u + 2u, 0);
    mv.audioRead = mv.audioQueued = 0;
    mv.audioPlayed = 0;
    mv.serial = mv.servedSerial = 0;
    mv.wallStart = std::chrono::steady_clock::now();
    if (!mv.streamOpen)
    {
        mv.stream = LoadAudioStream(kRate, 16, 2);
        SetAudioStreamCallback(mv.stream, g_audioCallbacks[ch]);
        PlayAudioStream(mv.stream);
        mv.streamOpen = true;
    }
    mv.active = true;
    mv.thread = std::thread(decodeThread, ch);
    return true;
}

extern "C" void hctpMovieStop(int ch)
{
    if (ch < 0 || ch >= kChannels)
        return;
    std::fprintf(stderr, "[nmovie] ch %d stop%c", ch, 10);
    std::unique_lock lock(g_movies[ch].m);
    stopLocked(ch, lock);
}

extern "C" void hctpMoviePause(int ch, bool pause)
{
    if (ch < 0 || ch >= kChannels)
        return;
    Movie &mv = g_movies[ch];
    std::lock_guard lock(mv.m);
    std::fprintf(stderr, "[nmovie] ch %d %s (pics %zu eof %d)%c", ch, pause ? "pause" : "continue", mv.pictures.size(), mv.eof ? 1 : 0, 10);
    if (mv.paused == pause)
        return;
    if (pause)
        mv.pausedAt = clockLocked(mv);
    else if (!mv.hasAudio)
        mv.wallStart = std::chrono::steady_clock::now() -
                       std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(mv.pausedAt));
    mv.paused = pause;
}

extern "C" bool hctpMovieActive(int ch)
{
    return ch >= 0 && ch < kChannels && g_movies[ch].active;
}

extern "C" int hctpMovieStatus(int ch)
{
    if (ch < 0 || ch >= kChannels)
        return 0;
    Movie &mv = g_movies[ch];
    std::lock_guard lock(mv.m);
    if (!mv.active)
        return 0;
    // ended: everything decoded and either the sound has run out (it is the clock: with it gone the last pictures
    // would never come due) or, without sound, the last picture is showing
    if (mv.ended || (mv.eof && (mv.hasAudio ? mv.audioQueued == 0 : mv.pictures.size() <= 1u)))
        return 3;
    return 2;
}

extern "C" void hctpMovieSetVolume(int ch, float volume)
{
    if (ch < 0 || ch >= kChannels)
        return;
    std::lock_guard lock(g_movies[ch].m);
    g_movies[ch].volume = std::clamp(volume, 0.0f, 2.0f);
}

extern "C" bool hctpMovieFrame(int ch, uint8_t *rgba, int w, int h)
{
    if (ch < 0 || ch >= kChannels || !rgba || w <= 0 || h <= 0)
        return false;
    Movie &mv = g_movies[ch];
    std::lock_guard lock(mv.m);
    if (static int idle[kChannels] = {}; (!mv.active || mv.pictures.empty()) && ++idle[ch] % 120 == 0)
        std::fprintf(stderr, "[nmovie] ch %d no picture (active %d eof %d ended %d)%c", ch, mv.active ? 1 : 0, mv.eof ? 1 : 0, mv.ended ? 1 : 0, 10);
    if (!mv.active || mv.pictures.empty())
        return false;
    if (!mv.started)
    {
        mv.started = true;
        mv.wallStart = std::chrono::steady_clock::now();
    }
    const double now = clockLocked(mv);
    if (static int calls[kChannels] = {}; ++calls[ch] % 120 == 0)
        std::fprintf(stderr, "[nmovie] ch %d clock %.2f pics %zu front %.2f audio q %zu played %llu primed %d eof %d serial %llu/%llu%c", ch,
                     now, mv.pictures.size(), mv.pictures.front().pts, mv.audioQueued,
                     static_cast<unsigned long long>(mv.audioPlayed), mv.audioPrimed ? 1 : 0, mv.eof ? 1 : 0,
                     static_cast<unsigned long long>(mv.servedSerial), static_cast<unsigned long long>(mv.serial), 10);
    // the newest picture that is due (drop the late ones); none due yet = keep showing the last one
    bool any = false;
    while (mv.pictures.size() > 1u && mv.pictures[1].pts <= now)
    {
        av_frame_free(&mv.pictures.front().frame);
        mv.pictures.pop_front();
        ++mv.serial;
    }
    if (mv.pictures.front().pts <= now || mv.serial == 0u)
        any = true;
    if (!any || mv.servedSerial == mv.serial + 1u)
        return false;
    AVFrame *f = mv.pictures.front().frame;
    if (!mv.sws || mv.swsW != w || mv.swsH != h || mv.swsSrcW != f->width || mv.swsSrcH != f->height || mv.swsFmt != f->format)
    {
        sws_freeContext(mv.sws);
        mv.sws = sws_getContext(f->width, f->height, static_cast<AVPixelFormat>(f->format), w, h, AV_PIX_FMT_RGBA,
                                SWS_BILINEAR, nullptr, nullptr, nullptr);
        mv.swsW = w;
        mv.swsH = h;
        mv.swsSrcW = f->width;
        mv.swsSrcH = f->height;
        mv.swsFmt = f->format;
    }
    if (!mv.sws)
        return false;
    static thread_local std::vector<uint8_t> linear;
    linear.resize(static_cast<size_t>(w) * h * 4u);
    uint8_t *dst[4] = {linear.data(), nullptr, nullptr, nullptr};
    int dstStride[4] = {w * 4, 0, 0, 0};
    sws_scale(mv.sws, f->data, f->linesize, 0, f->height, dst, dstStride);
    // the game's layout (Sofdec's output, uploaded by LoadImagePath3 / setLoadMovieTag): vertical strips 16 pixels
    // wide, strip after strip, each strip's rows top to bottom
    const int strips = w / 16;
    for (int s = 0; s < strips; ++s)
        for (int y = 0; y < h; ++y)
        {
            uint8_t *out = rgba + (static_cast<size_t>(s) * h * 16 + static_cast<size_t>(y) * 16) * 4u;
            std::memcpy(out, linear.data() + (static_cast<size_t>(y) * w + s * 16) * 4u, 16 * 4);
            for (int i = 0; i < 16; ++i)
                out[i * 4 + 3] = 0x80; // PS2 alpha 1.0
        }
    mv.servedSerial = mv.serial + 1u;
    mv.cv.notify_all();
    return true;
}
