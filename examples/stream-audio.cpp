#include "stream-audio.h"

#include <algorithm>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>

#ifdef WHISPER_COMMON_FFMPEG

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
}

#define STREAM_SAMPLE_RATE 16000

struct stream_audio_reader::impl {
    std::thread thread;
    mutable std::mutex mutex;
    std::condition_variable cv_data;
    std::condition_variable cv_space;

    std::deque<std::vector<float>> chunks;
    int64_t buffered     = 0;
    int64_t max_buffered = 0;

    bool eof     = false;
    bool stop    = false;
    bool failed  = false;
    double duration_s = 0.0;
    std::string err;

    void push(const float * data, int n) {
        if (n <= 0) {
            return;
        }

        std::vector<float> chunk(data, data + n);

        std::unique_lock<std::mutex> lock(mutex);
        cv_space.wait(lock, [this]() { return stop || buffered < max_buffered; });
        if (stop) {
            return;
        }
        buffered += n;
        chunks.push_back(std::move(chunk));
        cv_data.notify_one();
    }

    void fail(const std::string & message) {
        std::lock_guard<std::mutex> lock(mutex);
        failed = true;
        err = message;
        cv_data.notify_all();
    }

    void run(const std::string & path);
};

void stream_audio_reader::impl::run(const std::string & path) {
    AVFormatContext * fmt_ctx = nullptr;
    if (avformat_open_input(&fmt_ctx, path.c_str(), nullptr, nullptr) < 0) {
        fail("failed to open '" + path + "'");
        return;
    }

    if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
        fail("failed to find stream information in '" + path + "'");
        avformat_close_input(&fmt_ctx);
        return;
    }

    const AVCodec * codec = nullptr;
    const int stream_idx = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, &codec, 0);
    if (stream_idx < 0 || codec == nullptr) {
        fail("no decodable audio stream in '" + path + "'");
        avformat_close_input(&fmt_ctx);
        return;
    }

    {
        std::lock_guard<std::mutex> lock(mutex);
        if (fmt_ctx->duration != AV_NOPTS_VALUE) {
            duration_s = (double) fmt_ctx->duration / AV_TIME_BASE;
        }
    }

    AVCodecContext * codec_ctx = avcodec_alloc_context3(codec);
    if (codec_ctx == nullptr || avcodec_parameters_to_context(codec_ctx, fmt_ctx->streams[stream_idx]->codecpar) < 0) {
        fail("failed to allocate the audio decoder");
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&fmt_ctx);
        return;
    }

    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        fail("failed to open the audio decoder");
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&fmt_ctx);
        return;
    }

    AVChannelLayout out_layout = AV_CHANNEL_LAYOUT_MONO;

    SwrContext * swr = nullptr;
    if (swr_alloc_set_opts2(&swr, &out_layout, AV_SAMPLE_FMT_FLT, STREAM_SAMPLE_RATE,
                &codec_ctx->ch_layout, codec_ctx->sample_fmt, codec_ctx->sample_rate, 0, nullptr) < 0 ||
            swr_init(swr) < 0) {
        fail("failed to initialize the resampler");
        swr_free(&swr);
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&fmt_ctx);
        return;
    }

    std::vector<float> out;
    auto drain_frame = [&](AVFrame * frame) {
        const int max_out = std::max(1024, swr_get_out_samples(swr, frame->nb_samples));
        out.resize(max_out);

        uint8_t * out_data[1] = { (uint8_t *) out.data() };
        const int got = swr_convert(swr, out_data, max_out, (const uint8_t **) frame->extended_data, frame->nb_samples);
        if (got > 0) {
            push(out.data(), got);
        }
    };

    AVPacket * packet = av_packet_alloc();
    AVFrame  * frame  = av_frame_alloc();

    bool read_eof = false;
    while (!read_eof) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (stop) {
                break;
            }
        }

        if (av_read_frame(fmt_ctx, packet) < 0) {
            read_eof = true;
            break;
        }

        if (packet->stream_index != stream_idx) {
            av_packet_unref(packet);
            continue;
        }

        if (avcodec_send_packet(codec_ctx, packet) >= 0) {
            while (true) {
                const int ret = avcodec_receive_frame(codec_ctx, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    break;
                }
                if (ret < 0) {
                    break;
                }
                drain_frame(frame);
            }
        }
        av_packet_unref(packet);
    }

    // flush the decoder, then the resampler
    if (!read_eof) {
        // stopped early
    } else {
        avcodec_send_packet(codec_ctx, nullptr);
        while (avcodec_receive_frame(codec_ctx, frame) >= 0) {
            drain_frame(frame);
        }

        while (true) {
            const int max_out = std::max(1024, swr_get_out_samples(swr, 0));
            out.resize(max_out);
            uint8_t * out_data[1] = { (uint8_t *) out.data() };
            const int got = swr_convert(swr, out_data, max_out, nullptr, 0);
            if (got <= 0) {
                break;
            }
            push(out.data(), got);
        }
    }

    av_frame_free(&frame);
    av_packet_free(&packet);
    swr_free(&swr);
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&fmt_ctx);

    {
        std::lock_guard<std::mutex> lock(mutex);
        eof = true;
        cv_data.notify_all();
    }
}

stream_audio_reader::stream_audio_reader() : p(new impl) {}
stream_audio_reader::~stream_audio_reader() {
    {
        std::lock_guard<std::mutex> lock(p->mutex);
        p->stop = true;
        p->cv_space.notify_all();
        p->cv_data.notify_all();
    }
    if (p->thread.joinable()) {
        p->thread.join();
    }
}

bool stream_audio_reader::open(const std::string & path, int64_t max_buffered_samples) {
    std::lock_guard<std::mutex> lock(p->mutex);
    p->max_buffered = std::max<int64_t>(max_buffered_samples, STREAM_SAMPLE_RATE);
    p->thread = std::thread(&impl::run, p.get(), path);
    return true;
}

int stream_audio_reader::read(float * samples, int n_samples) {
    std::unique_lock<std::mutex> lock(p->mutex);
    p->cv_data.wait(lock, [this]() { return p->buffered > 0 || p->eof || p->failed; });

    if (p->buffered == 0) {
        return p->failed ? -1 : 0;
    }

    int done = 0;
    while (done < n_samples && !p->chunks.empty()) {
        auto & chunk = p->chunks.front();
        const int n = std::min<int>((int) chunk.size(), n_samples - done);
        std::copy(chunk.begin(), chunk.begin() + n, samples + done);
        done += n;
        p->buffered -= n;

        if (n == (int) chunk.size()) {
            p->chunks.pop_front();
        } else {
            chunk.erase(chunk.begin(), chunk.begin() + n);
        }
    }

    p->cv_space.notify_one();
    return done;
}

double stream_audio_reader::duration() const {
    std::lock_guard<std::mutex> lock(p->mutex);
    return p->duration_s;
}

std::string stream_audio_reader::error() const {
    std::lock_guard<std::mutex> lock(p->mutex);
    return p->err;
}

#else // WHISPER_COMMON_FFMPEG

struct stream_audio_reader::impl {};

stream_audio_reader::stream_audio_reader() : p(new impl) {}
stream_audio_reader::~stream_audio_reader() = default;

bool stream_audio_reader::open(const std::string & /*path*/, int64_t /*max_buffered_samples*/) {
    return false;
}

int stream_audio_reader::read(float * /*samples*/, int /*n_samples*/) {
    return -1;
}

double stream_audio_reader::duration() const {
    return 0.0;
}

std::string stream_audio_reader::error() const {
    return "streaming requires a build with WHISPER_COMMON_FFMPEG=ON";
}

#endif // WHISPER_COMMON_FFMPEG
