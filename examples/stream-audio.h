#pragma once

#include <cstdint>
#include <memory>
#include <string>

// Background audio decoder.
//
// Reads an encoded audio file with libav*, resamples it to mono 16 kHz f32 in a
// worker thread and exposes it as a bounded, blocking stream. The point is to keep
// transcoding/decoding off the whisper thread: while whisper decodes a batch, this
// keeps filling its buffer, so the GPU does not idle between batches.
//
// Requires a build with WHISPER_COMMON_FFMPEG=ON (open() fails otherwise).
struct stream_audio_reader {
    stream_audio_reader();
    ~stream_audio_reader();

    stream_audio_reader(const stream_audio_reader &) = delete;
    stream_audio_reader & operator=(const stream_audio_reader &) = delete;

    // max_buffered_samples bounds the decoded audio kept in RAM (default: 30 s).
    bool open(const std::string & path, int64_t max_buffered_samples = 30*16000);

    // Copy up to n_samples into samples, blocking until data is available.
    // Returns the number of samples copied, 0 at end of stream, < 0 on error.
    int read(float * samples, int n_samples);

    // File duration in seconds, 0 if unknown.
    double duration() const;

    std::string error() const;

private:
    struct impl;
    std::unique_ptr<impl> p;
};
