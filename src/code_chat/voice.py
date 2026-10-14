"""Live voice search: the microphone streams 16 kHz mono into a buffer, and
while someone talks, what they've said so far is embedded again as soon as
the previous embedding returns, so results refine as they speak. A pause ends
the search.
"""

import threading
import time

import numpy as np
import sounddevice as sd

RATE = 16000
MAX_SAMPLES = 30 * RATE
PRE_ROLL = int(0.3 * RATE)
# Wait for enough speech to mean something, then update after each further
# stretch rather than every chunk, so results don't churn.
MIN_SPEECH = int(1.5 * RATE)
UPDATE_EVERY = int(0.75 * RATE)
# Speech is louder than both this and NOISE_RATIO times the background level.
# Listening ends after PAUSE of quiet, once they've talked for MIN_TALK, so a
# breath between phrases doesn't cut them off.
VOICE_RMS = 0.004
NOISE_RATIO = 3
PAUSE = 2.5
MIN_TALK = 2.0
# Microphone errors, such as no input device or no permission to use it.
MicrophoneError = sd.PortAudioError


class Session:
    def __init__(self):
        self.lock = threading.Lock()
        self.chunks: list[np.ndarray] = []
        self.length = 0
        self.received = 0
        self.embedded = 0
        self.level: float | None = None
        self.floor: float | None = None
        self.speaking = False
        self.spoke_at = self.heard_at = 0.0
        self.stopped = False
        self.smoothed: np.ndarray | None = None
        self.closing: threading.Thread | None = None
        self.stream = sd.InputStream(samplerate=RATE, channels=1, dtype="float32", blocksize=RATE // 20, callback=self._hear)
        self.stream.start()
        assert self.stream.active, "the microphone stream didn't start; check the input device and permission to use it"
        assert self.stream.samplerate == RATE, f"the microphone runs at {self.stream.samplerate} Hz; embedding needs {RATE}"

    # sounddevice calls this with (data, frames, time, status); only the data matters.
    def _hear(self, data: np.ndarray, *_):
        assert data.ndim == 2 and data.shape[1] == 1, f"expected mono input blocks, got shape {data.shape}"
        if self.stopped:
            return
        samples = data[:, 0].copy()
        now = time.monotonic()
        self._listen_for_speech(samples, now)
        with self.lock:
            self.chunks.append(samples)
            self.length += len(samples)
            self.received += len(samples)
            # Keep a moment of lead-in before speech, and at most MAX_SAMPLES.
            keep = MAX_SAMPLES if self.speaking else PRE_ROLL
            while len(self.chunks) > 1 and self.length - len(self.chunks[0]) > keep:
                self.length -= len(self.chunks.pop(0))
            assert self.length <= self.received, f"buffered {self.length} samples but only received {self.received}"
        if self.speaking and now - self.heard_at > PAUSE and now - self.spoke_at > MIN_TALK:
            self.stop()

    def _listen_for_speech(self, samples: np.ndarray, now: float):
        """Track the level, smoothed over a few tens of milliseconds, and the
        room's background level, so speech is whatever stands well clear."""
        assert len(samples), f"an empty block of audio (shape {samples.shape}) has no level"
        rms = float(np.sqrt(np.mean(samples * samples)))
        self.level = rms if self.level is None else self.level * 0.8 + rms * 0.2
        self.floor = self.level if self.floor is None else self.floor
        if self.level > max(VOICE_RMS, self.floor * NOISE_RATIO):
            if not self.speaking:
                self.spoke_at = now
            self.speaking = True
            self.heard_at = now
        else:
            # Only between words, so talking doesn't raise it.
            self.floor = min(self.floor * 1.001, self.level)
        assert self.level >= 0 and self.floor >= 0, f"levels can't be negative: level {self.level}, floor {self.floor}"

    def stop(self) -> threading.Thread:
        """Stop listening. Returns the thread closing the microphone, to join
        when the process is about to exit."""
        assert self.stream is not None, "stop() needs the stream __init__ opened"
        if not self.stopped:
            self.stopped = True
            # Closing from the audio callback's own thread would deadlock it.
            self.closing = threading.Thread(target=self._close, daemon=True)
            self.closing.start()
        assert self.stopped and self.closing is not None, "stop() must leave the session stopped and closing"
        return self.closing

    def _close(self):
        assert self.stopped, "close the stream only once the session has stopped taking audio"
        self.stream.stop()
        self.stream.close()
        assert self.stream.closed, "the microphone stream is still open after closing it"

    def due(self) -> bool:
        """Whether there's enough new speech to embed again."""
        with self.lock:
            assert self.embedded <= self.received, f"embedded {self.embedded} samples of only {self.received}"
            if not self.speaking or self.length < MIN_SPEECH or self.received == self.embedded:
                return False
            fresh = self.received - self.embedded
            assert fresh > 0, f"no new samples since the last embedding ({self.received} received)"
            return self.stopped or fresh >= UPDATE_EVERY

    def audio(self) -> np.ndarray:
        with self.lock:
            assert self.chunks, "audio() called before any audio arrived; check due() first"
            self.embedded = self.received
            out = np.concatenate(self.chunks)
        assert len(out) == self.length, f"joined {len(out)} samples, but the buffer holds {self.length}"
        return out

    def loudness(self) -> float:
        """Microphone level on a decibel scale: quiet rooms near 0, speech 0.6–0.9."""
        level = self.level or 0.0
        assert level >= 0, f"microphone level {level} can't be negative"
        db = 20 * np.log10(level + 1e-6)
        out = float(min(1, max(0, (db + 55) / 40)))
        assert 0 <= out <= 1, f"loudness {out} must be between 0 and 1"
        return out
