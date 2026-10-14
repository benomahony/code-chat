"""Live voice search: the microphone streams 16 kHz mono into a buffer, and
while someone talks, what they've said so far is embedded again as soon as
the previous embedding returns, so results refine as they speak. A pause ends
the search.
"""

import threading
import time

import numpy as np

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


class Session:
    def __init__(self):
        import sounddevice as sd

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
        self.stream = sd.InputStream(samplerate=RATE, channels=1, dtype="float32", blocksize=RATE // 20, callback=self._hear)
        self.stream.start()

    def _hear(self, data, frames, at, status):
        if self.stopped:
            return
        samples = data[:, 0].copy()
        now = time.monotonic()
        # Smooth the level over a few tens of milliseconds, and track the
        # room's background level so speech is whatever stands well clear.
        rms = float(np.sqrt(np.mean(samples * samples)))
        self.level = rms if self.level is None else self.level * 0.8 + rms * 0.2
        if self.floor is None:
            self.floor = self.level
        with self.lock:
            if self.level > max(VOICE_RMS, self.floor * NOISE_RATIO):
                if not self.speaking:
                    self.spoke_at = now
                self.speaking = True
                self.heard_at = now
            else:
                # Only between words, so talking doesn't raise it.
                self.floor = min(self.floor * 1.001, self.level)
            self.chunks.append(samples)
            self.length += len(samples)
            self.received += len(samples)
            # Keep a moment of lead-in before speech, and at most MAX_SAMPLES.
            keep = MAX_SAMPLES if self.speaking else PRE_ROLL
            while self.chunks and self.length - len(self.chunks[0]) > keep:
                self.length -= len(self.chunks.pop(0))
        if self.speaking and now - self.heard_at > PAUSE and now - self.spoke_at > MIN_TALK:
            self.stop()

    def stop(self):
        if self.stopped:
            return
        self.stopped = True
        threading.Thread(target=self._close, daemon=True).start()

    def _close(self):
        self.stream.stop()
        self.stream.close()

    def due(self) -> bool:
        """Whether there's enough new speech to embed again."""
        with self.lock:
            if not self.speaking or self.length < MIN_SPEECH or self.received == self.embedded:
                return False
            return self.stopped or self.received - self.embedded >= UPDATE_EVERY

    def audio(self) -> np.ndarray:
        with self.lock:
            self.embedded = self.received
            return np.concatenate(self.chunks) if self.chunks else np.zeros(0, dtype=np.float32)

    def loudness(self) -> float:
        """Microphone level on a decibel scale: quiet rooms near 0, speech 0.6–0.9."""
        db = 20 * np.log10((self.level or 0) + 1e-6)
        return float(min(1, max(0, (db + 55) / 40)))
