"""Live voice search: the microphone streams 16 kHz mono into a buffer, and
while someone talks, what they've said so far is embedded again as soon as
the previous embedding returns, so results refine as they speak. A pause, or
silence from the start, ends the search.

The audio arrives on PortAudio's thread and is read on the caller's, so every
check of the session's state and the action it decides happen under one lock.
"""

import threading
import time

import numpy as np
import numpy.typing as npt
import sounddevice as sd

Audio = npt.NDArray[np.float32]

RATE = 16000
MAX_SAMPLES = 30 * RATE
PRE_ROLL = int(0.3 * RATE)
# Wait for enough speech to mean something, then update after each further
# stretch rather than every chunk, so results don't churn.
MIN_SPEECH = int(1.5 * RATE)
UPDATE_EVERY = int(0.75 * RATE)
# Speech is louder than both this and NOISE_RATIO times the background level.
# Listening ends after PAUSE of quiet, once they've talked for MIN_TALK, so a
# breath between phrases doesn't cut them off; or after SILENT_TIMEOUT if
# they never start.
VOICE_RMS = 0.004
NOISE_RATIO = 3
PAUSE = 4.0
MIN_TALK = 2.0
SILENT_TIMEOUT = 12.0
# Microphone errors, such as no input device or no permission to use it.
MicrophoneError = sd.PortAudioError


class Session:
    def __init__(self):
        self.lock = threading.Lock()
        self.chunks: list[Audio] = []
        self.length = 0
        self.received = 0
        self.embedded = 0
        self.level: float | None = None
        self.floor: float | None = None
        self.speaking = False
        self.started = time.monotonic()
        self.spoke_at = self.heard_at = 0.0
        self.stopped = False
        # Set once the final pass over everything said has been taken.
        self.finished = False
        self.smoothed: npt.NDArray[np.float32] | None = None
        self.closing: threading.Thread | None = None
        self.stream = sd.InputStream(samplerate=RATE, channels=1, dtype="float32", blocksize=RATE // 20, callback=self._hear)
        self.stream.start()
        assert self.stream.active, "the microphone stream didn't start; check the input device and permission to use it"
        assert self.stream.samplerate == RATE, f"the microphone runs at {self.stream.samplerate} Hz; pick an input device that supports {RATE} Hz"

    # sounddevice calls this with (data, frames, time, status); only the data matters.
    def _hear(self, data: Audio, *_: object):
        assert data.ndim == 2 and data.shape[1] == 1, f"expected mono input blocks, got shape {data.shape}; open the stream with channels=1"
        assert len(data) > 0, f"PortAudio delivered an empty block (shape {data.shape}); check the stream's blocksize"
        samples = data[:, 0].copy()
        now = time.monotonic()
        with self.lock:
            if self.stopped:
                return
            self._listen_for_speech(samples, now)
            self.chunks.append(samples)
            self.length += len(samples)
            self.received += len(samples)
            # Keep a moment of lead-in before speech, and at most MAX_SAMPLES.
            keep = MAX_SAMPLES if self.speaking else PRE_ROLL
            while len(self.chunks) > 1 and self.length - len(self.chunks[0]) > keep:
                self.length -= len(self.chunks.pop(0))
            paused = self.speaking and now - self.heard_at > PAUSE and now - self.spoke_at > MIN_TALK
            if paused or (not self.speaking and now - self.started > SILENT_TIMEOUT):
                self._stop()

    def _listen_for_speech(self, samples: Audio, now: float):
        """Track the level, smoothed over a few tens of milliseconds, and the
        room's background level, so speech is whatever stands well clear.
        Called with the lock held."""
        assert self.lock.locked(), "_listen_for_speech() changes shared state; call it with self.lock held"
        assert len(samples), f"an empty block of audio (shape {samples.shape}) has no level; skip empty blocks"
        rms = float(np.sqrt(np.mean(samples * samples)))
        level = rms if self.level is None else self.level * 0.8 + rms * 0.2
        floor = level if self.floor is None else self.floor
        if level > max(VOICE_RMS, floor * NOISE_RATIO):
            if not self.speaking:
                self.spoke_at = now
            self.speaking = True
            self.heard_at = now
        else:
            # Only between words, so talking doesn't raise it.
            floor = min(floor * 1.001, level)
        self.level, self.floor = level, floor

    def stop(self) -> threading.Thread:
        """Stop listening. Returns the thread closing the microphone, to join
        when the process is about to exit."""
        caller = threading.current_thread()
        assert caller is not self.closing, f"stop() from {caller.name}, the thread closing the stream, would wait on itself; return from it instead"
        with self.lock:
            closing = self._stop()
        assert closing is self.closing, f"stop() returned {closing}, not the session's closing thread {self.closing}; return _stop()'s thread"
        return closing

    def _stop(self) -> threading.Thread:
        """Stop, with the lock held, so a stop from the audio thread and one
        from the caller can't both close the stream."""
        assert self.lock.locked(), "_stop() changes shared state; call stop() instead"
        if self.closing is None:
            self.stopped = True
            # Closing from the audio callback's own thread would deadlock it.
            self.closing = threading.Thread(target=self._close, daemon=True)
            self.closing.start()
        assert self.stopped, "a session with a closing thread must be stopped; set stopped before starting it"
        return self.closing

    def _close(self):
        assert self.stopped, "close the stream only once the session has stopped; call stop()"
        caller = threading.current_thread()
        assert caller is self.closing, f"closing the stream from {caller.name}, not {self.closing}; let stop() start the closing thread"
        self.stream.stop()
        self.stream.close()

    def state(self) -> tuple[bool, bool]:
        """(speaking, done): whether speech was heard, and whether listening
        has stopped with nothing left to take."""
        with self.lock:
            speaking, stopped, finished = self.speaking, self.stopped, self.finished
        assert not finished or stopped, "the final pass was taken before listening stopped; only take() sets finished"
        assert not stopped or self.closing is not None, f"a stopped session has closing thread {self.closing}; stop through stop() so one starts"
        return speaking, stopped and (finished or not speaking)

    def take(self) -> tuple[Audio, bool] | None:
        """Everything said so far and whether it's the final pass, when
        there's enough new speech to embed again or listening has just
        stopped; else None."""
        with self.lock:
            assert self.embedded <= self.received, f"embedded {self.embedded} of only {self.received} samples; take() is the only place that sets embedded"
            fresh = self.received - self.embedded
            due = not self.stopped and self.speaking and self.length >= MIN_SPEECH and fresh >= UPDATE_EVERY
            final = self.stopped and self.speaking and not self.finished
            if not (due or final):
                return None
            self.embedded = self.received
            self.finished = final
            audio = np.concatenate(self.chunks)
        assert len(audio) > 0, f"took {len(audio)} samples of speech; only take() once speech has arrived"
        return audio, final

    def loudness(self) -> float:
        """Microphone level on a decibel scale: quiet rooms near 0, speech 0.6–0.9."""
        level = self.level or 0.0
        assert level >= 0, f"microphone level {level} can't be negative; check how _listen_for_speech() averages it"
        db = 20 * np.log10(level + 1e-6)
        out = float(min(1, max(0, (db + 55) / 40)))
        assert 0 <= out <= 1, f"loudness {out} must be between 0 and 1; check the clamp in loudness()"
        return out
