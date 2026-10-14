# The parts of sounddevice code-chat uses; the package ships no type information.
from collections.abc import Callable

class PortAudioError(Exception): ...

class InputStream:
    active: bool
    closed: bool
    samplerate: float
    def __init__(
        self,
        *,
        samplerate: float,
        channels: int,
        dtype: str,
        blocksize: int,
        callback: Callable[..., None],
    ) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...
