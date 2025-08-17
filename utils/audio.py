from pathlib import Path
from pydub import AudioSegment
from pydub.utils import which

# Use PATH first, fallback to C:\ffmpeg\bin
AudioSegment.converter = which("ffmpeg") or which(r"C:\ffmpeg\bin\ffmpeg.exe")
AudioSegment.ffprobe = which("ffprobe") or which(r"C:\ffmpeg\bin\ffprobe.exe")

if not AudioSegment.converter or not AudioSegment.ffprobe:
    raise FileNotFoundError(
        "ffmpeg or ffprobe not found. "
        "Install from https://ffmpeg.org/download.html and add to PATH."
    )

def ensure_wav_16k_mono(input_path: Path) -> Path:
    audio = AudioSegment.from_file(str(input_path))
    if input_path.suffix.lower() == ".wav" and audio.frame_rate == 16000 and audio.channels == 1:
        return input_path
    audio = audio.set_channels(1).set_frame_rate(16000)
    out_path = input_path.with_suffix(".16k.wav")
    audio.export(str(out_path), format="wav")
    return out_path
