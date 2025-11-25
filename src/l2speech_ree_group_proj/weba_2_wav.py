import subprocess
from pathlib import Path


def weba_2_wav(input_weba_path: Path, output_wav_path: Path) -> None:
    """
    Convert a .weba audio file to .wav format using ffmpeg.

    Args:
        input_weba_path (Path): Path to the input .weba file.
        output_wav_path (Path): Path to save the output .wav file.
    """
    command = [
        "ffmpeg",
        "-i",
        str(input_weba_path),
        str(output_wav_path)
    ]
    subprocess.run(command, check=True)

