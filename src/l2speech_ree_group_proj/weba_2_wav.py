import subprocess
from pathlib import Path


def weba_2_wav(input_weba_path: Path, output_wav_path: Path, print_logs: bool =False) -> None:
    """
    Convert a .weba audio file to .wav format using ffmpeg.

    Args:
        input_weba_path (Path): Path to the input .weba file.
        output_wav_path (Path): Path to save the output .wav file.
    """

    if output_wav_path.exists():
        print(f"{output_wav_path} already exists. Skipping conversion.")
        return

    try:
        command = [
            "ffmpeg",
            "-i",
            str(input_weba_path),
            str(output_wav_path)
        ]
        output_log = subprocess.run(command, capture_output=True, text=True).stdout
    except FileNotFoundError:
        command = [
            "/opt/homebrew/bin/ffmpeg",
            "-i",
            str(input_weba_path),
            str(output_wav_path)
        ]
        output_log = subprocess.run(command, capture_output=True, text=True).stdout
    except Exception as e:
        print(f"An error occurred: {e}")

    if print_logs:
        print(output_log) # type: ignore
