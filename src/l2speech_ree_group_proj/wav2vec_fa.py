"""
References:
- https://github.com/RusCucumber/wav2vec-forced-alignment
"""

from datetime import datetime
from pathlib import Path
from traceback import format_exc
from typing import List, Optional, Tuple

import pandas as pd
import torch
import torchaudio
import torchaudio.functional as F
from tqdm import tqdm

TARGET_CHANNEL = 0

def load_wav2vecfa_bundle(device) -> tuple:
    print("Loading wav2vec...", end=" ")
    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model(with_star=True).to(device)
    print("DONE!")

    return bundle, model

def check_files(input_dir: Path) -> int:
    if not input_dir.exists():
        raise FileNotFoundError(f"{str(input_dir)} was not found.")

    if not input_dir.is_dir():
        raise ValueError(f"argument \"input_dir\" must be directory. {str(input_dir)} is a file.")

    n_files = 0

    for wavpath in input_dir.glob("*.wav"):
        txtpath = input_dir / f"{wavpath.stem}.txt"
        if not txtpath.exists():
            raise FileNotFoundError(f"{str(txtpath)} was not found.")

        n_files += 1

    return n_files

def make_output_dir(output_dir: str|Path|None =None) -> Path:
    if output_dir is None:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path(__file__).parent / f"output_{current_time}"

        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            raise FileNotFoundError(f"{str(output_dir)} was not found.")

    return output_dir

def is_aligned(output_dir: Path, filename: str) -> bool:
    save_path = output_dir / f"{filename}.csv"

    return save_path.exists()

def load_audio_w_transcript(wavpath: Path) -> Tuple[torch.Tensor, int, str]:
    waveform, sample_rate = torchaudio.load(str(wavpath))

    txtpath = wavpath.parent / f"{wavpath.stem}.txt"
    with open(txtpath, "r") as f:
        transcript = f.readline()
        transcript = transcript.lower()

    return waveform, sample_rate, transcript

def resample(waveform: torch.Tensor, sample_rate: int, bundle: torchaudio.pipelines.Wav2Vec2FABundle) -> torch.Tensor:
    transformed_waveform = waveform[TARGET_CHANNEL, :].view(1, -1)

    target_sample_rate = int(bundle.sample_rate)
    if sample_rate == target_sample_rate:
        return transformed_waveform

    transformed_waveform = torchaudio.transforms.Resample(sample_rate, target_sample_rate)(transformed_waveform)
    return transformed_waveform

def tokenize(transcript: str, bundle: torchaudio.pipelines.Wav2Vec2FABundle) -> List[str]:
    dictionary = bundle.get_dict()

    tokenized_transcript = []
    for word in transcript.split():
        for char in word:
            tokenized_transcript.append(dictionary[char])

    return tokenized_transcript

def is_audio_shorter_than_txt(emission: torch.Tensor, targets: torch.Tensor) -> bool:
    n_audio_samples = emission.shape[1]
    n_tokens = targets.shape[1]

    return n_audio_samples <= n_tokens

def generate_pseudo_token_spans(emission: torch.Tensor, targets: torch.Tensor) -> List[List[F.TokenSpan]]:
    token_spans = []
    start = 0
    end = emission.shape[1] - 1
    for token in targets[0]:
        span = F.TokenSpan(token=token, start=start, end=end, score=0.0) # type: ignore
        token_spans.append(span)

    return token_spans

def align(
        waveform: torch.Tensor,
        tokens: List[str],
        model: torch.nn.Module,
        device: str
) -> Tuple[List[list], float]:
    with torch.inference_mode():
        emission, _ = model(waveform.to(device))

    targets = torch.tensor([tokens], dtype=torch.int32, device=device)
    if is_audio_shorter_than_txt(emission, targets):
        print("\nSpeech duration is too short for forced alignment.")
        print("Generate pseud token_spans.")

        token_spans = generate_pseudo_token_spans(emission, targets)
        ratio = waveform.size(1) / emission.size(1)
        return token_spans, ratio

    alignments, scores = F.forced_align(emission, targets, blank=0)

    alignments, scores = alignments[0], scores[0]  # remove batch dimension for simplicity
    scores = scores.exp()  # convert back to probability

    token_spans = F.merge_tokens(alignments, scores)

    ratio = waveform.size(1) / emission.size(1)

    return token_spans, ratio # type: ignore

def frame_2_sec(frame: int, ratio: float, sample_rate: int) -> float:
    return int(frame * ratio) / sample_rate

def to_dataframe(token_spans: list, transcript: str, ratio: float, sample_rate: int) -> pd.DataFrame:
    lengths = [len(word) for word in transcript.split()]
    if len(token_spans) != sum(lengths):
        raise RuntimeError(f"N tokens is not equal: {len(token_spans)} != {sum(lengths)}")

    i = 0
    data = []
    for length, chars in zip(lengths, transcript.split()):
        start_time = token_spans[i].start
        end_time = token_spans[i + length - 1].end

        start_time = frame_2_sec(start_time, ratio, sample_rate)
        end_time = frame_2_sec(end_time, ratio, sample_rate)

        row = [chars, start_time, end_time]
        data.append(row)

        i += length

    df_timestamp = pd.DataFrame(data, columns=["word", "start_time", "end_time"])

    return df_timestamp

def save_csv(df_timestamp: pd.DataFrame, filename: str, output_dir: Path) -> None:
    save_path = output_dir / f"{filename}.csv"

    df_timestamp.to_csv(save_path, index=False)

def wav2vec_fa(
        input_dir: str|Path,
        output_dir: Optional[str|Path] = None,
        skip_aligned: bool = True,
        device: Optional[str] = None
) -> None:
    input_dir = Path(input_dir)

    if skip_aligned:
        print("Skip aligned files!\n")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # check input data & count n_files
    n_files = check_files(input_dir)

    # make output dir
    output_dir = make_output_dir(output_dir)

    # load bundle
    bundle, model = load_wav2vecfa_bundle(device)

    # transcribe & save transcription
    with tqdm(input_dir.glob("*.wav"), total=n_files) as pbar:
        for wavpath in pbar:
            if skip_aligned & is_aligned(output_dir, wavpath.stem):
                pbar.set_description(f"Skip {wavpath.name}")
                continue
            pbar.set_description(f"Align {wavpath.name}")

            try:
                waveform, sample_rate, transcript = load_audio_w_transcript(wavpath)
                waveform = resample(waveform, sample_rate, bundle)
                tokens = tokenize(transcript, bundle)

                if len(tokens) == 0:
                    print(f"\n[Warn] No texts in {wavpath.stem}.")
                    print("Skip Forced Alignment\n\n")
                    continue

                token_spans, ratio = align(waveform, tokens, model, device)
                df_timestamp = to_dataframe(token_spans, transcript, ratio, bundle.sample_rate)

                save_csv(df_timestamp, wavpath.stem, output_dir)
            except Exception:
                print(f"\n[Error] {wavpath.name}")
                print("Stop Forced Alignmend\n\n")
                print(format_exc())
                return
