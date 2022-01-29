from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm.auto import tqdm

from examples.speech_synthesis.data_utils import (  # get_global_cmvn,
    extract_logmel_spectrogram, ipa_phonemize)
from fairseq.data.audio.audio_utils import convert_waveform


@dataclass
class AudioFeaturesConfig:
    normalize_volume: bool = False
    sample_rate: int = 16000
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    n_mels: int = 80
    f_min: int = 20
    f_max: int = 8000


def extract_features(path: Path, config: AudioFeaturesConfig):
    audio, sample_rate = sf.read(path)
    audio = torch.from_numpy(audio[None]).to(torch.float32)
    # print(audio)
    # print(sample_rate, audio.shape)
    audio, sample_rate = convert_waveform(
        audio,
        sample_rate,
        normalize_volume=config.normalize_volume,
        to_sample_rate=config.sample_rate,
    )

    features = extract_logmel_spectrogram(
        audio,
        sample_rate,
        output_path=None,
        win_length=config.win_length,
        hop_length=config.hop_length,
        n_fft=config.n_fft,
        n_mels=config.n_mels,
        f_min=config.f_min,
        f_max=config.f_max,
        target_length=None,
    )
    return features


def main():
    config = AudioFeaturesConfig()
    data_dir = Path("data") / "male"
    lang = "hi"
    with open(data_dir / "txt.done.data", "r") as f:
        texts = f.read()
        texts = texts.replace("(", "").replace(")", "").replace('"', "")
        texts = texts.split("\n")
    texts = [line.split() for line in texts]
    sample_ids_to_texts = {
        line[0]: " ".join(line[1:]) for line in texts if len(line) > 0
    }
    data_dir /= "wav"

    output_dir = Path("data-preprocessed")
    output_dir.mkdir(exist_ok=True)
    features_dir = output_dir / "features"
    features_dir.mkdir(exist_ok=True)
    tsv_data = {k: [] for k in ["id", "audio", "n_frames", "tgt_text"]}
    total = len([None for _ in data_dir.iterdir()])
    for sample_path in tqdm(data_dir.iterdir(), total=total):
        melspec = extract_features(sample_path, config).numpy()

        sample_id = sample_path.name[:-4]
        np.save(features_dir / f"{sample_id}.npy", melspec)

        text = sample_ids_to_texts[sample_id]

        tsv_data["audio"].append(sample_path.name)
        tsv_data["id"].append(sample_id)
        tsv_data["n_frames"].append(len(melspec))
        tsv_data["tgt_text"].append(text)

    tsv_data["tgt_text"] = ipa_phonemize(tsv_data["tgt_text"], lang=lang)

    tsv_data = pd.DataFrame.from_dict(tsv_data)
    print(tsv_data)
    tsv_data.to_csv(output_dir / "train.tsv")


if __name__ == "__main__":
    main()
