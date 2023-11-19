import os
from pathlib import Path

import torch
from g2p_en import G2p
from tqdm import tqdm

from src import waveglow
from src.utils import audio

raw_texts_old = [
    "A defibrillator is a device that gives a high energy"
    " electric shock to the heart of someone who is in "
    "cardiac arrest",
    "Massachusetts Institute of Technology may be best "
    "known for its math, science and engineering education",
    "Wasserstein distance or Kantorovich Rubinstein "
    "metric is a distance function defined between probability "
    "distributions on a given metric space",
]
g2p = G2p()
raw_texts = [g2p(i) for i in raw_texts_old]
# spaces actually corresponds to long pauses in speech, so we do not need them for short phrases
raw_texts = [[i for i in x if i != " "] for x in raw_texts]
combinations = [
    (1, 1, 1),
    (1, 1, 1.2),
    (1, 1, 0.8),
    (1, 1.2, 1),
    (1, 0.8, 1),
    (1.2, 1, 1),
    (0.8, 1, 1),
    (1.2, 1.2, 1.2),
    (0.8, 0.8, 0.8),
]


class Synthesizer:
    def __init__(
        self, waveglow, device, dir="results", use_arpa=False, samplerate=22050
    ):
        self.waveglow = waveglow
        self.device = device
        self.samplerate = samplerate
        self.cleaner_names: list = ["english_cleaners"]
        os.makedirs(dir, exist_ok=True)
        self.dir = Path(dir)
        self.g2p = G2p()
        # self.arpa = lambda x: ' '.join(['{' + ''.join(self.g2p(x)) + '}' for i in x]) if use_arpa else lambda x: x
        # self.symbols = {' ': 0} | {k: v + 1 for v, k in enumerate(valid_symbols)}
        self.symbols = {" ": 0, "": 0, "spn": 1} | {
            k: v for v, k in enumerate(self.g2p.phonemes)
        }
        self.inv_symbols = {v: k for k, v in self.symbols.items()}
        self.inv_symbols[0] = " "

    def __call__(self, model, raw_text, alpha=1.0, beta=1.0, gamma=1.0, idx=0):
        # phn = text.text_to_sequence(raw_text, self.cleaner_names)
        phn = [self.symbols.get(x, 0) for x in raw_text if x in self.symbols]
        text_tensor = torch.tensor(phn, dtype=torch.long, device=self.device).unsqueeze(
            0
        )
        src_pos = torch.arange(
            1, text_tensor.size(1) + 1, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        audio_path = (
            self.dir / f"speed={alpha}_pitch={beta}_energy={gamma}_id={idx}_.wav"
        )
        audio_path_waveglow = (
            self.dir
            / f"speed={alpha}_pitch={beta}_energy={gamma}_id={idx}_waveglow.wav"
        )
        with torch.no_grad():
            mel = model(text_tensor, src_pos, alpha=alpha, beta=beta, gamma=gamma)
            mel, mel_cuda = mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(
                1, 2
            )
            audio.tools.inv_mel_spec(mel, audio_path)
            waveglow.inference.inference(
                mel_cuda,
                self.waveglow,
                audio_path_waveglow,
                sampling_rate=self.samplerate,
            )
        return str(audio_path), str(audio_path_waveglow)

    def train_log(self, mel, duration):
        audio_path = self.dir / f"train_.wav"
        audio_path_waveglow = self.dir / f"train_waveglow.wav"
        with torch.no_grad():
            mel = mel[: duration.sum(), :].float()
            mel, mel_cuda = mel.cpu().transpose(0, 1), mel[
                None, :, :
            ].contiguous().transpose(1, 2)
            audio.tools.inv_mel_spec(mel, audio_path)
            waveglow.inference.inference(mel_cuda, self.waveglow, audio_path_waveglow)
        return str(audio_path), str(audio_path_waveglow)

    def create_audios(self, model):
        paths_standard, paths_waveglow, log_combinations, log_texts = [], [], [], []
        with tqdm(
            total=len(raw_texts) * len(combinations), desc="Synthesize audios"
        ) as pbar:
            for i, raw_text in enumerate(raw_texts):
                for alpha, beta, gamma in combinations:
                    standard_path, waveglow_path = self(
                        model, raw_text, alpha, beta, gamma, idx=i
                    )
                    paths_standard.append(standard_path)
                    paths_waveglow.append(waveglow_path)
                    log_combinations.append((alpha, beta, gamma))
                    log_texts.append(raw_texts_old[i])
                    pbar.update()
        return paths_standard, paths_waveglow, log_combinations, log_texts
