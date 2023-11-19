import sys
import numpy as np
from pathlib import Path
import pyworld as pw
import torchaudio
from scipy.interpolate import interp1d
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import src.utils.audio.hparams_audio as ha
import src.utils.audio.tools as tools


def scale_and_save_features(feature_dir, scaler=None):
    max_value = np.finfo(np.float64).min
    min_value = np.finfo(np.float64).max
    for file in Path(feature_dir).iterdir():
        data = np.load(file)
        scaled_data = (data - scaler.mean_) / scaler.scale_
        np.save(file, scaled_data, allow_pickle=False)
        max_value = max(max_value, max(scaled_data))
        min_value = min(min_value, min(scaled_data))
    print(min_value, max_value)


def remove_outlier(values):
    values = np.array(values)
    p25 = np.percentile(values, 25)
    p75 = np.percentile(values, 75)
    lower = p25 - 1.5 * (p75 - p25)
    upper = p75 + 1.5 * (p75 - p25)
    normal_indices = np.logical_and(values > lower, values < upper)
    return values[normal_indices]


def extract_features(audio_path):
    wave, sr = torchaudio.load(audio_path)
    wave = wave.numpy().flatten().astype(np.float64)
    pitch, t = pw.dio(wave, sr, frame_period=ha.hop_length / sr * 1000)
    pitch = pw.stonemask(wave, pitch, t, sr)
    mel_spec, energy = tools.get_mel_from_wav(wave)
    return mel_spec.T, pitch, energy.squeeze()


def save_features(feature_dir, feature_type, features, index):
    np.save(Path(feature_dir, f"{feature_type}-{index:05d}.npy"), features, allow_pickle=False)


def smooth_pitch(pitch, duration):
    nonzero_ids = np.where(pitch != 0)[0]
    interp_fn = interp1d(
        nonzero_ids,
        pitch[nonzero_ids],
        fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
        bounds_error=False,
    )
    pitch = interp_fn(np.arange(0, len(pitch)))
    pos = 0
    for i, d in enumerate(duration):
        if d > 0:
            pitch[i] = np.mean(pitch[pos: pos + d])
        else:
            pitch[i] = 0
        pos += d
    pitch = pitch[: len(duration)]
    return pitch


def smooth_energy(energy, duration):
    pos = 0
    for i, d in enumerate(duration):
        if d > 0:
            energy[i] = np.mean(energy[pos: pos + d])
        else:
            energy[i] = 0
        pos += d
    energy = energy[: len(duration)]
    return energy


def main(dataset_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    mel_dir, pitch_dir, energy_dir = [Path(output_dir, dir_name) for dir_name in ["mels", "pitch", "energy"]]
    for dir in [mel_dir, pitch_dir, energy_dir]:
        dir.mkdir(exist_ok=True)
    pitch_scaler, energy_scaler = StandardScaler(), StandardScaler()
    texts, idx = [], 1
    metadata_path = Path(dataset_dir) / "metadata.csv"
    with open(metadata_path, "r", encoding='utf-8') as metadata_file:
        for i, line in enumerate(tqdm(metadata_file.readlines())):
            if i > 10:
                break
            wave_path, _, text = line.strip().split('|')
            wav_path = Path(dataset_dir) / "wavs" / f"{wave_path}.wav"
            mel, pitch, energy = extract_features(str(wav_path))
            texts.append(text)
            # duration = np.load(Path("alignments") / f"{i}.npy")
            # pitch = smooth_pitch(pitch, duration)
            # energy = smooth_energy(energy, duration)
            save_features(mel_dir, "mel", mel, idx)
            save_features(pitch_dir, "pitch", pitch, idx)
            save_features(energy_dir, "energy", energy, idx)
            pitch_scaler.partial_fit(remove_outlier(pitch).reshape((-1, 1)))
            energy_scaler.partial_fit(remove_outlier(energy).reshape((-1, 1)))
            idx += 1
    (Path(output_dir) / "train.txt").write_text('\n'.join(texts))
    #
    # print("==========Pitch==========")
    # scale_and_save_features(pitch_dir, pitch_scaler)
    # print("==========Enegy==========")
    # scale_and_save_features(energy_dir, energy_scaler)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
