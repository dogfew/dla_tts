import numpy as np
from pathlib import Path
import librosa
import pyworld as pw
import textgrid
from tqdm import tqdm
import src.utils.audio.hparams_audio as ha
import src.utils.audio.tools as tools
from iteround import saferound
import argparse


def count_stats(feature_dir):
    max_value = np.finfo(np.float64).min
    min_value = np.finfo(np.float64).max
    for file in Path(feature_dir).iterdir():
        data = np.load(file).flatten()
        max_value = max(max_value, max(data))
        min_value = min(min_value, min(data))
    print(f"Min: {min_value}\nMax: {max_value}\n")


def calculate_durations(textgrid_file):
    tg = textgrid.TextGrid.fromFile(textgrid_file)
    durations = []
    phonemes = []
    for interval in tg[1].intervals:
        # this is constant for consistency of durations with other things
        duration = (interval.maxTime - interval.minTime) * 86.1328125
        durations.append(duration)
        phonemes.append(interval.mark)
    durations = np.array(durations)
    normalized_durations = durations / np.sum(durations) * np.ceil(np.sum(durations))
    return np.array(saferound(normalized_durations.tolist(), 0)), ' '.join(phonemes)


def extract_features(audio_path):
    wave, sr = librosa.load(audio_path)
    # wave = wave.numpy().flatten().astype(np.float64)
    wave = wave.flatten().astype(np.float64)
    pitch, t = pw.dio(wave, sr, frame_period=ha.hop_length / sr * 1000)
    pitch = pw.stonemask(wave, pitch, t, sr)
    mel_spec, energy = tools.get_mel_from_wav(wave)
    return mel_spec.T, pitch, energy.squeeze()


def save_features(feature_dir, feature_type, features, index):
    np.save(Path(feature_dir, f"{feature_type}-{index:05d}.npy"), features, allow_pickle=False)


def main(dataset_dir, output_dir="train_data", textgrid_dir='ljs_aligned2'):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    mel_dir, pitch_dir, energy_dir, alignment_dir = [Path(output_dir, dir_name)
                                                     for dir_name in ["mels", "pitch", "energy", "alignments"]]
    for dir in [mel_dir, pitch_dir, energy_dir, alignment_dir]:
        dir.mkdir(exist_ok=True)
    texts, idx = [], 1
    metadata_path = Path(dataset_dir) / "metadata.csv"
    dataset_dir2 = Path(dataset_dir) / "wavs"
    with open(metadata_path, "r", encoding='utf-8') as metadata_file:
        for i, line in enumerate(tqdm(metadata_file.readlines())):
            wave_path, _, _ = line.strip().split('|')
            try:
                durations, phonemes = calculate_durations(f"{textgrid_dir}/{wave_path}.TextGrid")
                wav_path = Path(dataset_dir2) / f"{wave_path}.wav"
                mel, pitch, energy = extract_features(str(wav_path))
            except FileNotFoundError:
                print(f"MFA could not generate alignments for {wave_path}. Skipping!")
                continue
            texts.append(phonemes)
            total_duration = durations.sum().astype(int)
            if total_duration != pitch.shape[0]:
                print('fixing duration')
                pitch = pitch[:total_duration]
                mel = mel[:total_duration, :]
                energy = energy[:total_duration]
            assert sum(durations) == pitch.shape[0], f"{durations.sum()}, {pitch.shape[0]}"
            save_features(alignment_dir, "alignment", durations, idx)
            save_features(mel_dir, "mel", mel, idx)
            save_features(pitch_dir, "pitch", pitch, idx)
            save_features(energy_dir, "energy", energy, idx)
            idx += 1
    (Path(output_dir) / "train.txt").write_text('\n'.join(texts))
    #
    print("==========Pitch Stats==========")
    count_stats(pitch_dir)
    print("==========Enegy Stats==========")
    count_stats(energy_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--dataset_dir', type=str, default='data/LJSpeech-1.1',
                        help='Directory of the dataset')
    parser.add_argument('--output_dir', type=str, default="train_data",
                        help='Output directory (default: train_data)')
    parser.add_argument('--textgrid_dir', type=str, default='ljs_aligned2',
                        help='TextGrid directory (default: ljs_aligned2)')

    args = parser.parse_args()
    main(args.dataset_dir, args.output_dir, args.textgrid_dir)
