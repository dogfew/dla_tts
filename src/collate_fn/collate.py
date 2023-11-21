import torch
from torch.nn.utils.rnn import pad_sequence


def create_positional_encoding(lengths, max_len):
    """Create positional encoding for a batch of sequences in a vectorized manner."""
    pos = torch.arange(1, max_len + 1, dtype=torch.long)
    encoded = (pos <= lengths.unsqueeze(1)).long() * pos
    return encoded


def reprocess_tensor(batch):
    raw_texts = [item["text"] for item in batch]
    mel_targets = [item["mel_target"] for item in batch]
    pitches_targets = [item["pitch_target"] for item in batch]
    energies_targets = [item["energy_target"] for item in batch]

    durations = [item["duration"] for item in batch]
    length_text = torch.tensor([text.size(0) for text in raw_texts])
    length_mel = torch.tensor([mel.size(0) for mel in mel_targets])

    # Positional encodings
    src_pos = create_positional_encoding(length_text, max_len=length_text.max())
    mel_pos = create_positional_encoding(length_mel, max_len=length_mel.max())

    # Padding
    texts = pad_sequence(raw_texts, batch_first=True, padding_value=0)
    durations = pad_sequence(durations, batch_first=True, padding_value=0)
    mel_targets = pad_sequence(mel_targets, batch_first=True, padding_value=0)
    pitches_targets = pad_sequence(pitches_targets, batch_first=True, padding_value=0)
    energies_targets = pad_sequence(energies_targets, batch_first=True, padding_value=0)
    out = {
        "src_seq": texts,
        "mel_target": mel_targets,
        "pitch_target": pitches_targets,
        "energy_target": energies_targets,
        "duration": durations,
        "mel_pos": mel_pos,
        "src_pos": src_pos,
        "mel_max_len": length_mel.max(),
        "raw_text": raw_texts,
    }

    return out


def collate_fn(batch):
    processed_batch = reprocess_tensor(batch)
    return processed_batch
