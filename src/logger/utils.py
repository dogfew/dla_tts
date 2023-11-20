import io
import matplotlib.pyplot as plt


def plot_spectrogram_to_buf(spectrogram_tensor, name=None):
    plt.figure(figsize=(20, 5))
    plt.imshow(spectrogram_tensor)
    plt.title(name)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return buf


def plot_spectrogram_and_pitch_and_energy_to_buf(
    spectrogram, pitch, energy, pitch_color="orange", energy_color="purple"
):
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.imshow(spectrogram, aspect="auto", origin="lower")
    pitch_normalized = pitch / 6.76
    energy_normalized = energy / 5.76
    height = spectrogram.shape[0]
    pitch_scaled = pitch_normalized * height
    energy_scaled = energy_normalized * height
    pitch_scaled = pitch_scaled
    energy_scaled = energy_scaled

    ax.plot(pitch_scaled, color=pitch_color, linewidth=2)
    ax.plot(energy_scaled, color=energy_color, linewidth=2)

    ax_energy = ax.twinx()
    ax_energy.set_ylim(0, 5.76)
    ax_energy.set_ylabel("Energy (log1p)", color=energy_color)
    ax_energy.tick_params(axis="y", labelcolor=energy_color)

    ax_pitch = ax.twinx()
    ax_pitch.spines["right"].set_position(("outward", 60))
    ax_pitch.set_ylim(0, 6.76)
    ax_pitch.set_ylabel("Pitch (log1p)", color=pitch_color)
    ax_pitch.tick_params(axis="y", labelcolor=pitch_color)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return buf
