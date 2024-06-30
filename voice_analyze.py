import librosa
import matplotlib.pyplot as plt
import numpy as np
import parselmouth
from parselmouth.praat import call

def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

def analyze_pitch_with_librosa(y, sr):
    # Extracting pitches using librosa's pitch tracking
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    pitch = [pitches[magnitudes[:, i].argmax(), i] for i in range(magnitudes.shape[1])]
    return pitch

def analyze_timbre(y, sr):
    # Spectral centroid as a simple timbre feature
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return spectral_centroid

def analyze_dynamics(y):
    # Root mean square (RMS) energy
    rms = librosa.feature.rms(y=y)
    return rms

def analyze_articulation(file_path):
    # Using Praat via parselmouth for intensity analysis
    snd = parselmouth.Sound(file_path)
    intensity = call(snd, "To Intensity", 75, 0.0, "yes")
    intensity_values = intensity.values.T
    return intensity_values

def analyze_rhythm(y, sr):
    # Beat tracking using librosa
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    return tempo, beats

def analyze_breath_control(y, sr):
    # Using RMS energy to analyze breath control
    rms = librosa.feature.rms(y=y)
    return rms

def analyze_vibrato(y, sr):
    # Vibrato analysis using librosa's pitch tracking
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    pitch = [pitches[magnitudes[:, i].argmax(), i] for i in range(magnitudes.shape[1])]
    pitch = np.array(pitch)
    pitch = pitch[pitch > 0]  # Filter out non-positive pitches
    return pitch

def plot_analysis_results(results):
    fig, axs = plt.subplots(len(results), 1, figsize=(10, 20))
    plt.subplots_adjust(hspace=0.5)  # Increase space between plots
    for i, (title, data) in enumerate(results.items()):
        axs[i].plot(data)
        axs[i].set_title(title, fontsize=10)
        axs[i].tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()
    plt.show()

def main(file_path):
    y, sr = load_audio(file_path)

    # Analyze pitch using librosa
    pitch_librosa = analyze_pitch_with_librosa(y, sr)

    # Analyze other characteristics
    timbre = analyze_timbre(y, sr)
    dynamics = analyze_dynamics(y)
    articulation = analyze_articulation(file_path)
    rhythm_tempo, rhythm_beats = analyze_rhythm(y, sr)
    breath_control = analyze_breath_control(y, sr)
    vibrato = analyze_vibrato(y, sr)

    # Prepare results for plotting
    results = {
        "Pitch (Librosa)": pitch_librosa,
        "Timbre (Spectral Centroid)": timbre[0],
        "Dynamics (RMS)": dynamics[0],
        "Articulation (Intensity)": articulation,
        "Rhythm (Tempo)": [rhythm_tempo] * len(rhythm_beats),
        "Breath Control (RMS)": breath_control[0],
        "Vibrato (Pitch)": vibrato
    }

    # Plot results
    plot_analysis_results(results)

# Example usage
file_path = "../FULL/male/male1/long_tones/straight/m1_long_straight_a.wav"
main(file_path)
