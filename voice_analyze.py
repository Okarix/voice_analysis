import sys
import os
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

def plot_analysis_results(results, output_path):
    num_plots = len(results)
    fig, axs = plt.subplots(num_plots, 1, figsize=(10, 2 * num_plots))
    plt.subplots_adjust(hspace=0.8)
    
    for i, (title, data) in enumerate(results.items()):
        axs[i].plot(data)
        axs[i].set_title(title, fontsize=8, loc='left', x=0.51)
        axs[i].tick_params(axis='both', which='major', labelsize=6)
        axs[i].title.set_fontsize(10)
        axs[i].title.set_size(8)
        axs[i].xaxis.label.set_size(6)
        axs[i].yaxis.label.set_size(6)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main(file_path):
    y, sr = load_audio(file_path)

    pitch_librosa = analyze_pitch_with_librosa(y, sr)
    timbre = analyze_timbre(y, sr)
    dynamics = analyze_dynamics(y)
    articulation = analyze_articulation(file_path)
    rhythm_tempo, rhythm_beats = analyze_rhythm(y, sr)
    breath_control = analyze_breath_control(y, sr)
    vibrato = analyze_vibrato(y, sr)

    results = {
        "Pitch (Librosa)": pitch_librosa,
        "Timbre (Spectral Centroid)": timbre[0],
        "Dynamics (RMS)": dynamics[0],
        "Articulation (Intensity)": articulation,
        "Rhythm (Tempo)": [rhythm_tempo] * len(rhythm_beats),
        "Breath Control (RMS)": breath_control[0],
        "Vibrato (Pitch)": vibrato
    }

    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')

    # Determine the next available result file number
    result_files = os.listdir('results')
    next_index = len(result_files) + 1
    output_path = f"results/analysis_results_{next_index}.png"

    plot_analysis_results(results, output_path)

    return output_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_audio.py <file_path>")
    else:
        file_path = sys.argv[1]
        main(file_path)