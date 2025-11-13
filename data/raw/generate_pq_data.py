import numpy as np
import pandas as pd
import os

# --- Configuration ---
SAMPLE_RATE = 3000  # Samples per second (Hz)
BASE_FREQUENCY = 60 # Base grid frequency (Hz)
N_SAMPLES = 200     # Number of samples per signal (e.g., ~3-4 cycles)
AMPLITUDE = 1.0     # Nominal amplitude (e.g., 1.0 p.u.)
N_FILES_PER_TYPE = 10 # Number of files to generate for each class
OUTPUT_DIR = 'data/raw'

# --- Signal Generation Functions ---

def generate_normal(n_samples=N_SAMPLES, noise_level=0.02):
    """Generates a normal sine wave with slight noise."""
    t = np.linspace(0, n_samples / SAMPLE_RATE, n_samples, endpoint=False)
    signal = AMPLITUDE * np.sin(2 * np.pi * BASE_FREQUENCY * t)
    noise = np.random.normal(0, noise_level, n_samples)
    return signal + noise

def generate_sag(n_samples=N_SAMPLES, sag_depth=0.5, duration=0.2):
    """Generates a signal with a voltage sag."""
    signal = generate_normal(n_samples)
    sag_samples = int(n_samples * duration)
    start = np.random.randint(0, n_samples - sag_samples)
    
    # Apply sag
    signal[start : start + sag_samples] *= (1 - sag_depth)
    return signal

def generate_swell(n_samples=N_SAMPLES, swell_height=0.4, duration=0.2):
    """Generates a signal with a voltage swell."""
    signal = generate_normal(n_samples)
    swell_samples = int(n_samples * duration)
    start = np.random.randint(0, n_samples - swell_samples)
    
    # Apply swell
    signal[start : start + swell_samples] *= (1 + swell_height)
    return signal

def generate_harmonics(n_samples=N_SAMPLES, harm_amplitude=0.2):
    """Generates a signal with 3rd and 5th harmonics."""
    t = np.linspace(0, n_samples / SAMPLE_RATE, n_samples, endpoint=False)
    
    # Fundamental frequency
    signal = AMPLITUDE * np.sin(2 * np.pi * BASE_FREQUENCY * t)
    
    # 3rd Harmonic
    harm_3 = harm_amplitude * np.sin(2 * np.pi * (3 * BASE_FREQUENCY) * t)
    
    # 5th Harmonic
    harm_5 = (harm_amplitude / 2) * np.sin(2 * np.pi * (5 * BASE_FREQUENCY) * t)
    
    return signal + harm_3 + harm_5 + np.random.normal(0, 0.01, n_samples)

# --- Main Script Logic ---

def create_dataset():
    """Generates and saves all signal files."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Generating synthetic PQ data in '{OUTPUT_DIR}'...")
    
    signals = []
    labels = []
    
    # Generate data
    for label in ['normal', 'sag', 'swell', 'harmonics']:
        for i in range(N_FILES_PER_TYPE):
            if label == 'normal':
                signal = generate_normal()
            elif label == 'sag':
                signal = generate_sag()
            elif label == 'swell':
                signal = generate_swell()
            elif label == 'harmonics':
                signal = generate_harmonics()
            
            signals.append(signal)
            labels.append(label)

    # Save as individual files (for the simulator)
    print(f"Saving {len(signals)} files for simulation...")
    for i, (signal, label) in enumerate(zip(signals, labels)):
        # The notebook will use the 'label' column for training
        # The pipeline will use the 'signal' column for inference
        df = pd.DataFrame({
            'signal': signal,
            'label': [label] * len(signal) # Add label to each row for training
        })
        
        filename = f"{label}_{i+1:03d}.parquet"
        filepath = os.path.join(OUTPUT_DIR, filename)
        df.to_parquet(filepath, index=False)

    print("Data generation complete.")
    print(f"Files saved in: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    create_dataset()
