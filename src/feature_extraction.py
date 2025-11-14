import numpy as np
import pywt
from scipy.stats import entropy

def calculate_statistics(coeffs):
    """Calculates energy and entropy from wavelet coefficients."""
    # Energy: Sum of squares of coefficients
    energy = np.sum(np.square(coeffs))
    
    # Entropy: Shannon entropy of the squared coefficients (as a probability distribution)
    # We square them to get a measure of energy distribution
    prob_dist = np.square(coeffs) / np.sum(np.square(coeffs))
    
    # Filter out zeros to avoid log(0)
    prob_dist = prob_dist[prob_dist > 0] 
    
    shannon_entropy = entropy(prob_dist, base=2)
    
    return energy, shannon_entropy

def extract_wavelet_features(signal, wavelet='db4', level=4):
    """
    Extracts statistical features (energy, entropy) from DWT coefficients.
    
    Parameters:
    - signal (list or np.array): The input signal.
    - wavelet (str): The wavelet to use (e.g., 'db4').
    - level (int): The decomposition level.
    
    Returns:
    - list: A flat list of statistical features (energy, entropy) 
            from each coefficient array.
    """
    # Ensure signal is a numpy array
    signal = np.array(signal)
    
    # Perform multilevel wavelet decomposition
    # Returns [cA_n, cD_n, cD_n-1, ..., cD_1]
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    features = []
    for c in coeffs:
        energy, sh_entropy = calculate_statistics(c)
        features.extend([energy, sh_entropy])
        
    return features
