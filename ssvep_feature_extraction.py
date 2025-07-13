import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

def compute_psd(epoch, sampling_rate=250):
    """
    Compute Power Spectral Density (PSD) using FFT.
    
    Args:
        epoch: Single epoch (samples, channels)
        sampling_rate: Sampling rate in Hz
    
    Returns:
        freqs: Frequency array
        psd: Power spectral density (frequencies, channels)
    """
    # Compute FFT for each channel
    fft_vals = np.zeros_like(epoch, dtype=complex)
    for ch in range(epoch.shape[1]):
        fft_vals[:, ch] = fft(epoch[:, ch])
    
    # Get frequency array
    freqs = fftfreq(epoch.shape[0], 1/sampling_rate)
    
    # Get positive frequencies only
    pos_freqs = freqs[:len(freqs)//2]
    psd = np.abs(fft_vals[:len(freqs)//2, :])**2
    
    return pos_freqs, psd

def extract_power_at_frequency(psd, freqs, target_freq, window_size=0.5):
    """
    Extract power at target frequency and its harmonics.
    
    Args:
        psd: Power spectral density (frequencies, channels)
        freqs: Frequency array
        target_freq: Target frequency in Hz
        window_size: Frequency window around target (Hz)
    
    Returns:
        power_fundamental: Power at fundamental frequency
        power_harmonic2: Power at 2nd harmonic
        power_harmonic3: Power at 3rd harmonic
    """
    # Find frequency bin closest to target frequency
    freq_idx = np.argmin(np.abs(freqs - target_freq))
    
    # Extract power at fundamental frequency
    power_fundamental = psd[freq_idx]
    
    # Extract power at harmonics
    freq_idx_h2 = np.argmin(np.abs(freqs - 2*target_freq))
    freq_idx_h3 = np.argmin(np.abs(freqs - 3*target_freq))
    
    power_harmonic2 = psd[freq_idx_h2]
    power_harmonic3 = psd[freq_idx_h3]
    
    return power_fundamental, power_harmonic2, power_harmonic3

def compute_snr(psd, freqs, target_freq, noise_window=2):
    """
    Compute Signal-to-Noise Ratio (SNR).
    
    Args:
        psd: Power spectral density (frequencies, channels)
        freqs: Frequency array
        target_freq: Target frequency in Hz
        noise_window: Number of bins around target for noise estimation
    
    Returns:
        snr: Signal-to-noise ratio
    """
    # Find target frequency bin
    target_idx = np.argmin(np.abs(freqs - target_freq))
    
    # Extract signal power
    signal_power = psd[target_idx]
    
    # Estimate noise power from neighboring bins
    start_idx = max(0, target_idx - noise_window)
    end_idx = min(len(freqs), target_idx + noise_window + 1)
    
    # Exclude the target frequency bin from noise estimation
    noise_bins = np.concatenate([
        psd[start_idx:target_idx],
        psd[target_idx+1:end_idx]
    ])
    
    noise_power = np.mean(noise_bins, axis=0)
    
    # Compute SNR
    snr = signal_power / (noise_power + 1e-10)  # Add small constant to avoid division by zero
    
    return snr

def aggregate_occipital_features(power_fundamental, power_harmonic2, power_harmonic3, snr, occipital_channels=None):
    """
    Aggregate features across occipital channels.
    
    Args:
        power_fundamental: Power at fundamental frequency (channels)
        power_harmonic2: Power at 2nd harmonic (channels)
        power_harmonic3: Power at 3rd harmonic (channels)
        snr: Signal-to-noise ratio (channels)
        occipital_channels: List of occipital channel indices
    
    Returns:
        features: Aggregated features
    """
    if occipital_channels is None:
        # Default occipital channels (adjust based on your montage)
        occipital_channels = slice(9, 16)  # Channels 9-15
    
    # Extract occipital channel data
    occ_power_fund = power_fundamental[occipital_channels]
    occ_power_h2 = power_harmonic2[occipital_channels]
    occ_power_h3 = power_harmonic3[occipital_channels]
    occ_snr = snr[occipital_channels]
    
    # Aggregate features
    features = [
        np.mean(occ_power_fund),    # Mean power across occipital
        np.std(occ_power_fund),     # Std power across occipital
        np.max(occ_power_fund),     # Max power across occipital
        np.mean(occ_power_h2),      # Mean 2nd harmonic power
        np.mean(occ_power_h3),      # Mean 3rd harmonic power
        np.mean(occ_snr),           # Mean SNR across occipital
        np.std(occ_snr),            # Std SNR across occipital
        np.max(occ_snr)             # Max SNR across occipital
    ]
    
    return np.array(features)

def extract_ssvep_features(epoch, stimulus_freqs, sampling_rate=250):
    """
    Extract SSVEP features from a single epoch.
    
    Args:
        epoch: Single epoch (samples, channels)
        stimulus_freqs: List of stimulus frequencies
        sampling_rate: Sampling rate in Hz
    
    Returns:
        features: Feature vector for this epoch
    """
    # Step 1: Compute PSD
    freqs, psd = compute_psd(epoch, sampling_rate)
    
    # Step 2: Extract features for each stimulus frequency
    all_features = []
    
    for freq in stimulus_freqs:
        # Extract power at fundamental and harmonics
        power_fund, power_h2, power_h3 = extract_power_at_frequency(psd, freqs, freq)
        
        # Compute SNR
        snr = compute_snr(psd, freqs, freq)
        
        # Aggregate features across occipital channels
        freq_features = aggregate_occipital_features(power_fund, power_h2, power_h3, snr)
        
        all_features.extend(freq_features)
    
    return np.array(all_features)

def normalize_features(feature_matrix, method='zscore'):
    """
    Normalize features to make them comparable across trials.
    
    Args:
        feature_matrix: Feature matrix (epochs, features)
        method: Normalization method ('zscore', 'minmax', 'robust')
    
    Returns:
        normalized_features: Normalized feature matrix
    """
    if method == 'zscore':
        # Z-score normalization
        normalized = (feature_matrix - np.mean(feature_matrix, axis=0)) / (np.std(feature_matrix, axis=0) + 1e-10)
    elif method == 'minmax':
        # Min-max normalization
        min_vals = np.min(feature_matrix, axis=0)
        max_vals = np.max(feature_matrix, axis=0)
        normalized = (feature_matrix - min_vals) / (max_vals - min_vals + 1e-10)
    elif method == 'robust':
        # Robust normalization using median and MAD
        median_vals = np.median(feature_matrix, axis=0)
        mad_vals = np.median(np.abs(feature_matrix - median_vals), axis=0)
        normalized = (feature_matrix - median_vals) / (mad_vals + 1e-10)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized

def feature_extraction_pipeline(processed_epochs, stimulus_freqs, filename_prefix='S5'):
    """
    Complete SSVEP feature extraction pipeline.
    
    Args:
        processed_epochs: List of processed epochs for each condition
        stimulus_freqs: List of stimulus frequencies
        filename_prefix: Prefix for saved files
    """
    print("=== SSVEP FEATURE EXTRACTION PIPELINE ===")
    
    all_features = []
    all_labels = []
    
    for cond_idx, epochs in enumerate(processed_epochs):
        print(f"\nProcessing Condition {cond_idx + 1}: {len(epochs)} epochs")
        
        condition_features = []
        
        for epoch_idx, epoch in enumerate(epochs):
            # Extract features from this epoch
            features = extract_ssvep_features(epoch, stimulus_freqs)
            condition_features.append(features)
            
            if (epoch_idx + 1) % 10 == 0:
                print(f"  Processed {epoch_idx + 1}/{len(epochs)} epochs")
        
        # Convert to numpy array
        condition_features = np.array(condition_features)
        
        # Use raw features (no normalization to preserve discriminative information)
        # condition_features_normalized = normalize_features(condition_features, method='zscore')
        
        # Save features for this condition
        np.save(f'{filename_prefix}_cond{cond_idx+1}_features.npy', condition_features)
        
        # Add to overall dataset
        all_features.append(condition_features)
        all_labels.extend([cond_idx] * len(epochs))
        
        print(f"Condition {cond_idx + 1}: Extracted {len(condition_features)} feature vectors")
        print(f"Feature shape: {condition_features.shape}")
    
    # Combine all conditions
    all_features = np.vstack(all_features)
    all_labels = np.array(all_labels)
    
    # Save combined feature matrix and labels
    np.save(f'{filename_prefix}_all_features.npy', all_features)
    np.save(f'{filename_prefix}_all_labels.npy', all_labels)
    
    print(f"\n=== FEATURE EXTRACTION SUMMARY ===")
    print(f"Total feature matrix shape: {all_features.shape}")
    print(f"Features per epoch: {all_features.shape[1]}")
    print(f"Total epochs: {all_features.shape[0]}")
    print(f"Stimulus frequencies: {stimulus_freqs} Hz")
    
    return all_features, all_labels

def main():
    """Main feature extraction pipeline for S5.mat."""
    
    # Load processed epochs
    print("Loading processed epochs...")
    processed_epochs = []
    for cond_idx in range(4):
        epochs = np.load(f'S5_cond{cond_idx+1}_processed.npy')
        processed_epochs.append(epochs)
        print(f"Condition {cond_idx + 1}: {len(epochs)} epochs")
    
    # Define stimulus frequencies (based on S5.mat analysis)
    stimulus_freqs = [12.0, 12.0, 11.7, 12.0]  # Hz - 4 conditions
    
    # Run feature extraction pipeline
    features, labels = feature_extraction_pipeline(processed_epochs, stimulus_freqs, 'S5')
    
    print("Feature extraction complete!")

if __name__ == "__main__":
    main() 