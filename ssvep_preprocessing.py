import numpy as np
import scipy.io
from scipy.signal import iirnotch, butter, filtfilt
from scipy.stats import kurtosis, zscore
import matplotlib.pyplot as plt

def notch_filter(data, sampling_rate=250, notch_freq=50, quality_factor=30):
    """
    Apply notch filter to remove mains frequency noise (50 Hz).
    
    Args:
        data: EEG data (samples, channels)
        sampling_rate: Sampling rate in Hz
        notch_freq: Notch frequency (50 Hz for Europe, 60 Hz for US)
        quality_factor: Quality factor for the notch filter
    
    Returns:
        notch_filtered: Data after notch filtering
    """
    # Design notch filter
    b_notch, a_notch = iirnotch(notch_freq, quality_factor, fs=sampling_rate)
    
    # Apply notch filter
    notch_filtered = filtfilt(b_notch, a_notch, data, axis=0)
    
    return notch_filtered

def bandpass_filter(data, sampling_rate=250, low_freq=5, high_freq=20, order=4):
    """
    Apply bandpass filter to isolate SSVEP frequency range (5-20 Hz).
    
    Args:
        data: EEG data (samples, channels)
        sampling_rate: Sampling rate in Hz
        low_freq: Lower cutoff frequency
        high_freq: Upper cutoff frequency
        order: Filter order
    
    Returns:
        bandpass_filtered: Data after bandpass filtering
    """
    # Design bandpass filter
    nyquist = sampling_rate / 2
    low_norm = low_freq / nyquist
    high_norm = high_freq / nyquist
    b_band, a_band = butter(order, [low_norm, high_norm], btype='band')
    
    # Apply bandpass filter
    bandpass_filtered = filtfilt(b_band, a_band, data, axis=0)
    
    return bandpass_filtered

def spatial_filter(data, method='car'):
    """
    Apply spatial filtering to enhance signal-to-noise ratio.
    
    Args:
        data: EEG data (samples, channels)
        method: 'car' for Common Average Reference, 'laplacian' for Laplacian
    
    Returns:
        spatially_filtered: Data after spatial filtering
    """
    if method == 'car':
        # Common Average Reference (CAR)
        spatially_filtered = data - np.mean(data, axis=1, keepdims=True)
    elif method == 'laplacian':
        # Laplacian filtering (simplified - subtract average of neighboring channels)
        spatially_filtered = data.copy()
        for ch in range(data.shape[1]):
            # Use all other channels as neighbors (simplified)
            neighbors = np.delete(data, ch, axis=1)
            spatially_filtered[:, ch] = data[:, ch] - np.mean(neighbors, axis=1)
    else:
        raise ValueError(f"Unknown spatial filtering method: {method}")
    
    return spatially_filtered

def artifact_rejection(epoch, amplitude_threshold=100, kurtosis_threshold=5):
    """
    Reject epochs contaminated by artifacts.
    
    Args:
        epoch: Single epoch (samples, channels)
        amplitude_threshold: Maximum allowed amplitude
        kurtosis_threshold: Maximum allowed kurtosis
    
    Returns:
        is_valid: Boolean indicating if epoch should be kept
    """
    # Amplitude-based rejection
    max_amplitude = np.max(np.abs(epoch))
    if max_amplitude > amplitude_threshold:
        return False
    
    # Kurtosis-based rejection (detects outliers)
    epoch_kurtosis = kurtosis(epoch, axis=0)
    if np.any(epoch_kurtosis > kurtosis_threshold):
        return False
    
    return True

def normalize_epoch(epoch, method='zscore'):
    """
    Normalize epoch to reduce inter-trial variability.
    
    Args:
        epoch: Single epoch (samples, channels)
        method: 'zscore' for z-score normalization
    
    Returns:
        normalized: Normalized epoch
    """
    if method == 'zscore':
        normalized = zscore(epoch, axis=0)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized

def preprocess_ssvep_pipeline(data, sampling_rate=250):
    """
    Complete SSVEP preprocessing pipeline.
    
    Args:
        data: Raw EEG data (channels, samples, conditions, trials)
        sampling_rate: Sampling rate in Hz
    
    Returns:
        processed_epochs: List of processed epochs for each condition
        valid_trials: List of valid trial indices for each condition
    """
    print("=== SSVEP PREPROCESSING PIPELINE ===")
    print(f"Input data shape: {data.shape}")
    
    processed_epochs = []
    valid_trials = []
    
    for cond_idx in range(data.shape[2]):
        print(f"\nProcessing Condition {cond_idx + 1}:")
        condition_epochs = []
        condition_valid_trials = []
        
        for trial_idx in range(data.shape[3]):
            # Extract epoch
            epoch = data[:, :, cond_idx, trial_idx].T  # Shape: (samples, channels)
            
            # Step 1: Notch filter (50 Hz mains noise)
            print(f"  Trial {trial_idx + 1}: Applying notch filter...")
            notch_filtered = notch_filter(epoch, sampling_rate)
            
            # Step 2: Bandpass filter (5-20 Hz SSVEP range)
            print(f"  Trial {trial_idx + 1}: Applying bandpass filter...")
            bandpass_filtered = bandpass_filter(notch_filtered, sampling_rate)
            
            # Step 3: Spatial filtering (Common Average Reference)
            print(f"  Trial {trial_idx + 1}: Applying spatial filtering...")
            spatially_filtered = spatial_filter(bandpass_filtered, method='car')
            
            # Step 4: Artifact rejection
            print(f"  Trial {trial_idx + 1}: Checking for artifacts...")
            is_valid = artifact_rejection(spatially_filtered)
            
            if is_valid:
                # Step 5: Normalization
                print(f"  Trial {trial_idx + 1}: Normalizing...")
                normalized = normalize_epoch(spatially_filtered)
                
                condition_epochs.append(normalized)
                condition_valid_trials.append(trial_idx)
                print(f"  Trial {trial_idx + 1}: ✓ Valid")
            else:
                print(f"  Trial {trial_idx + 1}: ✗ Rejected (artifacts)")
        
        processed_epochs.append(np.array(condition_epochs))
        valid_trials.append(condition_valid_trials)
        
        print(f"Condition {cond_idx + 1}: {len(condition_epochs)}/{data.shape[3]} trials kept")
    
    return processed_epochs, valid_trials

def save_processed_data(processed_epochs, valid_trials, filename_prefix='S5'):
    """
    Save processed epochs and trial information.
    
    Args:
        processed_epochs: List of processed epochs for each condition
        valid_trials: List of valid trial indices for each condition
        filename_prefix: Prefix for saved files
    """
    print(f"\n=== SAVING PROCESSED DATA ===")
    
    for cond_idx, (epochs, trials) in enumerate(zip(processed_epochs, valid_trials)):
        if len(epochs) > 0:
            # Save processed epochs
            np.save(f'{filename_prefix}_cond{cond_idx+1}_processed.npy', epochs)
            
            # Save trial information
            np.save(f'{filename_prefix}_cond{cond_idx+1}_trials.npy', np.array(trials))
            
            print(f"Condition {cond_idx + 1}: Saved {len(epochs)} epochs")
        else:
            print(f"Condition {cond_idx + 1}: No valid epochs to save")

def main():
    """Main preprocessing pipeline for S5.mat."""
    
    # Load S5.mat
    print("Loading S5.mat...")
    mat = scipy.io.loadmat('S5.mat')
    data_struct = mat['data'][0, 0]
    eeg_data = data_struct['EEG']  # Shape: (64, 750, 4, 40)
    
    print(f"Raw data shape: {eeg_data.shape}")
    print(f"Data format: (channels, samples, conditions, trials)")
    
    # Run preprocessing pipeline
    processed_epochs, valid_trials = preprocess_ssvep_pipeline(eeg_data)
    
    # Save processed data
    save_processed_data(processed_epochs, valid_trials, 'S5')
    
    # Summary
    print(f"\n=== PREPROCESSING SUMMARY ===")
    total_epochs = sum(len(epochs) for epochs in processed_epochs)
    total_trials = eeg_data.shape[2] * eeg_data.shape[3]
    print(f"Total trials processed: {total_trials}")
    print(f"Total epochs kept: {total_epochs}")
    print(f"Epoch retention rate: {total_epochs/total_trials*100:.1f}%")
    
    for cond_idx, epochs in enumerate(processed_epochs):
        print(f"Condition {cond_idx + 1}: {len(epochs)} epochs")
    
    print("Preprocessing complete!")

if __name__ == "__main__":
    main() 