# AURA SSVEP Classification System

This module provides a complete SSVEP (Steady-State Visual Evoked Potential) classification system for the AURA (Agentic Unified Robotics Architecture) project. It can classify up to 4 different visual stimuli from raw EEG data.

## Overview

SSVEP is a brain-computer interface technique that detects brain responses to visual stimuli flickering at specific frequencies. When a user focuses on a flickering object, their brain generates electrical activity at the same frequency, which can be detected in EEG signals.

## Features

- **Multi-class Classification**: Supports up to 4 different visual stimuli (classes 1-4)
- **Advanced Signal Processing**: 
  - Notch filtering for power line interference removal
  - Bandpass filtering (1-40 Hz)
  - Power Spectral Density (PSD) feature extraction
  - SSVEP-specific feature extraction around target frequencies
- **Machine Learning Pipeline**:
  - Support Vector Machine (SVM) classifier with RBF kernel
  - Hyperparameter optimization using GridSearchCV
  - Feature scaling and normalization
  - Cross-validation for robust performance evaluation
- **Real-time Ready**: Optimized for real-time classification
- **Model Persistence**: Save and load trained models
- **Visualization**: Confusion matrix plotting and performance metrics

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the following packages:
- numpy >= 1.21.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

## Usage

### Basic Usage

```python
from ssvep import SSVEPProcessor

# Initialize processor
processor = SSVEPProcessor(sampling_rate=250)  # 250 Hz sampling rate

# Load your EEG data (CSV format with labels in last column)
X, y = processor.load_eeg_data('your_eeg_data.csv')

# Define target frequencies (should match your visual stimuli)
target_frequencies = [7.5, 10.0, 12.0, 15.0]  # Hz

# Train the model
accuracy = processor.train_model(X, y, target_frequencies)

# Save the trained model
processor.save_model('aura_ssvep_model.pkl')

# Load model for real-time use
processor.load_model('aura_ssvep_model.pkl')

# Classify new EEG data
predicted_class, confidence = processor.predict(new_eeg_data, target_frequencies)
```

### Running Examples

```bash
# Run the main SSVEP system with synthetic data
python ssvep.py

# Run interactive examples
python ssvep_example.py
```

### Data Format

Your EEG data should be in CSV format with:
- Each row represents one EEG sample
- Each column represents one EEG channel
- The last column contains the class labels (1-4)

Example:
```csv
channel1,channel2,channel3,channel4,channel5,channel6,channel7,channel8,label
0.123,0.456,0.789,0.234,0.567,0.890,0.345,0.678,1
0.234,0.567,0.890,0.345,0.678,0.901,0.456,0.789,2
...
```

## Key Components

### SSVEPProcessor Class

The main class that handles all SSVEP processing:

- `__init__(sampling_rate, notch_freq)`: Initialize with EEG sampling rate
- `load_eeg_data(filepath)`: Load EEG data from CSV file
- `preprocess_data(data, target_frequencies)`: Complete preprocessing pipeline
- `train_model(X, y, target_frequencies)`: Train the classification model
- `predict(data, target_frequencies)`: Classify new EEG data
- `save_model(filepath)`: Save trained model
- `load_model(filepath)`: Load trained model

### Signal Processing

1. **Notch Filter**: Removes 50/60 Hz power line interference
2. **Bandpass Filter**: Filters EEG to 1-40 Hz range
3. **PSD Feature Extraction**: Extracts power in frequency bands (delta, theta, alpha, beta, gamma)
4. **SSVEP-Specific Features**: Extracts features around target frequencies

### Machine Learning

- **Feature Extraction**: Combines PSD and SSVEP-specific features
- **Feature Scaling**: StandardScaler for normalization
- **Classifier**: SVM with RBF kernel
- **Hyperparameter Tuning**: GridSearchCV for optimal parameters
- **Cross-validation**: 5-fold CV for robust evaluation

## Integration with AURA

### Visual Stimuli Setup

For optimal SSVEP detection, set up your visual stimuli with these frequencies:
- **Class 1**: 7.5 Hz (Red object)
- **Class 2**: 10.0 Hz (Blue object)
- **Class 3**: 12.0 Hz (Green object)
- **Class 4**: 15.0 Hz (Yellow object)

### Real-time Integration

```python
# In your AURA main loop
def process_ssvep_selection(eeg_data):
    """Process SSVEP selection in real-time."""
    predicted_class, confidence = processor.predict(eeg_data, target_frequencies)
    
    if confidence > 0.7:  # Confidence threshold
        object_mapping = {
            1: "red_object",
            2: "blue_object", 
            3: "green_object",
            4: "yellow_object"
        }
        return object_mapping.get(predicted_class, None)
    
    return None  # No clear selection
```

### Performance Optimization

For real-time performance:
1. Use pre-trained models
2. Process data in chunks (2-4 seconds)
3. Implement confidence thresholds
4. Use efficient feature extraction

## Performance Metrics

The system provides:
- **Accuracy**: Overall classification accuracy
- **Confusion Matrix**: Detailed classification results
- **Classification Report**: Precision, recall, F1-score per class
- **Confidence Scores**: Prediction confidence for each classification

## Troubleshooting

### Common Issues

1. **Low Accuracy**:
   - Check EEG data quality
   - Verify target frequencies match visual stimuli
   - Increase training data size
   - Adjust preprocessing parameters

2. **High Noise**:
   - Ensure good electrode contact
   - Check for power line interference
   - Verify sampling rate settings

3. **Model Loading Errors**:
   - Ensure all dependencies are installed
   - Check file paths and permissions

### Data Quality Tips

- Ensure good electrode contact
- Minimize movement artifacts
- Use consistent visual stimuli
- Record sufficient training data (200+ samples per class)
- Maintain consistent experimental conditions

## Advanced Usage

### Custom Feature Extraction

```python
# Add custom features
def extract_custom_features(data):
    # Your custom feature extraction
    return features

# Integrate with SSVEPProcessor
processor.custom_feature_extractor = extract_custom_features
```

### Different Classifiers

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Use different classifiers
processor.model = RandomForestClassifier(n_estimators=100)
# or
processor.model = MLPClassifier(hidden_layer_sizes=(100, 50))
```

## Research Applications

This SSVEP system is suitable for:
- Brain-computer interfaces
- Assistive technology
- Gaming applications
- Research studies
- Rehabilitation systems

## License

This code is part of the AURA project and follows the project's licensing terms.

## Contributing

For improvements or bug reports, please refer to the main AURA project guidelines. 