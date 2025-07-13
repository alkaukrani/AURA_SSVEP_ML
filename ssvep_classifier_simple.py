import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class SSVEPClassifier:
    """Comprehensive SSVEP classifier with multiple algorithms and decision smoothing."""
    
    def __init__(self, classifier_type='lda', smoothing_window=5, confidence_threshold=0.6):
        """
        Initialize SSVEP classifier.
        
        Args:
            classifier_type: 'lda' or 'svm'
            smoothing_window: Number of consecutive predictions for smoothing
            confidence_threshold: Minimum confidence for reliable prediction
        """
        self.classifier_type = classifier_type
        self.smoothing_window = smoothing_window
        self.confidence_threshold = confidence_threshold
        
        # Initialize classifier
        if classifier_type == 'lda':
            self.classifier = LinearDiscriminantAnalysis()
        elif classifier_type == 'svm':
            self.classifier = SVC(probability=True, random_state=42)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        # Initialize decision smoothing
        self.prediction_history = deque(maxlen=smoothing_window)
        self.confidence_history = deque(maxlen=smoothing_window)
        
        # Performance tracking
        self.training_accuracy = None
        self.validation_accuracy = None
        self.test_accuracy = None
        
    def load_data(self, features_file, labels_file):
        """Load features and labels."""
        print("Loading SSVEP features and labels...")
        self.features = np.load(features_file)
        self.labels = np.load(labels_file)
        
        print(f"Feature matrix shape: {self.features.shape}")
        print(f"Labels shape: {self.labels.shape}")
        print(f"Label distribution: {np.bincount(self.labels)}")
        
        return self.features, self.labels
    
    def preprocess_data(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets with stratification."""
        print("\n=== DATA PREPROCESSING ===")
        
        # Split data ensuring proportional representation of each class
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.labels, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=self.labels
        )
        
        # Scale features (important for SVM)
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        print(f"Training labels: {np.bincount(self.y_train)}")
        print(f"Test labels: {np.bincount(self.y_test)}")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def optimize_hyperparameters(self, cv_folds=5):
        """Optimize classifier hyperparameters using grid search."""
        print(f"\n=== HYPERPARAMETER OPTIMIZATION ===")
        
        if self.classifier_type == 'svm':
            # Define parameter grid for SVM
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'linear']
            }
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                SVC(probability=True, random_state=42),
                param_grid,
                cv=cv_folds,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(self.X_train_scaled, self.y_train)
            
            # Update classifier with best parameters
            self.classifier = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            self.best_score = grid_search.best_score_
            
            print(f"Best parameters: {self.best_params}")
            print(f"Best CV score: {self.best_score:.3f}")
            
        else:
            # LDA doesn't have hyperparameters to optimize
            print("LDA classifier - no hyperparameters to optimize")
            self.best_params = None
            self.best_score = None
    
    def train_classifier(self):
        """Train the classifier."""
        print(f"\n=== TRAINING {self.classifier_type.upper()} CLASSIFIER ===")
        
        # Train classifier
        self.classifier.fit(self.X_train_scaled, self.y_train)
        
        # Training accuracy
        self.training_accuracy = self.classifier.score(self.X_train_scaled, self.y_train)
        print(f"Training accuracy: {self.training_accuracy:.3f}")
        
        return self.classifier
    
    def cross_validate(self, cv_folds=5):
        """Perform cross-validation."""
        print(f"\n=== CROSS-VALIDATION ({cv_folds}-fold) ===")
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.classifier, 
            self.X_train_scaled, 
            self.y_train, 
            cv=cv_folds,
            scoring='accuracy'
        )
        
        self.validation_accuracy = cv_scores.mean()
        self.validation_std = cv_scores.std()
        
        print(f"Cross-validation accuracy: {self.validation_accuracy:.3f} (+/- {self.validation_std * 2:.3f})")
        print(f"CV scores: {cv_scores}")
        
        return cv_scores
    
    def evaluate_test_set(self):
        """Evaluate classifier on test set."""
        print(f"\n=== TEST SET EVALUATION ===")
        
        # Predictions
        self.y_pred = self.classifier.predict(self.X_test_scaled)
        self.y_prob = self.classifier.predict_proba(self.X_test_scaled)
        
        # Test accuracy
        self.test_accuracy = accuracy_score(self.y_test, self.y_pred)
        print(f"Test accuracy: {self.test_accuracy:.3f}")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(self.y_test, self.y_pred))
        
        # Confusion matrix
        self.cm = confusion_matrix(self.y_test, self.y_pred)
        print(f"\nConfusion Matrix:")
        print(self.cm)
        
        return self.test_accuracy
    
    def predict_with_smoothing(self, feature_vector):
        """
        Predict class with decision smoothing.
        
        Args:
            feature_vector: Single feature vector
            
        Returns:
            predicted_class: Predicted class
            confidence: Prediction confidence
            smoothed_prediction: Smoothed prediction
        """
        # Scale feature vector
        feature_scaled = self.scaler.transform(feature_vector.reshape(1, -1))
        
        # Get prediction and probability
        prediction = self.classifier.predict(feature_scaled)[0]
        probabilities = self.classifier.predict_proba(feature_scaled)[0]
        confidence = np.max(probabilities)
        
        # Add to history
        self.prediction_history.append(prediction)
        self.confidence_history.append(confidence)
        
        # Apply smoothing if enough history
        if len(self.prediction_history) >= self.smoothing_window:
            # Majority vote for smoothing
            smoothed_prediction = self._majority_vote(list(self.prediction_history))
        else:
            smoothed_prediction = prediction
        
        return prediction, confidence, smoothed_prediction
    
    def _majority_vote(self, predictions):
        """Apply majority vote smoothing."""
        from collections import Counter
        return Counter(predictions).most_common(1)[0][0]
    
    def monitor_performance(self, window_size=10):
        """Monitor classification performance over time."""
        if len(self.prediction_history) < window_size:
            return None
        
        recent_confidence = list(self.confidence_history)[-window_size:]
        avg_confidence = np.mean(recent_confidence)
        
        # Alert if performance is poor
        if avg_confidence < self.confidence_threshold:
            print(f"⚠️  WARNING: Low confidence detected ({avg_confidence:.3f})")
            print("Consider adjusting stimulus parameters or retraining model")
        
        return avg_confidence
    
    def get_performance_summary(self):
        """Get comprehensive performance summary."""
        summary = {
            'classifier_type': self.classifier_type,
            'training_accuracy': self.training_accuracy,
            'validation_accuracy': self.validation_accuracy,
            'test_accuracy': self.test_accuracy,
            'best_params': self.best_params,
            'best_cv_score': self.best_score,
            'smoothing_window': self.smoothing_window,
            'confidence_threshold': self.confidence_threshold
        }
        
        print(f"\n=== PERFORMANCE SUMMARY ===")
        print(f"Classifier: {summary['classifier_type'].upper()}")
        print(f"Training Accuracy: {summary['training_accuracy']:.3f}")
        print(f"Validation Accuracy: {summary['validation_accuracy']:.3f}")
        print(f"Test Accuracy: {summary['test_accuracy']:.3f}")
        print(f"Smoothing Window: {summary['smoothing_window']}")
        print(f"Confidence Threshold: {summary['confidence_threshold']}")
        
        if summary['best_params']:
            print(f"Best Parameters: {summary['best_params']}")
        
        return summary

def main():
    """Main ML pipeline for SSVEP classification."""
    
    print("=== SSVEP MACHINE LEARNING PIPELINE ===")
    
    # Initialize classifiers
    lda_classifier = SSVEPClassifier(classifier_type='lda', smoothing_window=5, confidence_threshold=0.6)
    svm_classifier = SSVEPClassifier(classifier_type='svm', smoothing_window=5, confidence_threshold=0.6)
    
    # Load data
    lda_classifier.load_data('S5_all_features.npy', 'S5_all_labels.npy')
    svm_classifier.load_data('S5_all_features.npy', 'S5_all_labels.npy')
    
    # Preprocess data
    lda_classifier.preprocess_data()
    svm_classifier.preprocess_data()
    
    # Train and evaluate LDA
    print("\n" + "="*50)
    print("LDA CLASSIFIER")
    print("="*50)
    
    lda_classifier.optimize_hyperparameters()
    lda_classifier.train_classifier()
    lda_classifier.cross_validate()
    lda_classifier.evaluate_test_set()
    lda_summary = lda_classifier.get_performance_summary()
    
    # Train and evaluate SVM
    print("\n" + "="*50)
    print("SVM CLASSIFIER")
    print("="*50)
    
    svm_classifier.optimize_hyperparameters()
    svm_classifier.train_classifier()
    svm_classifier.cross_validate()
    svm_classifier.evaluate_test_set()
    svm_summary = svm_classifier.get_performance_summary()
    
    # Compare classifiers
    print("\n" + "="*50)
    print("CLASSIFIER COMPARISON")
    print("="*50)
    
    comparison_data = {
        'Classifier': ['LDA', 'SVM'],
        'Training Accuracy': [lda_summary['training_accuracy'], svm_summary['training_accuracy']],
        'Validation Accuracy': [lda_summary['validation_accuracy'], svm_summary['validation_accuracy']],
        'Test Accuracy': [lda_summary['test_accuracy'], svm_summary['test_accuracy']]
    }
    
    print("Performance Comparison:")
    for i, classifier in enumerate(comparison_data['Classifier']):
        print(f"{classifier}:")
        print(f"  Training: {comparison_data['Training Accuracy'][i]:.3f}")
        print(f"  Validation: {comparison_data['Validation Accuracy'][i]:.3f}")
        print(f"  Test: {comparison_data['Test Accuracy'][i]:.3f}")
    
    # Determine best classifier
    best_classifier = 'LDA' if lda_summary['test_accuracy'] > svm_summary['test_accuracy'] else 'SVM'
    print(f"\nBest Classifier: {best_classifier}")
    
    # Test real-time prediction simulation
    print("\n" + "="*50)
    print("REAL-TIME PREDICTION SIMULATION")
    print("="*50)
    
    # Simulate real-time predictions using test data
    best_classifier_obj = lda_classifier if best_classifier == 'LDA' else svm_classifier
    
    print("Simulating real-time predictions with decision smoothing:")
    for i in range(min(10, len(best_classifier_obj.X_test_scaled))):  # Test first 10 samples
        feature_vector = best_classifier_obj.X_test_scaled[i]
        prediction, confidence, smoothed = best_classifier_obj.predict_with_smoothing(feature_vector)
        actual = best_classifier_obj.y_test[i]
        
        print(f"Sample {i+1}: Predicted={prediction}, Actual={actual}, Confidence={confidence:.3f}, Smoothed={smoothed}")
        
        # Monitor performance
        avg_conf = best_classifier_obj.monitor_performance()
        if avg_conf is not None:
            print(f"  Average confidence (last 10): {avg_conf:.3f}")
    
    return lda_classifier, svm_classifier

if __name__ == "__main__":
    lda_classifier, svm_classifier = main() 