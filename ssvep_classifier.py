import numpy as np
import scipy.io
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class SSVEPClassifier:
    def __init__(self, classifier_type='lda', smoothing_window=3):
        """
        SSVEP Classifier with real-time prediction capabilities.
        
        Args:
            classifier_type: 'lda' or 'svm'
            smoothing_window: Number of consecutive predictions for majority voting
        """
        self.classifier_type = classifier_type
        self.smoothing_window = smoothing_window
        self.prediction_history = deque(maxlen=smoothing_window)
        self.scaler = StandardScaler()
        
        if classifier_type == 'lda':
            self.classifier = LinearDiscriminantAnalysis()
        elif classifier_type == 'svm':
            self.classifier = SVC(kernel='rbf', probability=True)
        else:
            raise ValueError("classifier_type must be 'lda' or 'svm'")
    
    def train(self, X, y, test_size=0.2, random_state=42, optimize_hyperparams=True):
        """
        Train the classifier with cross-validation and hyperparameter optimization.
        """
        print(f"Training {self.classifier_type.upper()} classifier...")
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.classifier, X_train_scaled, y_train, cv=5)
        print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Hyperparameter optimization for SVM
        if self.classifier_type == 'svm' and optimize_hyperparams:
            print("Optimizing SVM hyperparameters...")
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            }
            grid_search = GridSearchCV(SVC(kernel='rbf', probability=True), 
                                     param_grid, cv=5, scoring='accuracy')
            grid_search.fit(X_train_scaled, y_train)
            self.classifier = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        
        # Train final model
        self.classifier.fit(X_train_scaled, y_train)
        
        # Evaluate on test set
        y_pred = self.classifier.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Test accuracy: {test_accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(np.array(cm), test_accuracy)
        
        # Store test data for real-time simulation
        self.X_test = X_test_scaled
        self.y_test = y_test
        
        return test_accuracy
    
    def predict_single_epoch(self, features):
        """
        Predict class for a single epoch with confidence score.
        """
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get prediction and probability
        prediction = self.classifier.predict(features_scaled)[0]
        probabilities = self.classifier.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)
        
        return prediction, confidence, probabilities
    
    def predict_with_smoothing(self, features):
        """
        Predict class with decision smoothing using majority voting.
        """
        prediction, confidence, probabilities = self.predict_single_epoch(features)
        
        # Add to prediction history
        self.prediction_history.append(prediction)
        
        # Majority vote if we have enough predictions
        if len(self.prediction_history) == self.smoothing_window:
            smoothed_prediction = max(set(self.prediction_history), 
                                   key=list(self.prediction_history).count)
            return smoothed_prediction, confidence, probabilities
        else:
            return prediction, confidence, probabilities
    
    def simulate_real_time(self, num_epochs=20):
        """
        Simulate real-time classification using test data.
        """
        print(f"\nSimulating real-time classification for {num_epochs} epochs...")
        
        # Reset prediction history
        self.prediction_history.clear()
        
        correct_predictions = 0
        total_predictions = 0
        
        for i in range(min(num_epochs, len(self.X_test))):
            features = self.X_test[i]
            true_label = self.y_test[i]
            
            # Predict with smoothing
            prediction, confidence, probabilities = self.predict_with_smoothing(features)
            
            # Check if prediction is correct
            is_correct = prediction == true_label
            if is_correct:
                correct_predictions += 1
            total_predictions += 1
            
            # Print results
            print(f"Epoch {i+1}: True={true_label}, Predicted={prediction}, "
                  f"Confidence={confidence:.3f}, Correct={is_correct}")
            
            # Monitor confidence and suggest adaptations
            if confidence < 0.6:
                print(f"  ⚠️  Low confidence ({confidence:.3f}) - consider longer stimulus duration")
        
        real_time_accuracy = correct_predictions / total_predictions
        print(f"\nReal-time accuracy: {real_time_accuracy:.3f}")
        return real_time_accuracy
    
    def plot_confusion_matrix(self, cm, accuracy):
        """Plot confusion matrix."""
        # Convert to numpy array if needed
        cm_array = np.array(cm)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['10Hz', '12Hz', '15Hz', '20Hz'],
                   yticklabels=['10Hz', '12Hz', '15Hz', '20Hz'])
        plt.title(f'Confusion Matrix (Accuracy: {accuracy:.3f})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('ssvep_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, feature_names=None):
        """Plot feature importance (for LDA)."""
        if self.classifier_type == 'lda':
            plt.figure(figsize=(12, 6))
            coef = self.classifier.coef_
            
            if feature_names is None:
                stimulus_freqs = [10, 12, 15, 20]
                feature_names = []
                for freq in stimulus_freqs:
                    feature_names.extend([
                        f'{freq}Hz_fund', f'{freq}Hz_harm2', f'{freq}Hz_harm3',
                        f'{freq}Hz_snr', f'{freq}Hz_max_power', f'{freq}Hz_max_snr'
                    ])
            
            # Plot coefficients for each class
            for i in range(coef.shape[0]):
                plt.subplot(2, 2, i+1)
                plt.bar(range(len(feature_names)), coef[i])
                plt.title(f'Class {i+1} ({stimulus_freqs[i]}Hz) Feature Weights')
                plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
                plt.ylabel('Weight')
            
            plt.tight_layout()
            plt.savefig('ssvep_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()

def main():
    """Main function to run SSVEP classification."""
    
    # Load features and labels
    print("Loading SSVEP features...")
    features = np.load('S11_all_features.npy')
    labels = np.load('S11_all_labels.npy')
    
    print(f"Feature matrix shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Train LDA classifier
    print("\n" + "="*50)
    print("TRAINING LDA CLASSIFIER")
    print("="*50)
    lda_classifier = SSVEPClassifier(classifier_type='lda', smoothing_window=3)
    lda_accuracy = lda_classifier.train(features, labels)
    
    # Train SVM classifier
    print("\n" + "="*50)
    print("TRAINING SVM CLASSIFIER")
    print("="*50)
    svm_classifier = SSVEPClassifier(classifier_type='svm', smoothing_window=3)
    svm_accuracy = svm_classifier.train(features, labels)
    
    # Compare results
    print("\n" + "="*50)
    print("CLASSIFICATION RESULTS COMPARISON")
    print("="*50)
    print(f"LDA Accuracy: {lda_accuracy:.3f}")
    print(f"SVM Accuracy: {svm_accuracy:.3f}")
    
    # Choose best classifier
    if lda_accuracy > svm_accuracy:
        best_classifier = lda_classifier
        print(f"LDA performs better - using LDA for real-time simulation")
    else:
        best_classifier = svm_classifier
        print(f"SVM performs better - using SVM for real-time simulation")
    
    # Real-time simulation
    print("\n" + "="*50)
    print("REAL-TIME CLASSIFICATION SIMULATION")
    print("="*50)
    real_time_accuracy = best_classifier.simulate_real_time(num_epochs=20)
    
    # Plot feature importance (for LDA)
    if best_classifier.classifier_type == 'lda':
        print("\nPlotting feature importance...")
        best_classifier.plot_feature_importance()
    
    print(f"\nFinal Results:")
    print(f"Best classifier: {best_classifier.classifier_type.upper()}")
    print(f"Test accuracy: {best_classifier.classifier.score(best_classifier.X_test, best_classifier.y_test):.3f}")
    print(f"Real-time accuracy: {real_time_accuracy:.3f}")

if __name__ == "__main__":
    main() 