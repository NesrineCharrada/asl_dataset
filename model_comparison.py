import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import time
from feature_extractor import HandFeatureExtractor

class ModelComparison:
    def __init__(self, dataset_path="processed_dataset_no_crop"):
        self.dataset_path = dataset_path
        self.feature_extractor = HandFeatureExtractor()
        self.classes = self._detect_classes()
        self.models = {}
        self.results = {}
        
        print(f"Initialized with {len(self.classes)} classes: {self.classes}")
    
    def _detect_classes(self):
        classes = set()
        for split in ["train", "validation", "test"]:
            path = os.path.join(self.dataset_path, split)
            if os.path.exists(path):
                for folder in os.listdir(path):
                    if os.path.isdir(os.path.join(path, folder)):
                        classes.add(folder)
        return sorted(list(classes))
    
    def load_dataset(self):
        X, y = [], []
        print("Loading dataset...")
        
        for split in ["train", "validation", "test"]:
            path = os.path.join(self.dataset_path, split)
            if not os.path.exists(path):
                continue
            
            for cls in self.classes:
                folder = os.path.join(path, cls)
                if not os.path.exists(folder):
                    continue
                
                for fname in os.listdir(folder):
                    if fname.endswith('.jpg'):
                        img = cv2.imread(os.path.join(folder, fname))
                        if img is not None:
                            features = self.feature_extractor.extract_features(img)
                            if not np.all(features == 0):
                                X.append(features)
                                y.append(cls)
        
        print(f"Loaded {len(X)} samples")
        return np.array(X), np.array(y)
    
    def prepare_models(self):
        """Initialize all models to compare"""
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'SVM (RBF)': SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                random_state=42
            ),
            'SVM (Linear)': SVC(
                kernel='linear',
                C=1,
                random_state=42
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=15,
                min_samples_split=5,
                random_state=42
            ),
            'Naive Bayes': GaussianNB()
        }
        
        print(f"\nPrepared {len(self.models)} models for comparison")
    
    def train_and_evaluate_all(self):
        """Train and evaluate all models"""
        print("\n" + "="*70)
        print("TRAINING AND EVALUATING ALL MODELS")
        print("="*70)
        
        # Load data
        X, y = self.load_dataset()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\nTraining samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        # Train and evaluate each model
        for model_name, model in self.models.items():
            print(f"\n{'-'*70}")
            print(f"Training: {model_name}")
            print(f"{'-'*70}")
            
            # Training time
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Prediction time
            start_time = time.time()
            y_pred = model.predict(X_test)
            prediction_time = time.time() - start_time
            
            # Metrics
            train_acc = accuracy_score(y_train, model.predict(X_train))
            test_acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Store results
            self.results[model_name] = {
                'model': model,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'f1_score': f1,
                'training_time': training_time,
                'prediction_time': prediction_time,
                'predictions': y_pred,
                'y_test': y_test
            }
            
            print(f"Training Accuracy: {train_acc*100:.2f}%")
            print(f"Test Accuracy: {test_acc*100:.2f}%")
            print(f"F1 Score: {f1:.4f}")
            print(f"Training Time: {training_time:.2f}s")
            print(f"Prediction Time: {prediction_time:.4f}s")
        
        print("\n" + "="*70)
        print("ALL MODELS TRAINED AND EVALUATED")
        print("="*70)
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        print("\n" + "="*70)
        print("MODEL COMPARISON REPORT")
        print("="*70)
        
        # Create comparison table
        print(f"\n{'Model':<25} {'Train Acc':<12} {'Test Acc':<12} {'F1 Score':<12} {'Train Time':<12}")
        print("-"*73)
        
        for model_name, results in sorted(self.results.items(), 
                                         key=lambda x: x[1]['test_accuracy'], 
                                         reverse=True):
            print(f"{model_name:<25} "
                  f"{results['train_accuracy']*100:>10.2f}% "
                  f"{results['test_accuracy']*100:>10.2f}% "
                  f"{results['f1_score']:>10.4f}  "
                  f"{results['training_time']:>10.2f}s")
        
        # Best model
        best_model = max(self.results.items(), key=lambda x: x[1]['test_accuracy'])
        print(f"\n🏆 BEST MODEL: {best_model[0]}")
        print(f"   Test Accuracy: {best_model[1]['test_accuracy']*100:.2f}%")
    
    def plot_comparison_charts(self):
        """Generate comparison visualizations"""
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Accuracy Comparison
        ax1 = plt.subplot(2, 3, 1)
        models = list(self.results.keys())
        train_accs = [self.results[m]['train_accuracy']*100 for m in models]
        test_accs = [self.results[m]['test_accuracy']*100 for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax1.bar(x - width/2, train_accs, width, label='Train Acc', color='skyblue')
        ax1.bar(x + width/2, test_accs, width, label='Test Acc', color='orange')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Training vs Testing Accuracy')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. F1 Score Comparison
        ax2 = plt.subplot(2, 3, 2)
        f1_scores = [self.results[m]['f1_score'] for m in models]
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        ax2.barh(models, f1_scores, color=colors)
        ax2.set_xlabel('F1 Score')
        ax2.set_title('F1 Score Comparison')
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. Training Time Comparison
        ax3 = plt.subplot(2, 3, 3)
        train_times = [self.results[m]['training_time'] for m in models]
        ax3.bar(models, train_times, color='coral')
        ax3.set_ylabel('Time (seconds)')
        ax3.set_title('Training Time Comparison')
        ax3.set_xticklabels(models, rotation=45, ha='right')
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Overfitting Analysis
        ax4 = plt.subplot(2, 3, 4)
        overfitting = [(self.results[m]['train_accuracy'] - self.results[m]['test_accuracy'])*100 
                      for m in models]
        colors = ['red' if x > 10 else 'yellow' if x > 5 else 'green' for x in overfitting]
        ax4.bar(models, overfitting, color=colors)
        ax4.axhline(y=5, color='orange', linestyle='--', label='Warning (5%)')
        ax4.axhline(y=10, color='red', linestyle='--', label='High (10%)')
        ax4.set_ylabel('Gap (%)')
        ax4.set_title('Overfitting Analysis (Train-Test Gap)')
        ax4.set_xticklabels(models, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
        # 5. Speed vs Accuracy
        ax5 = plt.subplot(2, 3, 5)
        for i, model_name in enumerate(models):
            ax5.scatter(self.results[model_name]['prediction_time']*1000,
                       self.results[model_name]['test_accuracy']*100,
                       s=200, alpha=0.6, label=model_name)
        ax5.set_xlabel('Prediction Time (ms)')
        ax5.set_ylabel('Test Accuracy (%)')
        ax5.set_title('Speed vs Accuracy Trade-off')
        ax5.legend(fontsize=8)
        ax5.grid(alpha=0.3)
        
        # 6. Best Model Confusion Matrix
        ax6 = plt.subplot(2, 3, 6)
        best_model_name = max(self.results.items(), key=lambda x: x[1]['test_accuracy'])[0]
        y_test = self.results[best_model_name]['y_test']
        y_pred = self.results[best_model_name]['predictions']
        
        cm = confusion_matrix(y_test, y_pred, labels=self.classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.classes, yticklabels=self.classes, ax=ax6)
        ax6.set_title(f'Best Model: {best_model_name}\nConfusion Matrix')
        ax6.set_xlabel('Predicted')
        ax6.set_ylabel('True')
        
        plt.tight_layout()
        plt.savefig('model_comparison_report.png', dpi=300, bbox_inches='tight')
        print("\n📊 Comparison charts saved: model_comparison_report.png")
        plt.show()
    
    def save_best_model(self):
        """Save the best performing model"""
        best_model_name = max(self.results.items(), 
                             key=lambda x: x[1]['test_accuracy'])[0]
        best_model = self.results[best_model_name]['model']
        
        with open('best_asl_model.pkl', 'wb') as f:
            pickle.dump({
                'model': best_model,
                'model_name': best_model_name,
                'classes': self.classes,
                'test_accuracy': self.results[best_model_name]['test_accuracy']
            }, f)
        
        print(f"\n💾 Best model saved: {best_model_name}")
        print(f"   Test Accuracy: {self.results[best_model_name]['test_accuracy']*100:.2f}%")

def main():
    print("="*70)
    print("ASL RECOGNITION - MULTI-MODEL COMPARISON")
    print("="*70)
    
    comparison = ModelComparison()
    comparison.prepare_models()
    comparison.train_and_evaluate_all()
    comparison.generate_comparison_report()
    comparison.plot_comparison_charts()
    comparison.save_best_model()
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  1. model_comparison_report.png - Visual comparison charts")
    print("  2. best_asl_model.pkl - Best performing model")
    print("\nUse these results for your presentation!")

if __name__ == "__main__":
    main()