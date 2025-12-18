import os
import cv2
import numpy as np
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from feature_extractor import HandFeatureExtractor

class ASLRecognizer:
    def __init__(self, dataset_path="processed_dataset_no_crop"):
        self.dataset_path = dataset_path
        self.model = None
        self.feature_extractor = HandFeatureExtractor()
        self.classes = self._detect_classes()
        print(f"Classes: {self.classes}")

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
        total = 0
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
                                total += 1
        print(f"Loaded {total} images")
        return np.array(X), np.array(y)
    
    def train_model(self):
        print("\nLoading dataset...")
        X, y = self.load_dataset()
        
        if len(X) == 0:
            print("ERROR: No images found!")
            return False
        
        print(f"Total samples: {len(X)}")
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"Training on {len(X_train)} samples...")
        print(f"Testing on {len(X_test)} samples...")
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        print("\nCalculating accuracy...")
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        print("\n" + "="*50)
        print(f"Training Accuracy: {train_acc*100:.2f}%")
        print(f"Test Accuracy: {test_acc*100:.2f}%")
        print("="*50)
        
        with open('asl_model.pkl', 'wb') as f:
            pickle.dump({'model': self.model, 'classes': self.classes}, f)
        print("\nModel saved!")
        return True
    
    def load_model(self):
        try:
            with open('asl_model.pkl', 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.classes = data['classes']
            print("Model loaded!")
            return True
        except:
            return False
    
    def predict(self, img):
        if self.model is None:
            return None
        features = self.feature_extractor.extract_features(img)
        if np.all(features == 0):
            return None
        pred = self.model.predict([features])[0]
        prob = np.max(self.model.predict_proba([features])[0])
        return pred, prob
    
    def run_webcam(self):
        if self.model is None and not self.load_model():
            print("Train model first!")
            return
        
        cap = cv2.VideoCapture(0)
        history = []
        
        print("\nWebcam started! Press Q to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame = self.feature_extractor.detector.findHands(frame)
            
            result = self.predict(frame)
            if result:
                sign, conf = result
                if conf > 0.3:
                    history.append(sign)
                    if len(history) > 10:
                        history.pop(0)
                
                if history:
                    stable = Counter(history).most_common(1)[0][0]
                    color = (0,255,0) if conf>0.7 else (0,255,255) if conf>0.5 else (0,100,255)
                    cv2.putText(frame, f"{stable}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 5)
                    cv2.putText(frame, f"{conf:.2f}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            cv2.imshow("ASL", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()