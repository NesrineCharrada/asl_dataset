import numpy as np
from HandTrackingModule import handDetector

class HandFeatureExtractor:
    def __init__(self):
        self.detector = handDetector(detectionCon=0.8, maxHands=1)
        
    def extract_features(self, img):
        img = self.detector.findHands(img, draw=False)
        landmarks = self.detector.findPosition(img, draw=False)
        
        if len(landmarks) < 21:
            return np.zeros(84)
            
        lm_array = np.array(landmarks)[:, 1:]
        
        x_min, y_min = np.min(lm_array, axis=0)
        x_max, y_max = np.max(lm_array, axis=0)
        
        x_range = max(1, x_max - x_min)
        y_range = max(1, y_max - y_min)
        
        normalized_lm = (lm_array - [x_min, y_min]) / [x_range, y_range]
        raw_features = normalized_lm.flatten()
        
        engineered_features = []
        palm = normalized_lm[0]
        
        fingertips = [4, 8, 12, 16, 20]
        for tip in fingertips:
            dist = np.linalg.norm(normalized_lm[tip] - palm)
            engineered_features.append(dist)
        
        for i in range(len(fingertips)-1):
            dist = np.linalg.norm(normalized_lm[fingertips[i]] - normalized_lm[fingertips[i+1]])
            engineered_features.append(dist)
        
        for finger in range(5):
            base = finger * 4 + 1
            for joint in range(3):
                p1 = normalized_lm[base + joint]
                p2 = normalized_lm[base + joint + 1]
                angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
                engineered_features.append(angle)
        
        all_features = np.concatenate([raw_features, np.array(engineered_features)])
        return all_features