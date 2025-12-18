import os
import cv2
from HandTrackingModule import handDetector

detector = handDetector(detectionCon=0.7)

print("Testing hand detection on original photos...")
print("="*50)

for letter in sorted(os.listdir('train')):
    letter_path = os.path.join('train', letter)
    if not os.path.isdir(letter_path):
        continue
    
    images = [f for f in os.listdir(letter_path) if f.endswith('.jpg')]
    detected = 0
    
    for img_name in images:
        img = cv2.imread(os.path.join(letter_path, img_name))
        if img is not None:
            img_proc = detector.findHands(img, draw=False)
            lm = detector.findPosition(img_proc, draw=False)
            if len(lm) > 0:
                detected += 1
    
    print(f"{letter}: {detected}/{len(images)} detected ({len(images)-detected} FAILED)")