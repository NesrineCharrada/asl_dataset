import os
import cv2
from HandTrackingModule import handDetector

INPUT_DIR = r"C:\Users\Expert Gaming\Desktop\asl_dataset"
OUTPUT_DIR = "processed_dataset"

BRIGHTNESS = [0.7, 0.85, 1.0, 1.15, 1.3]
ROTATION_ANGLES = [-15, -10, -5, 5, 10, 15]
FLIP_MODES = [1]

def apply_augmentations(image):
    augmented = []
    h, w = image.shape[:2]
    
    for alpha in BRIGHTNESS:
        bright = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        augmented.append(bright)
    
    for angle in ROTATION_ANGLES:
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        rotated = cv2.warpAffine(image, M, (w, h))
        augmented.append(rotated)
    
    for flip_mode in FLIP_MODES:
        flipped = cv2.flip(image, flip_mode)
        augmented.append(flipped)
    
    return augmented

detector = handDetector(detectionCon=0.7)
os.makedirs(OUTPUT_DIR, exist_ok=True)

for split in ['train', 'validation', 'test']:
    split_input = os.path.join(INPUT_DIR, split)
    split_output = os.path.join(OUTPUT_DIR, split)
    
    if not os.path.exists(split_input):
        continue
    
    os.makedirs(split_output, exist_ok=True)
    print(f"\nProcessing {split}...")
    
    for letter in os.listdir(split_input):
        letter_input = os.path.join(split_input, letter)
        letter_output = os.path.join(split_output, letter)
        
        if not os.path.isdir(letter_input):
            continue
        
        os.makedirs(letter_output, exist_ok=True)
        
        for img_name in os.listdir(letter_input):
            if not img_name.endswith('.jpg'):
                continue
            
            img_path = os.path.join(letter_input, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            img_proc = detector.findHands(img, draw=False)
            landmarks = detector.findPosition(img_proc, draw=False)
            
            if len(landmarks) > 0:
                x_coords = [pt[1] for pt in landmarks]
                y_coords = [pt[2] for pt in landmarks]
                x_min = max(min(x_coords) - 20, 0)
                x_max = min(max(x_coords) + 20, img.shape[1])
                y_min = max(min(y_coords) - 20, 0)
                y_max = min(max(y_coords) + 20, img.shape[0])
                
                roi = img[y_min:y_max, x_min:x_max]
                if roi.size > 0:
                    roi_resized = cv2.resize(roi, (64, 64))
                    
                    # Save original
                    cv2.imwrite(os.path.join(letter_output, img_name), roi_resized)
                    
                    # Save augmented
                    augmented = apply_augmentations(roi_resized)
                    base = img_name.replace('.jpg', '')
                    for i, aug in enumerate(augmented):
                        aug_name = f"{base}_aug{i}.jpg"
                        cv2.imwrite(os.path.join(letter_output, aug_name), aug)
        
        count = len([f for f in os.listdir(letter_output) if f.endswith('.jpg')])
        print(f"  {letter}: {count} images")

print("\nDone! Check processed_dataset folder")