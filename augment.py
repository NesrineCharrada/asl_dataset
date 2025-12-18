import cv2
import os
import numpy as np

def augment_image(img):
    augmented = []
    
    # Brightness variations
    for alpha in [0.7, 0.85, 1.15, 1.3]:
        bright = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
        augmented.append(bright)
    
    # Rotations
    h, w = img.shape[:2]
    for angle in [-15, -10, 10, 15]:
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        rotated = cv2.warpAffine(img, M, (w, h))
        augmented.append(rotated)
    
    return augmented

print("=" * 50)
print("  Data Augmentation")
print("=" * 50)

total_new = 0

for folder in ["train", "validation", "test"]:
    if not os.path.exists(folder):
        continue
    
    print(f"\nProcessing {folder}...")
    
    for letter in os.listdir(folder):
        letter_path = os.path.join(folder, letter)
        if not os.path.isdir(letter_path):
            continue
        
        images = [f for f in os.listdir(letter_path) if f.endswith('.jpg') and '_aug' not in f]
        
        for img_name in images:
            img_path = os.path.join(letter_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            augmented = augment_image(img)
            base_name = img_name.replace('.jpg', '')
            
            for i, aug_img in enumerate(augmented):
                aug_name = f"{base_name}_aug{i}.jpg"
                aug_path = os.path.join(letter_path, aug_name)
                if not os.path.exists(aug_path):
                    cv2.imwrite(aug_path, aug_img)
                    total_new += 1
        
        count = len([f for f in os.listdir(letter_path) if f.endswith('.jpg')])
        print(f"  {letter}: {count} images")

print(f"\nDone! Added {total_new} augmented images")
print("Now run: python main.py and choose 1 to retrain")