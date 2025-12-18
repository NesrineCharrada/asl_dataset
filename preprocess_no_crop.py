import os
import cv2

INPUT_DIR = r"C:\Users\Expert Gaming\Desktop\asl_dataset"
OUTPUT_DIR = "processed_dataset_no_crop"

def augment(img):
    h, w = img.shape[:2]
    augs = []
    
    # Brightness
    for alpha in [0.7, 0.85, 1.0, 1.15, 1.3]:
        augs.append(cv2.convertScaleAbs(img, alpha=alpha, beta=0))
    
    # Rotations
    for angle in [-15, -10, -5, 5, 10, 15]:
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        augs.append(cv2.warpAffine(img, M, (w, h)))
    
    # Flip
    augs.append(cv2.flip(img, 1))
    
    return augs

# Clean start
if os.path.exists(OUTPUT_DIR):
    import shutil
    shutil.rmtree(OUTPUT_DIR)

os.makedirs(OUTPUT_DIR)

for split in ['train', 'validation', 'test']:
    input_split = os.path.join(INPUT_DIR, split)
    output_split = os.path.join(OUTPUT_DIR, split)
    
    if not os.path.exists(input_split):
        continue
    
    os.makedirs(output_split, exist_ok=True)
    print(f"\nProcessing {split}...")
    
    for letter in os.listdir(input_split):
        input_letter = os.path.join(input_split, letter)
        output_letter = os.path.join(output_split, letter)
        
        if not os.path.isdir(input_letter):
            continue
        
        os.makedirs(output_letter, exist_ok=True)
        
        images = [f for f in os.listdir(input_letter) if f.endswith('.jpg')]
        total = 0
        
        for img_name in images:
            img = cv2.imread(os.path.join(input_letter, img_name))
            
            if img is not None:
                # Resize without cropping
                img_resized = cv2.resize(img, (64, 64))
                
                # Save original
                cv2.imwrite(os.path.join(output_letter, img_name), img_resized)
                total += 1
                
                # Save augmented
                augs = augment(img_resized)
                base = img_name.replace('.jpg', '')
                for i, aug in enumerate(augs):
                    cv2.imwrite(os.path.join(output_letter, f"{base}_aug{i}.jpg"), aug)
                    total += 1
        
        print(f"  {letter}: {total} images")

print("\nDone! Dataset in:", OUTPUT_DIR)
print("\nNow update asl_recognizer.py to use this folder!")