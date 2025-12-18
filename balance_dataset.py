import os
import random

TARGET_IMAGES = 60  # Target number of images per class

train_path = "processed_dataset/train"

print("Balancing dataset...")
print("="*50)

for letter in os.listdir(train_path):
    letter_path = os.path.join(train_path, letter)
    if not os.path.isdir(letter_path):
        continue
    
    images = [f for f in os.listdir(letter_path) if f.endswith('.jpg')]
    current_count = len(images)
    
    if current_count > TARGET_IMAGES:
        # Keep original images (without _aug in name)
        originals = [f for f in images if '_aug' not in f]
        augmented = [f for f in images if '_aug' in f]
        
        # Calculate how many augmented to keep
        to_keep = TARGET_IMAGES - len(originals)
        
        if to_keep > 0:
            # Randomly select augmented images to keep
            random.shuffle(augmented)
            keep_augmented = augmented[:to_keep]
            delete_augmented = augmented[to_keep:]
        else:
            # Too many originals, keep all originals
            delete_augmented = augmented
        
        # Delete excess
        for img in delete_augmented:
            os.remove(os.path.join(letter_path, img))
        
        new_count = len(originals) + min(to_keep, len(augmented)) if to_keep > 0 else len(originals)
        print(f"{letter}: {current_count} -> {new_count} images (deleted {len(delete_augmented)})")
    else:
        print(f"{letter}: {current_count} images (no change needed)")

print("\n" + "="*50)
print("Dataset balanced!")
print("Run 'python check_data.py' to verify")