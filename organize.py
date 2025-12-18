import os
import shutil

folders = ["train", "validation", "test"]

for folder in folders:
    if not os.path.exists(folder):
        continue
    
    print(f"Organizing {folder}...")
    
    for filename in os.listdir(folder):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        # Get letter from filename (e.g., "A.jpg" -> "A", "C_1.jpg" -> "C")
        letter = filename.split('.')[0].split('_')[0]
        
        # Create subfolder for this letter
        subfolder = os.path.join(folder, letter)
        os.makedirs(subfolder, exist_ok=True)
        
        # Move file into subfolder
        src = os.path.join(folder, filename)
        dst = os.path.join(subfolder, filename)
        shutil.move(src, dst)
        print(f"  Moved {filename} -> {letter}/")

print("\nDone! Your photos are now organized.")