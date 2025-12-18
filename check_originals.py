import os

print("\nORIGINAL train folder (non-processed):")
print("="*40)

for letter in sorted(os.listdir('train')):
    letter_path = os.path.join('train', letter)
    if os.path.isdir(letter_path):
        count = len([f for f in os.listdir(letter_path) if f.endswith('.jpg')])
        print(f"{letter}: {count} images")