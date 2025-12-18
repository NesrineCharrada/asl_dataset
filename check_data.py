import os

print("Images per letter in TRAIN folder:")
print("="*40)

train_path = "processed_dataset/train"

for letter in sorted(os.listdir(train_path)):
    letter_path = os.path.join(train_path, letter)
    if os.path.isdir(letter_path):
        count = len([f for f in os.listdir(letter_path) if f.endswith('.jpg')])
        print(f"{letter}: {count} images")

print("\n" + "="*40)
print("If any letter has 0 or very few images,")
print("that's the problem!")