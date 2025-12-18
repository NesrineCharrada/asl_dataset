import cv2
import os
from HandTrackingModule import handDetector

detector = handDetector(detectionCon=0.7)

print("=" * 50)
print("  ASL Data Collection Tool")
print("=" * 50)

letter = input("Which letter to collect? (e.g., A): ").strip().upper()

folder = os.path.join("train", letter)
os.makedirs(folder, exist_ok=True)

existing = len([f for f in os.listdir(folder) if f.endswith('.jpg')])
count = existing

print(f"\nCollecting photos for letter: {letter}")
print(f"Existing photos: {count}")
print(f"Saving to: {folder}")
print("\nControls:")
print("  SPACE = Capture photo")
print("  Q = Quit")
print("\nShow your hand and press SPACE to capture...")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    frame = detector.findHands(frame)
    landmarks = detector.findPosition(frame)
    
    # Show info
    cv2.putText(frame, f"Letter: {letter}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Photos: {count}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if len(landmarks) > 0:
        cv2.putText(frame, "Hand OK - Press SPACE", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No hand detected", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.putText(frame, "SPACE=Capture | Q=Quit", (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    
    cv2.imshow("Data Collection", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        if len(landmarks) > 0:
            count += 1
            filename = f"{letter}_{count}.jpg"
            filepath = os.path.join(folder, filename)
            cv2.imwrite(filepath, frame)
            print(f"Saved: {filename}")
        else:
            print("No hand detected - not saved")

cap.release()
cv2.destroyAllWindows()
print(f"\nDone! Total photos for {letter}: {count}")