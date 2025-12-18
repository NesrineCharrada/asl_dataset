import cv2
import os
from HandTrackingModule import handDetector

detector = handDetector(detectionCon=0.7)

print("=" * 50)
print("  TEST Data Collection")
print("=" * 50)

letter = input("Which letter to collect? (e.g., A): ").strip().upper()

folder = os.path.join("test", letter)
os.makedirs(folder, exist_ok=True)

existing = len([f for f in os.listdir(folder) if f.endswith('.jpg')])
count = existing

print(f"\nCollecting TEST photos for: {letter}")
print(f"Existing photos: {count}")
print(f"Goal: 2-3 photos")
print("\nSPACE = Capture | Q = Quit\n")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    frame = detector.findHands(frame)
    landmarks = detector.findPosition(frame)
    
    cv2.putText(frame, f"TEST - Letter: {letter}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Photos: {count}/3", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    if len(landmarks) > 0:
        cv2.putText(frame, "Hand OK - Press SPACE", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No hand detected", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("Test Collection", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' ') and len(landmarks) > 0:
        count += 1
        filename = f"{letter}_{count}.jpg"
        cv2.imwrite(os.path.join(folder, filename), frame)
        print(f"Saved: {filename}")

cap.release()
cv2.destroyAllWindows()
print(f"\nDone! {letter} has {count} test photos")