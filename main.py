from asl_recognizer import ASLRecognizer

def main():
    print("=" * 50)
    print("  ASL Hand Sign Recognition System")
    print("=" * 50)
    
    recognizer = ASLRecognizer()
    
    while True:
        print("\nMenu:")
        print("1. Train model")
        print("2. Run webcam detection")
        print("3. Exit")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            recognizer.train_model()
        elif choice == "2":
            recognizer.run_webcam()
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()