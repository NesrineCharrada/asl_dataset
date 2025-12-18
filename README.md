# American Sign Hand Recognition System

<div align="center">

![ASL Banner](https://img.shields.io/badge/ASL-Hand_Recognition-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python)
![ML](https://img.shields.io/badge/Machine_Learning-Random_Forest-orange?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Accuracy-77.72%25-success?style=for-the-badge)

**AI-Powered American Sign Language Recognition for Improved Communication Accessibility**

*By Nesrine Charrada | AI Clinic Project*

</div>

---

## 📋 Overview

This project develops an intelligent system to recognize American Sign Language (ASL) hand signs for letters A-Z using machine learning. The system processes images, trains and compares 7 different classification models, and provides real-time webcam predictions to support sign language recognition and improve communication accessibility.

### 🎯 Project Goals

- **Accessibility**: Bridge communication gaps for the deaf and hard-of-hearing community
- **Real-time Recognition**: Enable instant ASL letter detection through webcam
- **High Accuracy**: Achieve reliable predictions with optimized machine learning models
- **Modular Design**: Create an extensible system for future enhancements

---

## ✨ Key Features

- ✅ **21 ASL Letter Recognition** (A-Z, excluding J and Z due to motion requirements)
- ✅ **Real-time Webcam Detection** with visual feedback
- ✅ **77.72% Test Accuracy** using Random Forest classifier
- ✅ **7 ML Models Compared** for optimal performance
- ✅ **Debug Mode** with top 3 predictions and confidence scores
- ✅ **Color-coded Confidence Indicators**:
  - 🟢 Green: >70% confidence
  - 🟡 Yellow: 50-70% confidence
  - 🔴 Red: <50% confidence

---

## 🗂️ Project Structure

\\\
asl_dataset/
├── train/                         # Training images (organized by letter)
├── validation/                    # Validation images
├── test/                          # Test images
├── models/                        # Saved trained models
├── HandTrackingModule.py         # Hand detection using MediaPipe
├── feature_extractor.py          # Feature extraction (84 features)
├── recognizer.py                 # ASL recognition engine
├── preprocess_no_crop.py         # Data augmentation script
├── train_models.py               # Model training and comparison
├── asl_recognizer.py             # Main application
├── vscode_backup/                # VS Code settings backup
├── my_extensions.txt             # VS Code extensions list
├── reinstall_extensions.ps1      # Extension reinstall script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
\\\

---

## 🔬 Technical Approach

### Data Collection

**Method:**
- Images captured via smartphone (960 x 1280 px resolution)
- Multiple participants for hand variation diversity
- Consistent lighting and background conditions

**Dataset Composition:**
- **21 letter classes** (A-Z, excluding J and Z)
- **10 images per letter** initially collected
- Organized into train, validation, and test folders

### Data Preprocessing & Augmentation

To increase dataset robustness and prevent overfitting, we applied multiple augmentation techniques:

**Augmentation Techniques:**
- 🔆 **Brightness adjustment** - Varying lighting conditions
- 🔄 **Rotation** - Different hand orientations
- ↔️ **Horizontal flipping** - Mirror images

**Results:**
- **13x data increase** per original image
- **Final dataset**: 920 feature samples
- **Split**: 736 training (80%) / 184 testing (20%)

### Hand Tracking & Feature Extraction

**Technology**: MediaPipe Hands
- Detects **21 hand landmarks** per frame
- Extracts **84 features** including:
  - X, Y coordinates of each landmark
  - Distances between key points
  - Angles between finger segments
  - Normalized values for scale invariance

### Machine Learning Models

We trained and compared **7 different algorithms** to find the optimal model:

| Model | Train Acc | Test Acc | F1 Score | Train Time |
|-------|-----------|----------|----------|------------|
| 🏆 **Random Forest** | 99.05% | **77.72%** | **0.7606** | **0.29s** |
| Gradient Boosting | 100.00% | 75.54% | 0.7446 | 42.48s |
| SVM (RBF) | 91.85% | 72.83% | 0.7069 | 0.05s |
| SVM (Linear) | 93.75% | 71.74% | 0.6994 | 0.05s |
| Naive Bayes | 72.42% | 67.39% | 0.6677 | 0.01s |
| K-Nearest Neighbors | 100.00% | 66.30% | 0.6547 | 0.00s |
| Decision Tree | 95.11% | 64.67% | 0.6391 | 0.04s |

### Why Random Forest? 🏆

Random Forest emerged as the clear winner for several critical reasons:

1. **Best Test Accuracy**: 77.72% - highest among all models
2. **Strong F1 Score**: 0.7606 - balanced precision and recall
3. **Fast Training**: 0.29s (145x faster than Gradient Boosting)
4. **Reliable Real-time Performance**: Consistent predictions in live webcam testing
5. **Low Overfitting**: Small gap between train (99.05%) and test (77.72%) accuracy

> **Note**: Models like K-Nearest Neighbors and Gradient Boosting showed perfect training accuracy but failed to generalize well in real-time testing, demonstrating overfitting.

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Webcam (for real-time recognition)
- VS Code (optional, for development)

### Installation Steps

1. **Clone the repository**
   \\\ash
   git clone https://github.com/NesrineCharrada/asl_dataset.git
   cd asl_dataset
   \\\

2. **Create and activate virtual environment**
   \\\powershell
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   \\\

3. **Install dependencies**
   \\\ash
   pip install -r requirements.txt
   \\\

### Required Libraries

\\\
opencv-python
mediapipe
numpy
scikit-learn
pillow
matplotlib
seaborn
\\\

---

## 💻 Usage

### Training the Model

\\\ash
# Preprocess and augment data
python preprocess_no_crop.py

# Train all models and compare
python train_models.py
\\\

### Real-time Recognition

\\\ash
# Run the ASL recognizer with webcam
python asl_recognizer.py

# With debug mode (shows top 3 predictions)
python asl_recognizer.py --debug
\\\

**Controls:**
- Press **'q'** to quit
- Press **'d'** to toggle debug mode
- Show your hand sign to the camera for recognition

---

## 📊 Performance Analysis

### Training vs Testing Accuracy
- High training accuracy across most models
- Random Forest shows best generalization to unseen data
- Minimal overfitting compared to other ensemble methods

### Speed vs Accuracy Trade-off
- Random Forest offers optimal balance
- K-Nearest Neighbors is fastest but least accurate in practice
- Gradient Boosting is most accurate in training but slowest (42.48s)

### Confusion Matrix Insights
- Strong performance on distinct letters (A, B, L, O)
- Some confusion between similar hand shapes (e.g., M/N, S/T)
- Opportunities for improvement with additional training data

---

## 🎯 Applications & Use Cases

### 🎓 Educational Applications
- **ASL Learning Tools**: Interactive applications for students learning sign language
- **Practice & Feedback**: Real-time feedback for ASL learners
- **Classroom Integration**: Teaching aids for educators

### ♿ Accessibility Solutions
- **Communication Interfaces**: Enable deaf individuals to communicate with non-signers
- **Public Service Kiosks**: Accessible information systems
- **Emergency Communication**: Quick communication in critical situations

### 🎮 Technology Integration
- **Gesture-controlled Applications**: Control devices using ASL signs
- **Gaming Interfaces**: Novel game control mechanisms
- **Smart Home Control**: Accessibility features for home automation

### 🔬 Research Applications
- **Human-Computer Interaction**: Advanced gesture recognition research
- **Computer Vision Studies**: Hand tracking and pose estimation
- **Machine Learning Benchmarks**: Dataset for ML algorithm comparison

---

## 🔧 VS Code Setup

This project includes complete VS Code configuration backup for development consistency.

### Restore VS Code Extensions

\\\powershell
.\reinstall_extensions.ps1
\\\

### Restore VS Code Settings

\\\powershell
copy .\vscode_backup\settings.json \C:\Users\Expert Gaming\AppData\Roaming\Code\User\
\\\

### Included Extensions
- **Azure Tools for Containers** - Container development
- **Python** - Full Python language support
- **Python Debugger** - Advanced debugging capabilities
- **Pylance** - Fast Python language server
- **Python Environments** - Virtual environment management
- **Remote Containers** - Container development support

---

## 🔮 Future Enhancements

- [ ] Expand to all 26 letters including dynamic signs (J, Z)
- [ ] Add word and phrase recognition
- [ ] Implement sentence building and translation
- [ ] Mobile application development (iOS/Android)
- [ ] Multi-language sign language support
- [ ] Deep learning models (CNN, LSTM) for improved accuracy
- [ ] Cloud deployment for web-based access
- [ ] Integration with video conferencing platforms

---

## 🤝 Contributing

Contributions are welcome! This project represents a commitment to creating technology that serves human values of connection, communication, and inclusion.

### How to Contribute

1. Fork the repository
2. Create a feature branch (\git checkout -b feature/AmazingFeature\)
3. Commit your changes (\git commit -m 'Add some AmazingFeature'\)
4. Push to the branch (\git push origin feature/AmazingFeature\)
5. Open a Pull Request

---

## 📄 License

[Add your license here - e.g., MIT License]

---

## 👤 Author

**Nesrine Charrada**

- GitHub: [@NesrineCharrada](https://github.com/NesrineCharrada)
- Project: AI Clinic Project

---

## 🙏 Acknowledgments

- **MediaPipe** team for the excellent hand tracking solution
- **Scikit-learn** for comprehensive machine learning tools
- ASL community for inspiration and purpose
- All contributors and testers

---

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please open an issue on GitHub or contact the author directly.

---

<div align="center">

**Made with ❤️ for improved communication accessibility**

⭐ Star this repository if you find it helpful!

</div>
