# 👁️ CIFAR-10 Deep Learning Object Recognition System

A production-ready, end-to-end image classification pipeline built with **TensorFlow** and **Keras**. This project implements a custom **Convolutional Neural Network (CNN)** architecture to classify images into 10 distinct categories with high accuracy. It features a modular codebase, real-time data augmentation, and an interactive web interface powered by **Gradio** for immediate inference.

## 🚀 Key Features

* **Modular Architecture:** Clean separation of concerns (Data Loading, Model Building, Training, Evaluation).
* **Custom CNN Design:** Optimized deep network with `BatchNormalization`, `Dropout`, and `MaxPooling` for robust feature extraction.
* **Data Augmentation:** Real-time image transformations (rotation, shift, flip) to prevent overfitting and improve generalization.
* **Comprehensive Evaluation:** Generates Confusion Matrix, Classification Reports, and Accuracy/Loss curves.
* **Interactive Deployment:** User-friendly web UI for testing the model with custom images.

## 🛠️ Tech Stack

* **Core:** Python 3.12, TensorFlow 2.x, Keras
* **Interface:** Gradio (Web UI)
* **Data Processing:** NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn
* **Metrics:** Scikit-Learn

## 📂 Project Structure

```text
NesneTanimaProjesi/
├── 📂 models/                # Stores the trained model artifacts (.h5)
├── 📂 src/                   # Core application logic
│   ├── 📄 data_loader.py     # Data fetching, pre-processing & augmentation
│   ├── 📄 model_builder.py   # CNN architecture definition
│   ├── 📄 train.py           # Training loop & model saving orchestration
│   ├── 📄 evaluate.py        # Performance visualization & metrics
│   └── 📄 utils.py           # Helper functions (seed, directory management)
├── 📄 app.py                 # Entry point for Gradio Web Interface
├── 📄 requirements.txt       # Project dependencies
└── 📄 README.md              # Project documentation
```

## ⚙️ Installation & Setup

Follow these steps to set up the environment and run the project locally.

### 1. Clone the Repository
```bash
git clone [https://github.com/HNurA/NesneTanimaProjesi.git]
cd NesneTanimaProjesi
```
### 2. Create Virtual Environment
It is recommended to use Python 3.10 or 3.12.
```bash
# Windows
py -3.12 -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```
### 3. Install Dependencies
```bash
pip install tensorflow numpy matplotlib seaborn gradio scikit-learn
```

## 🏃‍♂️ Usage

### Phase 1: Training the Model
Run the training script to download the dataset, train the CNN, and save the model.

```bash
python src/train.py
```
* The system will automatically download CIFAR-10 data.
* Training runs for 15 epochs (configurable).
* Best model is saved to models/cifar10_model.h5.
* Evaluation charts will be displayed after training.

### Phase 2: Deployment (Web Interface)
Launch the Gradio interface to test the model.

```bash
python app.py
```
* Click the local URL (e.g., http://127.0.0.1:7860) or the public URL generated in the terminal.
* Upload an image (e.g., a plane or a cat) to see the prediction scores.

## 📊 Model Performance

The model is evaluated on the test set using the following metrics:

* **Accuracy:** Overall percentage of correct predictions.
* **Confusion Matrix:** Heatmap showing misclassification patterns between classes.
* **Class-wise Precision & Recall:** Detailed performance analysis for each of the 10 categories.

## 📝 License

This project is open-source and available under the MIT License.
