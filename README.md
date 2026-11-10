# 🌍 Nature Image Classification Using Deep Learning

## 📌 Project Overview

This project implements a **multi-class image classification system** using Deep Learning and Transfer Learning to identify different types of natural and urban environments.

The model uses **EfficientNetB0**, a pre-trained Convolutional Neural Network (CNN), to classify uploaded images into six categories.

The trained model is integrated into a **Streamlit web application**, allowing users to upload an image and receive a predicted category with its confidence score.

## 🎯 Image Categories

The model classifies images into the following six categories:

1. Buildings
2. Forest
3. Glacier
4. Mountain
5. Sea
6. Street

## 🧠 Model Architecture

This project explores multiple CNN architectures and transfer learning approaches:

- Custom CNN
- ResNet50
- MobileNetV2
- EfficientNetB0

**Final Selected Model: EfficientNetB0**

EfficientNetB0 was selected based on its validation and test performance.

### Model Configuration

- Input image size: 180 × 180 pixels
- Model: EfficientNetB0
- Framework: TensorFlow / Keras
- Learning approach: Transfer Learning
- Output: 6 image classes

## 📊 Model Performance

| Metric | Result |
|---|---:|
| Validation Accuracy | 93.51% |
| Test Accuracy | 92.03% |

The final model was evaluated on 3,000 test images.

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- EfficientNetB0
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Pillow (PIL)
- Streamlit
- Jupyter Notebook

## 📁 Project Structure

```text
Nature-Image-Classification/
│
├── app.py
├── best_efficientnetb0.keras
├── natue_imageCV_predict.ipynb
├── requirements.txt
├── .gitignore
│
└── nature_classification_sample_images/
    ├── buildings_sample.jpg
    ├── forest_sample.jpg
    ├── glacier_sample.jpg
    ├── mountain_sample.jpg
    ├── sea_sample.jpg
    └── street_sample.jpg
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Hari-smile/Nature-Image-Classification.git
```

### 2. Navigate to the project directory

```bash
cd Nature-Image-Classification
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

Windows:

```powershell
venv\Scripts\activate
```

### 5. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
python -m streamlit run app.py
```

Open the application in your browser:

```text
http://localhost:8501
```

## 🖥️ Application Features

- Upload JPG, JPEG and PNG images.
- Predict image categories using EfficientNetB0.
- Display the predicted class.
- Show prediction confidence.
- Display prediction probabilities for all six categories.
- Interactive Streamlit user interface.

## 🚀 Future Enhancements

- Deploy the application using Streamlit Community Cloud.
- Add more image categories.
- Improve model performance through further experimentation.
- Explore real-time image classification using a webcam.

## 👨‍💻 Author

**Hari Haran**

Aspiring AI Engineer | Data Science & Machine Learning

GitHub: [Hari-smile](https://github.com/Hari-smile)
