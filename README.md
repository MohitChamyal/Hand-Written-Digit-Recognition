# 🧠 Handwritten Digit Recognition

This project is designed to recognize handwritten digits (0-9) using a deep learning-based web application. It uses a Convolutional Neural Network (CNN) model trained with TensorFlow/Keras and deployed with a Flask-powered frontend.

---

## 📊 Dataset

The dataset used is the **MNIST** (Modified National Institute of Standards and Technology) dataset, which consists of 28x28 grayscale images of handwritten digits. It is widely used for training image classification models.

- **Features**: 28x28 pixel grayscale images of handwritten digits
- **Target**: Digit classification (0–9)

---

## 🧠 Model Used

- **Architecture**: Convolutional Neural Network (CNN)
- **Framework**: TensorFlow/Keras
- **Input Shape**: 28x28 grayscale images
- **Output**: Class probabilities for digits 0–9

---

## 🧪 Evaluation Metrics

- **Accuracy**: ~99% on validation data
- **Loss Function**: Sparse Categorical Crossentropy
- **Optimization**: Adam Optimizer
- **Metrics**: Accuracy

---

## 📈 Key Results

- The model achieves high accuracy (~99%) on the validation dataset.
- It can classify handwritten digits (0-9) from images with near-human accuracy.

---

## 🛠️ Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - TensorFlow, Keras (for model building)
  - Flask (for the web app)
  - NumPy, Pandas (for data manipulation)
  - Matplotlib, Seaborn (for visualization)
- **Frontend**: HTML (for web interface)
- **Hosting**: Localhost for testing

