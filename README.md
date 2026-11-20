# 👕 Fashion MNIST Clothing Classification with TensorFlow & Keras

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14%2B-FF6F00.svg?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000.svg?style=flat&logo=keras&logoColor=white)](https://keras.io)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?style=flat&logo=python&logoColor=white)](https://python.org)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-91.4%25-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Deep Learning classification pipeline built with TensorFlow and Keras to classify articles of clothing from the Fashion MNIST benchmark dataset across 10 discrete categories.

---

## 📊 Dataset & Categories

The **Fashion MNIST** dataset consists of 70,000 28x28 grayscale images (60,000 training, 10,000 testing):
* `0`: T-shirt / top
* `1`: Trouser
* `2`: Pullover
* `3`: Dress
* `4`: Coat
* `5`: Sandal
* `6`: Shirt
* `7`: Sneaker
* `8`: Bag
* `9`: Ankle boot

---

## 🛠 Pipeline & Methodology

1. **Pre-processing:** Min-max pixel normalization (`[0, 255] -> [0.0, 1.0]`) and data augmentation (random flips and minor translations).
2. **Convolutional Architecture:** Dual Conv2D blocks with spatial Max Pooling, followed by dense representation layers with Dropout to prevent overfitting.
3. **Evaluation:** Confusion matrix analysis highlighting nuanced inter-class distinctions (*e.g., Coat vs. Pullover vs. Shirt*).

---

## 🚀 Quickstart

```bash
# Clone repository
git clone https://github.com/quvonchbek212/fashion-mnist-classifier.git
cd fashion-mnist-classifier

# Install dependencies
pip install tensorflow matplotlib numpy scikit-learn

# Train and evaluate model
python train.py
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
