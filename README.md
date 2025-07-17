# DeepFake-Image-Detector
# DeepFake Image Detection using MobileNetV2

This project implements a deep learning-based classifier to detect deepfake images using the **MobileNetV2 architecture**, fine-tuned on real and fake facial datasets. The goal is to provide a lightweight, fast, and accurate model suitable for edge devices and real-time detection.

---

## 📌 Overview

- **Model**: MobileNetV2 (pretrained on ImageNet)
- **Dataset**: Real and Fake Face Detection Dataset from [Kaggle](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection)
- **Framework**: PyTorch
- **Accuracy Achieved**: ~76%

---

## 📁 Project Structure

deepfake-mobilenetv2/
├── model/
│ └── mobilenet_model.py # MobileNetV2 model definition
├── scripts/
│ ├── train_mobilenet.py # Training script
│ └── infer_image.py # Inference script
├── data/
│ └── (Real and Fake images)
├── mobilenetv2_deepfake.pth # Trained model weights
├── requirements.txt
└── README.md

---

## 🧠 Model Details

- Uses `torchvision.models.mobilenet_v2(pretrained=True)`
- Fine-tuned final classification layer for binary output (Real vs Fake)
- Optimized for low memory usage and inference speed

---

## 🔧 Training Configuration

- Optimizer: `Adam` (lr = 0.0001)
- Loss Function: `CrossEntropyLoss`
- Batch Size: 32
- Epochs: 10
- Image Size: 224x224

---

## 🖼️ Inference

To classify a new image as real or fake:

```bash
python scripts/infer_image.py --image path_to_image.jpg
