# Traffic Sign Classification using CNN (PyTorch)

This project implements a Convolutional Neural Network (CNN) from scratch using PyTorch to classify traffic sign images from the GTSRB dataset.

The goal of this project is to practice the complete computer vision pipeline, including data loading, model design, training, and evaluation.

---

## Dataset
- German Traffic Sign Recognition Benchmark (GTSRB)
- Image size: 32×32 RGB
- Number of classes: 43

---

## Model Architecture
- 2 Convolutional layers (Conv2D + ReLU + MaxPooling)
- Fully connected classifier
- CrossEntropy loss
- Adam optimizer

---

## Results
- Test Accuracy: **~90%**
- Evaluation using confusion matrix
- Strong performance with expected confusion between visually similar signs

---

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt


## Dataset
Due to GitHub file size limits, the dataset is not included in this repository.

You can download the GTSRB dataset from:
https://benchmark.ini.rub.de/gtsrb_news.html

After downloading, place the data files inside the `data/` directory.



