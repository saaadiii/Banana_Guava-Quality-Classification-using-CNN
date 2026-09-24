# Banana and Guava Quality Classification Using CNN

This project uses a **Convolutional Neural Network (CNN)** built with **PyTorch** to classify images of **banana and guava** into six quality categories.

The model distinguishes between different quality grades and defective fruits using image classification.

## Classes

The dataset contains the following six classes:

- `Banana_Class_A`
- `Banana_Class_B`
- `Banana_Defect`
- `Guava_Class_A`
- `Guava_Class_B`
- `Guava_Defect`

## Dataset

The dataset contains a total of **1,748 images**.

The images are divided into:

| Dataset | Images | Percentage |
|---|---:|---:|
| Training | 1,223 | 70% |
| Validation | 262 | 15% |
| Testing | 263 | 15% |

The random split uses a fixed seed (`42`) so the same train/validation/test split can be reproduced.

### Dataset Folder Structure

The notebook expects the dataset to be stored in Google Drive as:

```text
Fruits_data/
├── Banana_Class_A/
├── Banana_Class_B/
├── Banana_Defect/
├── Guava_Class_A/
├── Guava_Class_B/
└── Guava_Defect/
```

In the notebook, the dataset path is:

```python
dataset_path = "/content/drive/MyDrive/Fruits_data"
```

## Image Preprocessing

All images are resized to:

```text
128 × 128 pixels
```

The basic preprocessing pipeline contains:

- Resize to `128 × 128`
- Convert the image to a PyTorch tensor

The batch size is:

```text
32
```

## CNN Architecture

The project uses a custom CNN named `SimpleCNN`.

The network contains:

1. Convolutional layer: `3 → 16` channels
2. ReLU activation
3. Max Pooling
4. Convolutional layer: `16 → 32` channels
5. ReLU activation
6. Max Pooling
7. Convolutional layer: `32 → 64` channels
8. ReLU activation
9. Max Pooling
10. Flatten layer
11. Dropout
12. Fully connected layer: `16384 → 128`
13. ReLU activation
14. Output layer: `128 → 6`

The model has approximately:

```text
2,121,638 trainable parameters
```

## Training Configuration

The model is trained using:

- **Loss Function:** Cross Entropy Loss
- **Optimizer:** Adam
- **Learning Rate:** `0.001`
- **Batch Size:** `32`
- **Device:** CUDA GPU when available, otherwise CPU

Random seeds are set for PyTorch, NumPy, and Python to improve reproducibility.

## Training Round 1 — Basic CNN

The first model is trained for:

```text
15 epochs
```

No data augmentation is used and dropout is set to `0.0`.

### Round 1 Results

- Best Training Accuracy: **95.01%**
- Best Validation Accuracy: **85.11%**
- Test Accuracy: **82%**
- Macro F1-score: **0.81**
- Weighted F1-score: **0.82**

The notebook identifies a noticeable gap between training and validation performance, indicating likely overfitting.

## Training Round 2 — Data Augmentation + Dropout

To reduce overfitting and improve generalization, a second model is trained with data augmentation and dropout.

### Data Augmentation

The training images use:

- Random horizontal flipping
- Random rotation up to `15°`
- Random brightness adjustment
- Random contrast adjustment
- Resize to `128 × 128`
- Conversion to tensor

Validation and test images are not randomly augmented.

### Dropout

The second model uses:

```text
Dropout = 0.5
```

### Training

Round 2 is trained for:

```text
30 epochs
```

### Round 2 Results

- Best Validation Accuracy: **85.50%**
- Test Accuracy: **82%**
- Macro Precision: **0.86**
- Macro Recall: **0.82**
- Macro F1-score: **0.81**
- Weighted F1-score: **0.81**

The best validation accuracy increased slightly from **85.11%** in Round 1 to **85.50%** in Round 2.

## Round 2 Classification Results

| Class | Precision | Recall | F1-score | Test Images |
|---|---:|---:|---:|---:|
| Banana Class A | 0.68 | 0.94 | 0.79 | 64 |
| Banana Class B | 0.84 | 0.41 | 0.55 | 51 |
| Banana Defect | 0.94 | 0.95 | 0.94 | 61 |
| Guava Class A | 0.74 | 1.00 | 0.85 | 28 |
| Guava Class B | 0.94 | 0.67 | 0.78 | 24 |
| Guava Defect | 1.00 | 0.94 | 0.97 | 35 |

## Evaluation and Visualization

The notebook includes several visualizations and evaluation methods:

- Sample images from the dataset
- Training and validation loss curves
- Training and validation accuracy curves
- Learned filters from the first convolutional layer
- Feature maps from the first convolutional layer
- Feature maps from the third convolutional layer
- Confusion matrix
- Precision, recall, and F1-score
- Comparison of Round 1 and Round 2 validation accuracy

## Requirements

The notebook uses the following Python libraries:

```text
torch
torchvision
numpy
matplotlib
scikit-learn
Pillow
```

If running locally, the required packages can be installed with:

```bash
pip install torch torchvision numpy matplotlib scikit-learn pillow
```

Google Colab is recommended because the notebook is configured to access the dataset from Google Drive and can use a GPU when available.

## How to Run

1. Open `cnn.ipynb` in Google Colab.
2. Upload the `Fruits_data` folder to your Google Drive.
3. Make sure the dataset is located at:

   ```text
   MyDrive/Fruits_data
   ```

4. Enable GPU in Colab if available.
5. Run the notebook cells in order.
6. The notebook will:
   - Load the dataset
   - Split it into training, validation, and test sets
   - Display sample images
   - Build the CNN
   - Train the basic model
   - Evaluate overfitting
   - Display learned filters and feature maps
   - Evaluate the model using a confusion matrix and classification report
   - Apply augmentation and dropout
   - Train the improved model
   - Compare both training rounds
   - Evaluate the second model on the test set

## Project Objective

The objective of this project is to use deep learning and computer vision to automatically classify **banana and guava images according to their fruit type and quality condition**.

Rather than only identifying whether an image contains a banana or guava, the current model performs a more detailed **six-class classification** by identifying the fruit together with its quality category.

## Technologies Used

- Python
- PyTorch
- Torchvision
- Google Colab
- Google Drive
- NumPy
- Matplotlib
- Scikit-learn
- Pillow

## Notebook

Main notebook:

```text
cnn.ipynb
```

## Summary

This project demonstrates the complete workflow of a CNN-based image classification system, including dataset preparation, model building, training, validation, overfitting analysis, data augmentation, dropout, visualization of CNN features, and final performance evaluation on an unseen test set.
