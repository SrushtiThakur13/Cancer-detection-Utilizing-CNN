# Histopathologic Cancer Detection Using Deep Learning

This project implements a deep learning model to automate the detection of metastatic cancer in histopathologic scans of lymph nodes. The solution uses a Convolutional Neural Network (CNN) built with PyTorch and aims to accurately classify cancerous and non-cancerous tissue patches.

---

## 📊 Project Highlights

- **Dataset**: [Kaggle - Histopathologic Cancer Detection]
- **Image Size**: 96x96 pixels
- **Classes**: Cancerous (1) and Non-Cancerous (0)
- **Evaluation Metric**: Area Under the ROC Curve (AUC)
- **Sampling**: 160,000 images sampled (80,000 from each class)
- **Data Augmentation**:
  - Random flips and rotations
  - Normalization

---

## 🧠 Project Workflow

1. **Data Preparation**:
   - Load images and labels.
   - Apply train-validation split.
   - Perform data augmentation.

2. **Model Architecture**:
   - 5 Convolutional Layers
   - Batch Normalization and ReLU Activation
   - MaxPooling after each conv block
   - Fully Connected Layers with Dropout
   - Sigmoid Activation for binary classification

3. **Training**:
   - Loss: Binary Cross-Entropy Loss
   - Optimizer: Adam
   - Learning Rate: 0.00015
   - Early Stopping based on validation loss

4. **Testing**:
   - Predict labels on test dataset
   - Save submission CSV for Kaggle evaluation

5. **Evaluation**:
   - Achieved AUC ~0.95

---

## 🛠 Technologies Used

- Python 3
- PyTorch
- OpenCV
- Pandas
- Matplotlib
- Scikit-learn

---

## 📌 Key Insights

- CNNs can effectively detect cancerous patterns in small histopathology image patches.
- Data augmentation is crucial to generalize the model on unseen data.
- Using balanced sampling and early stopping prevents overfitting.

---

## 📜 Notes
- Data is sourced from public datasets [Kaggle - Histopathologic Cancer Detection].
