# Deep Learning Projects
## This folder contains 4 Project based on the Deep Learning and Computer Vision

---
1) Image Classifcation
    * Developed a deep learning model for classifying images into 10 classes of flora and fauna using a fine-tuned pretrained EfficientNet_V2_M model. Employed ImageFolder and DataLoader for data handling, CrossEntropyLoss for training, and optimized with AdamW (LR: 1e-4) and CosineAnnealingLR. Leveraged DataParallel for multi-GPU training with 4 workers. Achieved a 96.6% accuracy.

2) Depth Prediction 
    * Developed a computer vision model using a fine-tuned YOLOv11 to detect and classify mosquitoes into six types. Prepared datasets with automated folder creation and a custom YAML file for class labels and data paths. Trained the model using SGD (momentum 0.9), CosineAnnealingLR, and dropout (0.2) with data augmentation for 100 epochs. Achieved a mAP score of 0.58.

3) Mosquito Detection
    * Designed a custom UNet with 19 convolutional layers to predict depth maps from RGB images under challenging conditions (e.g., low light, noise). Used ReLU, group normalization, dropout, and a sigmoid activation for outputs. Trained the model with MSE loss, AdamW optimizer (LR: 1e-4), and StepLR (step size: 4, gamma: 0.01) for 110 epochs. Predictions were scaled (0–1) and saved as grayscale images. Achieved an RMSE loss of 0.089.

4) Low-Light Image Denoising and Super-Resolution
    * Developed a custom "DenoiseSuperResolution" model to denoise and enhance low-light images, using multiple convolutional layers, 4 residual blocks, and an attention mechanism. The model employed Leaky ReLU, batch normalization, and upsampling blocks for resolution enhancement, with output refinement via convolution. Trained using MSE loss, evaluated with PSNR, and optimized with Adam and GradScaler for mixed-precision. Achieved a PSNR value of 36.45 dB.