###
An AI-powered medical screening system designed to detect early signs of preeclampsia in pregnant women through retinal fundus image analysis. The system uses deep learning to classify retinal images and identify hypertension-related patterns that may indicate preeclampsia risk.

### Overview

This is a **proof-of-concept demo**. While the current dataset includes general hypertension cases (which the model was trained on), this demo demonstrates how such a system may look and function for early preeclampsia detection.

### Technology

- **Model:** RETFound (Retinal Foundation Model) - Vision Transformer architecture
- **Training:** Fine-tuned on retinal fundus images for hypertension classification
- **API:** FastAPI backend for model inference
- **Frontend:** React web application for image upload and results display


### Note

Due to file size limitations, the following are **not included** in this repository:
- Training dataset (retinal images)
- Model weights (`.pth` files)
- RETFound model code
- oculGravida-main/        # React frontend application
