[![CICD](https://github.com/joaquinorradre/MLOps-Lab2/actions/workflows/CICD.yml/badge.svg)](https://github.com/joaquinorradre/MLOps-Lab2/actions/workflows/CICD.yml)

# MLOps Lab 2 - Image Processing API

[![API on Render](https://img.shields.io/badge/Render-API-blue)](https://mlops-lab2-latest.onrender.com)
[![HuggingFace Space](https://img.shields.io/badge/HuggingFace-Frontend-purple)](https://huggingface.co/spaces/joaquinorradre/mlops)


## Overview
This project provides an Image Processing API built with FastAPI. It allows users to perform various operations on images, such as classification, resizing, and grayscale conversion.

To facilitate interaction, a Gradio interface has been developed as a frontend. This frontend is deployed on HuggingFace Spaces and communicates with the backend API, which is hosted on Render.

### Key Capabilities
- **Predict**: Identify image classes using a pre-trained model.
- **Resize**: Adjust images to custom width and height dimensions.
- **Grayscale**: Convert color images to grayscale.

## 🚀 Live Demo
- **Backend API (Swagger UI)**: [Render API](https://mlops-lab2-latest.onrender.com/docs)
- **Frontend Interface (Gradio)**: [HuggingFace Space]((https://huggingface.co/spaces/joaquinorradre/mlops))

## ✨ Features

### API Endpoints (Backend)
| Endpoint     | Method | Description                                      |
|--------------|--------|--------------------------------------------------|
| `/predict`   | POST   | Upload an image and get a predicted class.       |
| `/resize`    | POST   | Upload an image and resize it to specified width & height. |
| `/grayscale` | POST   | Upload an image and convert it to grayscale.     |

### Frontend (Gradio GUI)
The graphical interface connects directly to the Render API and features three dedicated tabs:
- **Predict Tab**: Upload an image to classify it.
- **Resize Tab**: Upload an image and specify dimensions.
- **Grayscale Tab**: Upload an image to remove color information.

## 🛠️ Deployment Architecture
This project follows a decoupled architecture:

### Render (API):
- The FastAPI application is dockerized and deployed here.
- It handles all the heavy lifting (image processing logic).
- Public URL: [https://mlops-lab2-latest.onrender.com](https://mlops-lab2-latest.onrender.com)

### HuggingFace Space (Frontend):
- Contains the `app.py` (Gradio) application.
- It serves as the user interface and sends requests to the Render API.
- Branch: `hf-space`.

### How to Use (GUI)
1. Go to the HuggingFace Space.
2. Select the operation tab (Predict, Resize, or Grayscale).
3. Upload an image (drag and drop or file selector).
4. (For Resize) Enter the target width and height.
5. Click the Submit/Run button.
6. View the result instantly on the right side of the screen.
