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
- **Backend API (Swagger UI)**: [https://mlops-lab2-latest.onrender.com/docs](https://mlops-lab2-latest.onrender.com/docs)
- **Frontend Interface (Gradio)**: [HuggingFace Space](https://huggingface.co/spaces/your-username/your-space)

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

## 💻 Local Installation & Usage

### Prerequisites
- Python >= 3.10
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/joaquinorradre/MLOps-Lab2
cd MLOps-Lab2
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Frontend Locally
If you want to run the Gradio interface locally but connect to the deployed API:
```bash
# Switch to the frontend branch/folder if necessary
git checkout hf-space
python app.py
```

### 4. How to Use (GUI)
1. Go to the HuggingFace Space.
2. Select the operation tab (Predict, Resize, or Grayscale).
3. Upload an image (drag and drop or file selector).
4. (For Resize) Enter the target width and height.
5. Click the Submit/Run button.
6. View the result instantly on the right side of the screen.

## 📂 Folder Structure
```
├── api/                 # FastAPI backend source code
│   └── api.py           # Main entry point for the API
├── hf-space/            # Gradio frontend code (for HuggingFace)
│   └── app.py           # Gradio application entry point
├── outputs/             # Directory for generated images (local testing)
├── templates/           # HTML templates (for API homepage)
├── requirements.txt     # Python dependencies
├── Dockerfile           # Configuration for Docker deployment
└── README.md            # Project documentation
```

## 📝 Notes
- **API Connection**: Ensure the `API_URL` variable in `hf-space/app.py` points to the correct Render deployment URL.
- **Processing**: All image processing is performed server-side (Render); the frontend merely sends files and displays the returned results.
- **CI/CD**: The HuggingFace Space is configured to update automatically via CI/CD pipelines whenever changes are pushed to the main branch.

## 👤 Author
Joaquín Orradre
