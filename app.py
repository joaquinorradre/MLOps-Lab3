import gradio as gr
import requests
import os
from io import BytesIO
from PIL import Image

API_URL = "https://mlops-lab3-latest.onrender.com"

def predict_image(file_path):
    if not file_path:
        return None
    try:
        with open(file_path, "rb") as f:
            filename = os.path.basename(file_path)
            files = {"file": (filename, f, "image/png")}
            
            response = requests.post(f"{API_URL}/predict", files=files, timeout=20)
            response.raise_for_status()
            data = response.json()
            return f"Prediction: {data.get('prediction')}"
    except Exception as e:
        raise gr.Error(f"Error en predicción: {str(e)}")

def resize_image(file_path, width, height):
    if not file_path:
        return None
    try:
        with open(file_path, "rb") as f:
            filename = os.path.basename(file_path)
            files = {"file": (filename, f, "image/png")}
            
            data = {"width": int(width), "height": int(height)}
            response = requests.post(f"{API_URL}/resize", files=files, data=data, timeout=20)
            
            if response.status_code != 200:
                try:
                    detail = response.json().get('detail')
                except:
                    detail = response.text
                raise gr.Error(f"Error API ({response.status_code}): {detail}")
            
            return Image.open(BytesIO(response.content)).convert("RGB")
            
    except Exception as e:
        raise gr.Error(f"Fallo en resize: {str(e)}")

def grayscale_image(file_path):
    if not file_path:
        return None
    try:
        with open(file_path, "rb") as f:
            filename = os.path.basename(file_path)
            files = {"file": (filename, f, "image/png")}
            
            response = requests.post(f"{API_URL}/grayscale", files=files, timeout=20)
            
            if response.status_code != 200:
                try:
                    detail = response.json().get('detail')
                except:
                    detail = response.text
                raise gr.Error(f"Error API ({response.status_code}): {detail}")
            
            return Image.open(BytesIO(response.content)).convert("RGB")
            
    except Exception as e:
        raise gr.Error(f"Fallo en grayscale: {str(e)}")

with gr.Blocks() as demo:
    gr.Markdown("## MLOps Lab 3 - Image Prediction & Processing (Resize and Grayscale) API")
    
    with gr.Tab("Predict"):
        img_input = gr.Image(type="filepath", label="Upload Image")
        predict_btn = gr.Button("Predict")
        predict_output = gr.Textbox(label="Prediction")
        predict_btn.click(predict_image, inputs=img_input, outputs=predict_output)
    
    with gr.Tab("Resize"):
        with gr.Row():
            resize_img = gr.Image(type="filepath", label="Upload Image")
            resize_output = gr.Image(label="Resized Image")
        with gr.Row():
            width_input = gr.Number(label="Width", value=256, precision=0)
            height_input = gr.Number(label="Height", value=256, precision=0)
        resize_btn = gr.Button("Resize")
        resize_btn.click(resize_image, inputs=[resize_img, width_input, height_input], outputs=resize_output)
    
    with gr.Tab("Grayscale"):
        with gr.Row():
            gray_img = gr.Image(type="filepath", label="Upload Image")
            gray_output = gr.Image(label="Grayscale Image")
        gray_btn = gr.Button("Convert to Grayscale")
        gray_btn.click(grayscale_image, inputs=gray_img, outputs=gray_output)

if __name__ == "__main__":
    demo.launch()