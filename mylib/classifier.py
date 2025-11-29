"""
classifier.py
----------------
ONNX-based pet classifier wrapper.
"""
import json
import os
import numpy as np
from PIL import Image
import io
import onnxruntime as ort


class PetClassifier:
    """
    Wrapper for ONNX model inference for pet classification.
    """
    def __init__(self, model_path: str = None, labels_path: str = None):
        """
        Initialize the classifier with the ONNX model and class labels.
        
        Args:
            model_path: Path to the ONNX model file
            labels_path: Path to the JSON file with class labels
        """
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            model_path = os.path.join(project_root, "pet_classifier_model.onnx")
        
        if labels_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            labels_path = os.path.join(project_root, "class_labels.json")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        
        self.input_name = self.session.get_inputs()[0].name
        
        with open(labels_path, 'r') as f:
            self.class_labels = json.load(f)
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Input name: {self.input_name}")
        print(f"✓ Number of classes: {len(self.class_labels)}")
    
    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        """
        Preprocess the input image to match the format used during training.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Preprocessed image as numpy array with shape (1, 3, 224, 224)
        """
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert('RGB')
        
        img = img.resize((224, 224))
        
        img_array = np.array(img).astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std
        
        img_array = np.transpose(img_array, (2, 0, 1))
        
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image_bytes: bytes) -> str:
        """
        Predict the class label for an input image.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Predicted class label as string
        """
        preprocessed_img = self.preprocess(image_bytes)
        
        inputs = {self.input_name: preprocessed_img}
        
        outputs = self.session.run(None, inputs)
        
        logits = outputs[0]
        
        predicted_idx = np.argmax(logits, axis=1)[0]
        
        predicted_class = self.class_labels[str(predicted_idx)]
        
        return predicted_class
    