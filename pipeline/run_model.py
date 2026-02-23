from ultralytics import YOLO
import cv2
import json
import os
import random

def load_model(model_path='yolo11n.pt'):
    """
    Loads the YOLOv11 model. Downloads it if not present.
    """
    print(f"Loading YOLOv11 model from {model_path}...")
    # 'yolo11n.pt' will be downloaded automatically by ultralytics if needed
    model = YOLO(model_path)
    return model

def run_inference(image_path, model):
    """
    Runs inference on a single image.
    """
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None

    # Run inference
    # classes: filter for relevant classes if using generic COCO model
    results = model(image_path, verbose=False) 
    result = results[0]
    
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = result.names[cls_id]
        
        detections.append({
            "label": label,
            "confidence": conf,
            "bbox": box.xyxy[0].tolist()
        })
        
    return detections

def classify_status(detections, mock_mode=False):
    """
    Determines if a business is Open or Closed based on detections.
    """
    if mock_mode:
        # Randomly assign status for demo purposes if no custom model is available
        status = "Closed" if random.random() < 0.2 else "Open"
        return status, 0.85

    status = "Open"
    confidence = 0.0
    
    # Logic for custom trained model
    for det in detections:
        # Assuming we have a class 'for_lease' or 'closed_sign'
        if det['label'] in ['for_lease', 'closed', 'vacant'] and det['confidence'] > 0.6:
            status = "Closed"
            confidence = det['confidence']
            break
            
    return status, confidence

if __name__ == "__main__":
    # Example usage
    model = load_model('yolo11n.pt') # Using nano model for speed
    
    # Dummy run on a generated image if none exists
    if not os.path.exists("test_image.jpg"):
        # create a black image
        import numpy as np
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.imwrite("test_image.jpg", img)
        
    dets = run_inference("test_image.jpg", model)
    print("Detections:", dets)
    
    status, conf = classify_status(dets, mock_mode=True)
    print(f"Status: {status}")
