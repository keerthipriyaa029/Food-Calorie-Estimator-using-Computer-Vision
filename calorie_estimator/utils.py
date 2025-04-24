# type: ignore
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import tempfile
import os
import time

def preprocess_image(image_path):
    """
    Read an image from file, resize to 224x224, and preprocess for CNN input.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        torch.Tensor: Preprocessed image tensor ready for model input
    """
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert BGR to RGB (OpenCV loads images in BGR format)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image for torchvision transforms
    img_pil = Image.fromarray(img)
    
    # Define preprocessing steps
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Apply preprocessing
    img_tensor = preprocess(img_pil)
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor

def load_image_for_display(image_path, size=(400, 400)):
    """
    Load an image for display purposes.
    
    Args:
        image_path (str): Path to the image file
        size (tuple): Size to resize the image to
        
    Returns:
        numpy.ndarray: Image as numpy array
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    
    return img 

def capture_image_from_webcam(preview_placeholder=None):
    """
    Capture an image from the webcam.
    
    Args:
        preview_placeholder: Optional Streamlit placeholder for displaying webcam preview
        
    Returns:
        str: Path to the captured image file
    """
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        raise ValueError("Could not open webcam. Please check your camera connection.")
    
    # Create a temporary file to save the captured image
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_file_path = temp_file.name
    temp_file.close()
    
    # Check if we're using Streamlit
    if preview_placeholder is not None:
        preview_placeholder.info("Webcam activated. Click 'Capture Image' when ready.")
        
        # Capture frames until told to stop
        stop_capture = False
        
        # Create container for webcam feed and capture button
        webcam_container = preview_placeholder.container()
        
        # Static image placeholder to avoid flickering
        img_placeholder = webcam_container.empty()
        
        try:
            # Give the camera a moment to warm up
            time.sleep(1)
            
            while not stop_capture:
                # Capture a frame
                ret, frame = cap.read()
                if not ret:
                    preview_placeholder.error("Failed to capture frame from webcam")
                    break
                
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Update the image placeholder
                img_placeholder.image(frame_rgb, caption="Webcam Feed", use_column_width=True)
                
                # Check for capture trigger file
                if os.path.exists(temp_file_path + ".trigger"):
                    # Save the current frame
                    cv2.imwrite(temp_file_path, frame)
                    # Remove the trigger file
                    try:
                        os.remove(temp_file_path + ".trigger")
                    except:
                        pass
                    stop_capture = True
                    break
                
                # Add a small delay to prevent too many UI updates
                time.sleep(0.1)
        except Exception as e:
            preview_placeholder.error(f"Webcam error: {e}")
        finally:
            # Release the camera
            cap.release()
    else:
        # Basic camera capture (non-Streamlit)
        try:
            # Give the camera a moment to warm up
            time.sleep(1)
            
            # Capture a single frame
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(temp_file_path, frame)
            else:
                raise ValueError("Failed to capture image from webcam")
        finally:
            # Release the camera
            cap.release()
    
    return temp_file_path

def save_uploaded_image(uploaded_file):
    """
    Save an uploaded file to a temporary location.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        str: Path to the saved image file
    """
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_file_path = temp_file.name
    
    # Write the uploaded file to the temporary file
    temp_file.write(uploaded_file.getvalue())
    temp_file.close()
    
    return temp_file_path 