# utils.py (YOLOv5 version)
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
# Define your class names (MUST match the order in your data.yaml for YOLOv5)
CLASS_NAMES = ['Front-Windscreen-Damage', 'Headlight-Damage', 'Major-Rear-Bumper-Dent', 'Rear-windscreen-Damage',
               'RunningBoard-Dent', 'Sidemirror-Damage', 'Signlight-Damage', 'Taillight-Damage',
               'bonnet-dent', 'doorouter-dent', 'fender-dent', 'front-bumper-dent', 'pillar-dent',
               'quaterpanel-dent', 'rear-bumper-dent', 'roof-dent']

# Path to your YOLOv5 model weights
MODEL_PATH = "best.pt" # Make sure this file is in the same directory as utils.py

# Initialize YOLOv5 model (loads on first call)
model = None

def load_yolov5_model():
    """Loads the YOLOv5 model from a .pt file."""
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"YOLOv5 model file not found at: {MODEL_PATH}")
        print(f"Loading YOLOv5 model from {MODEL_PATH}...")
        # Force PyTorch hub to load the model.
        # This will download YOLOv5 architecture if not already present.
        # It needs the 'yolov5' repo from GitHub.
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)
        model.eval() # Set model to evaluation mode
        print("YOLOv5 model loaded successfully.")
    return model

def predict_objects(image_pil: Image.Image, confidence_threshold=0.25):
    """
    Performs object detection using the loaded YOLOv5 model.
    Args:
        image_pil (PIL.Image.Image): The input image as a PIL Image object.
        confidence_threshold (float): Minimum confidence to consider a detection.
    Returns:
        A list of dictionaries, each representing a detected object
        (e.g., {'box': [x1, y1, x2, y2], 'class': 'bonnet-dent', 'confidence': 0.95}).
    """
    yolo_model = load_yolov5_model() # Ensure model is loaded

    # Convert PIL image to numpy array (RGB to BGR for OpenCV functions if needed later)
    img_np = np.array(image_pil)
    # YOLOv5 expects RGB, so no BGR conversion needed here.
    # If using OpenCV for drawing, remember to convert back to BGR.

    # Perform inference
    results = yolo_model(img_np) # Directly pass numpy array or PIL image

    detections = []
    # Process results. For YOLOv5 with ultralytics, results.xyxy[0] contains detections
    # format: [x1, y1, x2, y2, confidence, class_id]
    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)
        class_id = int(cls)

        if conf >= confidence_threshold:
            detections.append({
                "box": [x1, y1, x2, y2],
                "class": CLASS_NAMES[class_id],
                "confidence": round(float(conf), 4) # Round for cleaner output
            })
    return detections

def draw_boxes_on_image(image_pil: Image.Image, detections: list):
    """
    Draws bounding boxes and labels on a PIL image.
    Args:
        image_pil (PIL.Image.Image): The original PIL image.
        detections (list): List of detection dictionaries from predict_objects.
    Returns:
        PIL.Image.Image: The image with bounding boxes drawn.
    """
    draw = ImageDraw.Draw(image_pil)
    font_path = "arial.ttf" # Or any other font path if available
    try:
        font = ImageFont.truetype(font_path, size=16)
    except IOError:
        # Fallback to default if font not found (modern way)
        font = ImageFont.load_default()
        print("Could not load custom font, using default PIL font.") # Optional: print a message

    for det in detections:
        x1, y1, x2, y2 = det['box']
        label = f"{det['class']} ({det['confidence']:.2f})"
        color = "red"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Calculate text size for background rectangle (optional, but good practice)
        # PIL's textbbox is the modern way; textsize is deprecated
        try:
            text_bbox = draw.textbbox((x1 + 5, y1 + 5), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except AttributeError: # Fallback for older PIL versions
            text_width, text_height = draw.textsize(label, font=font)


        # Draw a background rectangle for the text for better readability
        # Adjust padding as needed
        text_bg_x1 = x1
        text_bg_y1 = y1
        text_bg_x2 = x1 + text_width + 10 # Add some padding
        text_bg_y2 = y1 + text_height + 5 # Add some padding

        # Ensure background is above the box if drawing below, or below if drawing above
        # Here drawing text on top of the box
        draw.rectangle([text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2], fill=color)

        # Draw text
        draw.text((x1 + 5, y1), label, fill="white", font=font) # Text directly on the box, adjusted Y position

    return image_pil

if __name__ == "__main__":
    # Test block for utils.py
    # This will assume you have a 'test_image.jpg' in the same directory
    # and 'best.pt' is also there.
    test_image_path = "test_image.jpg" # Make sure this path points to a real image!

    # Create a dummy image if it doesn't exist for testing
    if not os.path.exists(test_image_path):
        print(f"Creating a dummy image at {test_image_path} for testing.")
        dummy_img = Image.new('RGB', (640, 480), color = 'white')
        d = ImageDraw.Draw(dummy_img)
        d.text((10,10), "This is a dummy image for testing.", fill=(0,0,0))
        dummy_img.save(test_image_path)
        print("Please replace with a real car image for accurate testing.")

    try:
        # Load the model explicitly for testing the utility
        load_yolov5_model()

        # Load test image
        img = Image.open(test_image_path).convert('RGB')
        print(f"Loaded test image: {test_image_path}")

        # Predict
        predictions = predict_objects(img)
        print("\nPrediction Results:")
        for det in predictions:
            print(det)

        # Draw boxes and save result (optional for CLI test)
        if predictions:
            img_with_boxes = draw_boxes_on_image(img.copy(), predictions)
            output_path = "test_image_with_detections.jpg"
            img_with_boxes.save(output_path)
            print(f"Image with detections saved to: {output_path}")
        else:
            print("No detections found for the test image.")

    except Exception as e:
        print(f"Error during utils.py test: {e}")
        print("Ensure 'best.pt' is in the same directory and 'test_image.jpg' exists.")