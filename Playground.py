import torch
from torchvision import transforms
from PIL import Image
import streamlit as st
import cv2
import numpy as np
import os
import json
from facenet_pytorch import MTCNN
from Model.convnextv2 import convnextv2_atto  # Adjust the import path as needed

def draw_bounding_boxes_from_json(json_file, input_image_name, image_cv):
    """Annotate image with bounding boxes from the JSON file."""
    # Load the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Extract relevant information from the JSON structure
    images_info = {image['id']: image for image in data['images']}
    annotations_info = data['annotations']

    # Find the corresponding image by file name (with prefix)
    image_info = None
    for image in data['images']:
        if image['file_name'] == input_image_name:
            image_info = image
            break

    if not image_info:
        return None, None  # If image is not found in the JSON

    image_id = image_info['id']
    for annotation in annotations_info:
        if annotation['image_id'] == image_id:
            bbox = annotation['bbox']  # Bounding box is in the format [x, y, width, height]
            category_id = annotation['category_id']

            # Set color based on category ID
            if category_id == 0:  # Real
                color = (0, 255, 0)  # Green
                category_name = "Real"
            elif category_id == 1:  # Fake
                color = (0, 0, 255)  # Red
                category_name = "Fake"
            else:
                color = (255, 255, 255)  # White (unknown)
                category_name = "Unknown"

            # Draw the bounding box
            x, y, w, h = [int(coord) for coord in bbox]
            image_cv = cv2.rectangle(image_cv, (x, y), (x + w, y + h), color, 2)

            # Annotate with the category name
            cv2.putText(image_cv, category_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return image_cv, category_name



def show():
    # Set up Streamlit page
    st.title("Face Detection and Deepfake Classification")
    
    # Sidebar: Upload an image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if not uploaded_file:
        st.warning("Please upload an image to proceed.")
        return
    
    # Load pre-trained ConvNeXt model
    st.write("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join(os.path.dirname(__file__), "Model", "checkpoint-best-Finetune.pth")  # Update this path as needed
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except FileNotFoundError:
        st.error(f"Checkpoint not found at {checkpoint_path}. Please check the path.")
        return
    
    model = convnextv2_atto(num_classes=2)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    # Load MTCNN for face detection
    mtcnn = MTCNN(keep_all=True, device=device, thresholds=[0.7, 0.6, 0.6], min_face_size=20)

    # Transformation for input preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Process uploaded image
    st.write("Processing image...")
    image = Image.open(uploaded_file).convert("RGB")
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert to OpenCV format

    # Detect faces using MTCNN
    boxes, _ = mtcnn.detect(image)
    if boxes is None:
        st.warning("No faces detected in the uploaded image.")
        return
    
    # st.write(f"Detected {len(boxes)} face(s).")

    # Prepare faces for classification
    face_tensors = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cropped_face = image.crop((x1, y1, x2, y2))  # Crop face region
        face_tensor = transform(cropped_face).to(device)
        face_tensors.append(face_tensor)
    
    # Stack tensors and make predictions
    faces_batch = torch.stack(face_tensors).to(device)
    with torch.no_grad():
        outputs = model(faces_batch)
        predictions = torch.sigmoid(outputs).cpu().numpy()

    # Annotate image with classification results using OpenCV
    results = ["Fake" if pred[1] < 0.5 else "Real" for pred in predictions]

    # Get the image filename to check against the JSON file
    # uploaded_image_name = os.path.basename(uploaded_file.name)

    # # Add the prefix to the uploaded image filename for the JSON lookup
    # image_prefix = "Images/Test-Dev/"
    # full_image_name = image_prefix + uploaded_image_name

    # # Load JSON file
    # json_file_path = "D:/DSP/Streamlit/pages/Test-Dev_poly.json"  # Adjust path to your JSON annotations file

    # Check and annotate image using JSON
    # json_annotated_image, json_label = draw_bounding_boxes_from_json(json_file_path,  full_image_name, image_cv.copy())

    # Annotate the image based on predictions
    pred_annotated_image = image_cv.copy()
    for (box, result, prob) in zip(boxes, results, predictions[:, 1]):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 0, 255) if result == "Fake" else (0, 255, 0)  # Red for Fake, Green for Real
        
        # Draw bounding box
        cv2.rectangle(pred_annotated_image, (x1, y1), (x2, y2), color, 2)
        
        # Put text with classification result
        label = f"{result} ({prob:.2f})"
        cv2.putText(pred_annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, color, 2, cv2.LINE_AA)

    # Display the images
    # if json_annotated_image is not None:
    #     st.subheader("Annotated Image (Based on JSON)")
    #     json_annotated_image_rgb = cv2.cvtColor(json_annotated_image, cv2.COLOR_BGR2RGB)
    #     st.image(json_annotated_image_rgb, caption="Image annotated based on JSON", use_container_width=True)

    st.subheader("Annotated Image (Based on Prediction)")
    pred_annotated_image_rgb = cv2.cvtColor(pred_annotated_image, cv2.COLOR_BGR2RGB)
    st.image(pred_annotated_image_rgb, caption="Image annotated based on prediction", use_container_width=True)

    # # If image is found in JSON, display the labels
    # if json_label:
    #     st.write(f"Actual Label (from JSON): {json_label}")
    # st.write(f"Predicted Label: {results[0]}")

# Run the Streamlit function
if __name__ == "__main__":
    show()
