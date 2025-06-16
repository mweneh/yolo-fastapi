# app.py (Logo Customization)
import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont # Import Image from PIL
import io
import numpy as np
import os
import sys

# Add the directory containing inference_utils.py to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference_utils import draw_boxes_on_image

# --- Streamlit UI Configuration ---
st.set_page_config(
    page_title="Car Damage Object Detector",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="auto",
)

# --- Custom CSS for Sidebar & Main Content Colors ---
# ... (your existing CSS styling here) ...
st.markdown("""
<style>
    /* Overall App Background and Default Text Color */
    .stApp {
        background-color: #0A1828; /* Light gray background for the entire app */
        color: white; /* Default text color for the app */
    }

    /* Main Content Area Specific Styles */
    .main [data-testid="stMarkdownContainer"],
    .main .stAlert,
    .main .stRadio > label,
    .main .stButton > button,
    .main .stText,
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {
        color: #333333; /* Dark text for the main content area */
    }
     .main [data-testid="stImage"] {
        background-color: #f0f2f6;
    }

    /* --- Sidebar Specific Styles --- */
    [data-testid="stSidebar"] {
        background-color: black; /* Dark background for the sidebar */
        color: white;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6 {
        color: #FFD700;
    }
    [data-testid="stSidebar"] a {
        color: #ADD8E6 !important;
    }
    [data-testid="stSidebar"] .stRadio > label {
        color: white;
    }

    /* --- General Styling --- */
    h1 {
        text-align: center;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    div.stAlert {
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


st.title("Car Damage Object Detector (YOLOv5)")
st.markdown("Upload an image of a car or provide an image URL to detect and locate specific damage types.")

# --- FastAPI Endpoint Configuration ---
FASTAPI_URL = "http://127.0.0.1:8000/predict/"

# --- Main Content Area (same as before) ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Image Input")
    input_method = st.radio("Choose image input method:", ("Upload File", "Enter Image URL"), key="input_method_radio")

    original_image = None
    image_bytes = None
    image_filename = "uploaded_image.jpg"

    if input_method == "Upload File":
        uploaded_file = st.file_uploader("Choose a car image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            original_image = Image.open(uploaded_file).convert("RGB")
            image_bytes = io.BytesIO()
            original_image.save(image_bytes, format=uploaded_file.type.split('/')[-1].upper())
            image_bytes.seek(0)
            image_filename = uploaded_file.name
    elif input_method == "Enter Image URL":
        image_url = st.text_input("Enter Image URL:")
        if image_url:
            try:
                st.write("Fetching image from URL...")
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(image_url, headers=headers, timeout=10)
                response.raise_for_status()
                image_bytes = io.BytesIO(response.content)
                original_image = Image.open(image_bytes).convert("RGB")
                image_bytes.seek(0)

                image_filename = image_url.split('/')[-1].split('?')[0]
                if not image_filename or '.' not in image_filename:
                    image_filename = "url_image.jpg"

            except requests.exceptions.MissingSchema:
                st.error("Invalid URL. Please ensure it starts with 'http://' or 'https://'")
                original_image = None
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the URL. Please check your internet connection or the URL.")
                original_image = None
            except requests.exceptions.Timeout:
                st.error("Request timed out while fetching image from URL. The URL might be slow or unresponsive.")
                original_image = None
            except requests.exceptions.HTTPError as e:
                st.error(f"HTTP Error fetching image: {e.response.status_code} - {e.response.reason}. The server might be blocking automated requests.")
                original_image = None
            except Exception as e:
                st.error(f"Error fetching image from URL: {e}")
                original_image = None

if original_image is not None and image_bytes is not None:
    st.image(original_image, caption='Image to be analyzed', use_container_width=True)
    st.write("")
    if st.button("Detect Damage", key="detect_button"):
        st.write("Detecting objects...")

        image_format = original_image.format.lower() if original_image.format else "jpeg"
        mime_type = f"image/{image_format}"
        files = {"file": (image_filename, image_bytes, mime_type)}

        try:
            response = requests.post(FASTAPI_URL, files=files)
            response.raise_for_status()

            prediction_data = response.json()
            detections = prediction_data.get("detections", [])

            with col2:
                if detections:
                    st.success("Detections Found!")
                    st.subheader("Detected Damage:")

                    image_with_boxes = draw_boxes_on_image(original_image.copy(), detections)
                    st.image(image_with_boxes, caption='Image with Detections', use_container_width=True)

                    st.markdown("---")
                    st.subheader("Details:")
                    for det in detections:
                        st.write(f"- **{det['class']}**: Conf: {det['confidence']:.2f}, Box: {det['box']}")
                else:
                    st.info("No damage detected with the current confidence threshold.")

        except requests.exceptions.ConnectionError:
            st.error(f"Error: Could not connect to the FastAPI server at {FASTAPI_URL}. "
                     "Please ensure the backend server is running and accessible.")
        except requests.exceptions.HTTPError as e:
            st.error(f"Error from API: {e.response.status_code} - {e.response.json().get('detail', 'Unknown error')}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
else:
    st.info("")

# --- Sidebar Content ---
with st.sidebar:
    # --- Custom Logo Display ---
    LOGO_PATH = "src/golden.png"  # Adjust this path if 'src' is one level above app.py
    DESIRED_WIDTH = 150 # You can adjust this width in pixels

    try:
        logo = Image.open(LOGO_PATH)
        # Calculate new height to maintain aspect ratio
        original_width, original_height = logo.size
        new_height = int(DESIRED_WIDTH * (original_height / original_width))
        resized_logo = logo.resize((DESIRED_WIDTH, new_height))
        st.image(resized_logo, use_container_width=False) # Set to False as we explicitly sized it
    except FileNotFoundError:
        st.error(f"Logo file not found at {LOGO_PATH}. Please check the path.")
        st.image("https://via.placeholder.com/150/FFD700/000000?text=Logo+Missing", use_container_width=True) # Fallback placeholder
    except Exception as e:
        st.error(f"Error loading or resizing logo: {e}")
        st.image("https://via.placeholder.com/150/FFD700/000000?text=Logo+Error", use_container_width=True) # Fallback placeholder

    st.header("About This App")
    st.info("This application leverages a YOLOv5 model trained on car damage data to detect and classify various types of vehicle damage in real-time.")
    st.markdown("---")
    st.subheader("Contact")
    st.write("For inquiries, contact: [david.mauti@strathmore.edu](mailto:david.mauti@strathmore.edu)")
    st.markdown("[GitHub](https://github.com/mweneh)")