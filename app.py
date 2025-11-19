import streamlit as st
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(
    page_title="Spectra | Object Recognition",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Aesthetic Interface
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: white;
    }
    h1, h2, h3, h4, h5, h6, p, div, span, label, .stMarkdown {
        color: white !important;
    }
    .stButton>button {
        background-color: #00d2ff;
        color: white !important;
        border-radius: 20px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #3a7bd5;
        transform: scale(1.05);
    }
    .card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    /* Fix input fields text color */
    .stTextInput input, .stTextArea textarea {
        color: white !important;
    }
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("Spectra 👁️")
    st.markdown("---")
    st.markdown("### Model Settings")
    model_id = st.text_input("Model ID", "a-01a/spectra")
    
    st.markdown("### Classes")
    default_classes = "person, cat, dog, car, tree, house, phone, laptop, book, cup, chair, table, bird, flower"
    class_input = st.text_area("Object Classes (comma separated)", default_classes, height=150)
    
    st.markdown("---")
    st.info("Spectra uses a CLIP-based architecture for zero-shot object recognition.")

# Main Content
st.title("Spectra Object Recognition")
st.markdown("### Upload an image or use your webcam to detect objects.")

# Model Loading
@st.cache_resource
def load_model(model_id: str) -> tuple[CLIPModel | None, CLIPProcessor | None, str | None]:
    """
    Load the CLIP model and processor.

    Args:
        model_id (str): The Hugging Face model ID.

    Returns:
        tuple: (model, processor, error_trace) containing the loaded artifacts or error details.
    """
    try:
        model = CLIPModel.from_pretrained(model_id)
        processor = CLIPProcessor.from_pretrained(model_id)
        return model, processor, None
    except Exception:
        return None, None, traceback.format_exc()

model, processor, error_trace = load_model(model_id)

if not model:
    st.warning("Model not loaded. You can still browse the interface, but analysis will be disabled.")
    if error_trace:
        with st.expander("See error details"):
            st.code(error_trace)
else:
    st.success("Model loaded successfully!")

# Input Method
input_method = st.radio("Select Input", ["Upload Image", "Webcam"], horizontal=True)

image = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
else:
    camera_image = st.camera_input("Take a picture")
    if camera_image:
        image = Image.open(camera_image)

if image:
    # Display Image
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Input Image", use_column_width=True)
    
    with col2:
        st.subheader("Analysis Results")
        
        if st.button("Analyze Image", disabled=(model is None)):
            with st.spinner("Analyzing..."):
                # Prepare classes
                classes = [c.strip() for c in class_input.split(",") if c.strip()]
                if not classes:
                    classes = ["object"]
                    
                # Inference
                inputs = processor(text=classes, images=image, return_tensors="pt", padding=True)
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image # this is the image-text similarity score
                probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
                
                # Get top 5
                top_probs, top_labels = probs.cpu().topk(min(5, len(classes)))
                
                # Display results
                for i in range(top_probs.shape[1]):
                    label = classes[top_labels[0][i]]
                    score = top_probs[0][i].item()
                    st.markdown(f"""
                    <div style="margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between;">
                            <span><b>{label.title()}</b></span>
                            <span>{score:.1%}</span>
                        </div>
                        <div style="background-color: rgba(255,255,255,0.2); height: 8px; border-radius: 4px;">
                            <div style="background-color: #00d2ff; width: {score*100}%; height: 100%; border-radius: 4px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
    st.markdown("</div>", unsafe_allow_html=True)
