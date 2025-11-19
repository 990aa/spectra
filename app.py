import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
from transformers import CLIPProcessor, CLIPModel
import traceback
from dotenv import load_dotenv
import tempfile
from pathlib import Path

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
    model_id = st.text_input("Model ID", "990aa/spectra")
    
    st.markdown("### Object Classes")
    default_classes = """person, face, cat, dog, car, truck, bicycle, motorcycle, airplane, boat, 
tree, plant, flower, house, building, phone, laptop, computer, book, cup, 
bottle, chair, table, sofa, bed, bird, horse, sheep, cow, elephant, 
food, pizza, apple, orange, banana, sports ball, tennis racket, clock, 
keyboard, mouse, tv, backpack, umbrella, handbag, traffic light, fire hydrant"""
    class_input = st.text_area("Enter object classes (comma separated)", default_classes, height=200)
    
    st.markdown("### Detection Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.15, 0.05)
    top_k = st.slider("Top K Results", 1, 10, 5)
    
    st.markdown("---")
    st.info("🎯 Spectra uses CLIP-based architecture for zero-shot object recognition trained on LAION-5B, COCO, Visual Genome, and more.")

# Main Content
st.title("🎯 Spectra Object Recognition")
st.markdown("### Detect and describe objects in images, webcam, or videos")

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
        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        return model, processor, None
    except Exception:
        return None, None, traceback.format_exc()

def analyze_image(image: Image.Image, classes: list, model, processor, threshold: float = 0.15, top_k: int = 5):
    """
    Analyze an image and return detected objects with confidence scores.
    
    Args:
        image: PIL Image to analyze
        classes: List of object class names
        model: CLIP model
        processor: CLIP processor
        threshold: Minimum confidence threshold
        top_k: Number of top results to return
        
    Returns:
        List of tuples (class_name, confidence_score)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Prepare inputs
    inputs = processor(text=classes, images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu()
    
    # Get top results above threshold
    top_probs, top_indices = probs[0].topk(min(top_k, len(classes)))
    
    results = []
    for prob, idx in zip(top_probs, top_indices):
        if prob.item() >= threshold:
            results.append((classes[idx], prob.item()))
    
    return results

def process_video_frame(frame, classes, model, processor, threshold):
    """
    Process a single video frame and return annotated frame.
    
    Args:
        frame: OpenCV frame (BGR format)
        classes: List of object classes
        model: CLIP model
        processor: CLIP processor
        threshold: Confidence threshold
        
    Returns:
        Annotated frame
    """
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    
    # Analyze
    results = analyze_image(pil_image, classes, model, processor, threshold, top_k=3)
    
    # Annotate frame
    y_offset = 30
    for label, confidence in results:
        text = f"{label}: {confidence:.1%}"
        cv2.putText(frame, text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30
    
    return frame

model, processor, error_trace = load_model(model_id)

if not model:
    st.error("⚠️ Model not loaded. Please check the model ID or your internet connection.")
    if error_trace:
        with st.expander("📋 See error details"):
            st.code(error_trace)
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.success(f"✅ Model loaded successfully on {device.upper()}!")

# Input Method Selection
st.markdown("---")
input_method = st.radio("📥 Select Input Method", 
                        ["Upload Image", "Webcam Capture", "Upload Video"], 
                        horizontal=True)

# Prepare classes
classes = [c.strip() for c in class_input.split(",") if c.strip()]
if not classes:
    classes = ["object", "item", "thing"]

# ====================
# IMAGE UPLOAD
# ====================
if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg", "webp", "bmp"])
    
    if uploaded_file and model:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="📸 Uploaded Image", use_column_width=True)
        
        with col2:
            st.subheader("🔍 Detection Results")
            
            with st.spinner("🔄 Analyzing image..."):
                results = analyze_image(image, classes, model, processor, 
                                       confidence_threshold, top_k)
                
                if results:
                    for label, confidence in results:
                        st.markdown(f"""
                        <div style="margin-bottom: 15px;">
                            <div style="display: flex; justify-content: space-between;">
                                <span style="font-size: 18px;"><b>🎯 {label.title()}</b></span>
                                <span style="font-size: 18px; color: #00d2ff;"><b>{confidence:.1%}</b></span>
                            </div>
                            <div style="background-color: rgba(255,255,255,0.2); height: 10px; border-radius: 5px; margin-top: 5px;">
                                <div style="background: linear-gradient(90deg, #00d2ff, #3a7bd5); width: {confidence*100}%; height: 100%; border-radius: 5px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Generate description
                    st.markdown("---")
                    st.subheader("📝 Description")
                    top_objects = [label for label, _ in results[:3]]
                    description = f"This image contains: {', '.join(top_objects)}."
                    st.write(description)
                else:
                    st.warning("No objects detected above the confidence threshold.")

# ====================
# WEBCAM CAPTURE
# ====================
elif input_method == "Webcam Capture":
    st.markdown("### 📷 Capture from Webcam")
    camera_image = st.camera_input("Take a picture")
    
    if camera_image and model:
        image = Image.open(camera_image)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="📸 Captured Image", use_column_width=True)
        
        with col2:
            st.subheader("🔍 Detection Results")
            
            with st.spinner("🔄 Analyzing..."):
                results = analyze_image(image, classes, model, processor, 
                                       confidence_threshold, top_k)
                
                if results:
                    for label, confidence in results:
                        st.markdown(f"""
                        <div style="margin-bottom: 15px;">
                            <div style="display: flex; justify-content: space-between;">
                                <span style="font-size: 18px;"><b>🎯 {label.title()}</b></span>
                                <span style="font-size: 18px; color: #00d2ff;"><b>{confidence:.1%}</b></span>
                            </div>
                            <div style="background-color: rgba(255,255,255,0.2); height: 10px; border-radius: 5px; margin-top: 5px;">
                                <div style="background: linear-gradient(90deg, #00d2ff, #3a7bd5); width: {confidence*100}%; height: 100%; border-radius: 5px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Generate description
                    st.markdown("---")
                    st.subheader("📝 Description")
                    top_objects = [label for label, _ in results[:3]]
                    description = f"The webcam shows: {', '.join(top_objects)}."
                    st.write(description)
                else:
                    st.warning("No objects detected above the confidence threshold.")

# ====================
# VIDEO UPLOAD
# ====================
elif input_method == "Upload Video":
    st.markdown("### 🎥 Upload and Analyze Video")
    uploaded_video = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_video and model:
        # Save uploaded video to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        video_path = tfile.name
        
        # Video processing options
        col1, col2 = st.columns(2)
        with col1:
            process_video = st.button("🎬 Process Video", type="primary")
        with col2:
            frame_skip = st.slider("Process every Nth frame", 1, 30, 5, 
                                   help="Higher values = faster processing, lower accuracy")
        
        if process_video:
            # Open video
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            st.info(f"📊 Video Info: {total_frames} frames @ {fps} FPS")
            
            # Create output video
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = None
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            frame_display = st.empty()
            
            frame_count = 0
            processed_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Initialize output writer
                if out is None:
                    height, width = frame.shape[:2]
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                # Process frame at intervals
                if frame_count % frame_skip == 0:
                    processed_frame = process_video_frame(frame, classes, model, 
                                                         processor, confidence_threshold)
                    out.write(processed_frame)
                    
                    # Update display
                    if processed_count % 10 == 0:
                        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        frame_display.image(rgb_frame, caption=f"Processing frame {frame_count}/{total_frames}", 
                                          use_column_width=True)
                    
                    processed_count += 1
                else:
                    out.write(frame)
                
                frame_count += 1
                progress_bar.progress(min(frame_count / total_frames, 1.0))
                status_text.text(f"Processing: {frame_count}/{total_frames} frames")
            
            cap.release()
            out.release()
            
            st.success("✅ Video processing complete!")
            
            # Display processed video
            st.video(output_path)
            
            # Download button
            with open(output_path, 'rb') as f:
                st.download_button(
                    label="📥 Download Processed Video",
                    data=f,
                    file_name="spectra_processed_video.mp4",
                    mime="video/mp4"
                )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.7);">
    <p>Spectra Object Recognition | Powered by CLIP & Transformers</p>
    <p>Trained on LAION-5B, COCO, Visual Genome, and more</p>
</div>
""", unsafe_allow_html=True)
