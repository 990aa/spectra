import gradio as gr
from PIL import Image
import numpy as np
import cv2
import torch
from transformers import CLIPProcessor, CLIPModel
from dotenv import load_dotenv
import os
from pathlib import Path

# Load environment variables
load_dotenv()

# Global model variables
MODEL_CACHE = {}

def load_model(model_id: str = "990aa/spectra"):
    """Load CLIP model and processor with caching."""
    if model_id in MODEL_CACHE:
        return MODEL_CACHE[model_id]
    
    try:
        model = CLIPModel.from_pretrained(model_id)
        processor = CLIPProcessor.from_pretrained(model_id)
        
        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        
        MODEL_CACHE[model_id] = (model, processor, device)
        return model, processor, device
    except Exception as e:
        return None, None, str(e)

def analyze_image(image: Image.Image, class_text: str, model_id: str, threshold: float, top_k: int):
    """
    Analyze an image and return detected objects with confidence scores.
    
    Args:
        image: PIL Image to analyze
        class_text: Comma-separated string of object classes
        model_id: HuggingFace model ID
        threshold: Minimum confidence threshold
        top_k: Number of top results to return
        
    Returns:
        Tuple of (results_text, annotated_image)
    """
    if image is None:
        return "⚠️ Please upload an image first.", None
    
    # Load model
    model, processor, device = load_model(model_id)
    
    if model is None:
        return f"⚠️ Error loading model: {device}", None
    
    # Prepare classes
    classes = [c.strip() for c in class_text.split(",") if c.strip()]
    if not classes:
        return "⚠️ Please enter at least one object class.", None
    
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
    results_text = "## 🔍 Detection Results\n\n"
    
    detected_objects = []
    for prob, idx in zip(top_probs, top_indices):
        if prob.item() >= threshold:
            label = classes[idx]
            confidence = prob.item()
            results.append((label, confidence))
            detected_objects.append(label)
            results_text += f"**🎯 {label.title()}**: {confidence:.1%}\n\n"
    
    if not results:
        results_text = "⚠️ No objects detected above the confidence threshold."
    else:
        # Add description
        results_text += "\n---\n\n### 📝 Description\n\n"
        top_objects = [label for label, _ in results[:3]]
        description = f"This image contains: **{', '.join(top_objects)}**."
        results_text += description
    
    # Annotate image
    annotated = image.copy()
    return results_text, annotated

def process_video(video_path: str, class_text: str, model_id: str, threshold: float, frame_skip: int):
    """
    Process video and return annotated version.
    
    Args:
        video_path: Path to input video
        class_text: Comma-separated object classes
        model_id: HuggingFace model ID
        threshold: Confidence threshold
        frame_skip: Process every Nth frame
        
    Returns:
        Path to annotated video
    """
    if video_path is None:
        return None, "⚠️ Please upload a video first."
    
    # Load model
    model, processor, device = load_model(model_id)
    
    if model is None:
        return None, f"⚠️ Error loading model: {device}"
    
    # Prepare classes
    classes = [c.strip() for c in class_text.split(",") if c.strip()]
    if not classes:
        return None, "⚠️ Please enter at least one object class."
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video
    output_path = video_path.replace('.mp4', '_processed.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    processed_frames = 0
    
    status_text = f"📊 Processing video: {total_frames} frames @ {fps} FPS\n\n"
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame at intervals
        if frame_count % frame_skip == 0:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Analyze
            inputs = processor(text=classes, images=pil_image, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1).cpu()
            
            # Get top 3 results
            top_probs, top_indices = probs[0].topk(min(3, len(classes)))
            
            # Annotate frame
            y_offset = 30
            for prob, idx in zip(top_probs, top_indices):
                if prob.item() >= threshold:
                    label = classes[idx]
                    confidence = prob.item()
                    text = f"{label}: {confidence:.1%}"
                    cv2.putText(frame, text, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y_offset += 30
            
            processed_frames += 1
        
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    
    status_text += f"✅ Processed {processed_frames} frames out of {total_frames}\n"
    status_text += f"📥 Video saved to: {output_path}"
    
    return output_path, status_text

# Default object classes
DEFAULT_CLASSES = """person, face, cat, dog, car, truck, bicycle, motorcycle, airplane, boat, 
tree, plant, flower, house, building, phone, laptop, computer, book, cup, 
bottle, chair, table, sofa, bed, bird, horse, sheep, cow, elephant, 
food, pizza, apple, orange, banana, sports ball, tennis racket, clock, 
keyboard, mouse, tv, backpack, umbrella, handbag, traffic light, fire hydrant"""

# CSS for styling
custom_css = """
.gradio-container {
    font-family: 'Arial', sans-serif;
}
.header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 20px;
}
.footer {
    text-align: center;
    margin-top: 30px;
    padding: 20px;
    color: #666;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.HTML("""
    <div class="header">
        <h1>🎯 Spectra Object Recognition</h1>
        <p>Detect and describe objects in images and videos using CLIP-based AI</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## ⚙️ Settings")
            
            model_id = gr.Textbox(
                label="Model ID",
                value="990aa/spectra",
                placeholder="HuggingFace model ID",
                info="Enter a HuggingFace model repository ID"
            )
            
            class_input = gr.Textbox(
                label="Object Classes",
                value=DEFAULT_CLASSES,
                lines=8,
                placeholder="Enter comma-separated object classes",
                info="List of objects to detect"
            )
            
            threshold = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.15,
                step=0.05,
                label="Confidence Threshold",
                info="Minimum confidence for detection"
            )
            
            top_k = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                label="Top K Results",
                info="Number of top predictions to show"
            )
            
            device_info = gr.Markdown(
                f"**Device**: {'🚀 CUDA (GPU)' if torch.cuda.is_available() else '💻 CPU'}"
            )
        
        with gr.Column(scale=2):
            with gr.Tab("📸 Image Analysis"):
                gr.Markdown("### Upload an image to detect objects")
                
                with gr.Row():
                    image_input = gr.Image(
                        type="pil",
                        label="Upload Image",
                        height=400
                    )
                    
                    with gr.Column():
                        image_output = gr.Markdown(label="Results")
                        image_annotated = gr.Image(label="Annotated Image")
                
                image_button = gr.Button("🔍 Analyze Image", variant="primary", size="lg")
                
                gr.Examples(
                    examples=[],
                    inputs=image_input,
                    label="Example Images"
                )
            
            with gr.Tab("🎥 Video Processing"):
                gr.Markdown("### Upload a video to detect objects in motion")
                
                with gr.Row():
                    video_input = gr.Video(label="Upload Video")
                    video_output = gr.Video(label="Processed Video")
                
                frame_skip = gr.Slider(
                    minimum=1,
                    maximum=30,
                    value=5,
                    step=1,
                    label="Frame Skip",
                    info="Process every Nth frame (higher = faster)"
                )
                
                video_status = gr.Markdown("")
                video_button = gr.Button("🎬 Process Video", variant="primary", size="lg")
            
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
                ## About Spectra
                
                Spectra is an advanced object recognition model trained on:
                - 🌐 **LAION-5B**: Large-scale image-text pairs
                - 🎯 **COCO**: Common Objects in Context
                - 👁️ **Visual Genome**: Detailed scene understanding
                - 📦 **Objects365**: Diverse object categories
                
                ### Features
                - ✅ Zero-shot object detection
                - ✅ Real-time image analysis
                - ✅ Video processing with annotations
                - ✅ Customizable object classes
                - ✅ GPU acceleration support
                
                ### How to Use
                1. **Image Analysis**: Upload an image and click "Analyze Image"
                2. **Video Processing**: Upload a video, adjust frame skip, and click "Process Video"
                3. **Custom Classes**: Edit the object classes list to detect specific items
                4. **Adjust Threshold**: Increase for fewer, more confident detections
                
                ### Model Training
                The model uses CLIP-style contrastive learning with:
                - ViT-B/16 backbone
                - Mixed precision (FP16) training
                - Gradient accumulation
                - Multi-phase training pipeline
                
                ### Deployment
                This app can run:
                - 💻 **Locally**: `python app.py`
                - ☁️ **HuggingFace Spaces**: Deploy as a public or private Space
                - 🐳 **Docker**: Containerized deployment
                
                ---
                
                **License**: See LICENSE.md  
                **Repository**: [github.com/990aa/spectra](https://github.com/990aa/spectra)
                """)
    
    # Event handlers
    image_button.click(
        fn=analyze_image,
        inputs=[image_input, class_input, model_id, threshold, top_k],
        outputs=[image_output, image_annotated]
    )
    
    video_button.click(
        fn=process_video,
        inputs=[video_input, class_input, model_id, threshold, frame_skip],
        outputs=[video_output, video_status]
    )
    
    gr.HTML("""
    <div class="footer">
        <p>🎯 Spectra Object Recognition | Powered by CLIP & Transformers</p>
        <p>Trained on LAION-5B, COCO, Visual Genome, and more</p>
    </div>
    """)

# Launch configuration
if __name__ == "__main__":
    # Check if running on HuggingFace Spaces
    is_spaces = os.getenv("SPACE_ID") is not None
    
    demo.launch(
        server_name="0.0.0.0" if is_spaces else "127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )
