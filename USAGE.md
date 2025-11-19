# Spectra Usage Guide

## Overview
Spectra is an advanced object recognition model trained on LAION-5B, COCO, Visual Genome, and other large-scale datasets using CLIP-style contrastive learning.

## Features

### Training Pipeline (`spectra.ipynb`)
- ✅ **Multi-phase training** on diverse datasets
- ✅ **Streaming data** - zero disk usage for massive datasets
- ✅ **Robust checkpointing** - resumable training from HF Hub
- ✅ **Mixed precision** (FP16) training
- ✅ **Automatic error handling** for dataset loading

### Inference App (`app.py`)
- 📸 **Image Upload** - Analyze static images
- 📷 **Webcam Capture** - Real-time object detection
- 🎥 **Video Processing** - Annotate videos with detected objects

## Quick Start

### 1. Install Dependencies
```bash
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install all dependencies
pip install -e .
```

### 2. Set Up Environment
Create a `.env` file in the project root:
```env
HF_TOKEN=your_huggingface_token_here
```

### 3. Run the App
```bash
streamlit run app.py
```

### 4. Train the Model
Open `spectra.ipynb` and run all cells to start training.

## App Features

### Image Upload Mode
1. Select "Upload Image" 
2. Choose an image file (JPG, PNG, WEBP, BMP)
3. View detected objects with confidence scores
4. Get natural language descriptions

### Webcam Mode
1. Select "Webcam Capture"
2. Allow browser camera access
3. Capture a photo
4. Instant object detection results

### Video Processing Mode
1. Select "Upload Video"
2. Upload video file (MP4, AVI, MOV, MKV)
3. Adjust frame processing rate (higher = faster)
4. Click "Process Video"
5. Download annotated video

## Configuration Options

### Sidebar Settings
- **Model ID**: HuggingFace model repository
- **Object Classes**: Comma-separated list of objects to detect
- **Confidence Threshold**: Minimum detection confidence (0.0-1.0)
- **Top K Results**: Number of top predictions to show

### Default Object Classes
The app includes 40+ pre-configured object classes:
- People & Animals: person, face, cat, dog, bird, horse, etc.
- Vehicles: car, truck, bicycle, motorcycle, airplane, boat
- Indoor Objects: phone, laptop, book, cup, chair, table, sofa
- Outdoor Objects: tree, plant, flower, house, building
- Food Items: pizza, apple, orange, banana
- Sports & Activities: sports ball, tennis racket

## Training Details

### Phase 1: Large-scale Caption Datasets
- LAION-2B English subset
- Wikipedia Image-Text (WIT)
- Conceptual Captions

### Phase 2: Object Detection Datasets
- COCO (Common Objects in Context)
- Visual Genome
- Objects365

### Key Improvements in Notebook
- ✅ Fixed KeyError by adding safe field extraction
- ✅ Added proper exception handling for dataset loading
- ✅ Support for URL-based images (auto-download)
- ✅ Fallback mechanisms when datasets are unavailable
- ✅ Detailed logging of loaded datasets

## Troubleshooting

### Dataset Loading Errors
The notebook now includes fallback mechanisms. If a dataset fails to load, it will:
1. Log a warning
2. Continue with available datasets
3. Fall back to Phase 1 if Phase 2 datasets are unavailable

### Model Not Loading in App
Check:
- Valid HuggingFace token in `.env`
- Correct model ID
- Internet connection
- Sufficient RAM/VRAM

### Video Processing Slow
- Increase "frame skip" value (process every Nth frame)
- Use shorter videos for testing
- Ensure GPU is available (check sidebar for device info)

## Hardware Requirements

### Minimum
- CPU: 4+ cores
- RAM: 16GB
- Storage: 10GB free

### Recommended
- GPU: NVIDIA GPU with 8GB+ VRAM
- RAM: 32GB+
- Storage: 50GB+ for training

## Next Steps

1. **Fine-tune on custom data**: Add your own datasets to the training pipeline
2. **Deploy to production**: Use FastAPI or Gradio for production deployment
3. **Optimize for mobile**: Export to ONNX for edge devices
4. **Add more features**: Object tracking, segmentation, pose estimation

## Support

For issues or questions:
- Check error logs in the app's expander
- Review notebook cell outputs
- Verify environment setup

## License
See LICENSE.md for details.
