# Spectra Gradio App - Quick Start

## Run Locally

```bash
# Option 1: Direct run
python app.py

# Option 2: Using uv
uv run python app.py

# Option 3: Using Gradio CLI
gradio app.py
```

The app will be available at: **http://127.0.0.1:7860**

## Deploy to HuggingFace Spaces

See [README_SPACES.md](README_SPACES.md) for detailed deployment instructions.

Quick steps:
1. Go to https://huggingface.co/new-space
2. Select **Gradio** as SDK
3. Upload `app.py` and `requirements.txt`
4. Done! Space will auto-deploy

## Features

### 📸 Image Analysis
- Upload any image (JPG, PNG, WEBP, etc.)
- Instant object detection
- Confidence scores for each detection
- Natural language descriptions

### 🎥 Video Processing
- Upload videos (MP4, AVI, MOV, MKV)
- Frame-by-frame object detection
- Annotated video output
- Downloadable results

### ⚙️ Customization
- **Model ID**: Use any CLIP-compatible model
- **Object Classes**: Define custom detection categories
- **Confidence Threshold**: Filter low-confidence detections
- **Top K Results**: Limit number of predictions
- **Frame Skip**: Balance speed vs accuracy for videos

## Tips for Best Results

1. **Image Quality**: Use high-resolution images for better accuracy
2. **Specific Classes**: More specific class names improve detection
3. **Threshold Tuning**: Lower for exploratory, higher for precision
4. **GPU Acceleration**: Detected automatically if available
5. **Video Processing**: Increase frame skip for faster processing

## Model Information

Default model: `990aa/spectra`

Change in the UI or edit `app.py`:
```python
model_id = gr.Textbox(value="your-username/your-model")
```

## Troubleshooting

### App won't start
- Check Python version (3.9+)
- Install dependencies: `pip install -r requirements.txt`
- Check port 7860 is available

### Model loading fails
- Verify model ID is correct
- Check internet connection
- For private models, set `HF_TOKEN` in `.env`

### Slow inference
- GPU recommended for video processing
- Increase frame skip for videos
- Use smaller model (e.g., `openai/clip-vit-base-patch32`)

### Out of memory
- Reduce image resolution
- Increase frame skip for videos
- Use CPU instead of GPU for small tasks

## Environment Variables

Create `.env` file:
```env
HF_TOKEN=your_huggingface_token_here
```

Required only for:
- Private models
- Pushing to HuggingFace Hub
- Higher rate limits

## Development

Edit `app.py` and the app will auto-reload (Gradio feature).

Test different configurations:
- Modify `DEFAULT_CLASSES` for different object sets
- Adjust `threshold` default values
- Change UI layout in the Gradio Blocks

## Production Deployment

### Option 1: HuggingFace Spaces (Recommended)
- Free tier available
- Automatic scaling
- SSL included
- See README_SPACES.md

### Option 2: Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py .
EXPOSE 7860
CMD ["python", "app.py"]
```

### Option 3: Cloud VM
Deploy to AWS, GCP, Azure:
```bash
python app.py
# Or use Gunicorn/Uvicorn for production
```

## API Access

Gradio automatically provides an API endpoint.

Access at: `http://localhost:7860/api/`

Python client:
```python
from gradio_client import Client

client = Client("http://localhost:7860")
result = client.predict(
    image,          # PIL Image or filepath
    "cat, dog",     # classes
    "990aa/spectra", # model_id
    0.15,           # threshold
    5,              # top_k
    api_name="/analyze_image"
)
```

## Support

- Issues: [GitHub Issues](https://github.com/990aa/spectra/issues)
- Discussions: [GitHub Discussions](https://github.com/990aa/spectra/discussions)
- Docs: [USAGE.md](USAGE.md)
