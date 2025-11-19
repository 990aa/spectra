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

## Features

### Image Analysis
- Upload any image (JPG, PNG, WEBP, etc.)
- Instant object detection
- Confidence scores for each detection
- Natural language descriptions

### Video Processing
- Upload videos (MP4, AVI, MOV, MKV)
- Frame-by-frame object detection
- Annotated video output
- Downloadable results

### Customization
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

```
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

## API Access

Gradio automatically provides an API endpoint.

Access at: `http://localhost:7860/api/`

## Support

- Issues: [GitHub Issues](https://github.com/990aa/spectra/issues)
- Discussions: [GitHub Discussions](https://github.com/990aa/spectra/discussions)
- Docs: [USAGE.md](USAGE.md)
