# Spectra: Object Recognition Model

## Quick Start

### Prerequisites
- Python 3.10+
- Git
- UV package manager
- CUDA-compatible GPU (for training)
- HuggingFace account

### Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/990aa/spectra.git
cd spectra
```

#### 2. Install UV (if not already installed)
```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 3. Sync Dependencies
```bash
uv sync
```

This will install all required packages including:
- PyTorch
- Transformers
- Datasets
- Accelerate
- Streamlit
- And more...

### Training

#### Setup Environment Variables
Create a `.env` file in the project root:
```bash
HF_TOKEN=your_huggingface_token_here
```

Get token from: https://huggingface.co/settings/tokens

#### Run Training Notebook
```bash
# Launch Jupyter Lab
uv run jupyter lab spectra.ipynb

# Or use VS Code with Jupyter extension
code spectra.ipynb
```

**Training Features:**
- Automatic checkpointing to HuggingFace Hub
- Resume training from any checkpoint
- Multi-phase training (LAION → Detection datasets)
- Progress tracking with landmarks

### Running the Demo App

```bash
uv run streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

**Demo Features:**
- Upload images or use webcam
- Zero-shot object recognition
- Customizable object classes
- Real-time inference

## Usage Guide

### Using the Demo Interface

1. **Launch the app** (see above)
2. **Enter Model ID**: Default is `990aa/spectra`
3. **Define Object Classes**: Enter comma-separated classes (e.g., "cat, dog, car, tree")
4. **Choose Input**:
   - Upload Image: Click "Browse files"
   - Webcam: Enable camera and click capture
5. **Analyze**: Click "Analyze Image" button
6. **View Results**: See top 5 predictions with confidence scores

### Training from Scratch

1. Open `spectra.ipynb`
2. Run cells sequentially
3. Monitor training progress in the notebook
4. Check HuggingFace Hub for uploaded checkpoints

### Resuming Training

The notebook automatically detects previous checkpoints:
1. Run the notebook normally
2. It will download the latest `training_state.json` from HF
3. Training resumes from the saved step

## Project Structure

```
spectra/
├── spectra.ipynb          # Main training notebook
├── app.py                 # Streamlit demo application
├── README.md          # Detailed model documentation
├── README.md              # Model card for HuggingFace
├── INSTRUCTIONS.md        # This file
├── REPORT.md              # Technical report
├── .env                   # Environment variables (create this)
├── pyproject.toml         # UV dependencies
└── .venv/                 # Virtual environment (created by uv)
```

## Checking Your Model

### On HuggingFace Hub
Visit: https://huggingface.co/990aa/spectra

**What to check:**
- Model files (pytorch_model.bin)
- Training state (training_state.json)
- Model card (README.md)
- Configuration files

### On GitHub
Visit: https://github.com/990aa/spectra

**What to check:**
- Latest code updates
- Issues and discussions
- Documentation

## Troubleshooting

### Issue: `uv: command not found`
**Solution**: UV is not installed or not in PATH
```bash
# Reinstall UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (Linux/macOS)
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Issue: `HF_TOKEN not found`
**Solution**: Create `.env` file with your token
```bash
echo "HF_TOKEN=hf_your_token_here" > .env
```

### Issue: Out of Memory during training
**Solutions**:
1. Reduce batch size in CONFIG:
   ```python
   "batch_size": 32,  # Instead of 64
   ```
2. Increase gradient accumulation:
   ```python
   "grad_accum_steps": 8,  # Instead of 4
   ```
3. Use smaller model as fallback (already implemented)

### Issue: CUDA Out of Memory
**Solutions**:
1. Enable mixed precision (already enabled):
   ```python
   "mixed_precision": "fp16"
   ```
2. Clear cache:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```
3. Close other GPU applications

### Issue: Slow data loading
**Solutions**:
1. Reduce num_workers:
   ```python
   num_workers=2,  # Instead of 4
   ```
2. Check internet connection (streaming datasets)
3. Use smaller buffer size:
   ```python
   dataset.shuffle(buffer_size=500)  # Instead of 1000
   ```

### Issue: Streamlit app not loading model
**Solutions**:
1. **Check model ID**: Ensure `990aa/spectra` exists on HF
2. **Test with base model**: Use `laion/CLIP-ViT-B-16-laion2B-s34B-b88K`
3. **Check internet**: Model downloads require connectivity
4. **Clear cache**:
   ```bash
   rm -rf ~/.cache/huggingface/hub
   ```

### Issue: Dataset not found
**Solution**: Some datasets require authentication
```python
from huggingface_hub import login
login(token="your_token")
```

### Issue: Training interrupted
**Good News**: Training is automatically resumable!
1. Simply restart the notebook
2. Run all cells
3. Training continues from last checkpoint

### Issue: Push to Hub fails
**Solutions**:
1. Check token permissions (needs write access)
2. Verify repo exists: https://huggingface.co/990aa/spectra
3. Create repo manually if needed:
   ```python
   from huggingface_hub import create_repo
   create_repo("990aa/spectra", exist_ok=True)
   ```

## Performance Tips

### Training Optimization
1. **Use Colab Pro**: For better GPUs (V100/A100)
2. **Enable Torch Compile**: Already implemented, ~20% speedup
3. **Monitor GPU**: Use `nvidia-smi` to check utilization
4. **Batch Size**: Experiment with larger batches for better GPU usage

### Inference Optimization
1. **Model Quantization**: For faster inference
   ```python
   model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
   ```
2. **Batch Processing**: Process multiple images together
3. **ONNX Export**: For production deployment

## Support

- **Issues**: https://github.com/990aa/spectra/issues
- **Discussions**: https://github.com/990aa/spectra/discussions
- **HuggingFace**: https://huggingface.co/990aa/spectra/discussions

---