---
license: mit
tags:
- vision
- image-classification
- zero-shot-image-classification
- clip
datasets:
- laion/laion2B-en
- wikimedia/wit_base
- detection-datasets/coco
- huggingface/open-images-v7
base_model: laion/CLIP-ViT-B-16-laion2B-s34B-b88K
---

# Spectra: Robust Object Recognition Model

Spectra is a CLIP-based object recognition model trained on a massive scale using a combination of image-text pairs (LAION-5B, WIT) and converted detection datasets (COCO, OpenImages, Objects365).

## Model Details
- **Backbone**: `laion/CLIP-ViT-B-16-laion2B-s34B-b88K` (OpenCLIP ViT-B/16)
- **Training Strategy**: 
    - **Phase 1**: Contrastive pretraining on LAION-5B and WIT.
    - **Phase 2**: Fine-tuning on detection datasets (COCO, OpenImages) converted to image-text format.
- **Optimizations**: Mixed Precision (FP16), Gradient Accumulation, Torch Compile.

## Usage

### Zero-Shot Classification
```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

model = CLIPModel.from_pretrained("a-01a/spectra")
processor = CLIPProcessor.from_pretrained("a-01a/spectra")

image = Image.open("image.jpg")
inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)
print(probs)
```

## Training Data
- **LAION-5B**: ~5.85B image-text pairs.
- **WIT**: Wikipedia Image-Text.
- **COCO**: Common Objects in Context.
- **OpenImages V7**: Large-scale detection dataset.
- **Objects365**: 365 object categories.

## Demo
A Streamlit demo is available in the repository. Run `streamlit run app.py` to launch.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
Copyright (c) 2025 Abdul Ahad.
