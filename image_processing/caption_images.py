from transformers import AutoProcessor, Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image


def generate_image_caption_2(image_path):
# Load processor and model with half precision
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float32, device_map="auto"
    )

    # Optional: use torch.compile for speedup if using PyTorch 2.x
    raw_image = Image.open(image_path).convert('RGB').resize((224,224))
    try:
        model = torch.compile(model)
    except Exception:
        pass  # fallback if torch.compile is unavailable

    # Prepare inputs (single or batch can be used)
    inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float32)

    # Generate output with cache enabled for faster decoding
    with torch.no_grad():
        outputs = model.generate(**inputs, use_cache=True)

    # Decode and print the result
    caption = processor.decode(outputs[0], skip_special_tokens=True).strip()
    print(caption)
    raw_image