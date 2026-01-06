import torch
from PIL import Image
from lavis.models import load_model_and_preprocess

device = "cuda" if torch.cuda.is_available() else "cpu"
model, vis, txt = load_model_and_preprocess(
    name="blip_feature_extractor", model_type="base", is_eval=True, device=device
)

image_path = "~/ljj/imgflip_data/images/73ci94.jpg"
cap = "a test caption"

image = vis["eval"](Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
text_input = txt["eval"](cap)

sample = {"image": image, "text_input": [text_input]}

fi = model.extract_features(sample, mode="image")
ft = model.extract_features(sample, mode="text")

print("image keys:", list(fi.keys()))
print("text keys:", list(ft.keys()))
print("image_embeds_proj shape:", getattr(fi, "image_embeds_proj").shape)
print("text_embeds_proj shape:", getattr(ft, "text_embeds_proj").shape)