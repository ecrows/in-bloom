import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
from glob import glob
from pathlib import Path
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="large_coco", is_eval=True, device=device)


still_images = glob("pixelart\PixelArt\*.png")
still_images.extend(glob("pixelart\PixelArt\*.jpg"))
frameZeros = glob("pixelAnim/**/*frame0.png")
combo_files = still_images + frameZeros

breakdown = [s.split("\\")[-1].rsplit('.')[0].rsplit('_') for s in still_images]
breakdown.extend([a.split("\\frame0.png")[0].split("pixelAnim\\")[-1].rsplit('_') for a in frameZeros])
posts = []
for i, b in enumerate(breakdown):
    if len(b[-1]) <= 1:
        del(b[-1])

    if len(b[-1]) == 6:
        del(b[-1])

    posts.append({"title": b[-1], "author": '_'.join(b[:-1]), "image": combo_files[i], "img_id":i})

# load sample image

captioned_posts = []
for p in posts:
    raw_image = Image.open(p["image"]).convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    # generate caption
    caption = model.generate({"image": image})
    print(caption)
    p["caption"] = caption
    captioned_posts.append(p)
    
pd.DataFrame(captioned_posts).to_json("captioned_pixelart.json")

