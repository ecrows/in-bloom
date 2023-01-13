import torch
from PIL import Image
from glob import glob
from pathlib import Path
import pandas as pd
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

df = pd.read_json("captioned_albums.json")
for d in df:

