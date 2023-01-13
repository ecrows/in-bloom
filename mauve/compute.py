import mauve
import pandas as pd
import pickle

SEED = 0

NUM_SAMPLES = 3000

model_name="hfl/chinese-roberta-wwm-ext-large"
#model_name="uer/gpt2-chinese-cluecorpussmall",

p080 = pd.read_json("BLOOM/3K_BLOOM_p080.jsonl", lines=True)
p085 = pd.read_json("BLOOM/3K_BLOOM_p085.jsonl", lines=True)
p090 = pd.read_json("BLOOM/3K_BLOOM_p090.jsonl", lines=True)
p095 = pd.read_json("BLOOM/3K_BLOOM_p095.jsonl", lines=True)
p099 = pd.read_json("BLOOM/3K_BLOOM_p099.jsonl", lines=True)
mauve_p80 = p080["response"].apply(lambda a: a[0]["generated_text"]).str.split("下面是一首歌的中文歌词。\n歌词：\n").apply(lambda a: a[1])
mauve_p85 = p085["response"].apply(lambda a: a[0]["generated_text"]).str.split("下面是一首歌的中文歌词。\n歌词：\n").apply(lambda a: a[1])
mauve_p90 = p090["response"].apply(lambda a: a[0]["generated_text"]).str.split("下面是一首歌的中文歌词。\n歌词：\n").apply(lambda a: a[1])
mauve_p95 = p095["response"].apply(lambda a: a[0]["generated_text"]).str.split("下面是一首歌的中文歌词。\n歌词：\n").apply(lambda a: a[1])
mauve_p99 = p099["response"].apply(lambda a: a[0]["generated_text"]).str.split("下面是一首歌的中文歌词。\n歌词：\n").apply(lambda a: a[1])

real_text = pd.read_json("BLOOM/top_song_lyrics_cleaned_cn.jsonl", lines=True).sample(n=NUM_SAMPLES, random_state=SEED)

# out = mauve.compute_mauve(p_text=real_text["cleaned_lyrics"].to_list(), q_text=mauve_p80.to_list(), device_id=0, max_text_length=128, verbose=False, featurize_model_name=model_name, batch_size=32, seed=0)
# print(out)
# 
# with open("3k_p080mauve.pkl", "wb") as f:
#     pickle.dump(out, f)
# 
# out = mauve.compute_mauve(p_text=real_text["cleaned_lyrics"].to_list(), q_text=mauve_p85.to_list(), device_id=0, max_text_length=128, verbose=False, featurize_model_name=model_name, batch_size=32, seed=0)
# 
# print(out)
# 
# with open("3k_p085mauve.pkl", "wb") as f:
#     pickle.dump(out, f)
# 
# out = mauve.compute_mauve(p_text=real_text["cleaned_lyrics"].to_list(), q_text=mauve_p90.to_list(), device_id=0, max_text_length=128, verbose=False, featurize_model_name=model_name, batch_size=32, seed=0)
# print(out)
# 
# with open("3k_p090mauve.pkl", "wb") as f:
#     pickle.dump(out, f)
# 
# out = mauve.compute_mauve(p_text=real_text["cleaned_lyrics"].to_list(), q_text=mauve_p95.to_list(), device_id=0, max_text_length=128, verbose=False, featurize_model_name=model_name, batch_size=32, seed=0)
# print(out)
# 
# with open("3k_p095mauve.pkl", "wb") as f:
#     pickle.dump(out, f)
# 
out = mauve.compute_mauve(p_text=real_text["cleaned_lyrics"].to_list(), q_text=mauve_p99.to_list(), device_id=0, max_text_length=128, verbose=False, featurize_model_name=model_name, batch_size=32, seed=0)
print(out)

with open("3k_p099mauve.pkl", "wb") as f:
    pickle.dump(out, f)
