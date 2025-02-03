import pandas as pd

df = pd.read_csv("output_audio/metadata.csv", delimiter="|", header=None, names=["path", "sentence"], encoding="utf-8")
df['speaker'] = "speaker_name" 
df['speaker'] = "1" 
df.to_csv("output_audio/metadata_updt.csv", index=True)
