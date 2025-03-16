import re
import pandas as pd
from pe.logging import setup_logging, execution_logger


FILE_NAME = "lsun/bedroom_train/openai/gpt-4o-mini/pe/meta-llama/noise_multiplier=0/few_shot_chat/synthetic_text/000000010.csv"
MIN_LEN = 100

df = pd.read_csv(FILE_NAME).astype(str)

pattern = r'([^:]*:|^)(.*?)(?=\.$|$)'

text_data = list(df['text'])

text_data = [text.strip() for text in text_data if len(text)>MIN_LEN]

matches = [re.search(pattern,text,re.DOTALL) for text in text_data]

text_list = [match.group(2).strip() for match in matches]

cnt = 0
for i,(a,b) in enumerate(zip(text_data,text_list)):
    if a[:-1]!=b and a!=b:
        cnt+=1
        print(f"-----------------------------------------\nitem {i}:\n-------------------\n{a}\n-------------------\n{b}\n")

print(f"Total different count: {cnt}")