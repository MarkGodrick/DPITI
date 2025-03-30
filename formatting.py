import pandas as pd
import os

PATH = "results/cat/openai/gpt-4o-mini/pe/meta-llama/num_samples_schedule=200/epsilon=7.89/few_shot_chat_fill_in_the_blanks/fid_PE.EMBEDDING.T2I_embedding.stabilityai-sdxl-turbo_{'PE.VARIATION_API_FOLD_ID':0}.csv"

file = pd.read_csv(PATH)

metrics = []
metrics.append(float(file.columns[1]))
for num in file[file.columns[1]]:
    metrics.append(num)

format_str = """<td align="center">{:.4}</td>"""

for num in metrics:
    print(format_str.format(num))