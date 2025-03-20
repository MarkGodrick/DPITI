import pandas as pd
import os

PATH = "lsun/bedroom_train/baseline/pe/sdxl-turbo/variation_degree_version02/lookahead_degree=4"
FILE = "fid_PE.EMBEDDING.Inception.csv"

file = pd.read_csv(os.path.join(PATH,FILE))

metrics = []
metrics.append(float(file.columns[1]))
for num in file[file.columns[1]]:
    metrics.append(num)

format_str = """<td align="center">{:.4}</td>"""

for num in metrics:
    print(format_str.format(num))