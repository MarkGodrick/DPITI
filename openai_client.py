from datasets import load_dataset
from openai import OpenAI
from PIL import Image
import base64
import os
import io

api_key = os.environ["OPENAI_API_KEY"] 
client = OpenAI(api_key=api_key)
ds = load_dataset("jxie/camelyon17",split="id_train")
img = Image.Image()
for idx,item in enumerate(ds):
    if item['label']:
        img = ds[idx]['image']
        break
buffered = io.BytesIO()
img.save(buffered,format="JPEG")
encoded_img = base64.b64encode(buffered.getvalue()).decode("utf-8")
messages = [
    {
        "role":"user",
        "content":[
            {
                    "type":"text",
                    "text":"Describe the image with details so that diffusion models can generate a similar picture. Control token num within 100 tokens."
            },
            {
                    "type":"image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_img}"},
            },
        ],
    }
]

response = client.chat.completions.create(model="gpt-4o-mini",messages = messages,temperature=1.2,max_completion_tokens=100)

print(response.choices[0].message.content)
