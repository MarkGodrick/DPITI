from datasets import load_dataset
from openai import OpenAI
from PIL import Image
import base64
import os
import io

api_key = os.environ["OPENAI_API_KEY"] 
client = OpenAI(api_key=api_key)
img = Image.open("image.jpg")
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

# messages = [
#     {
#         "role":"user",
#         "content":[
#             {
#                     "type":"text",
#                     "text":"对于\frac{d^2}{dt^2}(\det(X+tV))^{1/n},$X\in\mathbb{S}^n_{++},V\in\mathbb{S}^n$?我不仅想要求值还要判定其正负性"
#             }
#         ],
#     }
# ]

response = client.chat.completions.create(model="gpt-4o-mini",messages = messages,temperature=1.2)

print(response.choices[0].message.content)
