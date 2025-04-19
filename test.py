import random
from transformers import GPT2Tokenizer

# Initialize the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# Given text
text = "A poised woman with curly blonde hair rests her chin on her hands, wearing a wide-brimmed black hat adorned with colorful flowers. Her serious gaze contrasts with the playful floral accessory."

# Function to mask words in the text with probability > 0.5
def mask_text_with_probability(text, mask_prob=0.5):
    tokens = tokenizer.encode(text)
    masked_text = []
    for word in tokens:
        if random.random() > mask_prob:
            masked_text.append('_')  # Mask the word
        else:
            masked_text.append(word)  # Keep the word as is

    return tokenizer.decode(masked_text)

# Apply masking
masked_text = mask_text_with_probability(text, mask_prob=0.5)
print(masked_text)
