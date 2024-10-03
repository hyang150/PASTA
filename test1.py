from pastalib.pasta import PASTA
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize pre-trained LLM
name = "huggyllama/llama-7b"
model = AutoModelForCausalLM.from_pretrained(name)
tokenizer = AutoTokenizer.from_pretrained(name, legacy=False)
tokenizer.pad_token = tokenizer.eos_token

# Add a pad token if the tokenizer does not have one
# Select the attention heads to be steered
head_config = {
    "3": [17, 7, 6, 12, 18], "8": [28, 21, 24], "5": [24, 4], 
    "0": [17], "4": [3], "6": [14], "7": [13], "11": [16], 
}

# Initialize the PASTA steerer
pasta = PASTA(
    model=model,
    tokenizer=tokenizer,
    head_config=head_config, 
    alpha=0.01,  # scaling coefficient
    scale_position="exclude",  # downweighting unselected tokens
)

# Model Input
texts = ["Mary is a doctor. She obtains her bachelor degree from UCSD. Answer the occupation of Mary and generate the answer as json format."]

# Tokenize the inputs manually to include attention_mask
inputs = tokenizer(
    texts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    return_attention_mask=True  # Ensure attention mask is generated
)

# Ensure attention_mask is passed to PASTA
if 'attention_mask' not in inputs:
    inputs['attention_mask'] = torch.ones(inputs['input_ids'].shape, dtype=torch.long)

# Call PASTA's inputs_from_batch without extra arguments
inputs, offset_mapping = pasta.inputs_from_batch(texts)

# User highlights specific input spans
emphasized_texts = ["Answer the occupation of Mary and generate the answer as json format"]

# PASTA registers the pre_forward_hook to edit attention
with pasta.apply_steering(
    model=model, 
    strings=texts, 
    substrings=emphasized_texts, 
    model_input=inputs, 
    offsets_mapping=offset_mapping
) as steered_model:
    outputs = steered_model.generate(**inputs, max_new_tokens=128)

# Print the outputs
print(outputs)
