from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_message(context):

    prompt = f"""
You are an assistive AI helping caregivers understand a non-verbal autistic child.

Signals detected:
{context}

Based on these signals, write ONE short sentence describing what the child might be communicating.

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response.split("Answer:")[-1].strip()