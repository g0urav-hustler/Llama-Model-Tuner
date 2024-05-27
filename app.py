


import streamlit as st




# Run text generation pipeline with our next model

def load_model():
    prompt = "What is cross-validation?"
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
    result = pipe(f"[INST] {prompt} [/INST]")
    print(result[0]['generated_text'])

