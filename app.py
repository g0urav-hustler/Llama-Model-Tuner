


import streamlit as st


def create_data(question, answer):
  text = f"<s> [INST] {question} [/INST] {answer} </s>"
  return text

# Run text generation pipeline with our next model

def load_model():
    prompt = "What is cross-validation?"
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
    result = pipe(f"[INST] {prompt} [/INST]")
    print(result[0]['generated_text'])

