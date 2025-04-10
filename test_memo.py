import streamlit as st
import nltk
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from difflib import SequenceMatcher
from html import escape
import time
import colorsys

# Load models
nltk.download('punkt')

@st.cache_resource
def load_model(model_name):
    def load_hfmodel(model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
        )
        return tokenizer, model

    print(f"Loading {model_name}. Please wait.")
    if model_name == "7B-Instruct-v0.2":
        llm_tokenizer, llm_model = load_hfmodel("mistralai/Mistral-7B-Instruct-v0.2")
    elif model_name == "Llama-2-7b-chat":
        llm_tokenizer, llm_model = load_hfmodel("meta-llama/Llama-2-7b-chat-hf")
    elif model_name == "Llama-2-70b-chat":
        llm_tokenizer, llm_model = load_hfmodel("meta-llama/Llama-2-70b-chat-hf")
    elif model_name == "Llama-3-8B":
        llm_tokenizer, llm_model = load_hfmodel("meta-llama/Meta-Llama-3-8B")
    elif model_name == "Llama-3-8B-Instruct":
        llm_tokenizer, llm_model = load_hfmodel("meta-llama/Meta-Llama-3-8B-Instruct")
    print(f"==== {model_name} loaded =====")
    st.success(f" {model_name} loaded")
    return llm_tokenizer, llm_model

# tokenizer, model = load_model("Llama-3-8B")
tokenizer, model = load_model("Llama-3-8B-Instruct")

# Function to calculate perplexity and log probabilities
def calculate_perplexity_and_logprobs(sentence):
    tokens = tokenizer.encode(sentence, return_tensors="pt")
    with torch.no_grad():
        output = model(tokens, labels=tokens)
        loss = output.loss
        logits = output.logits

    # Compute perplexity
    perplexity = torch.exp(loss).item()
    # Compute log probabilities
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_log_probs = [log_probs[0, i, token].item() for i, token in enumerate(tokens[0])]
    avg_log_prob = np.mean(token_log_probs)
    # Normalize log probabilities to a positive scale
    # normalized_log_prob = np.exp(avg_log_prob)  # Convert log prob to probability spa
    # print(avg_log_prob, normalized_log_prob)
    return perplexity, avg_log_prob

# Function to extract activations from the model
def extract_activations(sentence):
    tokens = tokenizer.encode(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(tokens, output_hidden_states = True)
        hidden_states = outputs.hidden_states  # Extract hidden states
    
    # Use the last layer's mean activation as a feature
    last_layer_activations = hidden_states[-1].mean(dim=1).squeeze().numpy()
    return last_layer_activations


# Function to analyze text familiarity
def analyze_text_familiarity(text):
    sentences = nltk.sent_tokenize(text)
    results = []
    
    for sentence in sentences:
        perplexity, normalized_log_prob = calculate_perplexity_and_logprobs(sentence)
        # familiarity_score = (1 / (1 + perplexity)) * normalized_log_prob  # Normalize score
        # familiarity_score = (1 / (1 + perplexity))  # Normalize score
        familiarity_score = np.clip(1 - (perplexity / 100), 0, 1)  # Adjusting scaling
        results.append((sentence, familiarity_score, perplexity, normalized_log_prob))
    
    return results

# Function to analyze text based on activation similarity
def analyze_text_factuality(text):
    sentences = nltk.sent_tokenize(text)
    results = []
    
    # Generate a baseline activation pattern for a factual knowledge reference
    reference_sentence = "The capital of France is Paris."
    reference_activation = extract_activations(reference_sentence)
    
    for sentence in sentences:
        activation = extract_activations(sentence)
        
        # Compute cosine similarity to the reference activation
        similarity = np.dot(reference_activation, activation) / (
            np.linalg.norm(reference_activation) * np.linalg.norm(activation)
        )
        
        # Scale similarity score for highlighting (normalize between 0 and 1)
        similarity_score = np.clip(similarity, 0, 1)
        # results.append((sentence, similarity_score))
        results.append(similarity_score)
    
    return results


def main():
    # Streamlit UI
    st.title("LLM Memorization")
    st.write(
        "Paste a long text and see which parts the LLM recognizes! ( Green = Unfamiliar, Red = Familiar)"
    )

    user_text = st.text_area("Paste your text here:", height = 100)
    if st.button("Analyze") and user_text:

        progress = st.progress(0, "Analyzing text")
        start = time.time()
        results = analyze_text_familiarity(user_text)
        activation_similarity = analyze_text_factuality(user_text)
        idx = 0        
        # HTML visualization
        highlighted_text = ""
        for sentence, score, perplexity, avg_log_prob in results:
            p = 1  - score
            act_sim = activation_similarity[idx]
            r, g, b = colorsys.hsv_to_rgb(p * (1/3), 1, 1)
            r, g, b = int(r * 255), int(g * 255), int(b * 255)
            color = f"rgba({r}, {g}, {b})"
            tooltip_text = f"Perplexity: {perplexity:.2f}, Log Prob: {avg_log_prob:.2f}, Activation Sim: {act_sim:.2f}, MS: {score:.2f}"
            highlighted_text += f'<span style="background-color: {color}; padding: 2px;" title="{tooltip_text}">{escape(sentence)}</span> '
            idx +=1

        st.markdown(f'<div style="font-size: 18px; line-height: 1.6;">{highlighted_text}</div>', unsafe_allow_html=True)
        duration = time.time() - start

        progress.progress(
            100, f"Processing time = {duration:.2f} seconds"
        )

if __name__ == "__main__":
    main()
