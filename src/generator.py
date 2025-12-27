from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B-Instruct"

def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="./"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer

def get_last_token_logits(model, tokenizer, in_str):
    in_ids = tokenizer(in_str, return_tensors="pt").input_ids.to(model.device) # (B, T)

    # Get logits
    with torch.no_grad():
        logits = model(in_ids).logits # (B, T, C)

    return logits[:, -1, :]

def get_next_token(logits, temperature=0.0):
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
    else:
        next_token = torch.argmax(logits, dim=-1)
    return next_token.item()

def generate_batch(model, tokenizer, prompt, n=1, max_length=50, temperature=0.7):
    model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_length,
        num_return_sequences=n,
        temperature=temperature,
        do_sample=(temperature > 0),
        top_k=50,
        pad_token_id=tokenizer.eos_token_id,
        stop_strings=["Question:", "Answer:", "<|im_ed|>"],
        tokenizer=tokenizer
    )

    decoded_response = []
    input_len = model_inputs.input_ids.shape[1]
    for i in range(n):
        new_tokens = generated_ids[i][input_len:]
        decoded_response.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return decoded_response

def generate_answer(model, tokenizer, prompt, temperature=0.0, max_length=30, verbose=False):
    if verbose:
        print("Prompt: ", prompt, end="\nResponse: ")
    output_text = ""
    for _ in range(max_length):
        logits = get_last_token_logits(model, tokenizer, prompt)
        next_token_id = get_next_token(logits, temperature)
        next_token_str = tokenizer.decode(next_token_id)
        
        if next_token_str in {"<|im_end|>", "\n\n"}: 
            break
        
        prompt += next_token_str
        output_text += next_token_str

        if verbose:
            print(next_token_str, end="", flush=True)
    if verbose: print()
    return output_text.strip()
        

if __name__ == "__main__":
    model, tokenizer = load_model()

    prompt = "The answer to 15 * 13 is"

    generate_answer(model, tokenizer, prompt)
        
