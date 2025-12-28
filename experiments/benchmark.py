from src import load_model, selection_loop, generate_batch
import time
import datasets
import re
import json

def clean_number(text):
    text = text.replace(",", "").replace("$", "").strip()
    try:
        return float(text)
    except ValueError:
        return None

def extract_answer(text):
    match = re.search(r"\\boxed\{(.+?)\}", text)
    if match:
        val = clean_number(match.group(1))
        if val is not None: return val

    match = re.search(r"####\s*(-?[\d,.]+)", text)
    if match:
        val = clean_number(match.group(1))
        if val is not None: return val
  
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if numbers:
        return clean_number(numbers[-1])
        
    return None

def is_correct(model_ans, gt_ans):
    if model_ans is None or gt_ans is None:
        return False
    return abs(model_ans - gt_ans) < 1e-6

def run_zero_shot(model, tokenizer, prompt):
    start = time.time()
    prompt = (
        "Solve the following math problem step by step without code (just math)\n"
        f"Math Question: {prompt}\n"
        "Output the final answer inside \\boxed at the end."
    )
    response = generate_batch(model, tokenizer, prompt, n=1, max_length=1024)  
    elapsed = time.time() - start
    print("Zero-shot: ", response[0])
    return {"answer": extract_answer(response[0]), "time": elapsed, "trace": response[0]}

def run_mcts(model, tokenizer, prompt, max_iter=8):
    start = time.time()
    current_state = (
        "Solve the following math problem step by step without code (just math)\n"
        f"Math Question: {prompt}\n"
        "Output the final answer inside \\boxed at the end."
    )
    for _ in range(max_iter):
        response = selection_loop(current_state, model, tokenizer, max_iter=8)
        if not response:
            break
        current_state = response.state
        if response.is_terminal:
            print("Terminal Node Value:", response.value)
            if response.value >= 0.8:
                break
    
    elapsed = time.time() - start
    print("MCTS: ", current_state)
    return {"answer": extract_answer(current_state), "time": elapsed, "trace": current_state}

def main():
    print("Loading model...")
    model, tokenizer = load_model()

    print("Loading dataset...")
    dataset = datasets.load_dataset("gsm8k", "main")

    test_data = dataset["test"]
    questions = test_data["question"][:50]
    answers = test_data["answer"][:50]

    output_file = "experiments/benchmark_results.json"

    correct_zero = 0
    correct_mcts = 0

    with open(output_file, "w") as f:
        pass

    print("Starting benchmark...")
    for i, (q, gt) in enumerate(zip(questions, answers)):
        gt_val = extract_answer(gt)
        zero_shot = run_zero_shot(model, tokenizer, q)
        mcts = run_mcts(model, tokenizer, q)

        correct_zero += is_correct(zero_shot["answer"], gt_val)
        correct_mcts += is_correct(mcts["answer"], gt_val)

        print("Question: ", q)
        print("Ground Truth: ", gt_val)
        print("Zero-shot: ", zero_shot["answer"], zero_shot["time"])
        print("MCTS: ", mcts["answer"], mcts["time"])
        print("=="*30, end="\n\n")
        results = {
            "question": q, 
            "ground_truth": gt_val,
            "zero_shot": {
                "answer": zero_shot["answer"],
                "correct": is_correct(zero_shot["answer"], gt_val),
                "time": zero_shot["time"],
            },
            "mcts": {
                "answer": mcts["answer"],
                "correct": is_correct(mcts["answer"], gt_val),
                "time": mcts["time"],
            }
        }

        with open(output_file, "a") as f:
            f.write(json.dumps(results) + "\n")

    print(f"Zero-shot accuracy: {correct_zero / len(questions)}")
    print(f"MCTS accuracy: {correct_mcts / len(questions)}")

    
if __name__ == "__main__":
    main()