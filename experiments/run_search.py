
from src import load_model, selection_loop
import time

def solve_math_problem(question, max_steps=10):
    print("Loading model...")
    model, tokenizer = load_model()

    current_state = f"Question: {question}\nLet's think step by step and output the final answer inside \\boxed{{}}."
    
    print(f"\nThinking about: {question}...\n")
    
    start_time = time.time()
    for step_num in range(max_steps):
        print(f"--- Step {step_num + 1} Analysis ---")
        
        best_node = selection_loop(current_state, model, tokenizer, max_iter=10)
        
        if not best_node:
            print("Error: MCTS could not find a valid next step.")
            break
            
        new_text = best_node.state[len(current_state):] # Get the difference
        print(f"Decided: {new_text.strip()} (Confidence: {best_node.visits} visits)")
        
        current_state = best_node.state
        
        if "\\boxed{" in new_text or "answer is" in new_text:
            print("\n!!! Solution Found !!!")
            break
            
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
    return current_state

if __name__ == "__main__":
    final_trace = solve_math_problem("What is 15 * 13?")
    
    print("\n" + "="*30)
    print("FINAL TRACE")
    print("="*30)
    print(final_trace)