from google import genai 
import dotenv
import os
import time
import re
from .generator import load_model, generate_batch
from google.api_core import exceptions
dotenv.load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def grade_batch(question, candidate_answers):
    message = [
        {
            "role": "user",
            "parts": [
                {
                    "text": (
                        "You are an expert math grader. I will show you a math question with current progress and several possible next steps.\n"
                        "Grade each candidate next step on a scale of 0.0 to 1.0 based on logic and correctness.\n"
                        "Rules:\n"
                        "1. Return ONLY a Python list of floats, e.g., [0.9, 0.4, 0.1].\n"
                        "2. Do not write any other text.\n"
                        f"Question + Current Progress: {question}\n\n"
                        f"Candidate Answers:\n{candidate_answers}"
                    )
                }
            ]
        }
    ]
    print("Grading answers: ", candidate_answers)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=message,
            )
            print("Grading response: ", response.text)
            match = re.search("\[(.*?)\]", response.text)
            if match:
                return [float(score) for score in match.group(1).split(",")]
            else:
                return [0.0] * len(candidate_answers)
        except exceptions.ResourceExhausted:
            print("hi")
            wait_time = 20 * (attempt + 1) 
            time.sleep(wait_time)
        except Exception as e:
            print("Exception:", e)
            return [0.0] * len(candidate_answers)
    return [0.0] * len(candidate_answers)
    
                    
    
def n_attempts(model, tokenizer, question, max_length=20, temperature=0.7, n=3):
    attempts = generate_batch(model, tokenizer, question, n=n, max_length=max_length, temperature=temperature)  

    attempt_score = grade_batch(question, attempts)
    attempts = list(zip(attempts, attempt_score))

    return attempts
    
if __name__ == "__main__":
    model, tokenizer = load_model()
    question = "What is 15*19?"
    attempts = n_attempts(model, tokenizer, question, max_length=50, temperature=0.7)
    print(attempts)