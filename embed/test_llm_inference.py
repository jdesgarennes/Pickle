import requests

OLLAMA_URL = "http://10.50.1.82:11434"

def run_completion(prompt):
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": "llama3.1:latest",
            "prompt": prompt,
            "stream": False   # disables streaming for easy parsing
        }
    )
    response.raise_for_status()
    return response.json()["response"]
if __name__ == "__main__":
    user_prompt = input("Enter your prompt for llama3.1: ")
    answer = run_completion(user_prompt)
    print(f"\nResponse:\n{answer}")
