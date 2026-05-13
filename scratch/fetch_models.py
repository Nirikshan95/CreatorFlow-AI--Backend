import requests
import json

def fetch_free_models():
    try:
        response = requests.get("https://openrouter.ai/api/v1/models")
        if response.status_code == 200:
            models = response.json().get("data", [])
            free_models = [m["id"] for m in models if "free" in m["id"]]
            print("\n".join(free_models))
        else:
            print(f"Error: {response.status_code}")
    except Exception as e:
        print(f"Exception: {str(e)}")

if __name__ == "__main__":
    fetch_free_models()
