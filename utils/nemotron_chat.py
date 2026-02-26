import requests
import json

# Configuration
SERVER_IP = "192.168.1.93" # <-- CHANGE THIS to your server's IP
PORT = "11434"
MODEL = "nemotron-orchestrator"
URL = f"http://{SERVER_IP}:{PORT}/api/chat"

def chat():
    print(f"--- Connected to {MODEL} on {SERVER_IP} ---")
    print("Type 'exit' to quit.\n")
    
    messages = []
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages.append({"role": "user", "content": user_input})
        
        payload = {
            "model": MODEL,
            "messages": messages,
            "stream": True # Stream tokens as they generate
        }
        
        try:
            print("Nemotron: ", end="", flush=True)
            response = requests.post(URL, json=payload, stream=True)
            response.raise_for_status()
            
            assistant_reply = ""
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    chunk = data.get("message", {}).get("content", "")
                    assistant_reply += chunk
                    print(chunk, end="", flush=True)
            
            print("\n")
            messages.append({"role": "assistant", "content": assistant_reply})
            
        except requests.exceptions.RequestException as e:
            print(f"\n[Error connecting to server: {e}]")

if __name__ == "__main__":
    chat()
