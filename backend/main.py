import os
import yaml
from fastapi import FastAPI, HTTPException
from litellm import Router
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="HomeLab AI API Gateway")

# Load config and initialize LiteLLM Router
with open("config.yaml") as f:
    config = yaml.safe_load(f)

router = Router(
    model_list=config["model_list"],
    routing_strategy="simple-shuffle"
)

@app.post("/v1/chat")
async def chat_endpoint(request: dict):
    """
    Accepts a standard OpenAI-style chat payload.
    Expected JSON: {"model": "orchestrator", "messages": [{"role": "user", "content": "Hi"}]}
    """
    model_alias = request.get("model", "orchestrator")
    messages = request.get("messages", [])
    
    try:
        # LiteLLM automatically routes to Ollama, TRT-LLM, or Gemini based on the alias
        response = await router.acompletion(
            model=model_alias,
            messages=messages,  
            stream=False # Set to True for SSE streaming to your future frontend
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
