from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Hello LLM Endpoint")

# Use the small distilgpt2 model
generator = pipeline("text-generation", model="distilgpt2")

class PromptRequest(BaseModel):
    prompt: str
    max_length: int = 50
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    do_sample: bool = True
    num_return_sequences: int = 1

class GenerateResponse(BaseModel):
    prompt: str
    generated_texts: list[str]
    model: str
    generation_settings: dict

# Convenience GET endpoint for quick testing in browser
@app.get("/hello-llm", response_model=GenerateResponse)
async def hello_llm_get(
    prompt: str = Query("Tell me a joke.", description="Prompt for the LLM"),
    max_length: int = Query(50, ge=1, le=1024),
    temperature: float = Query(1.0, ge=0.0, le=2.0),
    top_k: int = Query(50, ge=0),
    top_p: float = Query(0.95, ge=0.0, le=1.0),
    do_sample: bool = Query(True),
    num_return_sequences: int = Query(1, ge=1, le=5),
):
    # Reuse the pipeline to generate
    results = generator(
        prompt,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences,
    )
    texts = [r["generated_text"] for r in results]
    return GenerateResponse(
        prompt=prompt,
        generated_texts=texts,
        model="distilgpt2",
        generation_settings={
            "max_length": max_length,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "do_sample": do_sample,
            "num_return_sequences": num_return_sequences,
        },
    )
