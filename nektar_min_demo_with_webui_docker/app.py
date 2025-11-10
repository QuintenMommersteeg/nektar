from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ai_ollama import analyze_question, verbalize
from queries import execute_intent

app = FastAPI(title="Nektar Minimal Demo")

app.add_middleware(CORSMiddleware,
    allow_origins=['*'], allow_credentials=True,
    allow_methods=['*'], allow_headers=['*'])


class AskBody(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ask")
def ask(body: AskBody):
    route = analyze_question(body.question)
    intent = route.get("intent")
    params = route.get("params", {})
    exec_result = execute_intent(intent, params)
    if "error" in exec_result:
        return {"answer": f"Er ging iets mis: {exec_result['error']}", "intent": intent, "params": params}
    text = verbalize(body.question, exec_result["intent"], exec_result["result"])
    return {"answer": text, "intent": intent, "params": params, "raw": exec_result["result"]}

# Serve the web UI
app.mount('/', StaticFiles(directory='ui', html=True), name='ui')
