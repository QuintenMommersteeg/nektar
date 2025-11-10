from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
from ai_ollama import analyze_question, verbalize
from queries import execute_intent
from starlette.responses import Response

class CachedStaticFiles(StaticFiles):
    """Serve statics met agressieve caching voor assets en no-store voor HTML."""

    def file_response(self, path, *args, **kwargs) -> Response:
        response: Response = super().file_response(path, *args, **kwargs)
        if response.status_code == 200:
            lower = str(path).lower()
            if lower.endswith(".html"):
                response.headers.setdefault("Cache-Control", "no-cache, no-store, must-revalidate")
            else:
                response.headers.setdefault("Cache-Control", "public, max-age=31536000, immutable")
        return response


app = FastAPI(title="Nektar Minimal Demo")

app.add_middleware(GZipMiddleware, minimum_size=512)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
app.mount("/", CachedStaticFiles(directory="ui", html=True), name="ui")
