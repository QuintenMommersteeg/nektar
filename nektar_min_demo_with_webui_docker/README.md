# Minimale skeleton (automatisch aangemaakt omdat vorige zip niet was gedownload)


---

## Docker gebruiken

### 1) Build & run (zonder Ollama)
```bash
docker build -t nektar-demo .
docker run --rm -p 8000:8000 nektar-demo
# Open: http://localhost:8000
```

### 2) Docker Compose (met optionele Ollama-service)
```bash
# Alleen de app
docker compose up --build

# App + Ollama (lokaal LLM, gratis). Pas import in app.py aan naar ai_ollama eerst.
docker compose --profile ollama up --build
```

**Let op (Ollama):**
- Vervang in `app.py` de import:
  ```python
  from ai import analyze_question, verbalize
  # wordt
  from ai_ollama import analyze_question, verbalize
  ```
- Het endpoint van Ollama is in compose al `http://ollama:11434` via `OLLAMA_URL` env var.

### 3) Open de web UI
Ga naar **http://localhost:8000** en stel je vraag in het invoerveld. Het UI roept `POST /ask` aan.

### 4) Persistente data (optioneel)
Standaard staat `demo.db` in de container. Wil je die buiten de container bewaren (dev-mode)?
```yaml
# docker-compose.yml → service app:
volumes:
  - ./:/app
```
> In dat geval staat `demo.db` in je projectmap op de host.
