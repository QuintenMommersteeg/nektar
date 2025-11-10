# ai_ollama.py
import os, re, json, time, requests
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

# === Config ===
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
# "phi3" = licht/snel; "llama3.1:8b" = wat krachtiger
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3")
# AI-zinsbouw standaard AAN
USE_LLM_VERBALIZE = os.getenv("USE_LLM_VERBALIZE", "1").lower() in ("1", "true", "yes")

# === Helpers ===
def _balanced_json(text: str) -> Optional[dict]:
    """Pak het eerste geldige JSON-object uit een string (balans-gebaseerd)."""
    s = (text or "").strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    depth = 0
    start = -1
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                snippet = s[start:i + 1]
                try:
                    return json.loads(snippet)
                except Exception:
                    continue
    return None

def _ollama_chat(messages: list, timeout=30, retries=2) -> str:
    """Robuuste chat-call naar Ollama met retries; retourneert message.content."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
                timeout=timeout,
            )
            r.raise_for_status()
            return r.json().get("message", {}).get("content", "").strip()
        except Exception as e:
            last_err = e
            time.sleep(0.4 * (attempt + 1))
    return ""

# === Datum-normalisatie ===
def _normalize_date(question: str, raw_date: Optional[str]) -> Optional[str]:
    """
    Zet NL-datums om naar YYYY-MM-DD. Ondersteunt: 28/10, 28-10, 28-10-2025, 'vandaag', 'morgen'.
    Als jaar ontbreekt: huidig jaar.
    """
    q = (question or "").lower().strip()
    today = datetime.now()

    if not raw_date:
        if "vandaag" in q:
            return today.strftime("%Y-%m-%d")
        if "morgen" in q or "morg" in q:
            return (today + timedelta(days=1)).strftime("%Y-%m-%d")
        m = re.search(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b", q)
        if m:
            d, mth, y = m.groups()
            d = int(d); mth = int(mth)
            y = int(y) + 2000 if y and int(y) < 100 else (int(y) if y else today.year)
            try:
                return datetime(y, mth, d).strftime("%Y-%m-%d")
            except Exception:
                return None
        m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", q)
        if m:
            return m.group(1)
        return None

    rd = raw_date.strip().lower()
    if rd in ("vandaag", "today"):
        return today.strftime("%Y-%m-%d")
    if rd in ("morgen", "tomorrow"):
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")
    m = re.match(r"^(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?$", rd)
    if m:
        d, mth, y = m.groups()
        d = int(d); mth = int(mth)
        y = int(y) + 2000 if y and int(y) < 100 else (int(y) if y else today.year)
        try:
            return datetime(y, mth, d).strftime("%Y-%m-%d")
        except Exception:
            return None
    if re.match(r"^\d{4}-\d{2}-\d{2}$", rd):
        return rd
    return None

# === Naam-extractie voor '… voor/van Jan Jansen' ===
_NAME = r"[A-Za-zÀ-ÖØ-öø-ÿ'.-]+"

def _extract_name(question: str) -> Optional[str]:
    """
    Herkent patronen als:
    - … voor Jan Jansen / … van Jan Jansen
    Retourneert 'Voornaam Achternaam' of None.
    """
    if not question:
        return None
    q = question.strip()

    m = re.search(rf"\b(?:voor|van)\s+({_NAME})\s+({_NAME})\b", q, flags=re.IGNORECASE)
    if m:
        first, last = m.groups()
        name = f"{first} {last}".strip()
        return " ".join(part[:1].upper() + part[1:] for part in name.split())

    m = re.search(rf"\b(?:voor|van)\s+({_NAME}\s+{_NAME})\s*\??$", q, flags=re.IGNORECASE)
    if m:
        name = m.group(1)
        return " ".join(part[:1].upper() + part[1:] for part in name.split())

    return None

# === Fallback-router als LLM faalt ===
def _fallback_intent(question: str) -> Dict[str, Any]:
    q = (question or "").lower()
    date_norm = _normalize_date(question, None)
    person = _extract_name(question)

    if person and ("werkorder" in q or "order" in q):
        return {"intent": "work_orders_for_customer", "params": {"customer_name": person}}
    if person and ("afspraak" in q or "appointment" in q or "agenda" in q):
        return {"intent": "next_appointment_for_customer", "params": {"customer_name": person}}

    if date_norm and ("afspraak" in q or "agenda" in q or "appointment" in q):
        return {"intent": "appointments_on_date", "params": {"date": date_norm}}
    if date_norm and ("werkorder" in q or "order" in q):
        return {"intent": "work_orders_on_date", "params": {"date": date_norm}}
    if date_norm:
        return {"intent": "summary_on_date", "params": {"date": date_norm}}

    if "klant" in q and ("overzicht" in q or "alle" in q or "lijst" in q):
        return {"intent": "list_customers", "params": {}}
    if "werkorder" in q and ("laatste" in q or "recent" in q or "3" in q):
        return {"intent": "list_recent_work_orders", "params": {"limit": 3}}

    # vage vraag → brede zoekopdracht
    return {"intent": "search_all", "params": {"q": question}}

# === Router via LLM (met pre-checks & failsafes) ===
def analyze_question(question: str) -> Dict[str, Any]:
    # PRE-ROUTER: vang direct '… voor/van <Naam Naam>' af
    qlow = (question or "").lower()
    person = _extract_name(question)
    if person:
        if "werkorder" in qlow or "order" in qlow:
            return {"intent": "work_orders_for_customer", "params": {"customer_name": person}}
        if "afspraak" in qlow or "appointment" in qlow or "agenda" in qlow:
            return {"intent": "next_appointment_for_customer", "params": {"customer_name": person}}

    system = "Je bent een strikte JSON-router voor intents. Antwoord ALLEEN met JSON."
    prompt = """
Kies precies één intent:
- list_customers
- list_recent_work_orders
- work_orders_for_customer
- next_appointment_for_customer
- work_orders_on_date
- appointments_on_date
- summary_on_date
- search_all

JSON-schema (geen extra velden!):
{
  "intent": "<bovenstaande intent>",
  "params": {"customer_name": "string (opt)", "limit": getal (opt), "date": "YYYY-MM-DD (opt)", "q": "string (opt)"}
}

Regels:
- 'voor klant X' → intent=work_orders_for_customer + params.customer_name=X (Voornaam Achternaam).
- 'laatste/recent' → intent=list_recent_work_orders (+ limit=3 als niet opgegeven).
- Datum in vraag (28/10, 28-10-2025, 'vandaag', 'morgen'):
  - benoemt men afspraken → appointments_on_date;
  - benoemt men werkorders → work_orders_on_date;
  - generiek 'wat is er [datum] gepland?' → summary_on_date.
  - params.date ALTIJD 'YYYY-MM-DD'.
- Vraag is te vaag/vrij → intent=search_all met params.q = volledige vraag.
- Strict JSON. GEEN tekst buiten de JSON.

Voorbeelden:
Q: "Welke werkorders zijn er voor Jan Jansen?"
A: {"intent":"work_orders_for_customer","params":{"customer_name":"Jan Jansen"}}

Q: "Wat staat er op 28/10 gepland?"
A: {"intent":"summary_on_date","params":{"date":"2025-10-28"}}

Q: "Welke afspraken zijn er morgen?"
A: {"intent":"appointments_on_date","params":{"date":"YYYY-MM-DD"}}

Q: "Laatste 3 werkorders"
A: {"intent":"list_recent_work_orders","params":{"limit":3}}

Q: "Band lek, wie kan dit morgen doen?"
A: {"intent":"search_all","params":{"q":"Band lek, wie kan dit morgen doen?"}}
""".strip() + f"\n\nVraag: {question}"

    try:
        content = _ollama_chat(
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            timeout=30, retries=2
        )
        payload = _balanced_json(content) if content else None
        if not payload or "intent" not in payload:
            return _fallback_intent(question)

        intent = payload.get("intent")
        params = payload.get("params") or {}
        if not isinstance(params, dict):
            params = {}

        valid_intents = {
            "list_customers",
            "list_recent_work_orders",
            "work_orders_for_customer",
            "next_appointment_for_customer",
            "work_orders_on_date",
            "appointments_on_date",
            "summary_on_date",
            "search_all",
        }
        if intent not in valid_intents:
            return _fallback_intent(question)

        # guards
        if "limit" in params:
            try:
                l = int(params["limit"])
                params["limit"] = max(1, min(l, 20))
            except Exception:
                params.pop("limit", None)

        if "date" in params and params["date"] is not None:
            norm = _normalize_date(question, str(params["date"]))
            params["date"] = norm if norm else None

        # datum in vraag maar geen datum-intent? → corrigeer
        has_date_in_question = bool(_normalize_date(question, None) or re.search(r"\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?", question or ""))
        if has_date_in_question and intent not in {"work_orders_on_date", "appointments_on_date", "summary_on_date"}:
            chosen = "summary_on_date"
            if "afspraak" in qlow or "agenda" in qlow or "appointment" in qlow:
                chosen = "appointments_on_date"
            elif "werkorder" in qlow or "order" in qlow:
                chosen = "work_orders_on_date"
            params = {"date": _normalize_date(question, params.get("date")) or _normalize_date(question, None)}
            intent = chosen

        # niets passend? → zoekbreed
        if intent not in valid_intents:
            return {"intent": "search_all", "params": {"q": question}}

        return {"intent": intent, "params": params}
    except Exception:
        return _fallback_intent(question)

# === AI-zinsbouw ===
def _llm_verbalize(prompt: str) -> str:
    content = _ollama_chat(
        [
            {"role": "system", "content": "Schrijf 1–2 korte, duidelijke Nederlandse zinnen. Wees feitelijk; zeg 'geen resultaten' als er niets is. Geen aannames."},
            {"role": "user", "content": prompt.strip()},
        ],
        timeout=25, retries=1
    )
    return content or ""

# === Verwoorden van DB-resultaten → zin(nen) ===
def verbalize(question: str, intent: str, result) -> str:
    def fmt_dt(s: str) -> str:
        try:
            return datetime.strptime(s, "%Y-%m-%d %H:%M").strftime("%d-%m-%Y %H:%M")
        except Exception:
            return s or ""

    if intent == "list_customers":
        names = [r["name"] for r in (result or [])]
        if not names:
            text = "Er staan nog geen klanten in het systeem."
        elif len(names) == 1:
            text = f"Er is één klant: {names[0]}."
        elif len(names) == 2:
            text = f"Er zijn twee klanten: {names[0]} en {names[1]}."
        else:
            text = f"Er zijn {len(names)} klanten: {', '.join(names[:-1])} en {names[-1]}."

    elif intent == "list_recent_work_orders":
        rows = result or []
        if not rows:
            text = "Er zijn geen recente werkorders gevonden."
        else:
            parts = [f"#{r['id']} — {r['title']} ({r['status']}) voor {r['customer_name']} op {fmt_dt(r.get('created_at',''))}" for r in rows]
            text = ("Laatste werkorder: " + parts[0] + ".") if len(parts) == 1 else ("Recente werkorders: " + "; ".join(parts) + ".")

    elif intent == "work_orders_for_customer":
        rows = result or []
        if not rows:
            text = "Ik vond geen werkorders voor deze klant."
        else:
            cust = rows[0].get("customer_name", "deze klant")
            parts = [f"#{r['id']} — {r['title']} ({r['status']}, op {fmt_dt(r.get('created_at',''))})" for r in rows]
            text = (f"Voor {cust} staat één werkorder geregistreerd: {parts[0]}."
                    if len(parts) == 1 else f"Voor {cust} staan {len(rows)} werkorders geregistreerd: " + "; ".join(parts) + ".")

    elif intent == "next_appointment_for_customer":
        row = result or None
        if not row:
            text = "Er staat nog geen volgende afspraak gepland."
        else:
            text = f"De eerstvolgende afspraak voor {row.get('customer_name','de klant')} is op {fmt_dt(row.get('start_ts',''))} tot {fmt_dt(row.get('end_ts',''))} met {row.get('resource','een medewerker')}."

    elif intent == "work_orders_on_date":
        rows = result or []
        date_txt = (rows[0].get("created_at","")[:10] if rows else None) or _normalize_date(question, None) or "deze datum"
        if not rows:
            text = f"Er zijn geen werkorders gevonden op {date_txt}."
        else:
            parts = [f"#{r['id']} — {r['title']} ({r['status']}) voor {r['customer_name']} om {(r.get('created_at','')[11:16])}" for r in rows]
            text = f"Werkorders op {date_txt}: " + "; ".join(parts) + "."

    elif intent == "appointments_on_date":
        rows = result or []
        date_txt = (rows[0].get("start_ts","")[:10] if rows else None) or _normalize_date(question, None) or "deze datum"
        if not rows:
            text = f"Er zijn geen afspraken gevonden op {date_txt}."
        else:
            parts = [f"{r.get('customer_name','de klant')} {(r.get('start_ts','')[11:16])}–{(r.get('end_ts','')[11:16])} met {r.get('resource','een medewerker')}" for r in rows]
            text = f"Afspraken op {date_txt}: " + "; ".join(parts) + "."

    elif intent == "summary_on_date":
        payload = result or {}
        date_txt = payload.get("date") or _normalize_date(question, None) or "deze datum"
        work = payload.get("work_orders") or []
        appt = payload.get("appointments") or []
        if not work and not appt:
            text = f"Er is niets gepland op {date_txt}."
        else:
            parts = []
            if work:
                w = [f"#{r['id']} — {r['title']} ({r['status']}) voor {r['customer_name']} om {(r.get('created_at','')[11:16])}" for r in work][:5]
                parts.append("Werkorders: " + "; ".join(w))
            if appt:
                a = [f"{r.get('customer_name','de klant')} {(r.get('start_ts','')[11:16])}–{(r.get('end_ts','')[11:16])} met {r.get('resource','een medewerker')}" for r in appt][:5]
                parts.append("Afspraken: " + "; ".join(a))
            text = f"Op {date_txt}: " + " | ".join(parts) + ("." if not parts[-1].endswith(".") else "")

    elif intent == "search_all":
        payload = result or {}
        q = payload.get("query", "")
        work = payload.get("work_orders") or []
        appt = payload.get("appointments") or []
        if not work and not appt:
            text = f"Ik vond geen resultaten voor “{q}”."
        else:
            wtxt = f"{len(work)} werkorders" if work else "geen werkorders"
            atxt = f"{len(appt)} afspraken" if appt else "geen afspraken"
            examples = []
            for r in work[:2]:
                examples.append(f"WO#{r['id']} {r['title']} ({r['status']})")
            for r in appt[:2]:
                examples.append(f"Afspraak {r['customer_name']} {(r.get('start_ts','')[11:16])}")
            extra = "" if not examples else " Voorbeeld: " + "; ".join(examples) + "."
            text = f"Zoekresultaten voor “{q}”: {wtxt}, {atxt}.{extra}"

    else:
        text = "Ik heb geen relevante informatie gevonden."

    if USE_LLM_VERBALIZE:
        try:
            polished = _llm_verbalize(
                f"Vraag: {question}\nHuidige tekst: {text}\nZet dit om naar 1–2 duidelijke, natuurlijke NL-zinnen, feitelijk, zonder aannames."
            )
            return polished or text
        except Exception:
            return text
    return text