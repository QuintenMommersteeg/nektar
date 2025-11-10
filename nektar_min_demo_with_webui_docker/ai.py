import os, re, json
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def _fallback_intent(question: str) -> Dict[str, Any]:
    q = question.lower()
    if "klant" in q and ("overzicht" in q or "alle" in q or "lijst" in q):
        return {"intent": "list_customers", "params": {}}
    if "werkorder" in q and ("laatste" in q or "recent" in q or "3" in q):
        return {"intent": "list_recent_work_orders", "params": {"limit": 3}}
    m = re.search(r"voor\s+([A-Z][a-z]+\s+[A-Z][a-z]+)", question)
    if "werkorder" in q and m:
        return {"intent": "work_orders_for_customer", "params": {"customer_name": m.group(1)}}
    if ("afspraak" in q or "appointment" in q) and m:
        return {"intent": "next_appointment_for_customer", "params": {"customer_name": m.group(1)}}
    m2 = re.search(r"(jan jansen|piet pieters|sanne smit)", q)
    if m2:
        name = m2.group(1).title()
        return {"intent": "work_orders_for_customer", "params": {"customer_name": name}}
    return {"intent": "list_recent_work_orders", "params": {"limit": 3}}

def analyze_question(question: str) -> Dict[str, Any]:
    return _fallback_intent(question)

def verbalize(question: str, intent: str, result) -> str:
    if intent == "list_customers":
        if not result:
            return "Er zijn nog geen klanten in de database."
        names = ", ".join(r["name"] for r in result)
        return f"Dit zijn de klanten in het systeem: {names}."
    if intent == "list_recent_work_orders":
        if not result:
            return "Er zijn geen recente werkorders gevonden."
        lines = [f"#{r['id']} ({r['status']}): {r['title']} – {r['customer_name']}" for r in result]
        return "Recente werkorders:\n" + "\n".join(lines)
    if intent == "work_orders_for_customer":
        if not result:
            return "Geen werkorders gevonden voor deze klant."
        cust = result[0]["customer_name"]
        lines = [f"#{r['id']} ({r['status']}): {r['title']} – {r['created_at']}" for r in result]
        return f"Werkorders voor {cust}:\n" + "\n".join(lines)
    if intent == "next_appointment_for_customer":
        if not result:
            return "Er staat nog geen volgende afspraak gepland."
        return (f"Volgende afspraak voor {result['customer_name']}: "
                f"{result['start_ts']} tot {result['end_ts']} – resource {result['resource']}.")
    return "Ik heb je vraag verwerkt, maar vond geen duidelijke resultaten."
