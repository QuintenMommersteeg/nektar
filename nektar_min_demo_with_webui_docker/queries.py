from typing import Dict, Any
import db

def _summary_on_date(date: str):
    """Combineer werkorders + afspraken op één datum in één payload."""
    work = db.get_work_orders_on_date(date)
    appt = db.get_appointments_on_date(date)
    return {"date": date, "work_orders": work, "appointments": appt}

INTENTS = {
    "list_customers": {"fn": db.get_customers, "args": []},
    "list_recent_work_orders": {"fn": db.get_recent_work_orders, "args": ["limit"]},
    "work_orders_for_customer": {"fn": db.get_work_orders_for_customer, "args": ["customer_name"]},
    "next_appointment_for_customer": {"fn": db.get_next_appointment_for_customer, "args": ["customer_name"]},

    # Datum-intents
    "work_orders_on_date": {"fn": db.get_work_orders_on_date, "args": ["date"]},
    "appointments_on_date": {"fn": db.get_appointments_on_date, "args": ["date"]},
    "summary_on_date": {"fn": _summary_on_date, "args": ["date"]},

    # Vage/algemene vragen -> brede zoekopdracht
    "search_all": {"fn": db.search_all, "args": ["q"]},
}

def execute_intent(intent_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if intent_name not in INTENTS:
        return {"error": f"Unknown intent: {intent_name}"}
    entry = INTENTS[intent_name]
    fn = entry["fn"]
    call_kwargs = {}
    for arg in entry["args"]:
        if arg in params and params[arg] is not None:
            call_kwargs[arg] = params[arg]
    result = fn(**call_kwargs) if call_kwargs else fn()
    return {"intent": intent_name, "params": call_kwargs, "result": result}