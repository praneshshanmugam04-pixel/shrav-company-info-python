from fastapi import FastAPI
from model.request_model import Query
from services.filter_service import filter_employees
from services.security_service import sanitize_data, sanitize_for_full_query
from services.llm_service import generate_response
from services.data_service import fetch_employees
from services.rag_service import build_index, search
from services.query_services import is_full_data_query

app = FastAPI()

# ── Startup: Load employee data and build FAISS index ─────────────────────────
employees = fetch_employees()
build_index(employees)


@app.post("/chat")
def chat(query: Query):

    # ── Full-data queries: "list all employees", "everyone", etc. ─────────────
    if is_full_data_query(query.question):
        all_data = fetch_employees()
        safe_data = sanitize_for_full_query(all_data)   # Includes salary, joining date etc.
        answer = generate_response(query.question, safe_data)
        return {
            "answer": answer,
            "data": safe_data,
        }

    # ── Detect if query asks for sensitive fields explicitly ──────────────────
    sensitive_keywords = ["salary", "email", "phone", "joining date", "joined"]
    wants_sensitive = any(kw in query.question.lower() for kw in sensitive_keywords)

    # ── Normal RAG + Filter flow ──────────────────────────────────────────────
    results = search(query.question)
    filtered = filter_employees(query.question, results)
    limited = filtered[:5]  # Slightly wider window for better LLM context
    safe_data = sanitize_data(limited, include_sensitive=wants_sensitive)

    answer = generate_response(query.question, safe_data)

    return {
        "answer": answer,
        "data": safe_data,
    }