from groq import Groq
from config import GROQ_API_KEY
from datetime import date

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are an intelligent HR assistant for a company's internal employment management system.

Your responsibilities:
- Answer questions about employees, departments, assets, salaries, and roles
- Provide clear, structured, and professional responses
- Summarize data in a human-readable format when listing multiple employees
- Flag when data is insufficient to fully answer a question

Rules you must always follow:
- ONLY use the employee data provided in the context. Never fabricate details.
- Do NOT expose sensitive fields like joining_date or salary unless explicitly asked.
- When listing employees, use a clean bullet or numbered format.
- If the question is ambiguous, make a reasonable interpretation and state it briefly.
- Keep your tone professional, concise, and helpful.
- If no employees match the query, clearly say "No matching employees found" and suggest rephrasing.
- Today's date: {today}
"""

def generate_response(question: str, data: list) -> str:
    if not data:
        return "No matching employee records were found for your query. Please try rephrasing or broadening your search."

    today = date.today().strftime("%B %d, %Y")

    # Format employee data clearly for the LLM
    formatted_employees = []
    for i, emp in enumerate(data, 1):
        assets = emp.get("assets", {})
        assigned_assets = [
            asset.replace("_", " ").title()
            for asset, info in assets.items()
            if isinstance(info, dict) and info.get("assigned")
        ]
        unassigned_assets = [
            asset.replace("_", " ").title()
            for asset, info in assets.items()
            if isinstance(info, dict) and not info.get("assigned")
        ]

        emp_block = f"""Employee {i}:
  - Name       : {emp.get('name', 'N/A')}
  - Department : {emp.get('department', 'N/A')}
  - Role       : {emp.get('role', 'N/A')}
  - Salary     : {emp.get('salary', 'N/A')}
  - Status     : {emp.get('status', 'N/A')}
  - Email      : {emp.get('email', 'N/A')}
  - Phone      : {emp.get('phone', 'N/A')}
  - Joining Date: {emp.get('joining_date', 'N/A')}
  - Assigned Assets  : {', '.join(assigned_assets) if assigned_assets else 'None'}
  - Unassigned Assets: {', '.join(unassigned_assets) if unassigned_assets else 'None'}"""
        formatted_employees.append(emp_block)

    employee_context = "\n\n".join(formatted_employees)

    user_prompt = f"""Employee Data ({len(data)} record(s)):

{employee_context}

---

Question: {question}

Instructions:
- Answer the question directly and clearly using only the above data.
- If listing multiple employees, use a numbered or bullet format.
- If the answer involves assets, clearly state what is assigned and what is not.
- If the data doesn't fully answer the question, say so honestly.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(today=today)
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.3,       # Lower = more factual, less hallucination
        max_tokens=1024,
    )

    return response.choices[0].message.content.strip()