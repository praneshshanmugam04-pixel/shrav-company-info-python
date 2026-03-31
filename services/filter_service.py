"""
filter_service.py

Rule-based pre-filter applied on top of RAG search results.
Handles structured queries like asset checks, department filters,
role lookups, and salary comparisons before passing to the LLM.
"""

import re


# ─── Asset Filter ────────────────────────────────────────────────────────────

ASSET_KEYWORDS = ["laptop", "mouse", "monitor", "keyboard", "bag", "id card", "id_card"]

def _extract_asset(query: str):
    """Return the asset name mentioned in the query, or None."""
    for asset in ASSET_KEYWORDS:
        if asset in query:
            return asset.replace(" ", "_")
    return None

def _is_negated(query: str) -> bool:
    """Detect negation patterns like 'not assigned', 'no laptop', 'without'."""
    negation_patterns = [r"\bnot\b", r"\bno\b", r"\bwithout\b", r"\bunassigned\b", r"\bmissing\b"]
    return any(re.search(p, query) for p in negation_patterns)

def _filter_by_asset(query: str, employees: list) -> list | None:
    """Filter employees by asset assignment status. Returns None if no asset keyword found."""
    asset = _extract_asset(query)
    if not asset:
        return None

    negated = _is_negated(query)

    return [
        emp for emp in employees
        if asset in emp.get("assets", {})
        and emp["assets"][asset].get("assigned") != negated
    ]


# ─── Department Filter ────────────────────────────────────────────────────────

def _filter_by_department(query: str, employees: list) -> list | None:
    """Filter by department if a known department name is mentioned."""
    # Collect unique departments from data
    departments = {emp.get("department", "").lower() for emp in employees if emp.get("department")}

    for dept in departments:
        if dept and dept in query:
            return [emp for emp in employees if emp.get("department", "").lower() == dept]

    return None


# ─── Role Filter ─────────────────────────────────────────────────────────────

def _filter_by_role(query: str, employees: list) -> list | None:
    """Filter by role/job title if mentioned."""
    roles = {emp.get("role", "").lower() for emp in employees if emp.get("role")}

    for role in roles:
        if role and role in query:
            return [emp for emp in employees if emp.get("role", "").lower() == role]

    return None


# ─── Salary Filter ────────────────────────────────────────────────────────────

def _filter_by_salary(query: str, employees: list) -> list | None:
    """
    Handle queries like:
      'salary above 50000', 'earn more than 60k', 'salary below 40000'
    """
    above_match = re.search(r"(?:above|more than|greater than|over)\s*(\d+)", query)
    below_match = re.search(r"(?:below|less than|under)\s*(\d+)", query)

    if above_match:
        threshold = int(above_match.group(1))
        return [
            emp for emp in employees
            if _parse_salary(emp.get("salary")) is not None
            and _parse_salary(emp.get("salary")) > threshold
        ]

    if below_match:
        threshold = int(below_match.group(1))
        return [
            emp for emp in employees
            if _parse_salary(emp.get("salary")) is not None
            and _parse_salary(emp.get("salary")) < threshold
        ]

    return None

def _parse_salary(value) -> float | None:
    """Safely parse salary from string or number."""
    if value is None:
        return None
    try:
        # Strip currency symbols and commas: "$50,000" → 50000.0
        return float(re.sub(r"[^\d.]", "", str(value)))
    except ValueError:
        return None


# ─── Status Filter ────────────────────────────────────────────────────────────

def _filter_by_status(query: str, employees: list) -> list | None:
    """Handle 'active employees', 'inactive', 'on leave', etc."""
    status_keywords = {
        "active": ["active"],
        "inactive": ["inactive"],
        "on leave": ["on leave", "on_leave", "leave"],
        "terminated": ["terminated", "fired"],
    }

    for status, keywords in status_keywords.items():
        if any(kw in query for kw in keywords):
            return [
                emp for emp in employees
                if emp.get("status", "").lower() == status
            ]

    return None


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def filter_employees(query: str, employees: list) -> list:
    """
    Apply structured filters to employee list based on the query.
    Filters are tried in priority order; the first match wins.
    Falls back to the full list if no filter matches.

    Priority:
      1. Asset filter       (most specific hardware queries)
      2. Status filter      (active/inactive/on leave)
      3. Salary filter      (numeric comparisons)
      4. Department filter  (org unit)
      5. Role filter        (job title)
    """
    normalized = query.lower().strip()

    filter_pipeline = [
        _filter_by_asset,
        _filter_by_status,
        _filter_by_salary,
        _filter_by_department,
        _filter_by_role,
    ]

    for filter_fn in filter_pipeline:
        result = filter_fn(normalized, employees)
        if result is not None:       # A filter matched (even if result is empty)
            return result

    # No structured filter matched — return all, let RAG + LLM handle it
    return employees