"""
security_service.py

Handles data sanitization and field-level access control.
Controls which fields are returned based on the query context.
"""

# Fields always safe to return
PUBLIC_FIELDS = ["name", "department", "role"]

# Fields returned only when explicitly asked
SENSITIVE_FIELDS = ["salary", "joining_date", "email", "phone"]

# Asset fields that are always included (no PII)
ASSET_FIELDS = ["assets"]

# Status is operational info — safe to include
OPERATIONAL_FIELDS = ["status", "employee_id"]


def sanitize_data(data: list, include_sensitive: bool = False) -> list:
    """
    Sanitize a list of employee records.

    Args:
        data: Raw employee list from the data source.
        include_sensitive: If True, includes salary, joining_date, email, phone.
                           Set True only when user explicitly requests them.

    Returns:
        A sanitized list of employee dicts safe for LLM consumption.
    """
    safe = []

    for emp in data:
        record = {}

        # Always-safe fields
        for field in PUBLIC_FIELDS + OPERATIONAL_FIELDS:
            if field in emp:
                record[field] = emp[field]

        # Sensitive fields — only when explicitly requested
        if include_sensitive:
            for field in SENSITIVE_FIELDS:
                if field in emp:
                    record[field] = emp[field]

        # Assets — sanitize each asset entry
        raw_assets = emp.get("assets", {})
        if raw_assets:
            record["assets"] = _sanitize_assets(raw_assets)

        safe.append(record)

    return safe


def sanitize_for_full_query(data: list) -> list:
    """
    For full data queries (e.g., 'list all employees'),
    returns all fields including salary and joining date.
    """
    return sanitize_data(data, include_sensitive=True)


def _sanitize_assets(assets: dict) -> dict:
    """
    Sanitize the nested assets dict.
    Only keeps: assigned status and model (if present).
    Strips any internal IDs or metadata not meant for display.
    """
    sanitized = {}

    for asset_name, asset_info in assets.items():
        if isinstance(asset_info, dict):
            sanitized[asset_name] = {
                "assigned": asset_info.get("assigned", False),
            }
            # Include model/brand only if present and non-sensitive
            if "model" in asset_info:
                sanitized[asset_name]["model"] = asset_info["model"]
        else:
            # Handle flat boolean asset values
            sanitized[asset_name] = {"assigned": bool(asset_info)}

    return sanitized