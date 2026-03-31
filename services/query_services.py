def is_full_data_query(query):
    query = query.lower()

    keywords = [
        "all employees",
        "list all",
        "everyone",
        "all names"
    ]

    return any(k in query for k in keywords)