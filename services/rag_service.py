from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

employee_data = []
vectorizer = TfidfVectorizer()
vectors = None


def build_index(data):
    global employee_data, vectors

    employee_data = data

    texts = []

    for emp in data:
        text = f"""
        Name: {emp.get('name', '')}
        Department: {emp.get('department', '')}
        Role: {emp.get('role', '')}
        Salary: {emp.get('salary', '')}
        Joining Date: {emp.get('joining_date', '')}

        Assets:
        Laptop: {emp.get('assets', {}).get('laptop', {}).get('assigned', '')}
        Mouse: {emp.get('assets', {}).get('mouse', {}).get('assigned', '')}
        Monitor: {emp.get('assets', {}).get('monitor', {}).get('assigned', '')}
        Keyboard: {emp.get('assets', {}).get('keyboard', {}).get('assigned', '')}
        Bag: {emp.get('assets', {}).get('bag', {}).get('assigned', '')}
        ID Card: {emp.get('assets', {}).get('id_card', {}).get('assigned', '')}
        """
        texts.append(text)

    vectors = vectorizer.fit_transform(texts)


def search(query, top_k=3):
    global vectors

    if vectors is None:
        return []

    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, vectors).flatten()

    top_indices = similarity.argsort()[-top_k:][::-1]

    return [employee_data[i] for i in top_indices]