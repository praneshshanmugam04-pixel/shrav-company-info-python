from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

employee_data = []
index = None


def build_index(data):
    global employee_data, index

    employee_data = data

    texts = []

    for emp in data:
        text = f"""
        Name: {emp['name']}
        Department: {emp['department']}
        Role: {emp['role']}
        Salary: {emp['salary']}
        Joining Date: {emp['joining_date']}

        Assets:
        Laptop: {emp['assets']['laptop']['assigned']}
        Mouse: {emp['assets']['mouse']['assigned']}
        Monitor: {emp['assets']['monitor']['assigned']}
        Keyboard: {emp['assets']['keyboard']['assigned']}
        Bag: {emp['assets']['bag']['assigned']}
        ID Card: {emp['assets']['id_card']['assigned']}
        """
        texts.append(text)

    embeddings = model.encode(texts)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))


def search(query, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)

    return [employee_data[i] for i in indices[0]]