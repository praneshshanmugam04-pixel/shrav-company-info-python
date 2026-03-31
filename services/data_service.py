import requests
from config import DATA_API_URL


def fetch_employees():
    response = requests.get(DATA_API_URL)
    return response.json()