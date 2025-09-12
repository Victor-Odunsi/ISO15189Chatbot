import requests
import json, os

def get_api_response_stream(prompt: str, session_id: str):
    """
    Generator function to stream API response tokens.
    """
    base_url = os.getenv('API_URL', 'http://localhost:8000')
    url = f'{base_url}/chat'

    payload = {"question": prompt}
    if session_id:
        payload["session_id"] = session_id

    with requests.post(url, json=payload, stream=True) as response:
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    yield {"type": "error", "content": line}