import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

def get_rewrite_suggestions(text):
    API_URL = "https://api-inference.huggingface.co/models/t5-small"
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = f"paraphrase: {text} </s>"
    payload = {
        "inputs": prompt,
        "parameters": {
            "num_return_sequences": 2,
            "num_beams": 5
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    data = response.json()
    
    suggestions = []
    for item in data:
        suggestions.append({
            "type": "Rewritten Suggestion",
            "text": item.get("generated_text", "")
        })
    return suggestions

@app.route("/rewrite", methods=["POST"])
def rewrite():
    data = request.get_json()
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "No text provided"})
    try:
        suggestions = get_rewrite_suggestions(text)
        return jsonify({"suggestions": suggestions})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/")
def home():
    return jsonify({"message": "Retone AI Lite Backend is running with remote rewrite support!"})

if __name__ == "__main__":
    app.run(debug=True)