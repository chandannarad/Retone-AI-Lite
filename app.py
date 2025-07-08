from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

API_KEY = os.getenv("PERSPECTIVE_API_KEY")

def analyze_tone(text, context="chat"):
    url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={API_KEY}"

    data = {
        "comment": {"text": text},
        "languages": ["en"],
        "requestedAttributes": {
            "TOXICITY": {},
            "INSULT": {},
            "PROFANITY": {},
            "THREAT": {}
        }
    }

    try:
        response = requests.post(url, json=data)
        result = response.json()

        if "attributeScores" not in result:
            return {"error": "API failed or returned no scores."}

        scores = {}
        for attr in data["requestedAttributes"]:
            scores[attr] = round(result["attributeScores"][attr]["summaryScore"]["value"] * 100, 2)

        # Add context information to the response
        scores["context"] = context
        return scores
        
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "")
    context = data.get("context", "chat")  # Default to chat if no context provided
    
    if not text.strip():
        return jsonify({"error": "No text provided"})
    
    result = analyze_tone(text, context)
    return jsonify(result)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Retone AI Lite Backend is running!"})

if __name__ == "__main__":
    app.run(debug=True)