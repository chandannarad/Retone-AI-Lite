from flask import Flask, request, jsonify
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ NEW
import os
app = Flask(__name__)
CORS(app)  # ✅ This enables CORS for all routes

API_KEY = os.getenv("PERSPECTIVE_API_KEY")

def analyze_tone(text):
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

    response = requests.post(url, json=data)
    result = response.json()

    if "attributeScores" not in result:
        return {"error": "API failed or returned no scores."}

    scores = {}
    for attr in data["requestedAttributes"]:
        scores[attr] = round(result["attributeScores"][attr]["summaryScore"]["value"] * 100, 2)

    return scores

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "")
    result = analyze_tone(text)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)