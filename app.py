from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
import os
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# API Key for Perspective API (toxicity)
PERSPECTIVE_API_KEY = os.getenv("PERSPECTIVE_API_KEY")

# Load T5 paraphrasing pipeline once
paraphraser = pipeline("text2text-generation", model="t5-small")

def analyze_tone(text, context="chat"):
    url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={PERSPECTIVE_API_KEY}"
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

        scores["context"] = context
        return scores

    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

def generate_rewrite_suggestions(text, context, scores):
    prompt = f"paraphrase: {text}"
    try:
        # Generate 3 paraphrased outputs
        results = paraphraser(prompt, max_length=100, num_return_sequences=3, do_sample=True, temperature=0.7, top_p=0.9)

        suggestions = []
        for i, res in enumerate(results):
            suggestions.append({
                "type": f"Suggestion {i+1}",
                "text": res['generated_text']
            })

        return suggestions

    except Exception as e:
        # Fallback suggestions if something goes wrong
        fallback = [
            {"type": "Respectful Alternative", "text": "Let me rephrase this in a kinder way."},
            {"type": "Professional Version", "text": "Here's a more polite way to express this."}
        ]
        return fallback

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "")
    context = data.get("context", "chat")

    if not text.strip():
        return jsonify({"error": "No text provided"})

    result = analyze_tone(text, context)
    return jsonify(result)

@app.route("/rewrite", methods=["POST"])
def rewrite():
    data = request.get_json()
    text = data.get("text", "")
    context = data.get("context", "chat")
    scores = data.get("scores", {})

    if not text.strip():
        return jsonify({"error": "No text provided"})

    try:
        suggestions = generate_rewrite_suggestions(text, context, scores)
        return jsonify({
            "suggestions": suggestions,
            "original_text": text,
            "context": context
        })
    except Exception as e:
        return jsonify({"error": f"Rewrite failed: {str(e)}"})

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Retone AI Lite Backend is running with local T5 rewrite model!"})

if __name__ == "__main__":
    app.run(debug=True)