from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
import os
import json
import logging
import re

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY = os.getenv("PERSPECTIVE_API_KEY")


def rewrite_text_fallback(text, context="chat"):
    """Enhanced fallback rewrite using improved rules"""
    try:
        # Comprehensive word replacement rules
        rules = {
            "fuck": "very", "fucking": "very", "shit": "stuff", "damn": "darn",
            "hell": "heck", "bitch": "person", "ass": "butt", "bastard": "person",
            "piss": "annoy", "crap": "stuff", "douche": "person", "turd": "thing",
            "stupid": "uninformed", "idiot": "person", "moron": "individual",
            "dumb": "uninformed", "retard": "person", "jerk": "person",
            "loser": "individual", "freak": "unique person", "weirdo": "individual",
            "creep": "person", "psycho": "person", "lunatic": "person",
            "hate": "strongly dislike", "sucks": "is not ideal", "terrible": "less than ideal",
            "awful": "unpleasant", "worst": "most challenging", "horrible": "difficult",
            "disgusting": "unpleasant", "pathetic": "concerning", "worthless": "undervalued",
            "useless": "currently not helpful", "failure": "learning experience",
            "kill": "stop", "die": "go away", "destroy": "change", "attack": "address",
            "hurt": "affect", "harm": "affect", "violence": "conflict", "weapon": "tool",
        }

        rewritten = text

        # Apply word replacements (case-insensitive, whole words only)
        for bad_word, replacement in rules.items():
            pattern = r'\b' + re.escape(bad_word) + r'\b'
            rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)

        # Context-specific improvements
        if context == "email":
            if not rewritten.strip().endswith(('.', '!', '?')):
                rewritten += "."
            rewritten = f"I would like to express that {rewritten.strip()}"
        elif context == "social":
            if not rewritten.strip().endswith(('.', '!', '?')):
                rewritten += "."
            rewritten = f"Here's my perspective: {rewritten.strip()}"
        else:
            if not rewritten.strip().endswith(('.', '!', '?')):
                rewritten += "."
            rewritten = f"I think {rewritten.strip()}"

        rewritten = re.sub(r'\s+', ' ', rewritten).strip()
        if rewritten:
            rewritten = rewritten[0].upper() + rewritten[1:] if len(rewritten) > 1 else rewritten.upper()

        return {
            "rewritten_text": rewritten,
            "model_used": "Local Rewrite Engine"
        }

    except Exception as e:
        logger.error(f"Fallback rewrite failed: {str(e)}")
        return {"error": f"Fallback rewrite failed: {str(e)}"}


def analyze_tone(text, context="chat"):
    if not API_KEY:
        logger.error("Perspective API key not configured")
        return {"error": "Perspective API key not configured"}

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
        response = requests.post(url, json=data, timeout=30)
        result = response.json()

        if "attributeScores" not in result:
            logger.error(f"API failed or returned no scores: {result}")
            return {"error": "API failed or returned no scores."}

        scores = {}
        for attr in data["requestedAttributes"]:
            scores[attr] = round(result["attributeScores"][attr]["summaryScore"]["value"] * 100, 2)

        scores["context"] = context
        return scores

    except requests.exceptions.Timeout:
        logger.error("Perspective API request timed out")
        return {"error": "Request timed out"}
    except Exception as e:
        logger.error(f"Perspective API request failed: {str(e)}")
        return {"error": f"Request failed: {str(e)}"}


def is_toxic(scores, context):
    thresholds = {
        "chat": {"toxicity": 40, "insult": 30, "threat": 20, "profanity": 30},
        "social": {"toxicity": 35, "insult": 25, "threat": 15, "profanity": 25},
        "email": {"toxicity": 25, "insult": 20, "threat": 10, "profanity": 20}
    }
    threshold = thresholds.get(context, thresholds["chat"])

    return (scores.get("TOXICITY", 0) >= threshold["toxicity"] or
            scores.get("INSULT", 0) >= threshold["insult"] or
            scores.get("THREAT", 0) >= threshold["threat"] or
            scores.get("PROFANITY", 0) >= threshold["profanity"])


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "")
    context = data.get("context", "chat")
    if not text.strip():
        return jsonify({"error": "No text provided"})
    result = analyze_tone(text, context)
    if "error" not in result:
        result["is_toxic"] = is_toxic(result, context)
    return jsonify(result)


@app.route("/rewrite", methods=["POST"])
def rewrite():
    data = request.get_json()
    text = data.get("text", "")
    context = data.get("context", "chat")
    if not text.strip():
        return jsonify({"error": "No text provided"})
    result = rewrite_text_fallback(text, context)
    return jsonify(result)


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()
    original_text = data.get("original_text", "")
    rewritten_text = data.get("rewritten_text", "")
    rating = data.get("rating", "")
    context = data.get("context", "chat")
    feedback_log = {
        "original": original_text,
        "rewritten": rewritten_text,
        "rating": rating,
        "context": context
    }
    logger.info(f"Feedback received: {feedback_log}")
    return jsonify({"status": "Feedback recorded", "message": "Thank you for your feedback!"})


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Retone AI Lite Backend is running!"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
