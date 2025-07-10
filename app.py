from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
import os
import json
import logging
import re

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY = os.getenv("PERSPECTIVE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

def rewrite_text_with_api(text, context="chat"):
    if not HUGGINGFACE_API_KEY:
        logger.error("Hugging Face API key not configured")
        return {"error": "Hugging Face API key not configured"}

    models = [
        "prithivida/parrot_paraphraser_on_T5",
        "Vamsi/T5_Paraphrase_Paws"
    ]

    for model in models:
        try:
            api_url = f"https://api-inference.huggingface.co/models/{model}"
            headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
            prompt = f"paraphrase: {text} </s>"

            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_length": 100,
                    "temperature": 0.7,
                    "do_sample": True,
                    "top_k": 120,
                    "top_p": 0.95,
                    "early_stopping": True,
                    "num_return_sequences": 1
                }
            }

            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            logger.info(f"Trying model {model}: status {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                rewritten = result[0].get("generated_text", "").strip() if isinstance(result, list) and result else None
                if rewritten and rewritten.lower() != text.lower():
                    return {"rewritten_text": rewritten, "model_used": model}
            elif response.status_code in [404, 503]:
                continue
        except Exception as e:
            logger.error(f"Error with model {model}: {str(e)}")
            continue

    logger.warning("All Hugging Face models failed, using fallback")
    return rewrite_text_fallback(text, context)

def rewrite_text_fallback(text, context="chat"):
    try:
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
            "hurt": "affect", "harm": "affect", "violence": "conflict", "weapon": "tool"
        }

        rewritten = text
        for bad_word, replacement in rules.items():
            pattern = r'\\b' + re.escape(bad_word) + r'\\b'
            rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)

        if context == "email":
            rewritten = f"I would like to express that {rewritten.strip().rstrip('.!?')}"
        elif context == "social":
            rewritten = f"Here's my perspective: {rewritten.strip().rstrip('.!?')}"
        else:
            rewritten = f"I think {rewritten.strip().rstrip('.!?')}"

        rewritten = re.sub(r'\\s+', ' ', rewritten).strip()
        if rewritten:
            rewritten = rewritten[0].upper() + rewritten[1:]

        return {"rewritten_text": rewritten, "model_used": "fallback"}

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

        scores = {k: round(result["attributeScores"][k]["summaryScore"]["value"] * 100, 2)
                  for k in data["requestedAttributes"]}
        scores["context"] = context
        return scores

    except Exception as e:
        logger.error(f"Perspective API request failed: {str(e)}")
        return {"error": f"Request failed: {str(e)}"}

def is_toxic(scores, context):
    thresholds = {
        "chat": {"toxicity": 40, "insult": 30, "threat": 20, "profanity": 30},
        "social": {"toxicity": 35, "insult": 25, "threat": 15, "profanity": 25},
        "email": {"toxicity": 25, "insult": 20, "threat": 10, "profanity": 20}
    }
    t = thresholds.get(context, thresholds["chat"])
    return any(scores.get(attr, 0) >= t[attr.lower()] for attr in ["TOXICITY", "INSULT", "THREAT", "PROFANITY"])

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "")
    context = data.get("context", "chat")
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400
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
        return jsonify({"error": "No text provided"}), 400
    result = rewrite_text_with_api(text, context)
    if "error" in result:
        logger.info(f"API failed: {result['error']}, trying fallback...")
        result = rewrite_text_fallback(text, context)
    return jsonify(result)

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()
    logger.info(f"Feedback received: {data}")
    return jsonify({"status": "Feedback recorded", "message": "Thank you for your feedback!"})

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Retone AI Lite Backend is running!"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)