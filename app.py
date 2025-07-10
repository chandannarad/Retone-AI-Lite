from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
import os
import json
import logging

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY = os.getenv("PERSPECTIVE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

def create_rewrite_prompt(text, context):
    """Create context-aware prompts for rewriting"""
    context_prompts = {
        "chat": f"Rewrite this message to be friendly, respectful, and free of toxic or offensive language for casual conversation: '{text}'",
        "social": f"Rewrite this social media post to be positive, engaging, and free of toxic or offensive language: '{text}'",
        "email": f"Rewrite this text to be professional, polite, and free of toxic or offensive language for business communication: '{text}'"
    }
    return context_prompts.get(context, context_prompts["chat"])

def rewrite_text_with_api(text, context="chat"):
    """Rewrite text using Hugging Face Inference API"""
    if not HUGGINGFACE_API_KEY:
        logger.error("Hugging Face API key not configured")
        return {"error": "Hugging Face API key not configured"}
    
    try:
        api_url = "https://api-inference.huggingface.co/models/Vamsi/T5_Paraphrase_Paws"
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        prompt = create_rewrite_prompt(text, context)
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 150,
                "temperature": 0.7,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        response = requests.post(api_url, headers=headers, json=payload)
        logger.info(f"Hugging Face API response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                rewritten = result[0].get("generated_text", "").strip()
            elif isinstance(result, dict):
                rewritten = result.get("generated_text", "").strip()
            else:
                logger.error("Unexpected API response format")
                return {"error": "Unexpected API response format"}
            
            if rewritten and rewritten.lower() != text.lower():
                return {"rewritten_text": rewritten}
            else:
                logger.warning("API returned empty or unchanged text")
                return {"error": "No valid rewrite generated"}
        
        elif response.status_code == 503:
            logger.warning("Model is loading")
            return {"error": "Model is loading, please try again in a moment"}
        else:
            logger.error(f"API request failed with status {response.status_code}")
            return {"error": f"API request failed with status {response.status_code}"}
            
    except Exception as e:
        logger.error(f"Rewrite failed: {str(e)}")
        return {"error": f"Rewrite failed: {str(e)}"}

def rewrite_text_fallback(text, context="chat"):
    """Fallback rewrite using enhanced rules"""
    try:
        rules = {
            "stupid": "unwise",
            "idiot": "individual",
            "hate": "disagree with",
            "terrible": "unfavorable",
            "awful": "unpleasant",
            "worst": "difficult",
            "sucks": "is less than ideal",
            "sucker": "",
            "fucker": "",
            "fuck": "",
            "damn": "",
            "hell": "",
            "dumb": "uninformed",
            "jerk": "person",
            "loser": "individual",
            "bad": "not ideal"
        }
        
        rewritten = text.lower()
        for bad_word, replacement in rules.items():
            rewritten = rewritten.replace(bad_word, replacement)
        
        # Basic rephrasing
        if context == "email":
            rewritten = f"Dear recipient, I believe that {rewritten}."
        elif context == "social":
            rewritten = f"Here's my take: {rewritten}."
        else:  # chat
            rewritten = f"Hey, I think:  {rewritten}."
        
        # Capitalize and clean up
        rewritten = ' '.join(rewritten.capitalize().split())
        
        return {"rewritten_text": rewritten}
        
    except Exception as e:
        logger.error(f"Fallback rewrite failed: {str(e)}")
        return {"error": f"Fallback rewrite failed: {str(e)}"}

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
        scores["context"] = context
        return scores
    except Exception as e:
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
    result = rewrite_text_with_api(text, context)
    if "error" in result:
        logger.info(f"API failed: {result['error']}, trying fallback...")
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
    app.run(debug=True)