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
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

def create_rewrite_prompt(text, context):
    """Create context-aware prompts for rewriting"""
    context_prompts = {
        "chat": f"Rewrite this message to be friendly and respectful: {text}",
        "social": f"Make this social media post more positive and engaging: {text}",
        "email": f"Make this text professional and polite for business: {text}"
    }
    return context_prompts.get(context, context_prompts["chat"])

def rewrite_text_with_api(text, context="chat"):
    """Rewrite text using Hugging Face Inference API"""
    if not HUGGINGFACE_API_KEY:
        logger.error("Hugging Face API key not configured")
        return {"error": "Hugging Face API key not configured"}
    
    try:
        # Try multiple models in order of preference
        models = [
            "facebook/blenderbot-400M-distill",
            "microsoft/DialoGPT-medium",
            "google/flan-t5-base",
            "t5-small"
        ]
        
        for model in models:
            try:
                api_url = f"https://api-inference.huggingface.co/models/{model}"
                headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
                
                # Different approaches for different models
                if "blenderbot" in model:
                    # Blenderbot is a conversational model
                    payload = {
                        "inputs": create_rewrite_prompt(text, context),
                        "parameters": {
                            "max_length": 100,
                            "temperature": 0.7,
                            "do_sample": True
                        }
                    }
                elif "flan-t5" in model or "t5" in model:
                    # T5 models work with text-to-text generation
                    payload = {
                        "inputs": f"paraphrase: {text}",
                        "parameters": {
                            "max_length": 100,
                            "temperature": 0.7,
                            "do_sample": True
                        }
                    }
                else:
                    # Default for other models
                    payload = {
                        "inputs": create_rewrite_prompt(text, context),
                        "parameters": {
                            "max_new_tokens": 100,
                            "temperature": 0.7,
                            "do_sample": True,
                            "return_full_text": False
                        }
                    }
                
                response = requests.post(api_url, headers=headers, json=payload, timeout=30)
                logger.info(f"Trying model {model}: status {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    rewritten = None
                    
                    # Handle different response formats
                    if isinstance(result, list) and len(result) > 0:
                        if "generated_text" in result[0]:
                            rewritten = result[0]["generated_text"].strip()
                        elif "translation_text" in result[0]:
                            rewritten = result[0]["translation_text"].strip()
                        else:
                            rewritten = str(result[0]).strip()
                    elif isinstance(result, dict):
                        if "generated_text" in result:
                            rewritten = result["generated_text"].strip()
                        elif "translation_text" in result:
                            rewritten = result["translation_text"].strip()
                    
                    # Clean up the rewritten text
                    if rewritten:
                        # Remove the original prompt if it's included
                        if text.lower() in rewritten.lower():
                            # Try to extract just the rewritten part
                            parts = rewritten.split(text)
                            if len(parts) > 1:
                                rewritten = parts[-1].strip()
                        
                        # Remove common prefixes that models add
                        prefixes_to_remove = [
                            "rewrite this message to be friendly and respectful:",
                            "make this social media post more positive and engaging:",
                            "make this text professional and polite for business:",
                            "paraphrase:",
                            "rewritten:",
                            "improved:"
                        ]
                        
                        for prefix in prefixes_to_remove:
                            if rewritten.lower().startswith(prefix):
                                rewritten = rewritten[len(prefix):].strip()
                        
                        # Basic quality check
                        if len(rewritten) > 5 and rewritten.lower() != text.lower():
                            return {
                                "rewritten_text": rewritten,
                                "model_used": model
                            }
                
                elif response.status_code == 503:
                    logger.warning(f"Model {model} is loading, trying next...")
                    continue
                elif response.status_code == 404:
                    logger.warning(f"Model {model} not found, trying next...")
                    continue
                else:
                    logger.warning(f"Model {model} failed with status {response.status_code}")
                    continue
                    
            except Exception as e:
                logger.error(f"Error with model {model}: {str(e)}")
                continue
        
        # If all models fail, use fallback
        logger.warning("All Hugging Face models failed, using fallback")
        return rewrite_text_fallback(text, context)
            
    except Exception as e:
        logger.error(f"Rewrite failed: {str(e)}")
        return rewrite_text_fallback(text, context)

def rewrite_text_fallback(text, context="chat"):
    """Enhanced fallback rewrite using improved rules"""
    try:
        # Comprehensive word replacement rules
        rules = {
            # Profanity
            "fuck": "very", "fucking": "very", "shit": "stuff", "damn": "darn",
            "hell": "heck", "bitch": "person", "ass": "butt", "bastard": "person",
            "piss": "annoy", "crap": "stuff", "douche": "person", "turd": "thing",
            
            # Insults
            "stupid": "uninformed", "idiot": "person", "moron": "individual",
            "dumb": "uninformed", "retard": "person", "jerk": "person",
            "loser": "individual", "freak": "unique person", "weirdo": "individual",
            "creep": "person", "psycho": "person", "lunatic": "person",
            
            # Negative words
            "hate": "strongly dislike", "sucks": "is not ideal", "terrible": "less than ideal",
            "awful": "unpleasant", "worst": "most challenging", "horrible": "difficult",
            "disgusting": "unpleasant", "pathetic": "concerning", "worthless": "undervalued",
            "useless": "currently not helpful", "failure": "learning experience",
            
            # Threats/aggressive language
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
        else:  # chat
            if not rewritten.strip().endswith(('.', '!', '?')):
                rewritten += "."
            rewritten = f"I think {rewritten.strip()}"
        
        # Clean up extra spaces and capitalize properly
        rewritten = re.sub(r'\s+', ' ', rewritten).strip()
        if rewritten:
            rewritten = rewritten[0].upper() + rewritten[1:] if len(rewritten) > 1 else rewritten.upper()
        
        return {
            "rewritten_text": rewritten,
            "model_used": "fallback"
        }
        
    except Exception as e:
        logger.error(f"Fallback rewrite failed: {str(e)}")
        return {"error": f"Fallback rewrite failed: {str(e)}"}

def analyze_tone(text, context="chat"):
    """Analyze text tone using Perspective API"""
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
    """Determine if text is toxic based on context-specific thresholds"""
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

# Test endpoint to verify API keys
@app.route("/test-apis", methods=["GET"])
def test_apis():
    """Test both APIs to verify they're working"""
    results = {
        "perspective_api": "not configured",
        "huggingface_api": "not configured"
    }
    
    # Test Perspective API
    if API_KEY:
        try:
            test_result = analyze_tone("This is a test message")
            if "error" not in test_result:
                results["perspective_api"] = "working"
            else:
                results["perspective_api"] = f"error: {test_result['error']}"
        except Exception as e:
            results["perspective_api"] = f"error: {str(e)}"
    
    # Test Hugging Face API
    if HUGGINGFACE_API_KEY:
        try:
            test_result = rewrite_text_with_api("This is a test message")
            if "error" not in test_result:
                results["huggingface_api"] = "working"
            else:
                results["huggingface_api"] = f"error: {test_result['error']}"
        except Exception as e:
            results["huggingface_api"] = f"error: {str(e)}"
    
    return jsonify(results)

@app.route("/analyze", methods=["POST"])
def analyze():
    """Analyze text toxicity"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
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
    """Rewrite toxic text to be more appropriate"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    text = data.get("text", "")
    context = data.get("context", "chat")
    
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400
    
    # Try API first, then fallback
    result = rewrite_text_with_api(text, context)
    if "error" in result:
        logger.info(f"API failed: {result['error']}, trying fallback...")
        result = rewrite_text_fallback(text, context)
    
    return jsonify(result)

@app.route("/feedback", methods=["POST"])
def feedback():
    """Record user feedback"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
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
    """Health check endpoint"""
    return jsonify({"message": "Retone AI Lite Backend is running!"})

@app.route("/health", methods=["GET"])
def health():
    """Detailed health check"""
    return jsonify({
        "status": "healthy",
        "perspective_api": "configured" if API_KEY else "not configured",
        "huggingface_api": "configured" if HUGGINGFACE_API_KEY else "not configured"
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)