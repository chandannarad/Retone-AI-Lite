from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
import os
import gc
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)
CORS(app)

API_KEY = os.getenv("PERSPECTIVE_API_KEY")

# Global variables for model management
model = None
tokenizer = None
model_loaded = False

def load_model():
    """Load the lightweight T5 model only when needed"""
    global model, tokenizer, model_loaded
    
    if not model_loaded:
        try:
            # Use a small, efficient model that fits in 512MB RAM
            model_name = "google/flan-t5-small"  # ~80MB model
            
            print("Loading rewrite model...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Move to CPU to save memory
            if not torch.cuda.is_available():
                model = model.to('cpu')
            
            model_loaded = True
            print("Model loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    return True

def unload_model():
    """Unload model to free memory"""
    global model, tokenizer, model_loaded
    
    if model_loaded:
        del model
        del tokenizer
        model = None
        tokenizer = None
        model_loaded = False
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Model unloaded to free memory")

def create_rewrite_prompt(text, context):
    """Create context-aware prompts for rewriting"""
    
    context_prompts = {
        "chat": f"Rewrite this message to be more friendly and respectful for casual conversation: {text}",
        "social": f"Rewrite this social media post to be more positive and engaging: {text}",
        "email": f"Rewrite this text to be more professional and polite for business communication: {text}"
    }
    
    return context_prompts.get(context, context_prompts["chat"])

def rewrite_text(text, context="chat"):
    """Rewrite toxic text to be more respectful"""
    
    if not load_model():
        return {"error": "Model loading failed"}
    
    try:
        # Create context-aware prompt
        prompt = create_rewrite_prompt(text, context)
        
        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate rewrite with controlled parameters
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=150,
                num_beams=2,  # Reduced beams to save memory
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the result
        rewritten = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the output
        rewritten = rewritten.strip()
        
        # Unload model after use to free memory
        unload_model()
        
        return {"rewritten_text": rewritten}
        
    except Exception as e:
        unload_model()  # Ensure model is unloaded on error
        return {"error": f"Rewrite failed: {str(e)}"}

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

def is_toxic(scores, context):
    """Check if text is toxic based on context-aware thresholds"""
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
    
    # Add toxicity flag for frontend
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
    
    result = rewrite_text(text, context)
    return jsonify(result)

@app.route("/feedback", methods=["POST"])
def feedback():
    """Store user feedback on rewrites"""
    data = request.get_json()
    original_text = data.get("original_text", "")
    rewritten_text = data.get("rewritten_text", "")
    rating = data.get("rating", "")  # 'helpful' or 'not_helpful'
    context = data.get("context", "chat")
    
    # For now, just log feedback (in production, you'd store in a database)
    feedback_log = {
        "original": original_text,
        "rewritten": rewritten_text,
        "rating": rating,
        "context": context
    }
    
    print(f"Feedback received: {feedback_log}")
    
    # You can store this in a database or file for later analysis
    return jsonify({"status": "Feedback recorded", "message": "Thank you for your feedback!"})

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Retone AI Lite Backend is running!"})

if __name__ == "__main__":
    app.run(debug=True)