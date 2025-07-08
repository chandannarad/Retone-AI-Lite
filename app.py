from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
import os
import json

app = Flask(__name__)
CORS(app)

# API Keys
PERSPECTIVE_API_KEY = os.getenv("PERSPECTIVE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Hugging Face API URL
HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large"

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
    """Generate rewrite suggestions using Hugging Face models"""
    
    # Context-specific prompts
    context_prompts = {
        "chat": "Rewrite this message to be more friendly and respectful while keeping it casual: ",
        "social": "Rewrite this social media post to be more positive and engaging: ",
        "email": "Rewrite this message to be professional and polite: "
    }
    
    base_prompt = context_prompts.get(context, context_prompts["chat"])
    
    suggestions = []
    
    try:
        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Generate multiple suggestions with different approaches
        prompts = [
            f"{base_prompt}{text}",
            f"Make this message more polite and respectful: {text}",
            f"Rewrite this to express the same idea in a kinder way: {text}"
        ]
        
        for i, prompt in enumerate(prompts):
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_length": min(len(text) + 50, 200),
                    "min_length": max(10, len(text) - 20),
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(HF_API_URL, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
                        generated_text = result[0]["generated_text"]
                    else:
                        generated_text = ""

                    
                    # Clean up the generated text
                    if generated_text.startswith(prompt):
                        generated_text = generated_text[len(prompt):].strip()
                    
                    if generated_text and generated_text != text:
                        suggestion_type = ["Respectful Version", "Polite Version", "Kind Version"][i]
                        suggestions.append({
                            "type": suggestion_type,
                            "text": generated_text.strip()
                        })
        
        # If no suggestions were generated, provide fallback suggestions
        if not suggestions:
            suggestions = get_fallback_suggestions(text, context)
            
        return suggestions[:3]  # Return max 3 suggestions
        
    except Exception as e:
        print(f"Hugging Face API error: {e}")
        return get_fallback_suggestions(text, context)

def get_fallback_suggestions(text, context):
    """Provide fallback suggestions when AI model fails"""
    
    fallback_templates = {
        "chat": [
            "I understand you might be frustrated, but let me rephrase this more kindly.",
            "I hear what you're saying. Let me express this more respectfully.",
            "I get your point. Here's a friendlier way to say it."
        ],
        "social": [
            "Here's a more positive way to share this thought.",
            "Let me reframe this in a more constructive way.",
            "Here's how I'd express this more diplomatically."
        ],
        "email": [
            "I would like to respectfully express my concerns about this matter.",
            "I hope you'll consider my perspective on this issue.",
            "I wanted to share my thoughts on this topic professionally."
        ]
    }
    
    templates = fallback_templates.get(context, fallback_templates["chat"])
    
    return [
        {
            "type": "Respectful Alternative",
            "text": templates[0]
        },
        {
            "type": "Professional Version", 
            "text": templates[1]
        }
    ]

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
    
    if not HUGGINGFACE_API_KEY:
        return jsonify({"error": "Hugging Face API key not configured"})
    
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
    return jsonify({"message": "Retone AI Lite Backend is running with rewrite support!"})

if __name__ == "__main__":
    app.run(debug=True)