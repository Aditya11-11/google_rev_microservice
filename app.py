
import os
import json
from flask import Flask, request, jsonify
from langchain_groq import ChatGroq
from flask_cors import CORS
from flask_cors import cross_origin
import ast 

app = Flask(__name__)

# Global dictionary to store the two prompt components.
system_prompt = {
            "manual": """
        You're an AI assistant that reviews customer feedback for restaurants,hotels etc.

        Your task is to:
        1. Identify the overall sentiment in the review (positive, negative, or mixed).
        2. Extract specific problems or complaints (if any).
        3. Suggest actionable, practical solutions to those problems.

        Your output should strictly follow this JSON format:
        {
        "sentiment": "...",
        "problems": ["..."],
        "solutions": ["..."]
        }
        """,           
}

API_KEY = "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"
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/api/get_prompt', methods=['GET'])
@cross_origin()
def get_prompt():
    global system_prompt
    if system_prompt["manual"] is None :
        return jsonify({"error": "No system prompt has been set yet."}), 404

    combined_prompt = ""
    if system_prompt["manual"]:
        combined_prompt += system_prompt["manual"]
    return jsonify({"system_prompt": combined_prompt}), 200

@app.route('/api/chat', methods=['POST'])
@cross_origin()
def chat():
    data = request.get_json()
    if not data or "human_message" not in data:
        return jsonify({"error": "Please provide a 'human_message' field in the request body."}), 400

    if not API_KEY:
        return jsonify({"error": "API key not set. Please set it via /api/set_api_key."}), 500

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=API_KEY,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    combined_prompt = "\n".join(filter(None, [system_prompt.get("manual")]))

    messages = []
    if combined_prompt:
        messages.append(("system", combined_prompt))

    messages.append(("human", data["human_message"]))

    try:
        ai_response = llm.invoke(messages)
        try:
            AI_MSG = json.loads(ai_response.content)
        except json.JSONDecodeError:
            try:
                AI_MSG = ast.literal_eval(ai_response.content)
            except Exception:
                AI_MSG = {"raw": ai_response.content}
    except Exception as e:
        return jsonify({"error": f"LLM invocation error: {e}"}), 500

    return jsonify({"response": AI_MSG}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
