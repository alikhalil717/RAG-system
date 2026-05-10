from flask import Flask, request, jsonify
import logging
from sentence_transformers import CrossEncoder
from .config import *
from .utils import generate_embedding


print("Loading reranker model")
global reranker
reranker = CrossEncoder(RERANKER_MODEL_NAME, device=RERANKER_DEVICE, cache_folder=MODELS_CACHE_PATH)
print("Reranker model loaded")
app = Flask(__name__)
logging.getLogger('werkzeug').setLevel(logging.ERROR)
# health check endpoint
@app.route('/.well-known/ready', methods=['GET'])
def readiness():
    return "Ready", 200

@app.route('/meta', methods=['GET'])
def meta():
    return jsonify({'status': 'Ready'}), 200

@app.route('/vectors', methods=['POST'])
def vectorize():
    try:
        data = request.get_json(force=True)
        if 'text' not in data:
            return jsonify({'error': 'Missing "text" field in request body'}), 400
        texts = data['text']
        if not isinstance(texts, list):
            return jsonify({'error': '"text" field must be a list of strings'}), 400
        embeddings = generate_embedding(texts)
        if len(embeddings) == 1:
            embeddings = embeddings[0]
        return jsonify({'vectors': embeddings}), 200
    except Exception as e:
        logging.exception("Error in /vectors endpoint")
        return jsonify({'error': str(e)}), 500
@app.route('/rerank', methods=['POST'])
def rerank_endpoint():
    try:
        data = request.get_json(force=True)
        if 'query' not in data or 'documents' not in data:
            return jsonify({'error': 'Missing "query" or "documents" field in request body'}), 400
        query = data['query']
        documents = data['documents']
        if not isinstance(documents, list):
            return jsonify({'error': '"documents" field must be a list of strings'}), 400
        pairs = [(query, doc) for doc in documents]
        scores = reranker.predict(pairs)
        results = [
            {"document": doc, "score": float(score)}
            for doc, score in zip(documents, scores)
        ]
        return jsonify({'scores': results})

    except Exception as e:
        logging.exception("Error in /rerank endpoint")
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)