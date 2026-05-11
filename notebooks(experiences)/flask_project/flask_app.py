from flask import Flask, request, jsonify
import logging
from sentence_transformers import CrossEncoder

print("Loading reranker model")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')
print("Reranker model loaded")

app = Flask(__name__)

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

@app.route('/.well-known/ready', methods=['GET'])
def readiness():
    
    return "Ready", 200

@app.route('/meta', methods=['GET'])
def meta():
    return jsonify({'status': 'Ready'}), 200

@app.route('/vectors', methods=['POST'])
def vectorize():
    try:
        from utils import generate_embedding
        data = request.get_json(force=True)
        if 'text' not in data:
            return jsonify({'error': 'Missing "text" field'}), 400
        texts = data['text']
        if isinstance(texts, str):
            texts = [texts]
        embeddings = generate_embedding(texts)
        if len(embeddings) == 1:
            return jsonify({'vector': embeddings[0]})
        return jsonify({'vectors': embeddings})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/rerank', methods=['POST'])
def rerank_endpoint():
    try:
        data = request.get_json(force=True)
        query = data.get('query')
        documents = data.get('documents', [])
        if not query or not documents:
            return jsonify({'scores': []})
        pairs = [(query, doc) for doc in documents]
        scores = reranker.predict(pairs)  
        results = [
            {"document": doc, "score": float(score)}
            for doc, score in zip(documents, scores)
        ]
        return jsonify({'scores': results})
    except Exception as e:
        print(f"Rerank error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)