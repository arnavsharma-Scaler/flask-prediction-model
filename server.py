from flask import Flask, request, jsonify
from model import EmbeddingClassi

app = Flask(__name__)

model = EmbeddingClassi()

@app.route('/api/',methods=['POST'])
def predict():
    data = request.get_json(force=True)
    if 'query' in data:
        prediction = model.query(data['query'])
        output = prediction[0]
        return jsonify(output)
    else:
        return jsonify({"error": "Missing 'query' in request data"}), 400

if __name__ == '__main__':
    try:
        app.run(port=5000, debug=True)
    except:
        print("Server is exited unexpectedly. Please contact server admin.")