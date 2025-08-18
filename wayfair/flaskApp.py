from flask import Flask, jsonify, request

app = Flask(__name__)

# Simple GET endpoint
@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({"message": "Hello, Flask API!"})

# Example GET with query parameter
@app.route('/greet', methods=['GET'])
def greet():
    name = request.args.get("name", "World")
    return jsonify({"message": f"Hello, {name}!"})

# Example POST endpoint
@app.route('/echo', methods=['POST'])
def echo():
    data = request.get_json()
    return jsonify({"you_sent": data})

if __name__ == '__main__':
    app.run(debug=True)  # Runs on http://127.0.0.1:5000
