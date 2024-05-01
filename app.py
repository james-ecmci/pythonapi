from flask import Flask, jsonify, request
from deepface import DeepFace

app = Flask(__name__)


# Define API endpoint for saying hello
@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello, World!'})


# Define API endpoint for face verification
@app.route('/api/face-verification', methods=['POST'])
def face_verification():
    # Check if base64-encoded images are provided
    img1_base64 = request.json.get('img1_base64')
    img2_base64 = request.json.get('img2_base64')

    # Perform face verification
    result = DeepFace.verify(img1_path=img1_base64, img2_path=img2_base64, enforce_detection=False)

    # Return the entire result
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
