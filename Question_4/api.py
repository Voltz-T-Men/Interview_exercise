from flask import Flask, request, jsonify
import numpy as np
from model import NeuralNetwork
from utils import extract_label_features, cosine_similarity, create_triplets
import config

app = Flask(__name__)

# Initialize the model and load weights
model = NeuralNetwork(config.INPUT_SIZE, config.HIDDEN_SIZE, config.OUTPUT_SIZE)
model.load_weights('model_weights.npy')  

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict the label of an input sample.

    Request:
    --------
    POST /predict
    Content-Type: application/json
    Body:
    {
        "input": [list of input feature values]
    }

    Response:
    ---------
    - On success:
      Status Code: 200 OK
      Content-Type: application/json
      Body:
      {
          "predicted_label": int
      }
    
    - On error (e.g., missing input):
      Status Code: 400 Bad Request
      Content-Type: application/json
      Body:
      {
          "error": "No input provided"
      }

    Notes:
    ------
    The `input` should be a list of numerical values representing the feature vector for which
    the prediction is to be made. The model performs inference to predict the label based
    on similarity with stored label features.
    """
    data = request.json
    if 'input' not in data:
        return jsonify({"error": "No input provided"}), 400
    
    input_sample = np.array(data['input'])
    
    if input_sample.ndim == 1:
        input_sample = input_sample.reshape(1, -1)
    
    # Perform inference
    label_features_list, unique_labels = extract_label_features(model, X_train, y_train)
    input_features = model.forward(input_sample)
    similarities = [cosine_similarity(input_features, label_feature) for label_feature in label_features_list]
    predicted_label = unique_labels[np.argmax(similarities)]
    
    return jsonify({"predicted_label": int(predicted_label)})

if __name__ == '__main__':
    app.run(debug=True)
