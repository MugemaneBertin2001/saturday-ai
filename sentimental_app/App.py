from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from model import SentimentModel
from utils import preprocess_audio
import torch
from flask_cors import CORS
from flasgger import Swagger, swag_from 

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}

app = Flask(__name__)
CORS(app)
Swagger(app)  # Initialize Swagger
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

model = SentimentModel()
model.load_state_dict(torch.load('model/best_model.pth', map_location=torch.device('cpu')))
model.eval()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
@swag_from('docs/predict.yml')  # Load Swagger documentation from an external file
def upload_and_predict():
    """
    Upload an audio file and get sentiment prediction.
    ---
    consumes:
      - multipart/form-data
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: The audio file to upload
    responses:
      200:
        description: The sentiment prediction result
        schema:
          id: SentimentPrediction
          properties:
            message:
              type: string
              description: Success message
            filename:
              type: string
              description: Name of the uploaded file
            sentiment:
              type: string
              description: Predicted sentiment (negative, neutral, positive)
      400:
        description: Bad request (e.g., missing file, invalid format)
      500:
        description: Internal server error (e.g., prediction error)
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # Preprocess the audio
            audio_tensor = preprocess_audio(file_path)
            
            # Run the model and get the prediction
            with torch.no_grad():
                output = model(audio_tensor)
                prediction = torch.argmax(output, dim=1).item()
            
            sentiment = ['negative', 'neutral', 'positive'][prediction]

            return jsonify({"message": "File uploaded successfully", "filename": filename, "sentiment": sentiment}), 200
        
        except Exception as e:
            return jsonify({"error": f"Error during prediction: {str(e)}"}), 500
    
    return jsonify({"error": "Invalid file format"}), 400

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=False, host='0.0.0.0')
