import joblib
from flask import Flask, request, jsonify
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Path ke model yang telah dilatih
model_path = "models/multi_output_model.h5"
model = load_model(model_path)

# Path ke scaler
scaler_path = "models/scaler.pkl"

try:
    scaler = joblib.load(scaler_path)
except Exception as e:
    scaler = None
    print(f"Error loading scaler: {e}")

# Dimensi kepribadian
dimensions = ["EXT", "AGR", "CSN", "EST", "OPN"]

@app.route("/", methods=["GET"])
def home():
    return "API is running."

@app.route("/personality", methods=["POST"])
def predict():
    try:
        if scaler is None:
            return jsonify({"error": "Scaler is not available. Please ensure the scaler file exists and is correctly configured."}), 500

        data = request.get_json()
        if "input" not in data:
            return jsonify({"error": "Input data missing"}), 400
        
        # Ekstraksi nilai dari JSON berbasis pertanyaan
        questions = [
            "I am the life of the party.", "I don't talk a lot.", "I feel comfortable around people.", "I keep in the background.",
            "I start conversations.", "I have little to say.", "I talk to a lot of different people at parties.",
            "I don't like to draw attention to myself.", "I don't mind being the center of attention.", "I am quiet around strangers.",
            "I get stressed out easily.", "I am relaxed most of the time.", "I worry about things.", "I seldom feel blue.",
            "I am easily disturbed.", "I get upset easily.", "I change my mood a lot.", "I have frequent mood swings.",
            "I get irritated easily.", "I often feel blue.", "I feel little concern for others.", "I am interested in people.",
            "I insult people.", "I sympathize with others' feelings.", "I am not interested in other people's problems.",
            "I have a soft heart.", "I am not really interested in others.", "I take time out for others.",
            "I feel others' emotions.", "I make people feel at ease.", "I am always prepared.", "I leave my belongings around.",
            "I pay attention to details.", "I make a mess of things.", "I get chores done right away.",
            "I often forget to put things back in their proper place.", "I like order.", "I shirk my duties.",
            "I follow a schedule.", "I am exacting in my work.", "I have a rich vocabulary.", "I have difficulty understanding abstract ideas.",
            "I have a vivid imagination.", "I am not interested in abstract ideas.", "I have excellent ideas.",
            "I do not have a good imagination.", "I am quick to understand things.", "I use difficult words.",
            "I spend time reflecting on things.", "I am full of ideas."
        ]
        
        # Pastikan semua pertanyaan tersedia
        input_data = [data["input"].get(q, 0) for q in questions]

        # Convert input to numpy array and reshape
        input_data = np.array(input_data).reshape(1, -1)
        
        # Scale the input
        scaled_input = scaler.transform(input_data)
        
        # Get predictions
        predictions = model.predict(scaled_input)
        
        # Convert numpy values to Python native types and round scores
        scores = {dimensions[i]: round(float(predictions[i][0]), 2) for i in range(len(dimensions))}
        
        # Format the response
        result = {
            "Hasil Prediksi Kepribadian": [
                {"dimension": "EXT", "score": scores["EXT"], "description": "Extraversion: Seberapa ramah dan energik Anda."},
                {"dimension": "AGR", "score": scores["AGR"], "description": "Agreeableness: Seberapa ramah dan penuh kasih sayang Anda."},
                {"dimension": "CSN", "score": scores["CSN"], "description": "Conscientiousness: Seberapa terorganisir dan dapat diandalkan Anda."},
                {"dimension": "EST", "score": scores["EST"], "description": "Emotional Stability: Seberapa baik Anda mengelola stres dan emosi."},
                {"dimension": "OPN", "score": scores["OPN"], "description": "Openness: Seberapa kreatif dan terbuka terhadap ide-ide baru Anda."}
            ]
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
