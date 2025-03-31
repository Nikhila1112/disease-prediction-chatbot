from flask import Flask, render_template
import gdown
import os
import pickle

app = Flask(__name__)

# Model file settings
model_path = "model.pkl"
file_id = "https://drive.google.com/file/d/1m5qXO2tTNCNbT4kbsfA1NGtO4b-8NJye/view?usp=sharing"  # Replace with your actual file ID

# Download the model if it doesn't exist
if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)

# Load model
print("Loading ML model...")
model = pickle.load(open(model_path, 'rb'))
print("Model loaded successfully.")

@app.route("/")  # Home Page
def home():
    return render_template("index.html")

@app.route("/chat")  # Chat Page Route
def chat_page():
    return render_template("chatpage.html")  # This file must be inside "templates/"

if __name__ == "__main__":
    app.run(debug=True)
