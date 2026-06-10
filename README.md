# Music Genre Classifier
An end-to-end machine learning pipeline that classifies music genres from audio files. The project compares classical ML approaches against a deep learning model trained on spectrogram representations, served via a REST API with prediction history logging.
Tech Stack

ML & Data — Python, PyTorch, scikit-learn, librosa, pandas
API — FastAPI, SQLAlchemy, SQLite
Testing — pytest
DevOps — Docker, GitHub Actions, Azure (in progress)

Project Structure
music-genre-classifier/
├── scripts/
│   ├── dataset.py        # PyTorch dataset class
│   ├── train_cnn.py      # CNN training pipeline
│   └── predictor.py      # Model inference
├── api/
│   ├── main.py           # FastAPI application
│   └── database.py       # SQLAlchemy models and logging
├── notebooks/            # Exploration and visualization
├── models/               # Saved model weights
├── tests/                # pytest test suite
└── data/                 # GTZAN dataset (not included)
Dataset
This project uses the GTZAN Dataset — 1000 audio clips across 10 genres (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock).
Download and place it under data/genres_original/.
Results
ModelTest AccuracySVM0.73Random Forest0.77CNN (PyTorch)0.62
The CNN underperformed classical ML due to the small dataset size (999 samples) — a deliberate finding that demonstrates when deep learning is and isn't the right tool.
Getting Started
Clone the repository:
bashgit clone https://github.com/mohamad/music-genre-classifier.git
cd music-genre-classifier
Create a virtual environment and install dependencies:
bashpython3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
Train the model:
bashpython scripts/train_cnn.py
Start the API:
bashuvicorn api.main:app --reload
API Endpoints
Health check:
bashcurl http://localhost:8000
Predict genre from audio file:
bashcurl -X POST "http://localhost:8000/predict" \
  -F "file=@your_audio_file.wav"
Response:
json{
  "genre": "blues",
  "confidence": 92.9
}
Prediction history:
bashcurl http://localhost:8000/history
Supported Audio Formats
.wav .mp3 .m4a .flac .ogg
Running Tests
bashpytest tests/ -v
Notes

AMD GPU (ROCm) was attempted during development — PyTorch pip builds do not currently support gfx1012 architecture. CPU training is used instead.
Training takes approximately 5-10 minutes on CPU for 30 epochs.
