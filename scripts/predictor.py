import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_cnn import GenreCNN 
import librosa
import numpy as np
import torch.nn.functional as F

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
#Load Model and Device
device = torch.device('cpu')
model = GenreCNN(num_classes=10)
model.load_state_dict(torch.load('models/cnn_model.pth', map_location=device, weights_only=True))
model.eval()  # Set the model to evaluation mode  

#Prediction Function
def predict_genre(file_path):
    waveform, _ = librosa.load(file_path, sr=22050, mono=True)  # Load the audio file
    mel_spec = librosa.feature.melspectrogram(y=waveform, sr=22050, 
                                              n_mels=128, 
                                              n_fft=2048)  # Compute the Mel spectrogram
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_tensor = torch.tensor(mel_spec_db).unsqueeze(0).float()
    # Resize to (128, 128)
    mel_spec_tensor = F.interpolate(mel_spec_tensor.unsqueeze(0).float(), size=(128, 128), 
                                    mode='bilinear', 
                                    align_corners=False).squeeze(0).unsqueeze(0)
    
    
    with torch.no_grad():
        output = model(mel_spec_tensor)  # Get the model's output
        probabilities = F.softmax(output, dim=1)  # Convert output to probabilities
        confidence, predicted_idx = torch.max(probabilities, dim=1)  # Get the predicted genre index and confidence
        genre = genres[predicted_idx.item()]

    return { "genre": genre, "confidence": round(confidence.item()*100, 2)}



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predictor.py <file_path>")
        sys.exit(1)
    file_path = sys.argv[1]
    predicted_genre = predict_genre(file_path)
    print(f"Predicted genre index: {predicted_genre}")

