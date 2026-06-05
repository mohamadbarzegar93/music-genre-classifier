import torch
import torch.utils.data as data
"""import torchaudio"""
import librosa
import numpy as np
import os
import pathlib

# Dataset class for GTZAN music genre classification
class GTZANDataset(data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = pathlib.Path(root_dir)
        self.transform = transform
        self.file_paths = []
        self.labels = []
        # Get sorted list of genre directories
        self.genres = sorted([g.name for g in self.root_dir.iterdir() if g.is_dir()])
        # GTZAN dataset has a sample rate of 22050 Hz
        self.sample_rate = 22050

        # Iterate through each genre directory and collect file paths and labels
        for idx, genre in enumerate(self.genres):
            genre_dir = self.root_dir / genre
            for file_name in os.listdir(genre_dir):
                # Exclude the corrupted file 'jazz.00054.wav' from the dataset
                if file_name.endswith('.wav') and file_name != 'jazz.00054.wav':
                    self.file_paths.append(genre_dir / file_name)
                    self.labels.append(idx)  # Use the index of the genre as the label
    def __len__(self):        return len(self.file_paths)
    def __getitem__(self, idx):
        
        file_path = self.file_paths[idx]
        waveform, _ = librosa.load(file_path, sr=self.sample_rate, mono=True)
        mel_spec = librosa.feature.melspectrogram(y=waveform, sr=self.sample_rate, n_mels=128, n_fft=2048)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_tensor = torch.tensor(mel_spec_db).unsqueeze(0).float()
        mel_spec_tensor = torch.nn.functional.interpolate(
            mel_spec_tensor.unsqueeze(0),
            size=(128, 128),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        if self.transform:
            mel_spec_tensor = self.transform(mel_spec_tensor)
        label = self.labels[idx]
        return mel_spec_tensor, label

if __name__ == "__main__":
    # Example usage
    dataset = GTZANDataset(root_dir='data/genres_original')
    print(f"Total files: {len(dataset)}")
    print(f"Genres: {dataset.genres}")
    sample, label = dataset[0]
    print(f"Sample shape: {sample.shape}, Label: {label}")
    






    
