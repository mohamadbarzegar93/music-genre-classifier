import torch
import torch.utils.data as data
import torchaudio
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
        # Initialize Mel spectrogram transformer
        self.mel_transformer = torchaudio.transforms.MelSpectrogram(
        sample_rate=self.sample_rate,
        n_mels=128,
        n_fft=2048
        )
        # Initialize amplitude to decibel transformer    
        self.db_transformer = torchaudio.transforms.AmplitudeToDB()
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
        #Load the audio file
        file_path = self.file_paths[idx]
        waveform, sample_rate = torchaudio.load(file_path)
        # If the audio has more than one channel, convert it to mono by averaging the channels
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        # Resample the audio if the sample rate is different from the expected sample rate
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        # Convert the audio waveform to a Mel spectrogram
        mel_spec = self.mel_transformer(waveform)
        # Convert the Mel spectrogram to decibel scale
        mel_spec_db = self.db_transformer(mel_spec)
        #Resize the Mel spectrogram to a fixed size (e.g., 128x128)
        mel_spec_db = torch.nn.functional.interpolate(mel_spec_db.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False).squeeze(0)
        # Apply any additional transformations if provided
        if self.transform:
            mel_spec_db = self.transform(mel_spec_db)
        label = self.labels[idx]
        return mel_spec_db, label

if __name__ == "__main__":
    # Example usage
    dataset = GTZANDataset(root_dir='data/genres_original')
    print(f"Total files: {len(dataset)}")
    print(f"Genres: {dataset.genres}")
    sample, label = dataset[0]
    print(f"Sample shape: {sample.shape}, Label: {label}")
    






    
