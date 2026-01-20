import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import tempfile
import os
from io import BytesIO
import base64

class AudioProcessor:
    def __init__(self, target_sr=22050):
        self.target_sr = target_sr
        self.supported_formats = ['.wav', '.mp3', '.m4a', '.ogg']
    
    def load_audio(self, audio_path):
        """Load audio file and convert to mono"""
        try:
            y, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
            return y, sr
        except Exception as e:
            raise Exception(f"Error loading audio: {str(e)}")
    
    def preprocess_audio(self, audio_data, sr):
        """Preprocess audio: normalize, remove noise, trim silence"""
        # Normalize audio
        audio_normalized = librosa.util.normalize(audio_data)
        
        # Trim silence
        audio_trimmed, _ = librosa.effects.trim(audio_normalized, top_db=20)
        
        # Apply noise reduction (simple high-pass filter)
        if len(audio_trimmed) > 100:  # Ensure we have enough samples
            sos = signal.butter(10, 100, 'hp', fs=sr, output='sos')
            audio_clean = signal.sosfilt(sos, audio_trimmed)
        else:
            audio_clean = audio_trimmed
        
        return audio_clean
    
    def extract_features(self, audio_data, sr):
        """Extract audio features for emotion recognition"""
        features = {}
        
        try:
            # MFCC features
            mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            # Mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)
            
            # RMS energy
            rms = librosa.feature.rms(y=audio_data)
            
            features.update({
                'mfcc_mean': mfcc_mean,
                'mfcc_std': mfcc_std,
                'mel_spectrogram': mel_spec_db,
                'spectral_centroid': np.mean(spectral_centroid),
                'spectral_rolloff': np.mean(spectral_rolloff),
                'zcr': np.mean(zcr),
                'rms': np.mean(rms),
                'duration': len(audio_data) / sr
            })
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            # Return basic features
            features.update({
                'rms': np.sqrt(np.mean(audio_data**2)),
                'zcr': np.mean(librosa.feature.zero_crossing_rate(audio_data)),
                'duration': len(audio_data) / sr
            })
        
        return features
    
    def create_spectrogram(self, audio_data, sr=22050):
        """Create mel spectrogram visualization"""
        try:
            plt.figure(figsize=(10, 4))
            mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel Spectrogram')
            plt.tight_layout()
            
            # Convert plot to image
            buf = BytesIO()
            plt.savefig(buf, format='png', transparent=True, dpi=100)
            buf.seek(0)
            plt.close()
            
            return buf
        except Exception as e:
            print(f"Spectrogram error: {e}")
            # Return empty image
            buf = BytesIO()
            plt.figure(figsize=(10, 4))
            plt.text(0.5, 0.5, 'Spectrogram\nNot Available', ha='center', va='center')
            plt.axis('off')
            plt.savefig(buf, format='png', transparent=True)
            buf.seek(0)
            plt.close()
            return buf
    
    def get_audio_duration(self, audio_data, sr):
        """Calculate audio duration"""
        return len(audio_data) / sr
    
    def validate_audio(self, audio_data, sr, min_duration=1.0, max_duration=10.0):
        """Validate audio for processing"""
        duration = self.get_audio_duration(audio_data, sr)
        
        if duration < min_duration:
            raise Exception(f"Audio too short: {duration:.2f}s. Minimum: {min_duration}s")
        
        if duration > max_duration:
            raise Exception(f"Audio too long: {duration:.2f}s. Maximum: {max_duration}s")
        
        return True