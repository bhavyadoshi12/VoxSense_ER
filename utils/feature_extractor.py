# utils/feature_extractor.py
import librosa
import numpy as np

class AudioFeatureExtractor:
    def __init__(self, sample_rate=22050, n_mfcc=13, n_mels=64, n_fft=2048, hop_length=512):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def extract_features(self, audio_path):
        """Extract comprehensive audio features"""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)

            # Ensure audio is not too short
            if len(audio) < self.n_fft:
                audio = np.pad(audio, (0, self.n_fft - len(audio)))

            features = {}

            # MFCC Features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc,
                                        n_fft=self.n_fft, hop_length=self.hop_length)
            features['mfcc_mean'] = np.mean(mfccs, axis=1)
            features['mfcc_std'] = np.std(mfccs, axis=1)

            # Mel Spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=self.n_mels,
                                                     n_fft=self.n_fft, hop_length=self.hop_length)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            features['mel_mean'] = np.mean(mel_spec_db, axis=1)
            features['mel_std'] = np.std(mel_spec_db, axis=1)

            # Chroma Features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=self.n_fft,
                                                hop_length=self.hop_length)
            features['chroma_mean'] = np.mean(chroma, axis=1)
            features['chroma_std'] = np.std(chroma, axis=1)

            # Spectral Features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr,
                                                                n_fft=self.n_fft, hop_length=self.hop_length)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr,
                                                                  n_fft=self.n_fft, hop_length=self.hop_length)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr,
                                                              n_fft=self.n_fft, hop_length=self.hop_length)

            features['spectral_centroid'] = np.mean(spectral_centroid)
            features['spectral_bandwidth'] = np.mean(spectral_bandwidth)
            features['spectral_rolloff'] = np.mean(spectral_rolloff)

            # Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(audio, frame_length=self.n_fft,
                                                   hop_length=self.hop_length)
            features['zcr'] = np.mean(zcr)

            # RMS Energy
            rms = librosa.feature.rms(y=audio, frame_length=self.n_fft, hop_length=self.hop_length)
            features['rms'] = np.mean(rms)

            # Combine all features into single vector
            feature_vector = []
            for key in ['mfcc_mean', 'mfcc_std', 'mel_mean', 'mel_std',
                       'chroma_mean', 'chroma_std', 'spectral_centroid',
                       'spectral_bandwidth', 'spectral_rolloff', 'zcr', 'rms']:
                if isinstance(features[key], np.ndarray):
                    feature_vector.extend(features[key])
                else:
                    feature_vector.append(features[key])

            return np.array(feature_vector)

        except Exception as e:
            print(f"❌ Error processing audio: {e}")
            return np.zeros(158)  # Return zeros with correct dimension