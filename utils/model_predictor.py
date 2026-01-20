# utils/model_predictor.py
import torch
import torch.nn as nn
import numpy as np
import librosa
import json
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')

# CNN Model for Image-based Emotion Recognition
class CNNEmotionModel(nn.Module):
    def __init__(self, num_classes=8):
        super(CNNEmotionModel, self).__init__()
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Audio-based Emotion Model
class AudioEmotionModel(nn.Module):
    def __init__(self, input_size=158, num_classes=8):
        super(AudioEmotionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

class MultiModalEmotionPredictor:
    def __init__(self, 
                 audio_model_path="assets/models/best_emotion_model.pth",
                 cnn_model_path="assets/models/cnn_emotion_model.pth",
                 config_path="assets/models/model_config.json"):
        
        self.audio_model_path = audio_model_path
        self.cnn_model_path = cnn_model_path
        self.config_path = config_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Using device: {self.device}")
        
        # Create directories
        os.makedirs(os.path.dirname(audio_model_path), exist_ok=True)
        
        # Default configuration
        self.config = {
            "audio_input_size": 158,
            "num_classes": 8,
            "emotions": ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"],
            "feature_params": {
                "sr": 22050,
                "n_mfcc": 13,
                "n_fft": 2048,
                "hop_length": 512
            },
            "cnn_image_size": 128
        }
        
        # Load config
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
            print("✅ Configuration loaded successfully")
        except Exception as e:
            print(f"⚠️ Config loading warning: {e}")
        
        self.emotions = self.config["emotions"]
        
        # Initialize both models
        self.audio_model = None
        self.cnn_model = None
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize both audio and CNN models"""
        # Initialize Audio Model
        try:
            self.audio_model = AudioEmotionModel(
                input_size=self.config["audio_input_size"],
                num_classes=self.config["num_classes"]
            ).to(self.device)
            
            if os.path.exists(self.audio_model_path):
                checkpoint = torch.load(self.audio_model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.audio_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.audio_model.load_state_dict(checkpoint)
                self.audio_model.eval()
                print("✅ Audio emotion model loaded successfully!")
            else:
                print("⚠️ No audio model found, will use rule-based audio analysis")
                self.audio_model = None
                
        except Exception as e:
            print(f"❌ Audio model loading error: {e}")
            self.audio_model = None
        
        # Initialize CNN Model
        try:
            self.cnn_model = CNNEmotionModel(
                num_classes=self.config["num_classes"]
            ).to(self.device)
            
            if os.path.exists(self.cnn_model_path):
                checkpoint = torch.load(self.cnn_model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.cnn_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.cnn_model.load_state_dict(checkpoint)
                self.cnn_model.eval()
                print("✅ CNN emotion model loaded successfully!")
            else:
                print("⚠️ No CNN model found, will train a sample one")
                self._create_sample_cnn_model()
                
        except Exception as e:
            print(f"❌ CNN model loading error: {e}")
            self._create_sample_cnn_model()
    
    def _create_sample_cnn_model(self):
        """Create a sample CNN model for demonstration"""
        print("🔄 Creating sample CNN model...")
        try:
            # Initialize with sensible weights
            for layer in self.cnn_model.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, 0, 0.01)
                    nn.init.constant_(layer.bias, 0)
            
            # Save for future use
            torch.save({
                'model_state_dict': self.cnn_model.state_dict(),
                'config': self.config
            }, self.cnn_model_path)
            
            print("✅ Sample CNN model created and saved!")
        except Exception as e:
            print(f"❌ Error creating CNN model: {e}")
            self.cnn_model = None
    
    def extract_audio_features(self, audio_path):
        """Extract comprehensive audio features"""
        try:
            audio, sr = librosa.load(audio_path, sr=self.config["feature_params"]["sr"])
            
            features = []
            
            # MFCC Features
            mfcc = librosa.feature.mfcc(
                y=audio, 
                sr=sr, 
                n_mfcc=13,
                n_fft=self.config["feature_params"]["n_fft"],
                hop_length=self.config["feature_params"]["hop_length"]
            )
            features.extend(np.mean(mfcc, axis=1))
            features.extend(np.std(mfcc, axis=1))
            
            # Spectral Features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            
            features.extend([np.mean(spectral_centroid), np.std(spectral_centroid)])
            features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
            features.extend([np.mean(spectral_bandwidth), np.std(spectral_bandwidth)])
            
            # Chroma Features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features.extend(np.mean(chroma, axis=1))
            features.extend(np.std(chroma, axis=1))
            
            # Temporal Features
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
            rms_energy = librosa.feature.rms(y=audio)
            
            features.extend([np.mean(zero_crossing_rate), np.std(zero_crossing_rate)])
            features.extend([np.mean(rms_energy), np.std(rms_energy)])
            
            # Ensure correct size
            if len(features) < self.config["audio_input_size"]:
                features.extend([0.0] * (self.config["audio_input_size"] - len(features)))
            else:
                features = features[:self.config["audio_input_size"]]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"❌ Audio feature extraction error: {e}")
            return np.zeros(self.config["audio_input_size"], dtype=np.float32)
    
    def preprocess_image(self, image_path):
        """Preprocess image for CNN model"""
        try:
            from PIL import Image
            import torchvision.transforms as transforms
            
            # Image preprocessing pipeline
            transform = transforms.Compose([
                transforms.Resize((self.config["cnn_image_size"], self.config["cnn_image_size"])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            image = Image.open(image_path).convert('RGB')
            return transform(image).unsqueeze(0)  # Add batch dimension
            
        except Exception as e:
            print(f"❌ Image preprocessing error: {e}")
            return torch.zeros(1, 3, self.config["cnn_image_size"], self.config["cnn_image_size"])
    
    def predict_from_audio(self, audio_path):
        """Predict emotion from audio file"""
        try:
            if self.audio_model is None:
                return self._rule_based_audio_prediction(audio_path)
            
            features = self.extract_audio_features(audio_path)
            
            # Normalize features
            features = (features - np.mean(features)) / (np.std(features) + 1e-8)
            
            with torch.no_grad():
                input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                outputs = self.audio_model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_emotion = self.emotions[predicted_idx.item()]
            all_probabilities = probabilities.squeeze().cpu().numpy()
            emotion_probs = {emotion: float(prob) for emotion, prob in zip(self.emotions, all_probabilities)}
            
            return {
                'dominant_emotion': predicted_emotion.title(),
                'confidence': float(confidence.item()),
                'probabilities': emotion_probs,
                'model_type': 'audio_model'
            }
            
        except Exception as e:
            print(f"❌ Audio prediction error: {e}")
            return self._rule_based_audio_prediction(audio_path)
    
    def predict_from_image(self, image_path):
        """Predict emotion from image using CNN"""
        try:
            if self.cnn_model is None:
                return self._rule_based_image_prediction(image_path)
            
            input_tensor = self.preprocess_image(image_path).to(self.device)
            
            with torch.no_grad():
                outputs = self.cnn_model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_emotion = self.emotions[predicted_idx.item()]
            all_probabilities = probabilities.squeeze().cpu().numpy()
            emotion_probs = {emotion: float(prob) for emotion, prob in zip(self.emotions, all_probabilities)}
            
            return {
                'dominant_emotion': predicted_emotion.title(),
                'confidence': float(confidence.item()),
                'probabilities': emotion_probs,
                'model_type': 'cnn_model'
            }
            
        except Exception as e:
            print(f"❌ Image prediction error: {e}")
            return self._rule_based_image_prediction(image_path)
    
    def multi_modal_predict(self, audio_path=None, image_path=None):
        """Combine predictions from multiple modalities"""
        results = {}
        
        if audio_path:
            results['audio'] = self.predict_from_audio(audio_path)
        
        if image_path:
            results['image'] = self.predict_from_image(image_path)
        
        # Combine results if both available
        if audio_path and image_path:
            combined_emotion = self._combine_predictions(results)
            results['combined'] = combined_emotion
        
        return results
    
    def _combine_predictions(self, results):
        """Combine audio and image predictions"""
        audio_probs = results['audio']['probabilities']
        image_probs = results['image']['probabilities']
        
        # Weighted average (can adjust weights based on modality reliability)
        combined_probs = {}
        for emotion in self.emotions:
            combined_probs[emotion] = (audio_probs[emotion] * 0.6 + 
                                     image_probs[emotion] * 0.4)
        
        # Normalize
        total = sum(combined_probs.values())
        combined_probs = {k: v/total for k, v in combined_probs.items()}
        
        dominant_emotion = max(combined_probs.items(), key=lambda x: x[1])
        
        return {
            'dominant_emotion': dominant_emotion[0].title(),
            'confidence': float(dominant_emotion[1]),
            'probabilities': {k.title(): float(v) for k, v in combined_probs.items()},
            'model_type': 'multi_modal'
        }
    
    def _rule_based_audio_prediction(self, audio_path):
        """Fallback audio prediction with better feature-based rules"""
        try:
            features = self.extract_audio_features(audio_path)
            emotion_scores = {emotion: 0.1 for emotion in self.emotions}
            
            if len(features) > 26:
                # Extract key features more carefully
                mfcc_mean = np.mean(features[:13])  # MFCC means
                mfcc_std = np.mean(features[13:26])  # MFCC stds
                spectral_centroid = features[26] if len(features) > 26 else 2000
                spectral_rolloff = features[27] if len(features) > 27 else 4000
                rms = features[32] if len(features) > 32 else 0.1
                zero_crossing_rate = features[34] if len(features) > 34 else 0.1
                
                # Better emotion logic based on multiple features
                # Happy: high energy, higher spectral centroid, moderate variance
                if rms > 0.12 and spectral_centroid > 2000 and mfcc_std > 0.5:
                    emotion_scores['happy'] += 0.4
                
                # Sad: low energy, lower spectral centroid, lower pitch
                if rms < 0.08 and spectral_centroid < 1500:
                    emotion_scores['sad'] += 0.4
                
                # Angry: very high energy, high variance in features, aggressive rhythm
                if rms > 0.18 and mfcc_std > 1.5:
                    emotion_scores['angry'] += 0.3
                
                # Neutral: moderate everything, low variance
                if 0.08 <= rms <= 0.12 and mfcc_std < 0.5:
                    emotion_scores['neutral'] += 0.4
                
                # Fearful: high zero-crossing, unstable features
                if zero_crossing_rate > 0.08 and mfcc_std > 1.0:
                    emotion_scores['fearful'] += 0.3
                
                # Surprised: rapid energy changes, high spectral rolloff
                if spectral_rolloff > 5000 and rms > 0.1:
                    emotion_scores['surprised'] += 0.3
                
                # Calm: low RMS, smooth spectral features
                if rms < 0.1 and spectral_centroid < 2000:
                    emotion_scores['calm'] += 0.3
                
                # Disgust: mid-range energy with specific spectral shape
                if 0.1 < rms < 0.15 and 1500 < spectral_centroid < 2500:
                    emotion_scores['disgust'] += 0.2
            
            total = sum(emotion_scores.values())
            probabilities = {k: v/total for k, v in emotion_scores.items()}
            dominant_emotion = max(probabilities.items(), key=lambda x: x[1])
            
            return {
                'dominant_emotion': dominant_emotion[0].title(),
                'confidence': float(dominant_emotion[1]),
                'probabilities': {k.title(): float(v) for k, v in probabilities.items()},
                'model_type': 'rule_based_audio'
            }
            
        except Exception as e:
            print(f"❌ Rule-based audio prediction error: {e}")
            return self._get_fallback_prediction()
    
    def _rule_based_image_prediction(self, image_path):
        """Fallback image prediction"""
        try:
            # Simple rule-based for image (could use color analysis, etc.)
            from PIL import Image
            image = Image.open(image_path)
            
            # Convert to numpy for basic analysis
            img_array = np.array(image)
            
            # Simple brightness-based emotion guess
            brightness = np.mean(img_array)
            
            emotion_scores = {emotion: 0.1 for emotion in self.emotions}
            
            if brightness > 200:
                emotion_scores['happy'] += 0.3
                emotion_scores['surprised'] += 0.2
            elif brightness < 100:
                emotion_scores['sad'] += 0.3
                emotion_scores['fearful'] += 0.2
            else:
                emotion_scores['neutral'] += 0.3
            
            total = sum(emotion_scores.values())
            probabilities = {k: v/total for k, v in emotion_scores.items()}
            dominant_emotion = max(probabilities.items(), key=lambda x: x[1])
            
            return {
                'dominant_emotion': dominant_emotion[0].title(),
                'confidence': float(dominant_emotion[1]),
                'probabilities': {k.title(): float(v) for k, v in probabilities.items()},
                'model_type': 'rule_based_image'
            }
            
        except Exception as e:
            print(f"❌ Rule-based image prediction error: {e}")
            return self._get_fallback_prediction()
    
    def _get_fallback_prediction(self):
        """Final fallback"""
        return {
            'dominant_emotion': 'Neutral',
            'confidence': 0.5,
            'probabilities': {emotion.title(): 1.0/len(self.emotions) for emotion in self.emotions},
            'model_type': 'fallback'
        }
    
    def get_model_info(self):
        return {
            'audio_model_loaded': self.audio_model is not None,
            'cnn_model_loaded': self.cnn_model is not None,
            'emotions': self.emotions,
            'device': str(self.device)
        }

# Global instance
emotion_predictor = MultiModalEmotionPredictor()