# test_prediction.py
from utils.model_predictor import emotion_predictor
import librosa
import soundfile as sf
import numpy as np

def test_prediction():
    print("🧪 Testing Emotion Prediction System...")
    
    # Get model info
    model_info = emotion_predictor.get_model_info()
    print("📊 Model Information:")
    for key, value in model_info.items():
        print(f"   {key}: {value}")
    
    # Create a test audio file
    test_audio_path = "test_audio.wav"
    
    # Generate a sample audio (1 second of sine wave)
    sr = 22050
    t = np.linspace(0, 1, sr)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # A4 note
    audio += 0.2 * np.sin(2 * np.pi * 880 * t)  # A5 note
    
    # Save test audio
    sf.write(test_audio_path, audio, sr)
    print(f"🎵 Created test audio: {test_audio_path}")
    
    # Test prediction
    print("\n🎯 Making prediction...")
    result = emotion_predictor.predict_emotion(test_audio_path)
    
    print("\n📈 Prediction Results:")
    print(f"🎭 Dominant Emotion: {result['dominant_emotion']}")
    print(f"📊 Confidence: {result['confidence']:.3f}")
    print(f"⏱️ Duration: {result['audio_duration']:.2f}s")
    print(f"🔧 Model Used: {result['model_used']}")
    
    print("\n📋 All Probabilities:")
    for emotion, prob in result['probabilities'].items():
        print(f"   {emotion}: {prob:.3f}")
    
    # Clean up
    import os
    if os.path.exists(test_audio_path):
        os.remove(test_audio_path)

if __name__ == "__main__":
    test_prediction()