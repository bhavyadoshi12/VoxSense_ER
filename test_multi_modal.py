# test_multi_modal.py
from utils.model_predictor import emotion_predictor
import soundfile as sf
import numpy as np
from PIL import Image
import os

def test_multi_modal():
    print("🧪 Testing Multi-Modal Emotion Prediction...")
    
    # Get model info
    model_info = emotion_predictor.get_model_info()
    print("📊 Model Information:")
    for key, value in model_info.items():
        print(f"   {key}: {value}")
    
    # Create test files
    test_audio = "test_audio.wav"
    test_image = "test_image.jpg"
    
    # Create test audio
    sr = 22050
    t = np.linspace(0, 2, sr * 2)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # A4 note
    sf.write(test_audio, audio, sr)
    
    # Create test image (solid color)
    img = Image.new('RGB', (128, 128), color='red')
    img.save(test_image)
    
    print(f"🎵 Created test files: {test_audio}, {test_image}")
    
    # Test individual predictions
    print("\n🎯 Testing Audio Prediction...")
    audio_result = emotion_predictor.predict_from_audio(test_audio)
    print(f"   Emotion: {audio_result['dominant_emotion']}")
    print(f"   Confidence: {audio_result['confidence']:.3f}")
    print(f"   Model: {audio_result['model_type']}")
    
    print("\n🎯 Testing Image Prediction...")
    image_result = emotion_predictor.predict_from_image(test_image)
    print(f"   Emotion: {image_result['dominant_emotion']}")
    print(f"   Confidence: {image_result['confidence']:.3f}")
    print(f"   Model: {image_result['model_type']}")
    
    print("\n🎯 Testing Multi-Modal Prediction...")
    multi_result = emotion_predictor.multi_modal_predict(
        audio_path=test_audio, 
        image_path=test_image
    )
    
    print("📈 Combined Results:")
    for modality, result in multi_result.items():
        print(f"   {modality.upper()}: {result['dominant_emotion']} "
              f"(Confidence: {result['confidence']:.3f})")
    
    # Clean up
    for file in [test_audio, test_image]:
        if os.path.exists(file):
            os.remove(file)

if __name__ == "__main__":
    test_multi_modal()