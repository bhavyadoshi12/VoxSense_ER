# test_predictor.py
from enhanced_predictor import EnhancedEmotionPredictor
import glob

def main():
    # Initialize predictor
    predictor = EnhancedEmotionPredictor()
    
    # Get model info
    model_info = predictor.get_model_info()
    print("Model Information:", model_info)
    
    # Test with multiple audio files
    audio_files = glob.glob("test_audio/*.wav") + glob.glob("test_audio/*.mp3")
    
    if not audio_files:
        print("No audio files found in test_audio/ folder")
        return
    
    print(f"Found {len(audio_files)} audio files for testing...")
    
    for audio_file in audio_files:
        print(f"\n{'='*50}")
        print(f"Analyzing: {audio_file}")
        
        result = predictor.predict_emotion(audio_file)
        
        print(f"🎭 Emotion: {result['dominant_emotion']}")
        print(f"📊 Confidence: {result['confidence']:.2%}")
        print(f"⏱ Duration: {result['audio_duration']:.2f}s")
        
        # Show top 3 emotions
        top_emotions = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]
        print("🏆 Top 3 Emotions:")
        for emotion, prob in top_emotions:
            print(f"  {emotion}: {prob:.2%}")

if __name__ == "__main__":
    main()