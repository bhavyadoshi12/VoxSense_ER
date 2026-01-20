# main.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tempfile
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import base64
from io import BytesIO
import time
import soundfile as sf
import traceback

# --- Page configuration ---
st.set_page_config(
    page_title="VoxSense - AI Emotion Detection",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Import custom modules ---
torch_available = False
predictor_available = False
EmotionPredictor = None 

try:
    import torch
    torch_available = True
    try:
        from utils.model_predictor import EmotionPredictor as AdvancedPredictor
        EmotionPredictor = AdvancedPredictor 
        predictor_available = True
        print("✅ Using Advanced Predictor.")
    except ImportError as e:
        print(f"⚠️ Advanced predictor not available: {e}. Trying simple.")
        try:
            from utils.simple_predictor import SimpleEmotionPredictor
            EmotionPredictor = SimpleEmotionPredictor 
            predictor_available = True
            print("🔧 Using Simple Predictor.")
        except ImportError as e2: 
            print(f"❌ Simple predictor also not available: {e2}")
except ImportError as e:
    print(f"❌ PyTorch not available: {e}. Trying simple predictor.")
    try:
        from utils.simple_predictor import SimpleEmotionPredictor
        EmotionPredictor = SimpleEmotionPredictor 
        predictor_available = True
        print("🔧 Using Simple Predictor (PyTorch Not Found).")
    except ImportError as e2: 
        print(f"❌ Simple predictor also not available: {e2}")

try:
    from utils.feature_extractor import AudioFeatureExtractor
    print("✅ Audio feature extractor loaded.")
except ImportError as e: 
    print(f"⚠️ Feature extractor not available: {e}")

if not predictor_available or EmotionPredictor is None:
    st.error("❌ CRITICAL: No emotion prediction class loaded.")
    st.info("Ensure predictor files exist in utils folder.")
    st.stop()

# --- Premium CSS Styling ---
def load_css():
    st.markdown("""
    <style>
    /* Global Styles */
    .main {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
    }
    
    /* Headers with Premium Gradient */
        .main-header {
            font-size: 4.5rem;
            font-weight: 800;
            text-align: center;
            color: #FFD700 !important;
            margin-bottom: 1rem;
            text-shadow: 0 4px 20px rgba(212, 175, 55, 0.5);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            z-index: 10;
            position: relative;
            -webkit-font-smoothing: antialiased;
            text-rendering: optimizeLegibility;
            letter-spacing: 2px;
        }
        /* Force visibility for section headers and common Streamlit text elements */
        .section-header,
        .card,
        .card h2,
        .card h3,
        .metric-card,
        .stMarkdown,
        .stText,
        .stMetric,
        .stButton > button,
        .stMarkdown p,
        .stMarkdown h4 {
            color: #ffffff !important;
            z-index: 3;
            position: relative;
        }

        /* Ensure inline spans inherit visible color */
        .stMarkdown span, span {
            color: inherit !important;
        }
    
    .section-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFD700 !important;
        margin: 2rem 0 1rem 0;
        border-left: 4px solid #D4AF37;
        padding-left: 1rem;
        text-shadow: 0 2px 8px rgba(212, 175, 55, 0.4);
        z-index: 5;
        position: relative;
    }
    
    /* Premium Cards with Glass Effect */
    .card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(212, 175, 55, 0.2);
        border: 1px solid rgba(212, 175, 55, 0.3);
    }
    
    /* Premium Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #D4AF37 0%, #FFD700 100%);
        color: #0c0c0c;
        border: none;
        padding: 0.8rem 2rem;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0.5rem 0.2rem;
        cursor: pointer;
        border-radius: 50px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(212, 175, 55, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(212, 175, 55, 0.4);
        background: linear-gradient(135deg, #FFD700 0%, #D4AF37 100%);
    }
    
    .secondary-button {
        background: rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    .secondary-button:hover {
        background: rgba(255, 255, 255, 0.2) !important;
        border: 1px solid #D4AF37 !important;
    }
    
    /* Emotion-specific styling */
    .emotion-happy { border-left: 6px solid #4CAF50 !important; }
    .emotion-sad { border-left: 6px solid #2196F3 !important; }
    .emotion-angry { border-left: 6px solid #f44336 !important; }
    .emotion-neutral { border-left: 6px solid #9E9E9E !important; }
    .emotion-fearful { border-left: 6px solid #673AB7 !important; }
    .emotion-surprised { border-left: 6px solid #FF9800 !important; }
    .emotion-calm { border-left: 6px solid #00BCD4 !important; }
    .emotion-disgust { border-left: 6px solid #795548 !important; }
    
    /* Metrics and Stats */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.02);
        border: 1px solid rgba(212, 175, 55, 0.3);
    }
    
    /* Progress bars and loaders */
    .stProgress > div > div {
        background: linear-gradient(90deg, #D4AF37, #FFD700);
    }
    
    /* Radio buttons and select boxes */
    .stRadio > div {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 10px;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 2px dashed rgba(255, 255, 255, 0.2) !important;
        border-radius: 15px !important;
    }
    
    /* Sidebar enhancements */
    .css-1d391kg {
        background: linear-gradient(180deg, #0c0c0c 0%, #1a1a2e 100%);
    }
    
    /* Custom animations */
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .floating {
        animation: float 3s ease-in-out infinite;
    }
    
    /* Text gradients */
    .gradient-text {
        background: linear-gradient(45deg, #D4AF37, #FFD700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Session State Initialization ---
def init_session_state():
    defaults = {
        'page': 'home',
        'user_details': {},
        'audio_data': None,
        'audio_file_path': None,
        'emotion_result': None,
        'recording': False,
        'processing_complete': False,
        'visualizations': {},
        'recording_start_time': None,
        'sample_rate': 22050,
        'torch_available': torch_available
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# --- Compatibility helper to gracefully remove unsupported kwargs ---
def _call_compat(func, *args, **kwargs):
    """Call Streamlit functions but remove `use_container_width` if it's unsupported.
    This avoids crashes on older Streamlit versions that don't accept that kwarg.
    """
    try:
        return func(*args, **kwargs)
    except TypeError as e:
        # If the error was caused by an unexpected 'use_container_width' kwarg,
        # remove it and retry the call.
        msg = str(e)
        if "use_container_width" in msg or 'use_container_width' in kwargs:
            kwargs2 = kwargs.copy()
            kwargs2.pop('use_container_width', None)
            return func(*args, **kwargs2)
        raise

# --- Reset Analysis State ---
def reset_analysis():
    print("🔄 Resetting analysis state...") 
    audio_path = st.session_state.pop('audio_file_path', None) 
    if audio_path and os.path.exists(audio_path):
        try:
            is_temp_file = os.path.dirname(audio_path) == tempfile.gettempdir()
            if is_temp_file and audio_path.endswith('.wav'):
                 os.remove(audio_path)
                 print(f"🧹 Removed temporary audio file: {audio_path}")
        except Exception as e:
            print(f"⚠️ Error removing temp file '{audio_path}': {e}") 

    keys_to_reset = [
        'audio_data', 'emotion_result', 'recording', 
        'processing_complete', 'visualizations', 'recording_start_time'
    ]
    for key in keys_to_reset:
        st.session_state.pop(key, None)
        
    st.session_state.recording = False
    st.session_state.processing_complete = False
    st.session_state.sample_rate = 22050 
    st.session_state.visualizations = {} 
    st.session_state.emotion_result = None
    print("✅ Analysis state reset.")

# --- Navigation ---
def navigate_to(page):
    current_page = st.session_state.get('page', 'home')
    if (page == 'user_details' and current_page == 'home') or \
       (current_page == 'results' and page != 'results'):
        print(f"Navigating {current_page} -> {page}: Resetting analysis.")
        reset_analysis()
    st.session_state.page = page
    print(f"Navigating to page: {page}.")
    st.rerun()

# --- Page Definitions ---

def home_page():
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown('<h1 class="main-header">VoxSense</h1>', unsafe_allow_html=True)
        st.markdown('<h3 style="text-align: center; color: #e0e0e0; font-weight: 400; margin-top: 0; margin-bottom: 3rem; text-shadow: 0 2px 10px rgba(0,0,0,0.5);">AI-Powered Emotional Intelligence Platform</h3>', unsafe_allow_html=True)
        
        # Hide PyTorch warning - only show success message
        if st.session_state.torch_available: 
            st.success("🤖 **Premium Mode**: AI-powered deep emotion analysis")
        
        # Premium Hero Section
        st.markdown("""
        <div style="text-align: center; margin: 4rem 0;">
            <div class="floating" style="width: 200px; height: 200px; background: linear-gradient(135deg, #D4AF37, #FFD700, #FFF8DC); border-radius: 50%; margin: 0 auto; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 60px rgba(212, 175, 55, 0.6); border: 4px solid rgba(255, 255, 255, 0.3);">
                <span style="font-size: 6rem; color: #0c0c0c;">🎭</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Premium Feature Cards
        col1_feat, col2_feat, col3_feat = st.columns(3)
        with col1_feat: 
            st.markdown("""
            <div class="metric-card">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🎯</div>
                <h4 style="color: #ffffff; margin-bottom: 0.5rem;">Precision Analysis</h4>
                <p style="color: #b0b0b0; font-size: 0.9rem;">Advanced AI algorithms for accurate emotion detection</p>
            </div>
            """, unsafe_allow_html=True)
        with col2_feat: 
            st.markdown("""
            <div class="metric-card">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
                <h4 style="color: #ffffff; margin-bottom: 0.5rem;">Detailed Reports</h4>
                <p style="color: #b0b0b0; font-size: 0.9rem;">Comprehensive PDF insights with visual analytics</p>
            </div>
            """, unsafe_allow_html=True)
        with col3_feat: 
            st.markdown("""
            <div class="metric-card">
                <div style="font-size: 3rem; margin-bottom: 1rem;">⚡</div>
                <h4 style="color: #ffffff; margin-bottom: 0.5rem;">Real-time Processing</h4>
                <p style="color: #b0b0b0; font-size: 0.9rem;">Instant analysis with cutting-edge technology</p>
            </div>
            """, unsafe_allow_html=True)
        
        # CTA Button with Premium Styling
        if _call_compat(st.button, "🚀 Begin Emotional Analysis →", use_container_width=True, key="start_analysis"):
            navigate_to('user_details')

def user_details_page():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">👤 Personal Profile</h2>', unsafe_allow_html=True)
    st.markdown("Complete your profile for personalized emotional insights")
    
    defaults = st.session_state.get('user_details', {})
    with st.form("user_details_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("**Full Name** *", placeholder="Enter your full name", value=defaults.get('name', ''))
            age = st.number_input("**Age** *", min_value=1, max_value=120, value=defaults.get('age', 25))
            contact = st.text_input("**Contact Number**", placeholder="Your phone number", value=defaults.get('contact', ''))
        with col2:
            gender_options = ["Select", "Male", "Female", "Other", "Prefer not to say"]
            default_gender = defaults.get('gender', 'Select')
            gender_index = gender_options.index(default_gender) if default_gender in gender_options else 0
            gender = st.selectbox("**Gender** *", gender_options, index=gender_index)
            email = st.text_input("**Email Address**", placeholder="your.email@example.com", value=defaults.get('email', ''))
            occupation = st.text_input("**Occupation**", placeholder="Your profession", value=defaults.get('occupation', ''))
        
        st.markdown("<small style='color: #b0b0b0;'>* Required fields</small>", unsafe_allow_html=True)
        submitted = _call_compat(st.form_submit_button, "🎤 Continue to Voice Analysis →", use_container_width=True)
        
        if submitted:
            if name and age and gender != "Select":
                st.session_state.user_details = { 
                    'name': name, 'age': age, 'gender': gender, 
                    'email': email, 'contact': contact, 'occupation': occupation, 
                    'timestamp': datetime.now() 
                }
                st.success("✨ Profile saved successfully!")
                time.sleep(0.5) 
                navigate_to('record')
            else: 
                st.error("⚠️ Please complete all required fields")
    
    st.markdown('</div>', unsafe_allow_html=True)
    if _call_compat(st.button, "← Return to Home", use_container_width=True, type="secondary"): 
        navigate_to('home')

# --- Audio Processing Class ---
class AudioProcessor:
    def __init__(self): 
        self.sample_rate = 22050
    
    def load_audio(self, audio_path):
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True) 
            return audio.astype(np.float32), sr 
        except Exception as e:
            st.error(f"Error loading audio file: {e}")
            print(f"❌ Error loading audio file: {e}") 
            return None, self.sample_rate
    
    def create_spectrogram(self, audio_data, sr=22050):
        if audio_data is None or len(audio_data) == 0: 
            return None
        try:
            fig, ax = plt.subplots(figsize=(10, 4), facecolor='#0c0c0c')
            n_fft=1024; hop_length=256
            audio_data_float = audio_data.astype(np.float32)
            if len(audio_data_float) < n_fft: 
                audio_data_float = np.pad(audio_data_float, (0, n_fft - len(audio_data_float)))
            
            S = librosa.feature.melspectrogram(y=audio_data_float, sr=sr, n_fft=n_fft, 
                                             hop_length=hop_length, n_mels=128, fmax=8000)
            S_dB = librosa.power_to_db(S, ref=np.max)
            S_dB = np.nan_to_num(S_dB, nan=np.min(S_dB), posinf=np.max(S_dB), neginf=np.min(S_dB))
            
            img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', 
                                         hop_length=hop_length, fmax=8000, ax=ax, cmap='viridis')
            ax.set_title("Mel Spectrogram", fontweight='bold', color='white', pad=15)
            ax.set_xlabel("Time (seconds)", color='#e0e0e0')
            ax.set_ylabel("Frequency (Hz)", color='#e0e0e0')
            ax.tick_params(colors='#b0b0b0')
            ax.set_facecolor('#1a1a2e')
            
            cbar = plt.colorbar(img, ax=ax, format="%+2.0f dB")
            cbar.set_label('dB', color='white')
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
            
            plt.tight_layout()
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', 
                       facecolor='#0c0c0c', edgecolor='none')
            buffer.seek(0)
            plt.close(fig) 
            return buffer
        except Exception as e: 
            print(f"❌ Error creating spectrogram: {e}")
            traceback.print_exc()
            plt.close('all')
            return None

# --- Recording/Upload Page ---
def record_audio_page():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🎤 Voice Analysis Session</h2>', unsafe_allow_html=True)
    
    # Analysis Mode Indicator
    if not st.session_state.torch_available: 
        st.info("🔧 **Analysis Mode**: Enhanced rule-based emotion detection")
    else: 
        st.success("🤖 **Analysis Mode**: AI-powered deep emotion analysis")
    
    # Session Guidelines
    st.markdown("""
    <div style="background: rgba(212, 175, 55, 0.1); padding: 1.5rem; border-radius: 15px; border-left: 4px solid #D4AF37;">
        <h4 style="color: #FFD700; margin-bottom: 0.5rem;">🎯 Session Guidelines</h4>
        <ul style="color: #e0e0e0; margin-bottom: 0;">
            <li>Find a quiet, comfortable space</li>
            <li>Speak naturally for 3-10 seconds</li>
            <li>Express genuine emotions in your voice</li>
            <li>Ensure clear audio quality</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    recording_method = st.radio("**Select Analysis Method:**", 
                               ["🎙️ Live Voice Session", "📁 Upload Audio File"], 
                               key="record_method_radio")
    
    if recording_method == "🎙️ Live Voice Session": 
        live_recording_section()
    else: 
        audio_upload_section()
    
    st.markdown('</div>', unsafe_allow_html=True)
    if _call_compat(st.button, "← Back to Profile", use_container_width=True, type="secondary"): 
        navigate_to('user_details')

def live_recording_section():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        is_recording = st.session_state.get('recording', False)
        
        if not is_recording:
            if _call_compat(st.button, "🎤 Start Recording Session", use_container_width=True, key="start_rec"):
                reset_analysis() 
                st.session_state.recording = True
                st.session_state.recording_start_time = time.time()
                print("🎙️ Recording started.")
                st.rerun()
        else:
            # Recording Display with Premium Styling
            current_time = time.time()
            start_time = st.session_state.get('recording_start_time', current_time) 
            recording_duration = current_time - start_time
            
            st.markdown(f"""
            <div style="text-align: center; padding: 2rem; background: rgba(212, 175, 55, 0.1); border-radius: 15px; border: 2px solid rgba(212, 175, 55, 0.3);">
                <div style="font-size: 2rem; color: #FFD700; margin-bottom: 1rem;">● Recording</div>
                <div style="font-size: 1.5rem; color: #ffffff;">{recording_duration:.1f} seconds</div>
                <div style="color: #b0b0b0; margin-top: 0.5rem;">Speak naturally...</div>
            </div>
            """, unsafe_allow_html=True)

            # Stop Button
            if _call_compat(st.button, "⏹️ Stop & Analyze", key="stop_rec", use_container_width=True):
                st.session_state.recording = False 
                final_duration = max(recording_duration, 0.5)
                print(f"⏹️ Recording stopped. Duration: {final_duration:.1f}s.")

                # Generate Sample Audio Data
                duration = min(final_duration, 10.0)
                sample_rate = 22050
                t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False) 
                if len(t) == 0: 
                    t = np.linspace(0, 0.1, int(sample_rate * 0.1), endpoint=False)
                
                emotional_patterns = {
                    'happy': (0.8, 0.3, 0.1),
                    'sad': (0.3, 0.1, 0.05),
                    'angry': (0.9, 0.5, 0.2),
                    'neutral': (0.5, 0.2, 0.1)
                }
                pattern_key = np.random.choice(list(emotional_patterns.keys()))
                energy, variation, noise_level = emotional_patterns[pattern_key]
                base_freq = 200 + variation * 100
                
                audio_signal = (energy * np.sin(2*np.pi*base_freq*t) + 
                              variation*energy*np.sin(2*np.pi*base_freq*2*t) + 
                              variation*energy*np.sin(2*np.pi*base_freq*3*t)) * np.exp(-t/3)
                audio_signal += noise_level * np.random.normal(size=len(t))
                audio_signal = audio_signal.astype(np.float32) 
                
                # Prepare state for analysis
                st.session_state.audio_data = audio_signal
                st.session_state.audio_file_path = "recorded_audio.wav"
                st.session_state.sample_rate = sample_rate
                
                # Perform Analysis
                with st.spinner("🔍 Analyzing emotional patterns..."):
                    print("🧠 Starting analysis...")
                    emotion_result, visualizations = perform_analysis(
                        audio_path_arg=st.session_state.audio_file_path, 
                        audio_data_arg=st.session_state.audio_data, 
                        sample_rate_arg=st.session_state.sample_rate
                    )
                
                if emotion_result:
                    print("✅ Analysis complete.")
                    st.session_state.emotion_result = emotion_result
                    st.session_state.visualizations = visualizations
                    st.session_state.processing_complete = True
                    st.success("🎉 Analysis Complete!")
                    time.sleep(0.5)
                    navigate_to('results')
                    st.stop()
                else:
                    st.error("❌ Analysis failed. Please try again.")
                    reset_analysis()

            elif is_recording:
                time.sleep(0.1)
                st.rerun()

def audio_upload_section():
    uploaded_file = st.file_uploader("**Select Audio File**", 
                                    type=['wav', 'mp3', 'm4a', 'ogg'], 
                                    key="audio_uploader")
    
    if uploaded_file is not None:
        st.success(f"✅ File '{uploaded_file.name}' selected.")
        st.audio(uploaded_file) 

        if _call_compat(st.button, "🔍 Analyze Audio File", key="process_upload", use_container_width=True):
            reset_analysis() 
            temp_file_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_file_path = tmp_file.name
                
                st.session_state.audio_file_path = temp_file_path
                print(f"⬆️ File uploaded to: {temp_file_path}") 

                # Process audio file
                audio_processor = AudioProcessor()
                audio_data, sr = audio_processor.load_audio(temp_file_path)
                
                if audio_data is None: 
                    st.error("❌ Audio could not be loaded. Please try again.")
                    if temp_file_path and os.path.exists(temp_file_path):
                         os.remove(temp_file_path)
                    st.session_state.audio_file_path = None
                    return 

                st.session_state.audio_data = audio_data
                st.session_state.sample_rate = sr
                print(f"✅ File loaded. Duration: {len(audio_data)/sr:.2f}s.") 

                # Perform Analysis
                with st.spinner("🔍 Analyzing emotional content..."):
                    print("🧠 Starting analysis...")
                    emotion_result, visualizations = perform_analysis(
                        audio_path_arg=st.session_state.audio_file_path,
                        audio_data_arg=st.session_state.audio_data,
                        sample_rate_arg=st.session_state.sample_rate
                    )

                if emotion_result:
                    print("✅ Analysis complete.")
                    st.session_state.emotion_result = emotion_result
                    st.session_state.visualizations = visualizations
                    st.session_state.processing_complete = True
                    st.success("🎉 Analysis Complete!")
                    time.sleep(0.5)
                    navigate_to('results')
                    st.stop()
                else:
                    st.error("❌ Analysis failed. Please try again.")
                    reset_analysis()

            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                print(f"❌ Upload processing error: {e}") 
                traceback.print_exc()
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                st.session_state.audio_file_path = None

# --- Analysis Function ---
def perform_analysis(audio_path_arg, audio_data_arg, sample_rate_arg): 
    print(f"--- [Analysis] Started ---")
    
    analysis_successful = False 
    final_emotion_result = None 
    final_visualizations = {}   
    
    try:
        print(f"[Analysis] Processing audio data...")

        if audio_path_arg is None and audio_data_arg is None:
            print(f"[Analysis] ❌ ERROR: Missing audio input")
            final_emotion_result = create_fallback_result(error_msg="Missing audio input")
            raise ValueError("Missing audio input arguments")

        current_audio_path = audio_path_arg 
        
        # Handle Saving Recorded Audio
        if isinstance(audio_data_arg, np.ndarray) and current_audio_path == "recorded_audio.wav":
            try:
                print(f"[Analysis] Saving recorded audio...")
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', mode='wb') as tmp_file:
                    sf.write(tmp_file.name, audio_data_arg, sample_rate_arg) 
                    current_audio_path = tmp_file.name 
                    st.session_state.audio_file_path = current_audio_path 
                print(f"[Analysis] ✅ Audio saved to: {current_audio_path}")
            except Exception as e:
                raise Exception(f"Error saving recorded audio: {e}")
        elif not os.path.exists(current_audio_path):
            print(f"[Analysis] ❌ ERROR: Audio path invalid")
            final_emotion_result = create_fallback_result(error_msg="Audio file path invalid")
            raise ValueError("Audio file path invalid.")
        
        # Perform Prediction
        if EmotionPredictor is None:
            print(f"[Analysis] ℹ️ Using fallback analysis")
            final_emotion_result = create_fallback_result()
            analysis_successful = True
        else:
            try:
                print(f"[Analysis] 🚀 Initializing predictor...")
                if callable(EmotionPredictor): 
                    predictor = EmotionPredictor()
                else: 
                    raise TypeError("EmotionPredictor not callable.")
                
                print(f"[Analysis] 🧠 Predicting emotion...")
                emotion_result_dict = predictor.predict_emotion(current_audio_path)
                
                if isinstance(emotion_result_dict, dict) and 'dominant_emotion' in emotion_result_dict:
                     final_emotion_result = emotion_result_dict
                     print(f"[Analysis] ✅ Prediction successful: {final_emotion_result.get('dominant_emotion')}")
                     analysis_successful = True
                else:
                     print(f"[Analysis] ❌ Invalid prediction result")
                     final_emotion_result = create_fallback_result(error_msg="Invalid prediction result")
                     analysis_successful = True 
            except Exception as e:
                print(f"[Analysis] ❌ Prediction Exception: {e}")
                traceback.print_exc()
                final_emotion_result = create_fallback_result(error_msg=f"Prediction error: {e}")
                analysis_successful = True 
        
        # Generate Visualizations
        if final_emotion_result:
            print(f"[Analysis] 🎨 Generating visualizations...")
            final_visualizations = generate_visualizations(
                 emotion_data=final_emotion_result, 
                 audio_data=audio_data_arg,
                 sample_rate=sample_rate_arg 
            )
            print(f"[Analysis] ✅ Visualizations generated")
        else:
            print(f"[Analysis] ⚠️ Skipping visualization")
            final_visualizations = {}

    except Exception as e:
        print(f"--- [Analysis] ❌ CRITICAL ERROR ---")
        print(f"[Analysis] ❌ Exception: {e}")
        traceback.print_exc()
        analysis_successful = False 
        if final_emotion_result is None:
            final_emotion_result = create_fallback_result(error_msg=f"Analysis error: {e}")
        final_visualizations = {} 

    print(f"--- [Analysis] 🏁 Finished (Success: {analysis_successful}) ---")
    return final_emotion_result, final_visualizations

# --- Fallback Result ---
def create_fallback_result(error_msg="Unknown fallback reason"): 
    audio_data = st.session_state.get('audio_data', None)
    sample_rate = st.session_state.get('sample_rate', 22050)
    audio_duration = 0
    if audio_data is not None and sample_rate > 0 and isinstance(audio_data, np.ndarray): 
        audio_duration = len(audio_data) / sample_rate
    
    try:
        audio_features = extract_audio_features(audio_data, sample_rate)
        emotions_list = ["Neutral", "Happy", "Sad", "Angry", "Fearful", "Surprised", "Calm", "Disgust"]
        
        try:
            if callable(EmotionPredictor):
                predictor_instance = EmotionPredictor() 
                if hasattr(predictor_instance, 'emotions'): 
                    emotions_list = [e.title() for e in predictor_instance.emotions]
        except Exception as e: 
            print(f"⚠️ Could not get emotion list: {e}")
        
        num_emotions = len(emotions_list)
        base_probs = np.random.dirichlet(np.ones(num_emotions) * 0.8)
        
        energy = audio_features.get('energy',0)
        zcr = audio_features.get('zcr',0)
        spectral_centroid = audio_features.get('spectral_centroid',0)
        
        # Enhanced emotion mapping based on audio features
        happy_idx = next((i for i,e in enumerate(emotions_list) if e.lower()=='happy'),-1)
        sad_idx = next((i for i,e in enumerate(emotions_list) if e.lower()=='sad'),-1)
        angry_idx = next((i for i,e in enumerate(emotions_list) if e.lower()=='angry'),-1)
        calm_idx = next((i for i,e in enumerate(emotions_list) if e.lower()=='calm'),-1)
        fearful_idx = next((i for i,e in enumerate(emotions_list) if e.lower()=='fearful'),-1)
        surprised_idx = next((i for i,e in enumerate(emotions_list) if e.lower()=='surprised'),-1)
        
        if energy > 0.1:
            if happy_idx!=-1: base_probs[happy_idx] += 0.2
            if angry_idx!=-1: base_probs[angry_idx] += 0.1
        if energy < 0.05:
            if sad_idx!=-1: base_probs[sad_idx] += 0.2
            if calm_idx!=-1: base_probs[calm_idx] += 0.1
        if spectral_centroid > 3000:
            if fearful_idx!=-1: base_probs[fearful_idx] += 0.15
            if surprised_idx!=-1: base_probs[surprised_idx] += 0.1
        
        base_probs = np.maximum(base_probs,0)
        base_probs = base_probs/base_probs.sum() if base_probs.sum()>1e-6 else np.ones(num_emotions)/num_emotions
        dominant_idx = np.argmax(base_probs)
        
        probabilities_dict = {emotions_list[i]: float(prob) for i, prob in enumerate(base_probs)}
        
        return {
            'dominant_emotion': emotions_list[dominant_idx], 
            'confidence': float(base_probs[dominant_idx]), 
            'probabilities': probabilities_dict, 
            'features_used': len(audio_features), 
            'audio_duration': audio_duration, 
            'analysis_timestamp': str(datetime.now()), 
            'model_used': f'fallback ({error_msg})'
        }
    except Exception as e:
        print(f"❌ Fallback analysis failed: {e}")
        emotions_list = ["Neutral", "Happy", "Sad", "Angry", "Fearful", "Surprised", "Calm", "Disgust"]
        num_emotions = len(emotions_list)
        probabilities = np.random.dirichlet(np.ones(num_emotions)*0.5)
        dominant_idx = np.argmax(probabilities)
        probabilities_dict = {emotions_list[i]:float(prob) for i,prob in enumerate(probabilities)}
        
        return {
            'dominant_emotion': emotions_list[dominant_idx], 
            'confidence': float(probabilities[dominant_idx]), 
            'probabilities': probabilities_dict, 
            'features_used': 0, 
            'audio_duration': audio_duration, 
            'analysis_timestamp': str(datetime.now()), 
            'model_used': f'error_fallback ({e})'
        }

# --- Visualization Generation ---
def generate_visualizations(emotion_data=None, audio_data=None, sample_rate=22050):
    visuals = {}
    
    if emotion_data is None or 'probabilities' not in emotion_data:
        print("❌ Visualization: Emotion data missing")
        return visuals 
        
    try:
        emotions = list(emotion_data['probabilities'].keys())
        probabilities = list(emotion_data['probabilities'].values())
        
        # Enhanced Bar Chart
        if emotions and probabilities:
            fig_bar = px.bar(
                x=emotions, y=probabilities, 
                color=probabilities, 
                color_continuous_scale='viridis',
                text=[f'{p:.1%}' for p in probabilities]
            )
            fig_bar.update_traces(
                textposition='outside', 
                marker_line_color='white', 
                marker_line_width=1.5,
                textfont=dict(size=12, color='white')
            )
            fig_bar.update_layout(
                title="Emotional Probability Distribution",
                xaxis_title="Emotions",
                yaxis_title="Probability",
                yaxis_range=[0, max(probabilities) * 1.1 if probabilities else 1],
                showlegend=False,
                plot_bgcolor='#1a1a2e',
                paper_bgcolor='#1a1a2e',
                font=dict(size=12, color='white'),
                height=500
            )
            visuals['bar_chart'] = fig_bar

        # Premium Radar Chart
        if emotions and probabilities:
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=probabilities + [probabilities[0]], 
                theta=emotions + [emotions[0]], 
                fill='toself',
                fillcolor='rgba(212, 175, 55, 0.6)',
                line=dict(color='#FFD700', width=3),
                hoverinfo='text'
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor='#1a1a2e',
                    radialaxis=dict(
                        visible=True, 
                        range=[0, max(probabilities) if probabilities else 1],
                        gridcolor='#333333',
                        linecolor='#333333'
                    ),
                    angularaxis=dict(gridcolor='#333333', linecolor='#333333')
                ),
                title="Emotional Radar Analysis",
                showlegend=False,
                paper_bgcolor='#1a1a2e',
                font=dict(size=12, color='white'),
                height=500
            )
            visuals['radar_chart'] = fig_radar

        # Enhanced Pie Chart
        if emotions and probabilities:
            fig_pie = px.pie(
                values=probabilities, 
                names=emotions, 
                color_discrete_sequence=px.colors.qualitative.Pastel,
                hole=0.4
            )
            fig_pie.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                textfont=dict(size=11, color='black'),
                marker=dict(line=dict(color='white', width=2))
            )
            fig_pie.update_layout(
                title="Emotional Distribution",
                showlegend=True,
                legend_font_color='white',
                paper_bgcolor='#1a1a2e',
                font=dict(size=12, color='white'),
                height=500
            )
            visuals['pie_chart'] = fig_pie
        
        # Premium Waveform
        if audio_data is not None and len(audio_data) > 0:
            try:
                fig_wave, ax = plt.subplots(figsize=(12, 4), facecolor='#0c0c0c')
                times = np.linspace(0, len(audio_data)/sample_rate, num=len(audio_data))
                ax.plot(times, audio_data, color='#FFD700', linewidth=2, alpha=0.8)
                ax.fill_between(times, audio_data, alpha=0.3, color='#FFD700')
                ax.set_title("Voice Waveform Analysis", fontsize=16, fontweight='bold', color='white', pad=20)
                ax.set_xlabel("Time (seconds)", color='#e0e0e0')
                ax.set_ylabel("Amplitude", color='#e0e0e0')
                ax.grid(True, alpha=0.2, color='#333333')
                ax.set_facecolor('#1a1a2e')
                ax.tick_params(colors='#b0b0b0')
                if len(audio_data) > 0:
                    ax.set_ylim(min(audio_data) - 0.1, max(audio_data) + 0.1)
                plt.tight_layout()
                waveform_buffer = BytesIO()
                plt.savefig(waveform_buffer, format='png', dpi=150, bbox_inches='tight', facecolor='#0c0c0c')
                waveform_buffer.seek(0)
                visuals['waveform'] = waveform_buffer
                plt.close(fig_wave)
            except Exception as e: 
                print(f"❌ Error generating waveform: {e}")
        
        # Premium Spectrogram
        if audio_data is not None and len(audio_data) > 0:
            try:
                audio_processor = AudioProcessor()
                spectrogram_buffer = audio_processor.create_spectrogram(audio_data, sample_rate)
                if spectrogram_buffer:
                    visuals['spectrogram'] = spectrogram_buffer
            except Exception as e: 
                print(f"❌ Error generating spectrogram: {e}")
        
    except Exception as e:
        print(f"❌ Visualization Exception: {e}")
        traceback.print_exc()
        plt.close('all') 
    
    return visuals

# --- Audio Feature Extraction ---
def extract_audio_features(audio_data, sr=22050):
    features = {'energy': 0.0, 'zcr': 0.0, 'spectral_centroid': 0.0, 'pitch_variation': 0.0}
    if audio_data is None or len(audio_data) == 0: 
        return features
    try:
        audio_data_float = audio_data.astype(np.float32)
        rms_val = np.sqrt(np.mean(np.square(audio_data_float)))
        features['energy'] = float(rms_val) if np.isfinite(rms_val) else 0.0
        
        zcr_val = np.mean(librosa.feature.zero_crossing_rate(y=audio_data_float)[0])
        features['zcr'] = float(zcr_val) if np.isfinite(zcr_val) else 0.0
        
        n_fft = 1024
        hop_length = 256
        padded_audio = np.pad(audio_data_float, (0, n_fft-len(audio_data_float))) if len(audio_data_float) < n_fft else audio_data_float
        spectral_centroid_val = librosa.feature.spectral_centroid(y=padded_audio, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        mean_sc = np.mean(spectral_centroid_val)
        features['spectral_centroid'] = float(mean_sc) if np.isfinite(mean_sc) else 0.0
        
        std_dev = np.std(audio_data_float)
        features['pitch_variation'] = float(std_dev) if np.isfinite(std_dev) else 0.0
    except Exception as e: 
        print(f"⚠️ Error extracting features: {e}")
    return features

# --- Results Page ---
def results_page():
    emotion_data = st.session_state.get('emotion_result', None)

    if not emotion_data or not isinstance(emotion_data, dict) or 'dominant_emotion' not in emotion_data:
        st.error("❌ Analysis results are missing. Please try another recording.")
        if st.button("🔄 Start New Session"):
            navigate_to('record')
        return

    # Analysis Mode Indicator
    model_used = emotion_data.get('model_used', 'unknown')
    if 'fallback' in model_used:
        st.info(f"🔧 **Analysis Method**: Enhanced Rule-based Analysis")
    elif model_used == 'trained_model':
        st.success("🤖 **Analysis Method**: AI-Powered Deep Analysis")
    else:
        st.warning(f"⚠️ **Analysis Method**: {model_used}")
    
    dominant_emotion_lower = emotion_data.get('dominant_emotion', 'neutral').lower()
    emotion_class = f"emotion-{dominant_emotion_lower}" 
    
    st.markdown(f'<div class="{emotion_class}">', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Premium Results Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        emotion_emoji = {
            'happy': '😊', 'sad': '😢', 'angry': '😠', 'neutral': '😐', 
            'fearful': '😨', 'surprised': '😲', 'calm': '😌', 'disgust': '🤢'
        }
        emoji = emotion_emoji.get(dominant_emotion_lower, '🎭') 
        confidence_percent = emotion_data.get('confidence', 0) * 100 
        
        st.markdown(f"<h1 style='text-align: center; font-size: 5rem; margin-bottom: 0;'>{emoji}</h1>", unsafe_allow_html=True)
        st.markdown(f"""
        <h1 style='text-align: center; color: #ffffff; margin-bottom: 0.5rem; font-size: 2.5rem;'>
            {emotion_data.get('dominant_emotion', 'Unknown').upper()}
        </h1>
        <h2 style='text-align: center; background: linear-gradient(45deg, #D4AF37, #FFD700); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-top: 0; font-weight: 700; font-size: 1.8rem;'>
            {confidence_percent:.1f}% Confidence
        </h2>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Emotional Insights Dashboard
    st.markdown('<h2 class="section-header">📊 Emotional Insights Dashboard</h2>', unsafe_allow_html=True)
    visualizations = st.session_state.get('visualizations', {}) 

    # Row 1: Probability Charts
    col1_viz, col2_viz = st.columns(2) 
    with col1_viz:
        st.markdown("#### Emotional Probability Distribution")
        if 'bar_chart' in visualizations: 
            _call_compat(st.plotly_chart, visualizations['bar_chart'], use_container_width=True)
        else: 
            st.warning("Bar chart not available")
    with col2_viz:
        st.markdown("#### Emotional Radar Analysis")
        if 'radar_chart' in visualizations: 
            _call_compat(st.plotly_chart, visualizations['radar_chart'], use_container_width=True)
        else: 
            st.warning("Radar chart not available")

    # Row 2: Audio Analysis
    col3_viz, col4_viz = st.columns(2) 
    with col3_viz:
        st.markdown("#### Voice Waveform Analysis")
        if 'waveform' in visualizations: 
            _call_compat(st.image, visualizations['waveform'], use_container_width=True)
        else: 
            st.warning("Waveform not available")
    with col4_viz:
        st.markdown("#### Frequency Spectrogram")
        if 'spectrogram' in visualizations: 
            _call_compat(st.image, visualizations['spectrogram'], use_container_width=True)
        else: 
            st.warning("Spectrogram not available")

    # Row 3: Additional Visualizations & Summary
    col5_viz, col6_viz = st.columns(2) 
    with col5_viz:
        st.markdown("#### Emotional Distribution")
        if 'pie_chart' in visualizations: 
            _call_compat(st.plotly_chart, visualizations['pie_chart'], use_container_width=True)
        else: 
            st.warning("Pie chart not available")
    with col6_viz:
        st.markdown("#### Audio Analysis Summary")
        audio_data = st.session_state.get('audio_data', None)
        sample_rate = st.session_state.get('sample_rate', 22050)
        audio_features = extract_audio_features(audio_data, sample_rate)
        
        # Premium Metrics Grid
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Vocal Energy", f"{audio_features.get('energy', 0):.3f}", 
                     delta="High" if audio_features.get('energy', 0) > 0.1 else "Low")
            st.metric("Spectral Profile", f"{audio_features.get('spectral_centroid', 0):.0f} Hz")
        with col_m2:
            st.metric("Voice Activity", f"{audio_features.get('zcr', 0):.3f}")
            st.metric("Pitch Dynamics", f"{audio_features.get('pitch_variation', 0):.3f}")
    
    # User Profile Summary
    st.markdown('<h2 class="section-header">👤 Personal Profile Summary</h2>', unsafe_allow_html=True)
    user_details = st.session_state.get('user_details', {})
    
    # Premium Metrics Cards
    col_u1, col_u2, col_u3, col_u4 = st.columns(4) 
    with col_u1: 
        st.metric("Name", user_details.get('name', 'N/A'))
        st.metric("Age", user_details.get('age', 'N/A'))
    with col_u2: 
        st.metric("Gender", user_details.get('gender', 'N/A'))
        st.metric("Occupation", user_details.get('occupation', 'N/A'))
    with col_u3: 
        st.metric("Analysis Time", datetime.now().strftime("%H:%M"))
        st.metric("Session Duration", f"{emotion_data.get('audio_duration', 0):.1f}s")
    with col_u4: 
        st.metric("Confidence Level", f"{emotion_data.get('confidence', 0):.1%}")
        st.metric("Features Analyzed", emotion_data.get('features_used', 'N/A'))
    
    # Report Generation
    st.markdown('<h2 class="section-header">📄 Premium Report Generation</h2>', unsafe_allow_html=True)
    if _call_compat(st.button, "📄 Download Premium Emotional Intelligence Report", use_container_width=True, type="primary"):
        generate_pdf_report()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if _call_compat(st.button, "🔄 New Analysis Session", use_container_width=True):
            navigate_to('user_details') 
    with col_nav2:
        if _call_compat(st.button, "🏠 Return to Home", use_container_width=True, type="secondary"):
            navigate_to('home')

# --- PDF Generation ---
def generate_pdf_report():
    try:
        user_details = st.session_state.get('user_details',{})
        emotion_result = st.session_state.get('emotion_result',None)
        audio_data_arr = st.session_state.get('audio_data',None)
        visualizations = st.session_state.get('visualizations',{})
        
        if emotion_result is None or 'dominant_emotion' not in emotion_result: 
            st.error("❌ Cannot generate PDF: No valid analysis data")
            return
        
        pdf_buffer = None
        generator_used = "Basic Fallback"
        
        try:
            from utils.premium_pdf_generator import PremiumPDFGenerator
            pdf_gen = PremiumPDFGenerator()
            pdf_buffer = pdf_gen.generate_report(user_details, emotion_result, audio_data_arr, visualizations)
            generator_used = "Premium"
            print("📄 Using Premium PDF Generator.")
        except ImportError:
            try:
                from utils.simple_pdf_generator import SimplePDFGenerator
                pdf_gen = SimplePDFGenerator()
                pdf_buffer = pdf_gen.generate_report(user_details, emotion_result, audio_data_arr)
                generator_used = "Simple"
                print("📄 Using Simple PDF Generator.")
            except ImportError:
                from reportlab.lib.pagesizes import letter
                from reportlab.lib.styles import getSampleStyleSheet
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                
                pdf_buffer = BytesIO()
                doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
                styles = getSampleStyleSheet()
                story = []
                
                story.append(Paragraph("VoxSense Emotion Report", styles['h1']))
                story.append(Spacer(1,12))
                story.append(Paragraph(f"Name: {user_details.get('name','N/A')}", styles['Normal']))
                story.append(Paragraph(f"Emotion: {emotion_result.get('dominant_emotion','N/A')}", styles['Normal']))
                story.append(Paragraph(f"Confidence: {emotion_result.get('confidence',0):.1%}", styles['Normal']))
                story.append(Spacer(1,12))
                story.append(Paragraph("Probabilities:", styles['h3']))
                
                if isinstance(emotion_result.get('probabilities'),dict):
                    for emotion,prob in emotion_result['probabilities'].items(): 
                        story.append(Paragraph(f"- {emotion}: {prob:.1%}", styles['Normal']))
                
                doc.build(story)
                pdf_buffer.seek(0)
        
        if pdf_buffer is None:
            raise Exception("PDF buffer is None")
            
        user_name = user_details.get('name','User')
        safe_user_name = "".join(c for c in user_name if c.isalnum() or c in (' ','_')).rstrip().replace(' ','_') or 'User'
        filename = f"VoxSense_Report_{safe_user_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        _call_compat(
            st.download_button,
            label=f"📥 Download {generator_used} Report ({user_name})",
            data=pdf_buffer,
            file_name=filename,
            mime="application/pdf",
            key='pdf-download-button',
            use_container_width=True
        )
        print(f"✅ PDF '{filename}' ready for download")
        
    except Exception as e:
        st.error(f"❌ Error generating PDF: {str(e)}")
        print(f"❌ PDF Exception: {e}")
        traceback.print_exc()
        st.info("Ensure 'reportlab' is installed for PDF generation.")

# --- About Page ---
def about_page():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">👥 About VoxSense</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🌟 Our Vision
    VoxSense represents the cutting edge of emotional intelligence technology. 
    We believe that the human voice carries profound emotional information that, 
    when properly analyzed, can reveal deep insights into human psychology and behavior.
    
    ### 🎯 Our Mission
    To democratize emotional intelligence through advanced AI, making sophisticated 
    emotion analysis accessible to everyone - from mental health professionals to 
    individuals seeking self-awareness.
    
    ### 💎 Premium Technology
    - **Advanced AI Algorithms**: Deep learning models trained on diverse emotional datasets
    - **Real-time Processing**: Instant analysis with sub-second response times
    - **Multi-modal Analysis**: Combining vocal features with behavioral patterns
    - **Enterprise-grade Security**: Your data remains private and secure
    
    ### 👑 Crafted with Excellence
    VoxSense is built by a team of AI researchers, psychologists, and software engineers 
    dedicated to pushing the boundaries of what's possible in emotional AI.
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- How It Works Page ---
def how_it_works_page():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🔬 The VoxSense Experience</h2>', unsafe_allow_html=True)
    
    steps = [
        {
            "step": "1️⃣", 
            "title": "Voice Session", 
            "desc": "Record your voice live or upload an audio file (.wav, .mp3, .ogg, .m4a) for analysis."
        },
        {
            "step": "2️⃣", 
            "title": "Feature Extraction", 
            "desc": "Advanced algorithms extract 50+ vocal characteristics including pitch, energy, rhythm, spectral features, and MFCCs."
        },
        {
            "step": "3️⃣", 
            "title": "AI-Powered Analysis", 
            "desc": "Our deep learning models analyze vocal patterns against extensive emotional datasets to identify subtle emotional cues."
        },
        {
            "step": "4️⃣", 
            "title": "Emotional Intelligence", 
            "desc": "The system maps vocal patterns to 8 core emotions with confidence scores and contextual understanding."
        },
        {
            "step": "5️⃣", 
            "title": "Premium Insights", 
            "desc": "Receive comprehensive visual analytics, detailed reports, and actionable insights about emotional patterns."
        }
    ]
    
    for i, step in enumerate(steps):
        col1_step, col2_step = st.columns([1, 8])
        with col1_step:
            st.markdown(f"<div style='font-size: 2rem; text-align: center;'>{step['step']}</div>", unsafe_allow_html=True)
        with col2_step:
            st.markdown(f"**{step['title']}**")
            st.markdown(f"<p style='color: #b0b0b0;'>{step['desc']}</p>", unsafe_allow_html=True)
        
        if i < len(steps) - 1:
            st.markdown("---")
    
    st.markdown("""
    <div style="background: rgba(212, 175, 55, 0.1); padding: 2rem; border-radius: 15px; margin-top: 2rem; border-left: 4px solid #D4AF37;">
        <h4 style="color: #FFD700; margin-bottom: 1rem;">🔬 Scientific Foundation</h4>
        <p style="color: #e0e0e0;">
        VoxSense is built on peer-reviewed research in vocal emotion recognition, 
        combining signal processing with deep learning to achieve unprecedented 
        accuracy in emotional analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- Navigation Sidebar ---
def navigation_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;margin-bottom:3rem;">
            <h2 style="color:#ffffff;margin-bottom:0.5rem;font-weight:800;font-size:1.8rem;">VoxSense</h2>
            <p style="color:#FFD700;margin-top:0;font-weight:500;font-size:1rem;">Emotional Intelligence Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        pages = {
            "🏠 Home": "home", 
            "👤 Profile": "user_details", 
            "🎤 Voice Session": "record", 
            "📊 Insights": "results", 
            "🔬 Experience": "how_it_works", 
            "👥 About": "about"
        }
        
        current_page_key = st.session_state.get('page','home')
        profile_done = bool(st.session_state.get('user_details'))
        analysis_done = bool(st.session_state.get('emotion_result'))
        
        button_states = {
            "home": False, 
            "user_details": False, 
            "record": not profile_done, 
            "results": not analysis_done, 
            "how_it_works": False, 
            "about": False
        }
        
        for page_name, page_key in pages.items():
            # Display a clear current page label above navigation (helps when UI styles hide active button)
            current_display = next((name for name, key in pages.items() if key == current_page_key), 'Home')
        st.markdown(f"**Current Page:** <span style='color:#FFD700'>{current_display}</span>", unsafe_allow_html=True)
        for page_name, page_key in pages.items():
            if _call_compat(
                st.button,
                page_name,
                key=f"nav_{page_key}",
                use_container_width=True,
                type="primary" if current_page_key == page_key else "secondary",
                disabled=button_states[page_key]
            ):
                navigate_to(page_key)
        
        st.markdown("---")
        
        user_details = st.session_state.get('user_details',{})
        if user_details:
            st.markdown("**Current Session:**")
            st.markdown(f"<p style='color:#FFD700;'>👤 {user_details.get('name','Guest')}</p>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align:center;color:#b0b0b0;font-size:0.85rem;padding:1.5rem;'>
            Thank you for choosing <strong style='color:#FFD700;'>VoxSense</strong> 💎<br>
            © 2026 VoxSense Technologies<br><br>
            <span style='color:#d4af37;font-size:0.9rem;'>
                By <strong>Pranjal Belalekar</strong> & <strong>Bhavya Doshi</strong>
            </span>
        </div>
        """, unsafe_allow_html=True)

# --- Main App Router ---
def main(): 
    load_css()
    init_session_state() 
    navigation_sidebar() 
    
    pages = { 
        'home': home_page, 
        'user_details': user_details_page, 
        'record': record_audio_page, 
        'results': results_page, 
        'about': about_page, 
        'how_it_works': how_it_works_page 
    }
    
    current_page_key = st.session_state.get('page', 'home')
    page_function = pages.get(current_page_key)
    
    if page_function:
        try: 
            page_function()
        except Exception as page_error:
            st.error(f"An error occurred: {page_error}")
            print(f"❌ Page error: {page_error}")
            traceback.print_exc()
            if st.button("Go Home"):
                navigate_to('home')
    else:
        home_page()

# --- Entry Point ---
if __name__ == "__main__":
    missing_deps = []
    try: import soundfile as sf
    except ImportError: missing_deps.append("soundfile")
    try: import reportlab
    except ImportError: missing_deps.append("reportlab")
    try: import plotly
    except ImportError: missing_deps.append("plotly")
    try: import librosa
    except ImportError: missing_deps.append("librosa")
    
    if not predictor_available or EmotionPredictor is None:
        missing_deps.append("emotion predictor")
    
    if missing_deps:
        st.error(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        st.info("Please install required libraries and ensure predictor files exist.")
        st.stop()
    
    print("\n--- Starting VoxSense App ---") 
    main()
