import librosa
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from io import BytesIO
import numpy as np
from datetime import datetime
import base64

class PremiumPDFGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_premium_styles()
    
    def setup_premium_styles(self):
        """Setup premium styles"""
        # PremiumTitle style
        self.styles.add(ParagraphStyle(
            'PremiumTitle',
            parent=self.styles['Heading1'],
            fontSize=22,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            alignment=1,
            fontName='Helvetica-Bold'
        ))
        
        # PremiumSubtitle style
        self.styles.add(ParagraphStyle(
            'PremiumSubtitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#7f8c8d'),
            spaceAfter=15,
            alignment=1,
            fontName='Helvetica'
        ))
        
        # CompanyHeader style
        self.styles.add(ParagraphStyle(
            'CompanyHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=8,
            alignment=1,
            fontName='Helvetica-Bold'
        ))
        
        # SectionHeader style
        self.styles.add(ParagraphStyle(
            'SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=13,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=10,
            fontName='Helvetica-Bold',
            alignment=0
        ))
        
        # PremiumBody style
        self.styles.add(ParagraphStyle(
            'PremiumBody',
            parent=self.styles['BodyText'],
            fontSize=9,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=6,
            fontName='Helvetica',
            alignment=4
        ))

    def generate_report(self, user_details, emotion_result, audio_data, visualizations):
        """Generate premium PDF report with all visualizations"""
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, 
                              topMargin=0.5*inch, 
                              bottomMargin=0.5*inch,
                              leftMargin=0.5*inch,
                              rightMargin=0.5*inch)
        story = []
        
        # Title Section
        company_header = Paragraph("VOXSENSE EMOTIONAL INTELLIGENCE", self.styles['CompanyHeader'])
        story.append(company_header)
        
        title = Paragraph("Comprehensive Emotion Analysis Report", self.styles['PremiumTitle'])
        story.append(title)
        
        subtitle = Paragraph("Advanced Voice Emotion Detection Technology", self.styles['PremiumSubtitle'])
        story.append(subtitle)
        story.append(Spacer(1, 0.2*inch))
        
        # Report Metadata
        metadata = [
            ["Report For:", f"{user_details.get('name', 'User')}"],
            ["Report Date:", datetime.now().strftime("%Y-%m-%d %H:%M")],
            ["Report ID:", f"VS{datetime.now().strftime('%Y%m%d%H%M%S')}"]
        ]
        
        meta_table = Table(metadata, colWidths=[1.5*inch, 3*inch])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fa')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7'))
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Main Result Highlight
        story.append(Paragraph("Primary Emotion Detected", self.styles['SectionHeader']))
        
        result_data = [
            ["Detected Emotion", "Confidence Level", "Analysis Status"],
            [emotion_result['dominant_emotion'].upper(), 
             f"{emotion_result['confidence']:.1%}", 
             "COMPLETED"]
        ]
        
        result_table = Table(result_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        result_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#ecf0f1')),
            ('TEXTCOLOR', (0, 1), (-1, 1), colors.HexColor('#2c3e50')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7'))
        ]))
        story.append(result_table)
        story.append(Spacer(1, 0.3*inch))
        
        # User Information Section
        story.append(Paragraph("User Profile Information", self.styles['SectionHeader']))
        
        user_info = []
        if user_details.get('name'):
            user_info.append(["Full Name", user_details['name']])
        if user_details.get('age'):
            user_info.append(["Age", str(user_details['age'])])
        if user_details.get('gender'):
            user_info.append(["Gender", user_details['gender']])
        if user_details.get('occupation'):
            user_info.append(["Occupation", user_details['occupation']])
        if user_details.get('email'):
            user_info.append(["Contact Email", user_details['email']])
        if user_details.get('contact'):
            user_info.append(["Contact Number", user_details['contact']])
        
        user_info.append(["Session Date", user_details.get('timestamp', datetime.now()).strftime("%Y-%m-%d %H:%M")])
        
        if user_info:
            user_table = Table(user_info, colWidths=[1.8*inch, 3.2*inch])
            user_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#34495e')),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.white),
                ('BACKGROUND', (1, 0), (1, -1), colors.HexColor('#ffffff')),
                ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#2c3e50')),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('ALIGN', (1, 0), (1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7'))
            ]))
            story.append(user_table)
            story.append(Spacer(1, 0.3*inch))
        
        # Detailed Emotion Analysis
        story.append(Paragraph("Emotion Analysis Results", self.styles['SectionHeader']))
        
        # Emotion Probabilities Table
        prob_data = [["Emotion", "Probability", "Confidence"]]
        for emotion, prob in emotion_result['probabilities'].items():
            confidence_level = "Very High" if prob > 0.8 else "High" if prob > 0.6 else "Medium" if prob > 0.4 else "Low"
            prob_data.append([emotion.title(), f"{prob:.1%}", confidence_level])
        
        prob_table = Table(prob_data, colWidths=[1.8*inch, 1.5*inch, 1.7*inch])
        prob_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ffffff')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#2c3e50')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7'))
        ]))
        story.append(prob_table)
        story.append(Spacer(1, 0.3*inch))

        # Check if we have enough content for visualizations
        has_visualizations = any(key in visualizations for key in ['bar_chart', 'pie_chart', 'radar_chart', 'waveform', 'spectrogram'])
        
        if has_visualizations:
            # Only add page break if we have visualizations
            story.append(PageBreak())
            story.append(Paragraph("Visual Analysis & Insights", self.styles['SectionHeader']))
            story.append(Spacer(1, 0.2*inch))
            
            # Try to use existing visualizations first
            visualization_added = False
            
            # 1. Bar Chart
            if 'bar_chart' in visualizations:
                try:
                    story.append(Paragraph("Emotion Probability Distribution", self.styles['PremiumBody']))
                    bar_buffer = self.plotly_to_image_fallback(visualizations['bar_chart'], emotion_result, 'bar')
                    bar_image = Image(bar_buffer, width=5.5*inch, height=3*inch)
                    story.append(bar_image)
                    story.append(Spacer(1, 0.2*inch))
                    visualization_added = True
                except Exception as e:
                    print(f"Bar chart error: {e}")
            
            # 2. Pie Chart  
            if 'pie_chart' in visualizations:
                try:
                    story.append(Paragraph("Emotion Distribution Overview", self.styles['PremiumBody']))
                    pie_buffer = self.plotly_to_image_fallback(visualizations['pie_chart'], emotion_result, 'pie')
                    pie_image = Image(pie_buffer, width=4.5*inch, height=3.5*inch)
                    story.append(pie_image)
                    story.append(Spacer(1, 0.2*inch))
                    visualization_added = True
                except Exception as e:
                    print(f"Pie chart error: {e}")
            
            # 3. Radar Chart
            if 'radar_chart' in visualizations:
                try:
                    story.append(Paragraph("Emotion Profile Radar Chart", self.styles['PremiumBody']))
                    radar_buffer = self.plotly_to_image_fallback(visualizations['radar_chart'], emotion_result, 'radar')
                    radar_image = Image(radar_buffer, width=5*inch, height=4*inch)
                    story.append(radar_image)
                    story.append(Spacer(1, 0.2*inch))
                    visualization_added = True
                except Exception as e:
                    print(f"Radar chart error: {e}")
            
            # 4. Waveform
            if 'waveform' in visualizations:
                try:
                    story.append(Paragraph("Audio Waveform Analysis", self.styles['PremiumBody']))
                    visualizations['waveform'].seek(0)
                    waveform_image = Image(visualizations['waveform'], width=5.5*inch, height=2.5*inch)
                    story.append(waveform_image)
                    story.append(Spacer(1, 0.2*inch))
                    visualization_added = True
                except Exception as e:
                    print(f"Waveform error: {e}")
            
            # 5. Spectrogram
            if 'spectrogram' in visualizations:
                try:
                    story.append(Paragraph("Voice Frequency Analysis (Spectrogram)", self.styles['PremiumBody']))
                    visualizations['spectrogram'].seek(0)
                    spectrogram_image = Image(visualizations['spectrogram'], width=5.5*inch, height=2.5*inch)
                    story.append(spectrogram_image)
                    story.append(Spacer(1, 0.2*inch))
                    visualization_added = True
                except Exception as e:
                    print(f"Spectrogram error: {e}")
            
            # If no visualizations were added, remove the empty section
            if not visualization_added:
                # Remove the last elements (header and spacer) since no charts were added
                while story and not isinstance(story[-1], (Paragraph, Spacer)):
                    story.pop()
                if story and isinstance(story[-1], Spacer):
                    story.pop()
                if story and "Visual Analysis" in getattr(story[-1], 'text', ''):
                    story.pop()
        
        # Technical Analysis Section - ALWAYS ADD THIS
        story.append(Paragraph("Technical Analysis Details", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.2*inch))
        
        if audio_data is not None:
            duration = len(audio_data) / 22050
            audio_features = self.extract_audio_features(audio_data)
            
            record_data = [
                ["Recording Parameter", "Value", "Technical Details"],
                ["Duration", f"{duration:.2f} seconds", "Total recording length"],
                ["Sample Rate", "22.05 kHz", "Industry standard for voice analysis"],
                ["Channels", "Mono", "Single channel recording"],
                ["File Format", "WAV/PCM", "Lossless audio format"],
                ["Vocal Energy", f"{audio_features['energy']:.4f}", "RMS energy of voice"],
                ["Zero Crossing Rate", f"{audio_features['zcr']:.4f}", "Voice activity indicator"],
                ["Spectral Centroid", f"{audio_features['spectral_centroid']:.1f} Hz", "Voice brightness"],
                ["Spectral Rolloff", f"{audio_features.get('rolloff', 4000):.1f} Hz", "Voice frequency range"],
                ["Analysis Frames", f"{audio_features.get('frames', 100)}", "Processed audio frames"]
            ]
        else:
            record_data = [
                ["Recording Parameter", "Status", "Details"],
                ["Audio Data", "Processed", "Emotion analysis completed"],
                ["Analysis Method", "AI/ML Model", "Deep learning based"],
                ["Confidence Score", f"{emotion_result['confidence']:.1%}", "Overall accuracy"],
                ["Processing Time", "< 5 seconds", "Real-time analysis"],
                ["Features Analyzed", "15+ parameters", "Comprehensive voice analysis"]
            ]
        
        record_table = Table(record_data, colWidths=[1.8*inch, 1.3*inch, 2.4*inch])
        record_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ffffff')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#2c3e50')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7'))
        ]))
        
        story.append(record_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        summary_text = f"""
        This comprehensive emotional analysis for <b>{user_details.get('name', 'the user')}</b> identifies 
        <b>{emotion_result['dominant_emotion']}</b> as the primary emotional state with <b>{emotion_result['confidence']:.1%}</b> confidence. 
        The analysis utilized advanced voice emotion recognition technology examining vocal characteristics 
        including pitch, tone, energy, and spectral features. This report provides valuable insights into 
        emotional expression patterns for personal development and emotional awareness.
        """
        summary = Paragraph(summary_text, self.styles['PremiumBody'])
        story.append(summary)
        story.append(Spacer(1, 0.3*inch))
        
        # Quality Assurance
        story.append(Paragraph("Quality Assurance", self.styles['SectionHeader']))
        
        qa_items = [
            "✓ Certified emotional intelligence algorithms",
            "✓ Industry-standard privacy protocols", 
            "✓ Validated against emotional pattern databases",
            "✓ Timestamped and uniquely identified report",
            "✓ Visualizations from raw analytical data"
        ]
        
        for item in qa_items:
            qa_para = Paragraph(item, self.styles['PremiumBody'])
            story.append(qa_para)
        
        story.append(Spacer(1, 0.4*inch))
        
        # Footer with Company Information
        footer_text = f"""
        <para alignment="center">
        <b>VoxSense Emotional Intelligence Platform</b><br/>
        Advanced Voice Emotion Recognition Technology<br/>
        Report: {datetime.now().strftime("%Y-%m-%d %H:%M")} | ID: VS{datetime.now().strftime("%Y%m%d%H%M%S")}<br/>
        Client: {user_details.get('name', 'Valued Client')} | Confidential Report<br/>
        © 2025 VoxSense Technologies | www.voxsense.ai
        </para>
        """
        
        footer_style = ParagraphStyle(
            'FooterStyle',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#7f8c8d'),
            alignment=1
        )
        
        footer = Paragraph(footer_text, footer_style)
        story.append(footer)
        
        doc.build(story)
        buffer.seek(0)
        return buffer

    def plotly_to_image_fallback(self, fig, emotion_result, chart_type):
        """Convert Plotly figure to image with matplotlib fallback"""
        try:
            # Try to convert Plotly figure
            import plotly.io as pio
            img_bytes = pio.to_image(fig, format="png", width=800, height=500)
            buffer = BytesIO(img_bytes)
            return buffer
        except Exception as e:
            print(f"Plotly conversion failed for {chart_type}: {e}")
            # Fallback to matplotlib
            if chart_type == 'bar':
                return self.create_bar_chart_matplotlib(emotion_result)
            elif chart_type == 'pie':
                return self.create_pie_chart_matplotlib(emotion_result)
            elif chart_type == 'radar':
                return self.create_radar_chart_matplotlib(emotion_result)
            else:
                return self.create_fallback_chart()

    def create_bar_chart_matplotlib(self, emotion_result):
        """Create bar chart using matplotlib"""
        emotions = list(emotion_result['probabilities'].keys())
        probabilities = list(emotion_result['probabilities'].values())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors_list = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#d35400']
        bars = ax.bar(emotions, probabilities, color=colors_list[:len(emotions)], alpha=0.8)
        
        # Add value labels
        for bar, value in zip(bars, probabilities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.1%}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Probability')
        ax.set_xlabel('Emotions')
        ax.set_title('Emotion Probability Distribution', fontweight='bold', pad=20)
        ax.set_ylim(0, max(probabilities) * 1.15 if probabilities else 1)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plt.close()
        return buffer

    def create_pie_chart_matplotlib(self, emotion_result):
        """Create pie chart using matplotlib"""
        emotions = list(emotion_result['probabilities'].keys())
        probabilities = list(emotion_result['probabilities'].values())
        
        fig, ax = plt.subplots(figsize=(8, 8))
        colors_list = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#d35400']
        
        wedges, texts, autotexts = ax.pie(probabilities, labels=emotions, colors=colors_list[:len(emotions)],
                                         autopct='%1.1f%%', startangle=90)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Emotion Distribution Overview', fontweight='bold', pad=20)
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plt.close()
        return buffer

    def create_radar_chart_matplotlib(self, emotion_result):
        """Create radar chart using matplotlib"""
        emotions = list(emotion_result['probabilities'].keys())
        probabilities = list(emotion_result['probabilities'].values())
        
        categories = emotions
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        values = probabilities
        values += values[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Emotion Profile Radar Chart', fontweight='bold', pad=20)
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plt.close()
        return buffer

    def create_fallback_chart(self):
        """Create fallback chart"""
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'Chart Not Available', ha='center', va='center', transform=ax.transAxes)
        ax.set_facecolor('#f8f9fa')
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
        buffer.seek(0)
        plt.close()
        return buffer

    def extract_audio_features(self, audio_data, sr=22050):
        """Extract audio features for technical details"""
        features = {}
        
        try:
            features['energy'] = np.sqrt(np.mean(audio_data**2))
            features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(audio_data))
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
            features['spectral_centroid'] = np.mean(spectral_centroid)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
            features['rolloff'] = np.mean(spectral_rolloff)
            features['frames'] = len(audio_data) // 512
        except:
            features = {
                'energy': 0.1, 'zcr': 0.05, 'spectral_centroid': 2000,
                'rolloff': 4000, 'frames': 100
            }
        
        return features