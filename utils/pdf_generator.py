# pdf_generator.py

import librosa
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-GUI environments
from io import BytesIO
import numpy as np
from datetime import datetime
import traceback # For detailed error logging

class PremiumPDFGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_premium_styles()

    def setup_premium_styles(self):
        """Setup premium styles"""
        common_props = {'fontName': 'Helvetica'}
        bold_props = {'fontName': 'Helvetica-Bold'}

        self.styles.add(ParagraphStyle('PremiumTitle', parent=self.styles['h1'], fontSize=20, alignment=1, spaceAfter=14, textColor=colors.HexColor('#1A1A2E'), **bold_props))
        self.styles.add(ParagraphStyle('PremiumSubtitle', parent=self.styles['h2'], fontSize=12, alignment=1, spaceAfter=18, textColor=colors.HexColor('#7f8c8d'), **common_props))
        self.styles.add(ParagraphStyle('CompanyHeader', parent=self.styles['h2'], fontSize=14, alignment=1, spaceAfter=6, textColor=colors.HexColor('#D4AF37'), **bold_props)) # Gold color
        self.styles.add(ParagraphStyle('SectionHeader', parent=self.styles['h2'], fontSize=12, alignment=0, spaceBefore=10, spaceAfter=8, textColor=colors.HexColor('#16213E'), **bold_props)) # Dark Blue
        self.styles.add(ParagraphStyle('PremiumBody', parent=self.styles['BodyText'], fontSize=9, alignment=4, spaceAfter=6, textColor=colors.HexColor('#2c3e50'), leading=12, **common_props)) # Justified
        self.styles.add(ParagraphStyle('FooterStyle', parent=self.styles['Normal'], fontSize=7, alignment=1, textColor=colors.HexColor('#95a5a6'), **common_props))
        self.styles.add(ParagraphStyle('ChartLabel', parent=self.styles['Normal'], fontSize=8, alignment=1, spaceBefore=2, spaceAfter=6, textColor=colors.dimgrey, **common_props))


    def generate_report(self, user_details, emotion_result, audio_data, visualizations):
        """Generate premium PDF report with all visualizations"""
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                                topMargin=0.7*inch, bottomMargin=0.7*inch,
                                leftMargin=0.7*inch, rightMargin=0.7*inch)
        story = []
        available_width = A4[0] - 1.4 * inch # Calculate available width

        try:
            # === Title Section ===
            story.append(Paragraph("VOXSENSE EMOTIONAL INTELLIGENCE", self.styles['CompanyHeader']))
            story.append(Paragraph("Comprehensive Emotion Analysis Report", self.styles['PremiumTitle']))
            story.append(Paragraph("Advanced Voice Emotion Detection Technology", self.styles['PremiumSubtitle']))
            story.append(Spacer(1, 0.1*inch))

            # === Report Metadata ===
            metadata = [
                [Paragraph("<b>Report For:</b>", self.styles['PremiumBody']), Paragraph(f"{user_details.get('name', 'User')}", self.styles['PremiumBody'])],
                [Paragraph("<b>Report Date:</b>", self.styles['PremiumBody']), Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M"), self.styles['PremiumBody'])],
                [Paragraph("<b>Report ID:</b>", self.styles['PremiumBody']), Paragraph(f"VS{datetime.now().strftime('%Y%m%d%H%M%S')}", self.styles['PremiumBody'])]
            ]
            meta_table = Table(metadata, colWidths=[1.5*inch, available_width - 1.5*inch])
            meta_table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#EAECEE')),
            ]))
            story.append(meta_table)
            story.append(Spacer(1, 0.2*inch))

            # === Main Result Highlight ===
            story.append(Paragraph("Primary Emotion Detected", self.styles['SectionHeader']))
            result_data = [
                [Paragraph(h, self.styles['PremiumBody']) for h in ["Detected Emotion", "Confidence Level", "Analysis Status"]],
                [Paragraph(f"<b>{emotion_result['dominant_emotion'].upper()}</b>", self.styles['PremiumBody']),
                 Paragraph(f"<b>{emotion_result['confidence']:.1%}</b>", self.styles['PremiumBody']),
                 Paragraph("COMPLETED", self.styles['PremiumBody'])]
            ]
            result_table = Table(result_data, colWidths=[available_width/3.0]*3)
            result_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16213E')), # Dark Blue Header
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(result_table)
            story.append(Spacer(1, 0.2*inch))

            # === User Information Section ===
            story.append(Paragraph("User Profile Information", self.styles['SectionHeader']))
            user_info = []
            if user_details.get('name'): user_info.append([Paragraph("<b>Full Name</b>", self.styles['PremiumBody']), Paragraph(user_details['name'], self.styles['PremiumBody'])])
            if user_details.get('age'): user_info.append([Paragraph("<b>Age</b>", self.styles['PremiumBody']), Paragraph(str(user_details['age']), self.styles['PremiumBody'])])
            if user_details.get('gender'): user_info.append([Paragraph("<b>Gender</b>", self.styles['PremiumBody']), Paragraph(user_details['gender'], self.styles['PremiumBody'])])
            if user_details.get('occupation'): user_info.append([Paragraph("<b>Occupation</b>", self.styles['PremiumBody']), Paragraph(user_details['occupation'], self.styles['PremiumBody'])])
            # Add more fields if needed
            user_info.append([Paragraph("<b>Session Date</b>", self.styles['PremiumBody']), Paragraph(user_details.get('timestamp', datetime.now()).strftime("%Y-%m-%d %H:%M"), self.styles['PremiumBody'])])

            if user_info:
                user_table = Table(user_info, colWidths=[1.8*inch, available_width - 1.8*inch])
                user_table.setStyle(TableStyle([
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                     ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#EAECEE')),
                ]))
                story.append(user_table)
                story.append(Spacer(1, 0.2*inch))

            # === Detailed Emotion Analysis ===
            story.append(Paragraph("Emotion Analysis Results", self.styles['SectionHeader']))
            prob_data = [[Paragraph(h, self.styles['PremiumBody']) for h in ["Emotion", "Probability", "Confidence Level"]]]
            sorted_probs = sorted(emotion_result['probabilities'].items(), key=lambda item: item[1], reverse=True)
            for emotion, prob in sorted_probs:
                level = "Very High" if prob > 0.7 else "High" if prob > 0.5 else "Medium" if prob > 0.3 else "Low"
                prob_data.append([Paragraph(emotion.title(), self.styles['PremiumBody']),
                                  Paragraph(f"{prob:.1%}", self.styles['PremiumBody']),
                                  Paragraph(level, self.styles['PremiumBody'])])

            prob_table = Table(prob_data, colWidths=[available_width/3.0]*3)
            prob_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')), # Blue Header
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
            ]))
            story.append(prob_table)
            story.append(Spacer(1, 0.2*inch))

            # === VISUALIZATIONS SECTION ===
            story.append(PageBreak())
            story.append(Paragraph("Visual Analysis & Insights", self.styles['SectionHeader']))
            story.append(Spacer(1, 0.1*inch))

            chart_width = available_width * 0.95 # Slightly smaller than full width

            # --- Error Handling for Charts ---
            def add_chart_to_story(chart_func, title, *args):
                try:
                    buffer = chart_func(*args)
                    if buffer:
                        img = Image(buffer, width=chart_width, height=(chart_width * 0.5)) # Maintain aspect ratio roughly
                        img.hAlign = 'CENTER'
                        story.append(img)
                        story.append(Paragraph(title, self.styles['ChartLabel']))
                        story.append(Spacer(1, 0.15*inch))
                    else:
                        story.append(Paragraph(f"<i>Error generating {title}.</i>", self.styles['PremiumBody']))
                        story.append(Spacer(1, 0.15*inch))
                except Exception as e:
                    print(f"Error adding chart '{title}': {e}")
                    traceback.print_exc() # Print detailed error
                    story.append(Paragraph(f"<i>Error generating {title}: {e}</i>", self.styles['PremiumBody']))
                    story.append(Spacer(1, 0.15*inch))

            # Add Charts Safely
            add_chart_to_story(self.create_bar_chart, "Emotion Probability Distribution", emotion_result)

            if 'waveform' in visualizations and visualizations['waveform']:
                 try:
                    waveform_image = Image(visualizations['waveform'], width=chart_width, height=(chart_width*0.4))
                    waveform_image.hAlign = 'CENTER'
                    story.append(waveform_image)
                    story.append(Paragraph("Audio Waveform Analysis", self.styles['ChartLabel']))
                    story.append(Spacer(1, 0.15*inch))
                 except Exception as e:
                    print(f"Error adding waveform: {e}")
                    story.append(Paragraph(f"<i>Error displaying waveform image.</i>", self.styles['PremiumBody']))

            if 'spectrogram' in visualizations and visualizations['spectrogram']:
                 try:
                    spectrogram_image = Image(visualizations['spectrogram'], width=chart_width, height=(chart_width*0.4))
                    spectrogram_image.hAlign = 'CENTER'
                    story.append(spectrogram_image)
                    story.append(Paragraph("Voice Frequency Analysis (Spectrogram)", self.styles['ChartLabel']))
                    story.append(Spacer(1, 0.15*inch))
                 except Exception as e:
                    print(f"Error adding spectrogram: {e}")
                    story.append(Paragraph(f"<i>Error displaying spectrogram image.</i>", self.styles['PremiumBody']))

            # Add Pie and Radar after Waveform/Spectrogram if they exist
            add_chart_to_story(self.create_pie_chart, "Emotion Distribution Overview", emotion_result)
            # Make Radar slightly smaller
            try:
                radar_buffer = self.create_radar_chart(emotion_result)
                if radar_buffer:
                    radar_img = Image(radar_buffer, width=chart_width*0.7, height=chart_width*0.7)
                    radar_img.hAlign = 'CENTER'
                    story.append(radar_img)
                    story.append(Paragraph("Emotion Profile Radar Chart", self.styles['ChartLabel']))
                    story.append(Spacer(1, 0.15*inch))
            except Exception as e:
                 print(f"Error adding radar chart: {e}")
                 story.append(Paragraph(f"<i>Error generating Radar Chart.</i>", self.styles['PremiumBody']))


            # === Recording Details Section ===
            story.append(PageBreak())
            story.append(Paragraph("Recording Details & Analysis Parameters", self.styles['SectionHeader']))
            story.append(Spacer(1, 0.1*inch))

            if audio_data is not None:
                duration = len(audio_data) / 22050
                audio_features = self.extract_audio_features(audio_data)
                record_data = [
                    [Paragraph(h, self.styles['PremiumBody']) for h in ["Parameter", "Value", "Details"]],
                    [Paragraph("Duration", self.styles['PremiumBody']), Paragraph(f"{duration:.2f} s", self.styles['PremiumBody']), Paragraph("Total recording length", self.styles['PremiumBody'])],
                    [Paragraph("Sample Rate", self.styles['PremiumBody']), Paragraph("22.05 kHz", self.styles['PremiumBody']), Paragraph("Standard for voice analysis", self.styles['PremiumBody'])],
                    [Paragraph("Vocal Energy (RMS)", self.styles['PremiumBody']), Paragraph(f"{audio_features['energy']:.4f}", self.styles['PremiumBody']), Paragraph("Overall loudness", self.styles['PremiumBody'])],
                    [Paragraph("Zero Crossing Rate", self.styles['PremiumBody']), Paragraph(f"{audio_features['zcr']:.4f}", self.styles['PremiumBody']), Paragraph("Related to voice activity/noise", self.styles['PremiumBody'])],
                    [Paragraph("Spectral Centroid", self.styles['PremiumBody']), Paragraph(f"{audio_features['spectral_centroid']:.1f} Hz", self.styles['PremiumBody']), Paragraph("Indicator of voice brightness", self.styles['PremiumBody'])],
                ]
            else:
                 record_data = [
                    [Paragraph(h, self.styles['PremiumBody']) for h in ["Parameter", "Value", "Details"]],
                    [Paragraph("Audio Data", self.styles['PremiumBody']), Paragraph("Unavailable", self.styles['PremiumBody']), Paragraph("Features extracted before processing", self.styles['PremiumBody'])],
                 ]
            # Add common analysis details
            record_data.extend([
                    [Paragraph("Analysis Method", self.styles['PremiumBody']), Paragraph(emotion_result.get('model_used', 'Rule-Based').replace('_',' ').title(), self.styles['PremiumBody']), Paragraph("Algorithm used for prediction", self.styles['PremiumBody'])],
                    [Paragraph("Features Analyzed", self.styles['PremiumBody']), Paragraph(str(emotion_result.get('features_used', 'N/A')), self.styles['PremiumBody']), Paragraph("Number of acoustic features", self.styles['PremiumBody'])],
            ])

            record_table = Table(record_data, colWidths=[1.8*inch, 1.5*inch, available_width - 3.3*inch])
            record_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#5D6D7E')), # Greyish Header
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                 ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
            ]))
            story.append(record_table)
            story.append(Spacer(1, 0.2*inch))

            # === Executive Summary ===
            story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
            summary_text = f"""
            This emotional analysis for <b>{user_details.get('name', 'the user')}</b> identified 
            <b>{emotion_result['dominant_emotion']}</b> as the primary emotional state detected in the provided voice sample, 
            with a model confidence of <b>{emotion_result['confidence']:.1%}</b>. 
            The analysis employed {emotion_result.get('model_used', 'standard voice emotion recognition techniques').replace('_',' ').title()}, examining key vocal characteristics. 
            This report provides a snapshot of the emotional expression for awareness and potential further exploration.
            """
            story.append(Paragraph(summary_text, self.styles['PremiumBody']))
            story.append(Spacer(1, 0.2*inch))

            # === Footer Placeholder (will be added on each page by build) ===
            # story.append(Paragraph("© 2025 VoxSense Technologies", self.styles['FooterStyle']))

            # --- Build the PDF ---
            doc.build(story, onFirstPage=self._header_footer, onLaterPages=self._header_footer)

        except Exception as e:
            print(f"FATAL PDF Generation Error: {e}")
            traceback.print_exc()
            # If build fails, return None or raise error
            return None # Indicate failure

        buffer.seek(0)
        return buffer

    # --- Header and Footer ---
    def _header_footer(self, canvas, doc):
        canvas.saveState()
        # Footer
        footer = Paragraph("© 2025 VoxSense Emotional Intelligence | Confidential Report | www.voxsense.ai", self.styles['FooterStyle'])
        w, h = footer.wrap(doc.width, doc.bottomMargin)
        footer.drawOn(canvas, doc.leftMargin, h) # Draw at bottom
        # Add page number
        page_num = canvas.getPageNumber()
        canvas.setFont('Helvetica', 7)
        canvas.setFillColor(colors.HexColor('#95a5a6'))
        canvas.drawRightString(doc.width + doc.leftMargin, 0.5 * inch, f"Page {page_num}")

        canvas.restoreState()

    # --- Chart Generation Methods (with error handling) ---
    def _create_chart_base(self, create_func, *args, **kwargs):
        """Base function to create charts safely."""
        buffer = BytesIO()
        try:
            fig, ax = create_func(*args, **kwargs)
            if fig:
                fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white', transparent=False)
                plt.close(fig) # Close the figure
                buffer.seek(0)
                return buffer
            else:
                plt.close() # Close if fig wasn't returned properly
                return None
        except Exception as e:
            print(f"Error creating chart: {e}")
            traceback.print_exc()
            plt.close() # Ensure plot is closed on error
            return None # Return None on error

    def create_bar_chart(self, emotion_result):
        def plot():
            emotions = list(emotion_result['probabilities'].keys())
            probabilities = list(emotion_result['probabilities'].values())
            fig, ax = plt.subplots(figsize=(7, 3.5), facecolor='white') # Adjusted size
            colors_list = plt.cm.viridis(np.linspace(0, 1, len(emotions))) # Viridis colormap
            bars = ax.bar(emotions, probabilities, color=colors_list, edgecolor='grey', linewidth=0.5)

            for bar, value in zip(bars, probabilities):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., min(height + 0.02, 0.98), f'{value:.0%}', ha='center', va='bottom', fontsize=7, color='black')

            ax.set_ylabel('Probability', fontsize=8)
            ax.set_xlabel('Emotions', fontsize=8)
            ax.set_title('Emotion Probability Distribution', fontsize=10, pad=10)
            ax.set_ylim(0, 1.05)
            ax.tick_params(axis='x', labelsize=7, rotation=30, ha='right')
            ax.tick_params(axis='y', labelsize=7)
            ax.grid(True, axis='y', linestyle='--', alpha=0.6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            fig.tight_layout(pad=0.5)
            return fig, ax
        return self._create_chart_base(plot)

    def create_pie_chart(self, emotion_result):
         def plot():
            emotions = list(emotion_result['probabilities'].keys())
            probabilities = list(emotion_result['probabilities'].values())
            fig, ax = plt.subplots(figsize=(5, 4), facecolor='white') # Adjusted size
            colors_list = plt.cm.viridis(np.linspace(0, 1, len(emotions)))
            # Explode the largest slice slightly
            explode = [0.05 if prob == max(probabilities) else 0 for prob in probabilities]

            wedges, texts, autotexts = ax.pie(probabilities, labels=[e.title() for e in emotions], colors=colors_list,
                                              autopct='%1.1f%%', startangle=90, pctdistance=0.85,
                                              wedgeprops={'edgecolor': 'white', 'linewidth': 0.5},
                                              explode=explode, textprops={'fontsize': 7})
            for autotext in autotexts: autotext.set_color('white')
            # Draw circle for donut chart effect
            centre_circle = plt.Circle((0,0),0.70,fc='white')
            fig.gca().add_artist(centre_circle)

            ax.set_title('Emotion Distribution', fontsize=10, pad=10)
            fig.tight_layout(pad=0.5)
            return fig, ax
         return self._create_chart_base(plot)


    def create_radar_chart(self, emotion_result):
        def plot():
            emotions = list(emotion_result['probabilities'].keys())
            probabilities = list(emotion_result['probabilities'].values())
            N = len(emotions)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]
            values = probabilities + probabilities[:1]

            fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True), facecolor='white') # Adjusted size
            ax.plot(angles, values, 'o-', linewidth=1.5, color='#3498db', markersize=4)
            ax.fill(angles, values, alpha=0.25, color='#3498db')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([e.title() for e in emotions], fontsize=7)
            ax.set_yticks(np.arange(0.1, 1.1, 0.2))
            ax.set_yticklabels([f"{i:.0%}" for i in np.arange(0.1, 1.1, 0.2)], fontsize=6)
            ax.set_ylim(0, 1)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_title('Emotion Profile Radar', fontsize=10, pad=15)
            fig.tight_layout(pad=1.0) # More padding for polar
            return fig, ax
        return self._create_chart_base(plot)


    def extract_audio_features(self, audio_data, sr=22050):
        """Extract audio features safely."""
        features = {'energy': 0, 'zcr': 0, 'spectral_centroid': 0} # Default values
        try:
            if audio_data is not None and len(audio_data) > 0:
                features['energy'] = np.sqrt(np.mean(np.square(audio_data)))
                features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(y=audio_data))
                features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr))
        except Exception as e:
            print(f"Error extracting audio features: {e}")
        return features