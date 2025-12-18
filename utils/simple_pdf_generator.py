from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
from datetime import datetime

class SimplePDFGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
    
    def generate_report(self, user_details, emotion_result, audio_data):
        """Generate professional PDF report"""
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch)
        story = []
        
        # Title Section
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=20,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=20,
            alignment=1
        )
        
        title = Paragraph("VoxSense Emotion Analysis Report", title_style)
        story.append(title)
        
        subtitle = Paragraph("Emotion Detection through Voice Analysis", self.styles['Heading2'])
        story.append(subtitle)
        story.append(Spacer(1, 0.3*inch))
        
        # Main Result Highlight
        result_text = f"Primary Emotion: <b>{emotion_result['dominant_emotion']}</b> - Confidence: <b>{emotion_result['confidence']:.1%}</b>"
        result_para = Paragraph(result_text, self.styles['Heading2'])
        story.append(result_para)
        story.append(Spacer(1, 0.4*inch))
        
        # User Information
        story.append(Paragraph("User Information", self.styles['Heading3']))
        
        user_data = [
            ["Field", "Details"],
            ["Full Name", user_details.get('name', 'N/A')],
            ["Age", str(user_details.get('age', 'N/A'))],
            ["Gender", user_details.get('gender', 'N/A')],
            ["Email", user_details.get('email', 'N/A')],
            ["Contact", user_details.get('contact', 'N/A')],
            ["Occupation", user_details.get('occupation', 'N/A')],
            ["Analysis Date", datetime.now().strftime("%Y-%m-%d %H:%M")]
        ]
        
        user_table = Table(user_data, colWidths=[1.8*inch, 4.2*inch])
        user_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        story.append(user_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Detailed Analysis
        story.append(Paragraph("Emotion Analysis Results", self.styles['Heading3']))
        
        # Emotion Probabilities Table
        prob_data = [["Emotion", "Probability", "Confidence Level"]]
        for emotion, prob in emotion_result['probabilities'].items():
            confidence_level = "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
            prob_data.append([emotion, f"{prob:.1%}", confidence_level])
        
        prob_table = Table(prob_data, colWidths=[1.8*inch, 1.8*inch, 2.4*inch])
        prob_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#764ba2')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        story.append(prob_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Audio Information
        story.append(Paragraph("Technical Details", self.styles['Heading3']))
        
        if audio_data is not None:
            duration = len(audio_data) / 22050
            audio_info = [
                ["Parameter", "Value"],
                ["Audio Duration", f"{duration:.2f} seconds"],
                ["Sample Rate", "22.05 kHz"],
                ["Channels", "Mono"],
                ["Analysis Method", "Advanced Feature Extraction"],
                ["Features Analyzed", "MFCC, Spectral, Pitch, Energy"],
                ["Model Type", "Rule-based Emotion Classification"]
            ]
        else:
            audio_info = [
                ["Parameter", "Value"],
                ["Status", "Audio analysis completed"],
                ["Analysis Method", "Advanced Feature Extraction"]
            ]
        
        audio_table = Table(audio_info, colWidths=[2.2*inch, 3.8*inch])
        audio_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#17a2b8')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#e3f2fd')),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        story.append(audio_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Visualization Chart
        story.append(Paragraph("Emotion Distribution", self.styles['Heading3']))
        
        # Create bar chart
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            emotions = list(emotion_result['probabilities'].keys())
            probabilities = list(emotion_result['probabilities'].values())
            
            colors_list = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFD93D', '#964B00', '#FFA500']
            bars = ax.bar(emotions, probabilities, color=colors_list, edgecolor='black', linewidth=0.5)
            
            # Add value labels on bars
            for bar, value in zip(bars, probabilities):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.1%}', ha='center', va='bottom', fontsize=9)
            
            ax.set_ylabel('Probability', fontweight='bold')
            ax.set_title('Emotion Detection Probabilities', fontweight='bold')
            ax.set_ylim(0, 1)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Convert to image
            chart_buffer = BytesIO()
            plt.savefig(chart_buffer, format='png', dpi=150, bbox_inches='tight')
            chart_buffer.seek(0)
            plt.close()
            
            chart_image = Image(chart_buffer, width=5*inch, height=3*inch)
            story.append(chart_image)
        except Exception as e:
            error_msg = Paragraph(f"Chart unavailable: {str(e)}", self.styles['Normal'])
            story.append(error_msg)
        
        story.append(Spacer(1, 0.3*inch))
        
        # Footer
        story.append(Spacer(1, 0.5*inch))
        footer_text = f"""
        <para alignment="center">
        <b>VoxSense Emotion Intelligence Platform</b><br/>
        Thank you for using VoxSense - Emotion Speaks Louder Than Words ❤️<br/>
        Report generated on: {datetime.now().strftime("%Y-%m-%d at %H:%M:%S")}<br/>
        Report ID: VS{datetime.now().strftime("%Y%m%d%H%M%S")}<br/>
        © 2025 VoxSense Technologies | Confidential Report
        </para>
        """
        
        footer_style = ParagraphStyle(
            'FooterStyle',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=1
        )
        
        footer = Paragraph(footer_text, footer_style)
        story.append(footer)
        
        doc.build(story)
        buffer.seek(0)
        return buffer