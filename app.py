# app.py
from flask import Flask, render_template, request, jsonify, Response
from main import (
    download_video,
    extract_audio,
    upload_file,
    request_transcript,
    poll_transcript,
    analyze_risks_with_ai,
    analyze_public_records,
    search_company_news,
    build_search_query,
    display_news_results
)
import os
import tempfile
import json
import time
import requests
from dotenv import load_dotenv
from julep import Julep

# --- Updated function to ensure directory exists before saving the transcript ---
def save_transcript_text(transcript_json, output_textfile="results/transcript.txt"):
    # Ensure the directory exists
    directory = os.path.dirname(output_textfile)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(output_textfile, 'w', encoding='utf-8') as f:
        f.write("Full Transcript:\n")
        f.write(transcript_json.get("text", "") + "\n\n")
        if transcript_json.get("utterances"):
            f.write("Speaker Breakdown:\n")
            for utterance in transcript_json["utterances"]:
                f.write(f"Speaker {utterance.get('speaker', 'N/A')}: {utterance.get('text', '')}\n")
    print(f"Transcript saved to {output_textfile}")
# --- End updated function ---

app = Flask(__name__)
progress_status = {}

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

load_dotenv()
JULEP_API_KEY = os.getenv("JULEP_API_KEY")

# Initialize Julep Client for risk analysis.
try:
    julep_client = Julep(api_key=JULEP_API_KEY)
    agent = julep_client.agents.create(
        name="Risk Analyzer",
        model="gpt-4o",
        about="Detects risk-related keywords and phrases in a transcript."
    )
    print("Julep Agent created successfully for risk analysis.")
except Exception as e:
    print(f"Error creating Julep Agent: {str(e)}")
    agent = None

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/media')
def media():
    return render_template('media.html')

@app.route('/public', methods=['GET', 'POST'])
def public():
    """
    Handles file uploads for public records analysis.
    When analysisType is 'company', it extracts company names.
    When analysisType is 'risk_single', it performs risk analysis for a single company
    (public cases such as bankruptcy, fraud, etc.), rather than contract extraction.
    """
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            analysis_type = request.form.get('analysisType')
            if analysis_type == 'company':
                summary, company_names = analyze_public_records(file_path)
                os.remove(file_path)
                return jsonify({
                    'summary': f"<h2>Company Name Extraction Summary</h2><p>{summary}</p>",
                    'company_names': company_names,
                    'risk_reports': []
                })
            elif analysis_type == 'risk_single':
                # For a single company risk analysis based on a public case document.
                # Adjust the prompt in analyze_public_records to perform risk analysis.
                summary, risk_report = analyze_public_records(file_path, analysis="risk_single")
                os.remove(file_path)
                return jsonify({
                    'summary': f"<h2>Risk Analysis Summary</h2><p>{summary}</p>",
                    'risk_report': risk_report,
                    'company_names': []  # Not needed in this mode
                })
    return render_template('public.html')


@app.route('/company', methods=['POST'])
def company_search():
    try:
        data = request.get_json()
        companies = data.get('companies', [])
        if not companies:
            return jsonify({'error': 'No company names provided.'}), 400
        risk_reports = []
        for company_name in companies:
            query = build_search_query(company_name, ["bankruptcy", "lawsuit", "fraud"])
            data = search_company_news(query)
            if data and "news_results" in data:
                formatted_news = display_news_results(data)
                risk_report = analyze_risks_with_ai(formatted_news)
                risk_reports.append({
                    'company_name': company_name,
                    'risk_report': risk_report,
                    'news_results': formatted_news
                })
            else:
                risk_reports.append({
                    'company_name': company_name,
                    'risk_report': 'No news results found.',
                    'news_results': ''
                })
        return jsonify({'success': True, 'risk_reports': risk_reports})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process', methods=['POST'])
def process_video():
    try:
        url = request.form.get('url')
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
        with tempfile.TemporaryDirectory() as temp_dir:
            progress_status['status'] = 'downloading'
            progress_status['message'] = 'Downloading video...'
            progress_status['progress'] = 10
            video_file = os.path.join(temp_dir, 'video.mp4')
            if not download_video(url, video_file):
                raise Exception('Video download failed')
            progress_status['status'] = 'processing_audio'
            progress_status['message'] = 'Extracting audio...'
            progress_status['progress'] = 30
            audio_file = os.path.join(temp_dir, 'audio.mp3')
            if not extract_audio(video_file, audio_file):
                raise Exception('Audio extraction failed')
            progress_status['status'] = 'transcribing'
            progress_status['message'] = 'Transcribing audio...'
            progress_status['progress'] = 50
            audio_url = upload_file(audio_file)
            transcript_id = request_transcript(audio_url)
            transcript_json = poll_transcript(transcript_id)
            progress_status['status'] = 'analyzing'
            progress_status['message'] = 'Analyzing risks...'
            progress_status['progress'] = 80
            transcript_filename = os.path.join('results', 'transcript.txt')
            save_transcript_text(transcript_json, transcript_filename)
            with open(transcript_filename, 'r', encoding='utf-8') as f:
                transcript_text = f.read()
            risk_report = analyze_risks_with_ai(transcript_text).replace("**", "")
            progress_status['status'] = 'completed'
            progress_status['message'] = 'Analysis complete!'
            progress_status['progress'] = 100
            return jsonify({
                'success': True,
                'transcript': transcript_text,
                'risk_report': risk_report
            })
    except Exception as e:
        progress_status['status'] = 'error'
        progress_status['message'] = str(e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
