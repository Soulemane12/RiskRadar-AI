# app.py
from flask import Flask, render_template, request, jsonify, Response
from main import (
    download_video,
    extract_audio,
    upload_file,
    request_transcript,
    poll_transcript,
    save_transcript_text,
    analyze_risks_with_ai,
    analyze_public_records,
    search_company_news,
    build_search_query,
    display_news_results  # imported for use in company search
)
import os
import tempfile
import json
import time
import requests
from dotenv import load_dotenv
from julep import Julep

app = Flask(__name__)
progress_status = {}

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load Environment Variables
load_dotenv()
JULEP_API_KEY = os.getenv("JULEP_API_KEY")

# Julep Client Initialization for risk analysis
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
    Handles file upload and (if needed) company search. 
    Note: Company searches are now directed to the /company endpoint.
    """
    if request.method == 'POST':
        if 'file' in request.files:
            # Handle file upload for public records analysis.
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400

            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            analysis_type = request.form.get('analysisType')  # Get the selected analysis type

            if analysis_type == 'company':
                summary, company_names = analyze_public_records(file_path)
                os.remove(file_path)
                return jsonify({
                    'summary': f"<h2>Company Name Extraction Summary</h2><p>{summary}</p>",
                    'company_names': company_names,  # Return company names
                    'risk_reports': []  # No risk reports for this option
                })
            elif analysis_type == 'risk':
                summary, company_names = analyze_public_records(file_path)
                risk_reports = []
                for company in company_names:
                    risk_report = analyze_risks_with_ai(company)  # Assuming this function takes a company name
                    risk_reports.append(f"Risk report for {company}: {risk_report}")

                os.remove(file_path)
                return jsonify({
                    'summary': f"<h2>Risk Analysis Summary</h2><p>{summary}</p>",
                    'company_names': company_names,  # Return company names
                    'risk_reports': risk_reports  # Return risk reports for each company
                })

    return render_template('public.html')  # GET: render the public page

@app.route('/company', methods=['POST'])
def company_search():
    try:
        data = request.get_json()  # Parse JSON body
        companies = data.get('companies', [])  # Extract 'companies' list

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

def analyze_risks_with_julep(data):
    """
    Analyze the news results using Julep for risk assessment.
    """
    if "news_results" not in data:
        return "No news results found."

    news_results = data["news_results"]
    if not news_results:
        return "No news results to analyze."

    prompt = "Analyze the following news articles for risk-related keywords or phrases:\n\n"
    for result in news_results:
        title = result.get("title", "No title")
        link = result.get("link", "No link provided")
        prompt += f"Title: {title}\nLink: {link}\n\n"
    prompt += "Return a risk assessment report."

    try:
        task = julep_client.tasks.create(
            agent_id=agent.id,
            name="Risk Analysis for Company News",
            description="Analyze news articles for risk-related keywords and phrases.",
            main=[{
                "prompt": [
                    {"role": "system", "content": "You are an assistant that identifies risk indicators in news articles."},
                    {"role": "user", "content": prompt}
                ],
                "return": {"result": "Risk Analysis Report."}
            }]
        )
        execution = julep_client.executions.create(task_id=task.id, input={})
        while True:
            result = julep_client.executions.get(execution.id)
            if result.status in ["succeeded", "failed"]:
                break

        if result.status == "succeeded":
            if "choices" in result.output and len(result.output["choices"]) > 0:
                return result.output["choices"][0]["message"]["content"]
            else:
                return "No risk indicators found."
        else:
            return f"Risk analysis failed: {result.error}"
    except Exception as e:
        return f"Error during risk analysis with Julep: {e}"

if __name__ == '__main__':
    app.run(debug=True)
