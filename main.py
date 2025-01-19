import os
import time
import requests
import yt_dlp
from moviepy.editor import VideoFileClip
from dotenv import load_dotenv
import chardet
import fitz  # PyMuPDF
from flask import request, jsonify
from julep import Julep

load_dotenv()
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
JULEP_API_KEY = os.getenv("JULEP_API_KEY")
SERPAPI_KEY = os.getenv('SERPAPI_KEY')
if not ASSEMBLYAI_API_KEY:
    raise ValueError("Please set the ASSEMBLYAI_API_KEY in your .env file")
if not JULEP_API_KEY:
    raise ValueError("Please set the JULEP_API_KEY in your .env file")
if not SERPAPI_KEY:
    raise ValueError("Please set the SERPAPI_KEY in your .env file")

UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"
AAI_HEADERS = {"authorization": ASSEMBLYAI_API_KEY}

try:
    julep_client = Julep(api_key=JULEP_API_KEY)
    agent = julep_client.agents.create(
        name="Risk Analyzer",
        model="o1-preview",
        about="Detects risk-related keywords and phrases in a transcript."
    )
    print("Julep Agent created successfully for risk analysis.")
except Exception as e:
    print(f"Error creating Julep Agent: {str(e)}")
    agent = None

def detect_platform(url):
    url_lower = url.lower()
    if "tiktok.com" in url_lower:
        return "TikTok"
    elif "instagram.com" in url_lower:
        return "Instagram"
    elif "twitter.com" in url_lower or "x.com" in url_lower:
        return "X/Twitter"
    else:
        raise ValueError("Unsupported URL domain. Please provide a TikTok, Instagram, or Twitter URL.")

def download_video(url, output_filename="video.mp4"):
    platform = detect_platform(url)
    print(f"Detected platform: {platform}")
    ydl_opts = {'format': 'mp4', 'outtmpl': output_filename, 'noplaylist': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Downloading video from: {url}")
            ydl.download([url])
            print("Video download completed!")
        return output_filename
    except Exception as e:
        print("An error occurred during video download:", e)
        return None

def extract_audio(video_filename, audio_filename="audio.mp3"):
    try:
        clip = VideoFileClip(video_filename)
        clip.audio.write_audiofile(audio_filename)
        clip.close()
        print(f"Audio extracted to {audio_filename}")
        return audio_filename
    except Exception as e:
        print("Error extracting audio:", e)
        return None

def upload_file(filename):
    CHUNK_SIZE = 5242880
    def read_file(filename, chunk_size=CHUNK_SIZE):
        with open(filename, 'rb') as _file:
            while True:
                data = _file.read(chunk_size)
                if not data: break
                yield data
    print(f"Uploading {filename} to AssemblyAI...")
    response = requests.post(UPLOAD_URL, headers=AAI_HEADERS, data=read_file(filename))
    if response.status_code != 200:
        raise Exception(f"Upload failed: {response.text}")
    upload_url = response.json()['upload_url']
    print("Upload completed!")
    return upload_url

def request_transcript(audio_url):
    json_data = { "audio_url": audio_url }
    response = requests.post(TRANSCRIPT_URL, json=json_data, headers=AAI_HEADERS)
    if response.status_code != 200:
        raise Exception(f"Transcription request failed: {response.text}")
    transcript_id = response.json()['id']
    return transcript_id

def poll_transcript(transcript_id, polling_interval=5):
    polling_url = f"{TRANSCRIPT_URL}/{transcript_id}"
    while True:
        response = requests.get(polling_url, headers=AAI_HEADERS)
        if response.status_code != 200:
            raise Exception(f"Error polling transcript: {response.text}")
        status = response.json()['status']
        if status == 'completed':
            print("Transcription completed!")
            return response.json()
        elif status == 'error':
            raise Exception(f"Transcription failed: {response.json().get('error')}")
        else:
            print(f"Transcription status: {status}. Waiting {polling_interval} seconds...")
            time.sleep(polling_interval)

def save_transcript_text(transcript_json, output_textfile="transcript.txt"):
    with open(output_textfile, 'w', encoding='utf-8') as f:
        f.write("Full Transcript:\n")
        f.write(transcript_json.get("text", "") + "\n\n")
        if transcript_json.get("utterances"):
            f.write("Speaker Breakdown:\n")
            for utterance in transcript_json["utterances"]:
                f.write(f"Speaker {utterance.get('speaker', 'N/A')}: {utterance.get('text', '')}\n")
    print(f"Transcript saved to {output_textfile}")

def analyze_risks_with_ai(transcript_text, output_file="risk_report.txt"):
    if not transcript_text:
        print("No transcript text available for AI risk analysis.")
        return ""
    if agent is None:
        print("Julep Agent not initialized. Cannot perform AI risk analysis.")
        return ""
    prompt = f"""
    Analyze the following transcript and identify any risk-related keywords or phrases relevant to reputation, legal issues, financial instability, or controversies. For each item you identify, provide a brief explanation of why it might be a risk indicator.

    Transcript:
    {transcript_text}

    Return the results as a numbered list.
    """
    try:
        task = julep_client.tasks.create(
            agent_id=agent.id,
            name="AI Risk Analysis",
            description="Analyze transcript for risk-related keywords and phrases.",
            main=[{
                "prompt": [
                    {"role": "system", "content": "You are an assistant that identifies risk indicators in transcripts of spoken content."},
                    {"role": "user", "content": prompt}
                ],
                "return": {"result": "Risk Analysis Report."}
            }]
        )
        print("Julep risk analysis task created. Waiting for result...")
        execution = julep_client.executions.create(task_id=task.id, input={})
        while True:
            result = julep_client.executions.get(execution.id)
            if result.status in ["succeeded", "failed"]:
                break
            print("Risk analysis processing...")
            time.sleep(1)
        if result.status == "succeeded":
            if "choices" in result.output and len(result.output["choices"]) > 0:
                risk_report = result.output["choices"][0]["message"]["content"]
                print("Risk analysis succeeded.")
            else:
                print("No 'choices' found in the Julep output.")
                return ""
        else:
            print(f"Julep task failed: {result.error}")
            return ""
    except Exception as e:
        print(f"Error during AI risk analysis with Julep: {e}")
        return ""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(risk_report)
        print(f"Risk analysis report saved to {output_file}")
    except Exception as e:
        print(f"Error saving risk report to {output_file}: {e}")
    return risk_report

def analyze_public_records(file_path, analysis="company"):
    """
    Analyze the uploaded file for public records.
    If analysis="company", extract company names.
    If analysis="risk_single", perform risk analysis on the document (e.g. checking for bankruptcy, fraud, etc.)
    for one company.
    """
    try:
        # Read file contents.
        if file_path.lower().endswith('.pdf'):
            import fitz
            document = fitz.open(file_path)
            file_content = ""
            for page in document:
                file_content += page.get_text()
            document.close()
        else:
            import chardet
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                encoding = chardet.detect(raw_data)['encoding']
            with open(file_path, 'r', encoding=encoding) as f:
                file_content = f.read()
        
        if analysis == "company":
            # Original prompt for company extraction (if needed)
            prompt = f"""
            You are an expert in public case analysis. Based on the text excerpt provided below from a public record, identify the companys involved in the case. and just output the companys names
            
            Text:
            {file_content}
            """
            task_name = "Public Record Company Extraction"
        elif analysis == "risk_single":
            # New prompt for risk analysis on public cases:
            prompt = f"""
            You are an expert in analyzing public case documents.
            Analyze the following document excerpt for risk-related indicators 
            (such as signs of bankruptcy, fraud, legal investigations, or other controversies) that might affect the reputation or financial stability of the company.
            Provide a detailed risk analysis report that summarizes the main risk factors.
            
            Document:
            {file_content}
            """
            task_name = "Public Record Risk Analysis"
        
        task = julep_client.tasks.create(
            agent_id=agent.id,
            name=task_name,
            description=task_name,
            main=[{
                "prompt": [
                    {"role": "system", "content": "You are an assistant specializing in public case risk analysis."},
                    {"role": "user", "content": prompt}
                ],
                "return": {"result": "Risk Analysis Report."}
            }]
        )
        
        print(f"Julep task '{task_name}' created. Waiting for result...")
        execution = julep_client.executions.create(task_id=task.id, input={})
        while True:
            result = julep_client.executions.get(execution.id)
            if result.status in ["succeeded", "failed"]:
                break
            print("Processing...")
            time.sleep(1)
            
        if result.status == "succeeded":
            if "choices" in result.output and len(result.output["choices"]) > 0:
                output_text = result.output["choices"][0]["message"]["content"]
                if analysis == "company":
                    # In company mode, we return two company names (if any were extracted).
                    companies = [line.strip() for line in output_text.splitlines() if line.strip()]
                    companies = companies[:2]
                    summary = "Public record processed successfully."
                    return summary, companies
                else:
                    # In risk analysis mode, output the risk report.
                    summary = "Risk analysis completed successfully."
                    return summary, output_text
            else:
                if analysis == "company":
                    return "No companies found.", []
                else:
                    return "No risk indicators found.", ""
        else:
            if analysis == "company":
                return "Company extraction failed.", []
            else:
                return "Risk analysis failed.", ""
    
    except Exception as e:
        print(f"Error during public records analysis: {e}")
        if analysis == "company":
            return "An error occurred during company extraction.", []
        else:
            return "An error occurred during risk analysis.", ""


def build_search_query(company_name: str, legal_keywords: list) -> str:
    keywords_query = " OR ".join(legal_keywords)
    full_query = f'"{company_name}" ({keywords_query})'
    return full_query

def search_company_news(query: str):
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_news",
        "q": query,
        "gl": "us",
        "hl": "en",
        "api_key": SERPAPI_KEY
    }
    print(f"Searching news for query: {query}")
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Error: Received status code {response.status_code}")
        return None
    return response.json()

def display_news_results(data):
    if "news_results" not in data:
        return "No news results found."
    news_results = data["news_results"]
    results = []
    for i, result in enumerate(news_results, start=1):
        title = result.get("title", "No title")
        link = result.get("link", "No link provided")
        date = result.get("date", "No date provided")
        source = result.get("source", {}).get("name", "Unknown source")
        results.append(f"Result {i}:")
        results.append(f"  Title : {title}")
        results.append(f"  Source: {source}")
        results.append(f"  Date  : {date}")
        results.append(f"  Link  : {link}")
        results.append("")
    return "\n".join(results)
    
def extract_company_names(text):
    import re
    pattern = r'\b[A-Z][a-zA-Z0-9&.\' ]+\b'
    company_names = re.findall(pattern, text)
    return list(set(company_names))

if __name__ == "__main__":
    url = input("Enter the TikTok, Instagram, or X/Twitter video URL: ").strip()
    try:
        video_file = download_video(url)
        if not video_file:
            exit()
        audio_file = extract_audio(video_file)
        if not audio_file:
            exit()
        audio_url = upload_file(audio_file)
        transcript_id = request_transcript(audio_url)
        print(f"Transcription requested. Transcript ID: {transcript_id}")
        transcript_json = poll_transcript(transcript_id)
        transcript_filename = "transcript.txt"
        save_transcript_text(transcript_json, transcript_filename)
        try:
            with open(transcript_filename, 'r', encoding='utf-8') as f:
                transcript_text = f.read()
        except Exception as e:
            print(f"Error reading transcript file {transcript_filename}: {e}")
            transcript_text = ""
        risk_report = analyze_risks_with_ai(transcript_text)
        print("AI-Generated Risk Analysis Report:")
        print(risk_report)
    except Exception as e:
        print("An error occurred:", e)
