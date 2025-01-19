import os
import time
import requests
import yt_dlp
from moviepy.editor import VideoFileClip
from dotenv import load_dotenv
import chardet
import fitz  # PyMuPDF
from flask import request, jsonify

# For Julep risk analysis
from julep import Julep

###########################################
# Load Environment Variables
###########################################
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

###########################################
# AssemblyAI API Endpoints and Headers
###########################################
UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"
AAI_HEADERS = {"authorization": ASSEMBLYAI_API_KEY}

###########################################
# Julep Client Initialization
###########################################
try:
    julep_client = Julep(api_key=JULEP_API_KEY)
    # Create a Julep agent for risk analysis.
    agent = julep_client.agents.create(
        name="Risk Analyzer",
        model="gpt-4o",
        about="Detects risk-related keywords and phrases in a transcript."
    )
    print("Julep Agent created successfully for risk analysis.")
except Exception as e:
    print(f"Error creating Julep Agent: {str(e)}")
    agent = None

###########################################
# Functions for Video Download, Audio Extraction,
# AssemblyAI Transcription, and Risk Analysis.
###########################################

def detect_platform(url):
    """Detect if the URL is from TikTok, Instagram, or X/Twitter based on the domain."""
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
    """Download video from TikTok, Instagram, or X/Twitter using yt-dlp."""
    platform = detect_platform(url)
    print(f"Detected platform: {platform}")
    
    # yt-dlp options; they work for multiple platforms
    ydl_opts = {
        'format': 'mp4',
        'outtmpl': output_filename,
        'noplaylist': True,
    }
    
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
    """Extract audio from a video file and save it as MP3."""
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
    """Upload a file to AssemblyAI and return the upload URL."""
    CHUNK_SIZE = 5242880  # 5MB

    def read_file(filename, chunk_size=CHUNK_SIZE):
        with open(filename, 'rb') as _file:
            while True:
                data = _file.read(chunk_size)
                if not data:
                    break
                yield data

    print(f"Uploading {filename} to AssemblyAI...")
    response = requests.post(UPLOAD_URL, headers=AAI_HEADERS, data=read_file(filename))
    if response.status_code != 200:
        raise Exception(f"Upload failed: {response.text}")
    upload_url = response.json()['upload_url']
    print("Upload completed!")
    return upload_url

def request_transcript(audio_url):
    """Request a transcription job from AssemblyAI."""
    json_data = {
        "audio_url": audio_url,
        # Uncomment the next line if you wish to enable speaker diarization:
        # "speaker_labels": True
    }
    response = requests.post(TRANSCRIPT_URL, json=json_data, headers=AAI_HEADERS)
    if response.status_code != 200:
        raise Exception(f"Transcription request failed: {response.text}")
    transcript_id = response.json()['id']
    return transcript_id

def poll_transcript(transcript_id, polling_interval=5):
    """Poll AssemblyAI until transcription is complete."""
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
    """Save the full transcript and, if available, speaker breakdown to a text file."""
    with open(output_textfile, 'w', encoding='utf-8') as f:
        f.write("Full Transcript:\n")
        f.write(transcript_json.get("text", "") + "\n\n")
        if transcript_json.get("utterances"):
            f.write("Speaker Breakdown:\n")
            for utterance in transcript_json["utterances"]:
                f.write(f"Speaker {utterance.get('speaker', 'N/A')}: {utterance.get('text', '')}\n")
    print(f"Transcript saved to {output_textfile}")

def analyze_risks_with_ai(transcript_text, output_file="risk_report.txt"):
    """
    Use the AI (via Julep) to analyze the transcript and identify risk-related keywords/phrases.
    Returns a risk report as generated by the AI.
    """
    if not transcript_text:
        print("No transcript text available for AI risk analysis.")
        return ""

    if agent is None:
        print("Julep Agent not initialized. Cannot perform AI risk analysis.")
        return ""

    # Prepare a prompt that instructs the AI to find and list risk-related words/phrases.
    # You can adjust the prompt to suit your specific requirements.
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
            main=[
                {
                    "prompt": [
                        {"role": "system", "content": "You are an assistant that identifies risk indicators in transcripts of spoken content."},
                        {"role": "user", "content": prompt}
                    ],
                    "return": {"result": "Risk Analysis Report."}
                }
            ]
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

def analyze_public_records(file_path):
    """
    Analyze the uploaded file for public records using Julep.
    This updated version instructs the AI to identify the two companies
    that are parties to a contract agreement.
    """
    try:
        # Read the file contents (PDFs handled differently from text files)
        if file_path.lower().endswith('.pdf'):
            import fitz  # PyMuPDF
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
        
        # Updated prompt for company extraction:
        prompt = f"""
        You are an expert in contract analysis. Based on the text excerpt provided below from a contract agreement, give the 2 companies that are parties to the agreement. Return your answer as exactly 2 separate lines, each line containing one company name.
        
        Text:
        {file_content}
        """
        
        task = julep_client.tasks.create(
            agent_id=agent.id,
            name="Contract Companies Extraction",
            description="Extract the two companies that are parties to the contract agreement.",
            main=[{
                "prompt": [
                    {"role": "system", "content": "You are an assistant specialized in contract analysis."},
                    {"role": "user", "content": prompt}
                ],
                "return": {"result": "Two Companies Involved in the Contract Agreement."}
            }]
        )
        
        print("Julep contract extraction task created. Waiting for result...")
        
        execution = julep_client.executions.create(task_id=task.id, input={})
        while True:
            result = julep_client.executions.get(execution.id)
            if result.status in ["succeeded", "failed"]:
                break
            print("Processing contract extraction...")
            time.sleep(1)
            
        if result.status == "succeeded":
            if "choices" in result.output and len(result.output["choices"]) > 0:
                extracted_text = result.output["choices"][0]["message"]["content"]
                companies = [line.strip() for line in extracted_text.splitlines() if line.strip()]
                companies = companies[:2]  # return at most 2 companies
                summary = "Contract companies extracted successfully."
                return summary, companies
            else:
                return "No companies found.", []
        else:
            return "Company extraction failed.", []
    
    except Exception as e:
        print(f"Error during public records analysis: {e}")
        return "An error occurred during analysis.", []


def format_public_record_output(output):
    """Format the public record output for better readability."""
    # Here you can implement your formatting logic
    # For example, you can use HTML to format the output
    formatted = "<h2>Public Records Analysis Report</h2>"
    formatted += "<p>" + output.replace("\n", "<br>") + "</p>"
    return formatted

def extract_company_names(text):
    """Extract company names from the provided text."""
    # Implement your logic to extract company names using regex or NLP libraries
    import re

    # Example regex pattern to find company names (this is a placeholder and may need adjustment)
    pattern = r'\b[A-Z][a-zA-Z0-9&.\' ]+\b'
    company_names = re.findall(pattern, text)

    # Remove duplicates and return the list
    return list(set(company_names))  # Return unique company names

###########################################
# Main Function: Execute All Steps
###########################################

def main():
    url = input("Enter the TikTok, Instagram, or X/Twitter video URL: ").strip()
    try:
        # Step 1: Download the video.
        video_file = download_video(url)
        if not video_file:
            return

        # Step 2: Extract audio from the video.
        audio_file = extract_audio(video_file)
        if not audio_file:
            return

        # Step 3: Upload audio file to AssemblyAI.
        audio_url = upload_file(audio_file)

        # Step 4: Request transcription from AssemblyAI.
        transcript_id = request_transcript(audio_url)
        print(f"Transcription requested. Transcript ID: {transcript_id}")

        # Step 5: Poll until the transcription is complete.
        transcript_json = poll_transcript(transcript_id)

        # Step 6: Save the transcript to a text file.
        transcript_filename = "transcript.txt"
        save_transcript_text(transcript_json, transcript_filename)

        # Read transcript text from file.
        try:
            with open(transcript_filename, 'r', encoding='utf-8') as f:
                transcript_text = f.read()
        except Exception as e:
            print(f"Error reading transcript file {transcript_filename}: {e}")
            transcript_text = ""

        # Step 7: Use the AI to analyze the transcript for risk-related keywords/phrases.
        risk_report = analyze_risks_with_ai(transcript_text)
        print("AI-Generated Risk Analysis Report:")
        print(risk_report)
        
    except Exception as e:
        print("An error occurred:", e)

def analyze_company():
    """Analyze news articles for a company using Julep."""
    company_name = request.form.get('company')
    if not company_name:
        return jsonify({'error': 'No company name provided'}), 400

    # Your logic to analyze the company goes here
    # For example, you might want to set risk_report based on your analysis
    risk_report = "Sample risk report based on analysis."

    return jsonify({
        'success': True,
        'risk_report': risk_report
    })

def build_search_query(company_name: str, legal_keywords: list) -> str:
    """
    Build a search query string using the company name and a list of legal-related keywords.
    """
    keywords_query = " OR ".join(legal_keywords)
    full_query = f'"{company_name}" ({keywords_query})'
    return full_query

def search_company_news(query: str):
    """
    Perform a Google News search using SerpApi with the provided query.
    """
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_news",
        "q": query,
        "gl": "us",     # Country: United States; adjust if needed
        "hl": "en",     # Language: English; adjust if needed
        "api_key": SERPAPI_KEY
    }
    
    print(f"Searching news for query: {query}")
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print(f"Error: Received status code {response.status_code}")
        return None
    
    return response.json()

def display_news_results(data):
    """
    Display key details from the news_results in the API response.
    """
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
        results.append("")  # Add a blank line for spacing

    return "\n".join(results)  # Join the results into a single string

if __name__ == "__main__":
    main()
