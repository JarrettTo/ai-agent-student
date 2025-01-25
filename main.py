from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
import pyaudio
from dotenv import load_dotenv
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from audio import transcribe_audio
from flask import Flask, request, jsonify
from vosk import Model, KaldiRecognizer
from threading import Thread
import os
from flask_cors import CORS
from google.cloud import speech
import wave
import requests
import speech_recognition as sr
import json
import signal
import sys

load_dotenv()
EMAIL = os.getenv("EMAIL")
PASSWORD = os.getenv("PASSWORD")
GMEET_LINK = os.getenv("GMEET_LINK")

# Configure Google Cloud Speech-to-Text

app = Flask(__name__)
CORS(app)
# Initialize Google Cloud Speech-to-Text client
speech_client = speech.SpeechClient()
model_path = "model"  # Path to your VOSK model directory
if not os.path.exists(model_path):
    raise ValueError("VOSK model not found. Download a model from https://alphacephei.com/vosk/models and extract it to the 'model' directory.")

# Load the model once at startup
#vosk_model = Model(model_path)

recording_frames = []
is_running = True

audio_buffer = b""
buffer_threshold = 16000 * 6  # Process ~2 seconds of audio (16kHz * 2 seconds)
def process_audio_with_google_streaming():
    """
    Process audio using Google Speech-to-Text Streaming API for real-time transcription.
    """
    global audio_buffer, is_running

    try:
        # Configure Google Speech-to-Text
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,  # Match your audio rate
            language_code="en-US",  # Change this to your desired language
            enable_automatic_punctuation=True,  # Optional: punctuation in output
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True,  # Get partial results
        )

        # Streaming generator function
        def audio_generator():
            global audio_buffer
            while is_running:
                if audio_buffer:
                    # Send the buffered audio
                    yield speech.StreamingRecognizeRequest(audio_content=audio_buffer)
                    audio_buffer = b""  # Clear the buffer after sending
                else:
                    # Wait a short time to avoid excessive looping
                    time.sleep(0.1)

        # Debugging: Ensure the generator is yielding properly
        def debug_generator():
            for request in audio_generator():
                print("Debug: Sending audio chunk to Google STT")
                yield request

        # Start streaming recognition
        responses = speech_client.streaming_recognize(streaming_config, debug_generator())

        # Process responses
        transcript = ""
        for response in responses:
            if not response.results:
                continue

            # Get the first result
            result = response.results[0]

            if result.is_final:  # Process finalized results
                transcript += result.alternatives[0].transcript + " "
                print(f"Transcript: {transcript.strip()}")

                # Append to transcript.txt
                with open("transcript.txt", "a") as transcript_file:
                    transcript_file.write(transcript.strip() + " ")

        return {"status": "Streaming transcription complete", "transcript": transcript.strip()}
    except Exception as e:
        print(f"Error in Google Streaming STT transcription: {e}")
        return {"error": str(e)}
def process_audio_with_vosk(audio_data):
    """
    Process audio using VOSK for offline speech-to-text transcription with buffered audio.
    """
    global audio_buffer

    try:
        # Append incoming audio data to the buffer
        audio_buffer += audio_data

        # Only process if the buffer exceeds the threshold
        if len(audio_buffer) >= buffer_threshold:
            recognizer = KaldiRecognizer(vosk_model, 16000)  # 16kHz sample rate
            transcript = ""

            # Feed the buffered audio data to the recognizer
            if recognizer.AcceptWaveform(audio_buffer):
                result = json.loads(recognizer.Result())
                if result.get("confidence", 0) >= 0.7:  # Adjust threshold as needed
                    transcript += result.get("text", "") + " "
            else:
                partial_result = json.loads(recognizer.PartialResult())
                transcript += partial_result.get("partial", "") + " "

            # Clear the buffer after processing
            audio_buffer = b""

            if transcript.strip():
                print(f"Transcript: {transcript.strip()}")

                # Append transcript to a file
                with open("transcript.txt", "a") as transcript_file:
                    transcript_file.write(transcript + " ")

                return {"status": "Audio processed", "transcript": transcript.strip()}

        return {"status": "Buffering audio..."}
    except Exception as e:
        print(f"Error in VOSK transcription: {e}")
        return {"error": str(e)}

@app.route('/process_audio_google', methods=['POST'])
def process_audio_google():
    """
    Flask endpoint for handling audio data and performing speech-to-text transcription.
    """
    global audio_buffer

    try:
        audio_data = request.data
        print(f"Received audio data: {len(audio_data)} bytes")

        # Append incoming audio data to the buffer
        audio_buffer += audio_data

        # Process audio using Google Streaming STT
        result = process_audio_with_google_streaming()

        return jsonify(result), 200
    except Exception as e:
        print(f"Error processing audio: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/process_audio', methods=['POST'])
def process_audio():
    """
    Flask endpoint for handling audio data and performing offline speech-to-text transcription using VOSK.
    """
    try:
        audio_data = request.data
        print(f"Received audio data: {len(audio_data)} bytes")

        # Process the audio using VOSK
        result = process_audio_with_google_streaming(audio_data)

        return jsonify(result), 200
    except Exception as e:
        print(f"Error processing audio: {e}")
        return jsonify({"error": str(e)}), 500


def start_flask_app():
    """
    Start Flask server in a separate thread.
    """
    app.run(port=5000, debug=False, use_reloader=False, threaded=True)


def capture_audio_from_virtual_device():
    """
    Captures audio from the VB-Audio Virtual Cable input device (CABLE Output) continuously.
    """
    global recording_frames, is_running

    p = pyaudio.PyAudio()
    virtual_device_index = None

    # Find the VB-Audio Virtual Cable device index
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if "CABLE Output" in device_info["name"]:
            virtual_device_index = i
            max_channels = device_info["maxInputChannels"]
            sample_rate = int(device_info["defaultSampleRate"])
            print(f"Virtual audio device found: {device_info['name']} (Index: {i})")
            print(f"Max Input Channels: {max_channels}")
            print(f"Default Sample Rate: {sample_rate}")
            break

    if virtual_device_index is None:
        raise ValueError("VB-Audio Virtual Cable device not found. Ensure it's installed and configured.")

    # Use optimal settings
    channels = 1  # Mono for speech recognition
    rate = 16000  # Use 16kHz for compatibility with speech models
    frames_per_buffer = 4096*3  # Larger buffer for more meaningful chunks

    print("CHANNELS USED:", channels)
    print("RATE USED:", rate)
    print("FRAMES_PER_BUFFER:", frames_per_buffer)

    try:
        # Open the audio stream
        stream = p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=virtual_device_index,
            frames_per_buffer=frames_per_buffer,
        )
        print("Listening for audio from virtual device...")

        # Continuously capture audio
        while is_running:
            audio_data = stream.read(frames_per_buffer, exception_on_overflow=False)
            recording_frames.append(audio_data)  # Save audio for the WAV file

            # Send the audio data to the Flask endpoint
            try:
                response = requests.post(
                    "http://127.0.0.1:5000/process_audio_google",
                    data=audio_data,
                    headers={"Content-Type": "application/octet-stream"},
                )
                if response.status_code == 200:
                    print(f"Audio sent: {len(audio_data)} bytes, Response: {response.status_code}")
            except Exception as e:
                print(f"Error sending audio data: {e}")
    except Exception as e:
        print(f"Error opening audio stream: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

        # Save the recording to a WAV file
        print("Saving audio to debug_audio.wav...")
        with wave.open("debug_audio.wav", "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(rate)
            wf.writeframes(b"".join(recording_frames))
        print("Audio saved to debug_audio.wav.")


def handle_exit(signum, frame):
    """
    Handle graceful shutdown when Ctrl+C is pressed.
    """
    global is_running
    print("\nGracefully shutting down...")
    is_running = False
    time.sleep(2)  # Allow threads to finish
    sys.exit(0)

def join_google_meet(meet_link, email, password,audio_thread):
    """
    Automates joining a Google Meet session.

    :param meet_link: URL of the Google Meet link
    :param email: Gmail address
    :param password: Gmail password
    """
    # Set up Selenium WebDriver
    chrome_options = Options()
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--use-fake-ui-for-media-stream")  # Grants microphone/camera permissions automatically
    chrome_options.add_argument("C:/Users/USER/AppData/Local/Google/Chrome/User Data/Default")
    chrome_options.add_argument("profile-directory=Default")  # Use the default profile
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    service = Service("C:/chromedriver/chromedriver-win64/chromedriver.exe")
    driver = webdriver.Chrome(service=service, options=chrome_options)

    # Navigate to the Google Meet link
    driver.get(meet_link)
    time.sleep(10)

    try:
        # Wait for the input field to be present
        name_field = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//input[@placeholder='Your name']"))
        )

        # Clear the field and input the name
        name_field.clear()
        name_field.send_keys("Justin To")
        print("Name entered successfully.")
    except Exception as e:
        print(f"Error entering name: {e}")

    try:
        # Wait for the "Ask to join" button and click it
        ask_to_join_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[.//span[text()='Ask to join']]"))
        )
        ask_to_join_button.click()
        print("Successfully clicked the 'Ask to join' button.")
        time.sleep(30)
        audio_thread.start()
    except Exception as e:
        print(f"Error clicking 'Ask to join' button: {e}")


    input("Press Enter to close the browser...")
    driver.quit()


if __name__ == "__main__":
    # Start Flask server in a separate thread
    signal.signal(signal.SIGINT, handle_exit)
    flask_thread = Thread(target=start_flask_app)
    flask_thread.start()

    # Start capturing audio in another thread
    audio_thread = Thread(target=capture_audio_from_virtual_device)
    

    # Join Google Meet

    join_google_meet(GMEET_LINK,EMAIL,PASSWORD,audio_thread)
