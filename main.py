from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from notion_client import Client
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
import numpy as np
import openai
from collections import deque
from openai import OpenAI
from audio_processing import preprocess_audio
from notion_integration import format_notes_for_notion,notes_to_notion
from openai_functions import check_and_respond_with_openai, openai_make_notes, chatgpt_transcript

load_dotenv()
EMAIL = os.getenv("EMAIL")
PASSWORD = os.getenv("PASSWORD")
GMEET_LINK = os.getenv("GMEET_LINK")
NOTION_API_TOKEN = os.getenv("NOTION_API_TOKEN")
NOTION_PAGE_ID = os.getenv("NOTION_PAGE_ID")

#Flask Configuration
app = Flask(__name__)
CORS(app)

#VOSK model initialization
speech_client = speech.SpeechClient()
model_path = "model"  # Path to your VOSK model directory
if not os.path.exists(model_path):
    raise ValueError("VOSK model not found. Download a model from https://alphacephei.com/vosk/models and extract it to the 'model' directory.")
vosk_model = Model(model_path)

#flags and audio buffer configuration
global message_tab_opened
global_driver = None
rolling_history = deque(maxlen=50)  
recording_frames = []
is_running = True
audio_buffer = b""
buffer_threshold = 16000 * 6  # Process ~2 seconds of audio (16kHz * 2 seconds)


def process_audio_with_vosk(audio_data):
    """
    Process audio using VOSK for offline speech-to-text transcription with buffered audio.
    """
    global audio_buffer, rolling_history

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

                # Append transcript to rolling history
                words = transcript.strip().split()
                rolling_history.extend(words)

                # Extract the last 15 words including the current transcript
                context_words = list(rolling_history)[-15:]
                context = " ".join(context_words)

                # Check for the words "Justin" or "just in"
                if "justin" in transcript.strip() or "just in" in transcript.strip():
                    print("Detected mention of 'Justin' or 'just in'!")

                    # Pass context to OpenAI for processing
                    response = check_and_respond_with_openai(context)
                    print(f"Response: {response}")
                    send_message_to_chat(response)

                # Append transcript to a file
                with open("transcript.txt", "a") as transcript_file:
                    transcript_file.write(transcript + " ")

                return {"status": "Audio processed", "transcript": transcript.strip()}

        return {"status": "Buffering audio..."}
    except Exception as e:
        print(f"Error in VOSK transcription: {e}")
        return {"error": str(e)} 

@app.route('/process_audio', methods=['POST'])
def process_audio():
    """
    Flask endpoint for handling audio data and performing offline speech-to-text transcription using VOSK.
    """
    try:
        audio_data = request.data

        # Process the audio using VOSK
        result = process_audio_with_vosk(audio_data)

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
            #preprocessed_audio = preprocess_audio(audio_data, rate)
            #recording_frames.append(preprocessed_audio)  # Save audio for the WAV file
            # Send the audio data to the Flask endpoint
            try:
                response = requests.post(
                    "http://127.0.0.1:5000/process_audio",
                    data=audio_data,
                    headers={"Content-Type": "application/octet-stream"},
                )
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
    time.sleep(30)  # Allow threads to finish
    sys.exit(0)

def join_google_meet(meet_link, email, password,audio_thread):
    """
    Automates joining a Google Meet session.

    :param meet_link: URL of the Google Meet link
    :param email: Gmail address
    :param password: Gmail password
    """
    # Set up Selenium WebDriver
    global global_driver
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
    global_driver = webdriver.Chrome(service=service, options=chrome_options)

    # Navigate to the Google Meet link
    global_driver.get(meet_link)
    time.sleep(10)

    try:
        # Wait for the input field to be present
        name_field = WebDriverWait(global_driver, 10).until(
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
        ask_to_join_button = WebDriverWait(global_driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[.//span[text()='Ask to join']]"))
        )
        ask_to_join_button.click()
        print("Successfully clicked the 'Ask to join' button.")
        time.sleep(10)
        audio_thread.start()
    except Exception as e:
        print(f"Error clicking 'Ask to join' button: {e}")


    input("Press Enter to close the browser...")
    
def send_message_to_chat(message):
    """
    Sends a message to the Google Meet chat using the existing Selenium WebDriver instance.

    :param message: The message to send in the Google Meet chat
    """
    global global_driver  # Access the global driver instance
    global message_tab_opened
    try:
        # Open the chat panel by clicking the "Chat with everyone" button
        if message_tab_opened != True:
            chat_button = WebDriverWait(global_driver, 20).until(
                EC.element_to_be_clickable(
                    (
                        By.XPATH,
                        "//button[@aria-label='Chat with everyone']",
                    )
                )
            )
            chat_button.click()
            print("Chat panel opened.")
            message_tab_opened = True

        # Wait for the chat input box to be present
        chat_input = WebDriverWait(global_driver, 20).until(
            EC.presence_of_element_located(
                (
                    By.XPATH,
                    "//textarea[@aria-label='Send a message to everyone']",
                )
            )
        )

        # Enter the message into the chat input box
        chat_input.send_keys(message)
        print("Message entered into the chat box.")

        # Click the send button to send the message
        send_button = WebDriverWait(global_driver, 20).until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    "//button[@aria-label='Send a message to everyone']",
                )
            )
        )
        send_button.click()
        print("Message sent to chat.")

    except Exception as chat_error:
        print(f"Error sending message to chat: {chat_error}")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_exit)
    global message_tab_opened
    message_tab_opened = False
    flask_thread = Thread(target=start_flask_app)
    flask_thread.start()
    audio_thread = Thread(target=capture_audio_from_virtual_device)
    join_google_meet(GMEET_LINK,EMAIL,PASSWORD,audio_thread)
    is_running = False  # Stop the audio recording thread
    transcript = chatgpt_transcript()
    if transcript:
        print("Final Transcript:")
        print(transcript)
        notes=openai_make_notes(transcript)
        notion_blocks=format_notes_for_notion(notes)
        notes_to_notion(NOTION_API_TOKEN, NOTION_PAGE_ID,notion_blocks)
