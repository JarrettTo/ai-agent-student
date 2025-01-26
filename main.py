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
import numpy as np
import noisereduce as nr
import scipy.io.wavfile as wav
from scipy.signal import butter, lfilter
import openai
from collections import deque
from openai import OpenAI

load_dotenv()
EMAIL = os.getenv("EMAIL")
PASSWORD = os.getenv("PASSWORD")
GMEET_LINK = os.getenv("GMEET_LINK")

# Configure Google Cloud Speech-to-Text
global_driver = None
app = Flask(__name__)
CORS(app)
# Initialize Google Cloud Speech-to-Text client
speech_client = speech.SpeechClient()
model_path = "model"  # Path to your VOSK model directory
if not os.path.exists(model_path):
    raise ValueError("VOSK model not found. Download a model from https://alphacephei.com/vosk/models and extract it to the 'model' directory.")

# Load the model once at startup
vosk_model = Model(model_path)
global message_tab_opened = False
openai.api_key = os.getenv("OPENAI_API_KEY")
rolling_history = deque(maxlen=50)  # Adjust the size for how far back you want to look
recording_frames = []
is_running = True

audio_buffer = b""
buffer_threshold = 16000 * 6  # Process ~2 seconds of audio (16kHz * 2 seconds)
def check_and_respond_with_openai(context):
    
    """
    Use OpenAI to check if there's a question and respond appropriately.
    """
    print("CONTEXT:", context)
    client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    prompt = f"""
    You are a helpful transcript assistant. Analyze the following context which contains a speech to text transcript that is rough and uncleaned.
    "{context}"

    If any subset of words in the sentence contains or even resembles a question, reply with the best answer to that question, but only give the answer (for example if you detect the question What is a Tomato, your answer should be "It is a fruit.".  If the question you find is what is _____, answer it). Do not reply with a question or relay the question back. If it contains keywords that indicate the presence of a question such as "what, where, when, why, who, how", try your best to piece together question based on words that sound similar or just fill in the gaps of possible missing words. If you really can't make out a question, reply with "My mic is broken po but I'm here."
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use a suitable OpenAI GPT model
            messages=[
                {"role": "system", "content": "You are a helpful assistant reading a speech to text transcript that is rough and uncleaned."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,   # Adjust based on the expected response length
            top_p=1.0,        # Full probability distribution
            frequency_penalty=0,  # No penalty for frequent tokens
            presence_penalty=0    # No penalty for introducing new topics
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error querying OpenAI: {e}")
        return "I'm sorry, there was an error processing the request."

def bandpass_filter(audio_data, rate, lowcut=300, highcut=3400):
    """
    Apply a bandpass filter to the audio.
    """
    nyquist = 0.5 * rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(1, [low, high], btype="band")
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    filtered_audio = lfilter(b, a, audio_array)
    return filtered_audio.astype(np.int16).tobytes()
def reduce_noise(audio_data, rate):
    """
    Reduce noise in the audio data using noisereduce library.
    """
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    # Use the first 1 second of audio as a noise profile
    noise_profile = audio_array[:rate] if len(audio_array) > rate else audio_array
    reduced_audio = nr.reduce_noise(y=audio_array, sr=rate, y_noise=noise_profile)
    return reduced_audio.tobytes()
def normalize_audio(audio_data):
    """
    Normalize the audio to have consistent volume levels.
    """
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    max_val = np.max(np.abs(audio_array))
    normalized_audio = (audio_array / max_val * 32767).astype(np.int16)
    return normalized_audio.tobytes()
def preprocess_audio(audio_data, rate):
    """
    Preprocess audio by reducing noise, normalizing, downsampling, and applying bandpass filtering.
    """
    # Noise reduction
    audio_data = reduce_noise(audio_data, rate)
    
    # Downsample to 16 kHz
    if rate != 16000:
        audio_data = downsample_audio(audio_data, rate, 16000)
        rate = 16000  # Update the rate after downsampling

    # Normalize audio
    audio_data = normalize_audio(audio_data)

    # Bandpass filter for human speech frequencies
    audio_data = bandpass_filter(audio_data, rate)

    return audio_data
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
            message_tab_opened = False

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
    # Start Flask server in a separate thread
    signal.signal(signal.SIGINT, handle_exit)
    flask_thread = Thread(target=start_flask_app)
    flask_thread.start()

    # Start capturing audio in another thread
    audio_thread = Thread(target=capture_audio_from_virtual_device)
    

    # Join Google Meet

    join_google_meet(GMEET_LINK,EMAIL,PASSWORD,audio_thread)
