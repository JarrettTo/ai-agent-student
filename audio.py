import pyaudio
from google.cloud import speech

speech_client = speech.SpeechClient()

def transcribe_audio():
    """
    Captures audio from the microphone and transcribes it in real-time.
    """
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        input=True,
                        frames_per_buffer=1024)
    print(audio.get_device_info_by_index())

    print("Listening for audio...")

    # Configure the request
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US"
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True
    )

    # Stream audio data and send it to the Speech API
    def generate_audio_chunks():
        while True:
            yield speech.StreamingRecognizeRequest(audio_content=stream.read(1024))

    # Perform the streaming recognition
    responses = speech_client.streaming_recognize(
        config=streaming_config,
        requests=generate_audio_chunks()
    )

    # Process the responses
    try:
        for response in responses:
            for result in response.results:
                if result.is_final:
                    transcript = result.alternatives[0].transcript
                    print(f"Transcript: {transcript}")
    except Exception as e:
        print(f"Error during streaming recognition: {e}")
