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

    print("Listening for audio...")

    with speech_client.streaming_recognize(
        speech.StreamingRecognizeRequest(
            config=speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="en-US"
            ),
            interim_results=True
        )
    ) as stream_recognizer:

        for audio_chunk in iter(lambda: stream.read(1024), b""):
            stream_recognizer.write(audio_chunk)
            response = stream_recognizer.recognize()
            if response.results:
                transcript = response.results[0].alternatives[0].transcript
                print(f"Transcript: {transcript}")
