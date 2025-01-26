import os
import openai
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=os.getenv("OPENAI_API_KEY"),
    )
def check_and_respond_with_openai(context):
    
    """
    Use OpenAI to check if there's a question and respond appropriately.
    """
    print("CONTEXT:", context)
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
def openai_make_notes(transcript):
    
    """
    Use OpenAI to generate notes given a transcript.
    """
    prompt = f"""
    You are a helpful notetaker. Analyze the following transcript from a university class.
    "{transcript}"

    Format the notes with headings, bullet points, and key action items where applicable. Make it comprehensive and do your best to clean and adjust any possible mistranscripted words based on potential context. Only answer with the notes format, don't say anything else
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use a suitable OpenAI GPT model
            messages=[
                {"role": "system", "content": "You are a helpful notetaker. Analyze the following transcript from a university class."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=3000,   # Adjust based on the expected response length
            top_p=1.0,        # Full probability distribution
            frequency_penalty=0,  # No penalty for frequent tokens
            presence_penalty=0    # No penalty for introducing new topics
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error querying OpenAI: {e}")
        return "I'm sorry, there was an error processing the request."
def chatgpt_transcript():
    """
    Opens debug_audio.wav, transcribes it using OpenAI Whisper, and returns the text.
    """
    try:
        audio_file_path = "debug_audio.wav"  # Path to your audio file
        with open(audio_file_path, "rb") as audio_file:
            transcription = client.audio.translations.create(
                model="whisper-1",
                file=audio_file,
            )
        print("Transcription: ", transcription.text)
        return transcription.text
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None
