from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
import pyaudio
from google.cloud import speech
import os
from dotenv import load_dotenv
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
load_dotenv()
EMAIL = os.getenv("EMAIL")
PASSWORD = os.getenv("PASSWORD")
GMEET_LINK = os.getenv("GMEET_LINK")

# Configure Google Cloud Speech-to-Text


def join_google_meet(meet_link, email, password):
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
    # Click "Join now"
    try:
        # Wait for the "Ask to join" button and click it
        ask_to_join_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[.//span[text()='Ask to join']]"))
        )
        ask_to_join_button.click()
        print("Successfully clicked the 'Ask to join' button.")
    except Exception as e:
        print(f"Error clicking 'Ask to join' button: {e}")
        print("Joined Google Meet successfully.")

    # Start listening to audio and transcribing
    #transcribe_audio()
    input("Press Enter to close the browser...")


# Usage
if __name__ == "__main__":
    join_google_meet(GMEET_LINK, EMAIL, PASSWORD)
