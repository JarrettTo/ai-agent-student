
# AI Agent for Online Classes

## **Overview**

This is an AI agent that automatically:
1. **Joins online classes** via Google Meet.
2. **Handles attendance** by sending a message via chat when the name is called or a question is detected.
3. **Records meeting transcripts** and uses AI to generate notes.
4. **Creates a Notion document** filled with well-formatted notes, including headings, bullet points, and other key formatting.

### **Disclaimer**
This project is purely an experiment and should **NOT** be used for actual classes.

---

## **How to Set Up**

### **1. Install and Configure VB-Audio Virtual Cable**
If you're on Windows, VB-Audio Virtual Cable is required to route audio from Google Meet into the AI for processing.

#### **Steps to Install VB-Audio Virtual Cable:**
1. Go to the [VB-Audio Virtual Cable download page](https://vb-audio.com/Cable/).
2. Download and install the Virtual Cable driver.
3. Open the Windows "Sound Settings":
   - Set **CABLE Input** as your **Default Playback Device**.
   - Set **CABLE Output** as your **Default Recording Device**.
4. Test the setup:
   - Play audio on your computer and verify that it routes through the "CABLE Output" device in your Sound Control Panel.

#### **Alternatives for Non-Windows Users:**
- For Mac: Use **Loopback** by Rogue Amoeba or **BlackHole** (free) as a virtual audio device.
- For Linux: Use **PulseAudio** or **JACK Audio** to set up virtual audio routing.

---

### **2. Set Up Python Environment**
1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. Install Python dependencies:
   - Create and activate a virtual environment:
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     ```
   - Install the required packages:
     ```bash
     pip install -r requirements.txt
     ```

---

### **3. Set Up API Keys**
- **OpenAI API**:
  - Create an account at [OpenAI](https://openai.com/).
  - Generate an API key from the OpenAI dashboard.
- **Notion API**:
  - Create a Notion Integration at [Notion Developers](https://www.notion.so/my-integrations).
  - Copy your Integration Token and share your target Notion page with the integration.

Create a `.env` file in the root of your project and add the following variables:
```env
OPENAI_API_KEY=your_openai_api_key
NOTION_API_TOKEN=your_notion_api_token
NOTION_PAGE_ID=your_notion_page_id
EMAIL=your_google_account_email
PASSWORD=your_google_account_password
GMEET_LINK=google_meet_link
```

---

### **4. Install Google Chrome and ChromeDriver**
- Install Google Chrome from [here](https://www.google.com/chrome/).
- Download the ChromeDriver version that matches your Chrome version from [ChromeDriver Downloads](https://sites.google.com/chromium.org/driver/).
- Place the `chromedriver` executable in your project folder or a directory in your system PATH.

---

### **5. Run the AI Agent**
1. Start the AI agent:
   ```bash
   python main.py
   ```
2. Follow the prompts to join a Google Meet session.
3. The agent will:
   - Join the Google Meet session.
   - Handle attendance by listening for your name and responding in chat.
   - Detect and respond to questions in the meeting chat.
   - Record the meeting transcript, generate notes, and upload them to Notion.

---

### **6. Debugging and Logs**
- **Audio Troubleshooting**: If audio isn't being processed, verify the Virtual Cable configuration and ensure it is set as the default device.
- **Error Logs**: Check the console for error messages related to API keys, Selenium setup, or audio input.

---

## **Features**
1. **Automation**: Hands-free attendance and participation in online classes.
2. **Transcript and Notes**: AI-generated notes from meeting transcripts stored in Notion for easy reference.
3. **Customizable Responses**: The agent uses OpenAI to answer questions or send default responses if no question is detected.

---

### **Future Improvements**
- Add support for platforms other than Google Meet.
- Extend functionality to handle breakout rooms or multiple participants.
- Add a graphical user interface (GUI) for easier configuration. 

---

Enjoy experimenting with this AI agent!
