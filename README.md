# test
# repository for talking agents

import os
import uuid
import json
from datetime import datetime
import sounddevice as sd
import scipy.io.wavfile as wav
import pygame
import numpy as np
import time
from openai import OpenAI

import re

def is_hallucinated_silence(text):
    """
    Detects if Whisper likely hallucinated a response due to silence or noise.
    Catches most common variants with fuzzy, substring, and short length matching.
    """
    t = re.sub(r'[^\w\s]', '', text).strip().lower()
    hallucinations = [
        "thank you for watching", "thanks for watching", "hello everyone",
        "thank you so much for watching", "thank you", "thanks", ""
    ]
    for h in hallucinations:
        if h and h in t:
            return True
    # Also consider very short output as likely silence hallucination
    if len(t) <= 6:
        return True
    return False

# Initialize OpenAI client for API usage
client = OpenAI()

# Location to store the transcript
DESKTOP = "/Users/Desktop"
TRANSCRIPT_FILE = os.path.join(DESKTOP, "fany_assistantapi_transcript.txt")

# Set your OpenAI Assistant's ID here (get this from OpenAI's Assistants dashboard)
ASSISTANT_ID = "asst_wEoD6VMuJEjh1bjDNIVDJiXl"  # <--- CHANGE THIS TO YOUR ASSISTANT ID!

# List of core questions to use for the interview
QUESTIONS = [
    "What interests you about joining our communications team?",
    "Can you describe a challenging communication situation you handled and how you resolved it?",
    "How do you approach communicating complex information to non-expert audiences?",
    "What tools or technologies have you used to manage communication or media relations?",
    "What do you believe is the most important quality for someone on a communications team, and why?",
]

def record_with_silence_detection(filename, max_duration=180, silence_threshold=500, max_silence=5, fs=44100):
    """
    Records audio from the microphone up to max_duration seconds.
    Ends early if there are more than max_silence seconds of continuous silence.
    The WAV file is saved to filename.
    """
    print(f"🎙️ Recording... You have up to {max_duration} seconds. If you are silent for {max_silence} seconds, Fany will move on.")
    block_duration = 0.5  # Number of seconds per silence check block
    silence_blocks = int(max_silence / block_duration)  # How many blocks count as silence
    max_blocks = int(max_duration / block_duration)
    audio = []
    # Configure sounddevice for microphone
    sd.default.samplerate = fs
    sd.default.channels = 1
    stream = sd.InputStream(dtype='int16')
    stream.start()
    silence_count = 0
    start_time = time.time()
    try:
        for i in range(max_blocks):
            # Read a small chunk of audio
            block, _ = stream.read(int(fs * block_duration))
            audio.append(block)
            # If block is below threshold (quiet), count as silence
            if np.abs(block).mean() < silence_threshold:
                silence_count += 1
                if silence_count >= silence_blocks:
                    print("Detected silence threshold. Stopping recording.")
                    break
            else:
                silence_count = 0
    except KeyboardInterrupt:
        print("Recording interrupted by user.")
    finally:
        stream.stop()
        stream.close()
    # Combine all blocks to a single waveform
    audio_np = np.concatenate(audio, axis=0)
    wav.write(filename, fs, audio_np)
    print("✅ Recording saved.")
    total_time = time.time() - start_time
    return total_time

def transcribe_audio(filename):
    """
    Uses OpenAI Whisper API to transcribe a WAV audio file.
    Returns the transcript string.
    """
    print("📝 Transcribing with Whisper...")
    with open(filename, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    print("👤 You said:", transcript.text)
    return transcript.text

def stream_speak_text(text, voice="sage"):
    """
    Uses OpenAI TTS API to generate speech for Fany's response.
    Plays audio using pygame mixer.
    """
    print(f"🔊 Fany (voice '{voice}') is streaming...")
    speech = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text
    )
    temp_mp3 = f"temp_{uuid.uuid4()}.mp3"
    with open(temp_mp3, "wb") as f:
        f.write(speech.content)
    pygame.mixer.init()
    pygame.mixer.music.load(temp_mp3)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pass
    if os.path.exists(temp_mp3):
        os.remove(temp_mp3)

def save_full_history(history):
    """
    Writes the full conversation transcript to the user's Desktop.
    """
    with open(TRANSCRIPT_FILE, "w", encoding="utf-8") as f:
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        f.write(f"=== Fany AI Interview - Assistants API ({timestamp}) ===\n\n")
        for turn in history:
            role = turn['role'].upper()
            content = turn['content']
            f.write(f"{role}: {content}\n\n")

def run_assistant_thread(assistant_id, user_input, thread_id=None):
    """
    Sends the user input to a persistent OpenAI Assistant thread and streams the response.
    Returns: (reply_text, thread_id) for session continuity.
    """
    if thread_id is None:
        # Start a new persistent thread (conversation session)
        thread = client.beta.threads.create()
        thread_id = thread.id
    else:
        thread = client.beta.threads.retrieve(thread_id)
    # Add the user's latest message to the thread
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_input
    )
    # Start a new assistant run for the thread, streaming output
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        stream=True
    )
    reply_text = ""
    print("🤖 Fany is replying (Assistants API, streaming)...")
    for event in run:
        # Stream in all event deltas as they arrive
        if event.data.object == "thread.message.delta":
            content = event.data.delta.content[0].text.value if event.data.delta.content else ""
            print(content, end="", flush=True)
            reply_text += content
    print()  # Print newline after streaming
    return reply_text, thread_id

def is_follow_up(reply):
    """
    Checks if Fany's reply is a follow-up question or clarification (should stay on the same question).
    """
    reply_lower = reply.lower()
    if reply.strip().endswith("?"):
        return True
    follow_words = ["could you", "can you", "would you", "please elaborate", "tell me more", "clarify", "example", "share more", "specifically"]
    return any(w in reply_lower for w in follow_words)

def is_user_prompted(reply):
    """
    Checks if Fany is inviting the candidate to ask a question or continue, not move on.
    """
    keywords = [
        "ask your question", "go ahead", "please ask", "feel free to ask", "what's your question",
        "i'm listening", "sure", "please continue", "how can i help"
    ]
    reply_lower = reply.lower()
    return any(k in reply_lower for k in keywords)

def is_farewell(reply):
    """
    Checks if Fany's reply is a farewell (signals the session should end).
    """
    farewells = [
        "thank you for your time", "goodbye", "have a great day", "if you ever want to continue",
        "it was nice talking", "take care", "see you", "reach out"
    ]
    reply_lower = reply.lower()
    return any(word in reply_lower for word in farewells)

def main():
    # === Main Conversation Loop ===
    # This is the main control loop for the adaptive, voice-based interview.
    # Each main question is presented to the user, allowing for unlimited follow-ups and clarification until:
    #   - Fany's reply is a summary or move-on,
    #   - The user or Fany terminates,
    #   - Or 5 minutes elapse per question (whichever comes first).

    print("\n🟣 Fany AI Interviewer - OpenAI Assistants/Threads API Demo\n")
    print("Say 'terminate interview' anytime to stop the session.\n")
    voice = "sage"  # Fany will use OpenAI's 'sage' TTS voice
    transcript = []  # Conversation transcript
    intro = "Hello! I'm Fany AI, your digital interviewer for the Communications Team at GE Appliances. Let's start your interview."
    stream_speak_text(intro, voice)
    print("Fany: " + intro)
    thread_id = None  # Persistent conversation thread
    try:
        for question in QUESTIONS:
            if 'terminate_now' in locals() and terminate_now:
                break
            current_q = question
            waiting_for_completion = True
            question_start_time = time.time()  # Track start of this main question
            while waiting_for_completion:
                # Speak and display Fany's current question or follow-up
                stream_speak_text(current_q, voice)
                print("Fany:", current_q)
                transcript.append({"role": "assistant", "content": current_q})
                # Record answer (ends on silence)
                filename = f"temp_{uuid.uuid4()}.wav"
                answer_time = record_with_silence_detection(
                    filename, max_duration=180, silence_threshold=500, max_silence=5)
                user_input = transcribe_audio(filename).strip()
                if os.path.exists(filename):
                    os.remove(filename)
                # Robust hallucination and silence detection (captures variations)
                if is_hallucinated_silence(user_input):
                    print("Detected silence or hallucinated phrase. No real input received.")
                    stream_speak_text("I didn't hear anything. Please try again or let me know if you'd like to continue.", voice)
                    continue  # Re-ask the same question, do not proceed
                # Termination check: user says terminate
                terminate_now = False
                if "terminate interview" in user_input.lower() or user_input.lower() == "terminate":
                    farewell = "Thank you for your time. Goodbye!"
                    print("Fany:", farewell)
                    stream_speak_text(farewell, voice)
                    transcript.append({"role": "user", "content": user_input})
                    save_full_history(transcript)
                    terminate_now = True
                    break
                transcript.append({"role": "user", "content": user_input})
                # ---- Run Assistant thread (persistent server-side memory)
                reply, thread_id = run_assistant_thread(ASSISTANT_ID, user_input, thread_id)
                transcript.append({"role": "assistant", "content": reply})
                # Termination check: Fany gives a farewell
                if is_farewell(reply):
                    stream_speak_text(reply, voice)
                    waiting_for_completion = False
                    terminate_now = True
                    break
                # Check if we've spent too long (5 min) on this question
                elapsed = time.time() - question_start_time
                if elapsed >= 300:
                    print("Fany: Time's up for this question. Let's continue to the next one.")
                    stream_speak_text("Thank you for your thoughtful answers. Let's move to the next question.", voice)
                    waiting_for_completion = False
                    break
                # Adaptive loop: stay on same question for follow-ups or if Fany prompts you
                if is_follow_up(reply) or is_user_prompted(reply):
                    current_q = reply
                    continue
                else:
                    stream_speak_text(reply, voice)
                    waiting_for_completion = False
    except KeyboardInterrupt:
        print("\n⚠️ Interview terminated by user (Ctrl+C). Saving transcript...")
        save_full_history(transcript)
        print("Transcript saved to your Desktop. Goodbye.")
    print("\n✅ Interview finished! Transcript saved to your Desktop.")
    save_full_history(transcript)

if __name__ == "__main__":
    main()
