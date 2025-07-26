import gradio as gr
import sounddevice as sd
import numpy as np
import whisper
import cohere
from gtts import gTTS
from playsound import playsound
import tempfile
import os

class ArabicVoiceAssistant:
    def __init__(self):
        self.audio_model = whisper.load_model("small")
        self.co = cohere.Client("")
        self.response_path = None

    def record_and_process(self, duration):
        try:
            print("\n بدء التسجيل...")
            audio = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype='float32')
            sd.wait()
            audio = np.squeeze(audio)

            print(" تحويل الصوت إلى نص...")
            result = self.audio_model.transcribe(audio, language='ar')
            user_text = result["text"].strip()

            if "توقف" in user_text:
                reply = "مع السلامة!"
            else:
                print("🤖 توليد الرد...")
                reply = self.co.chat(model="command-r7b-arabic-02-2025", message=user_text).text.strip()

            print(" تحويل الرد إلى صوت...")
            tts = gTTS(text=reply, lang='ar')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                self.response_path = fp.name

            return user_text, reply, self.response_path

        except Exception as e:
            return f" خطأ: {e}", "", None

    def clear_audio(self):
        if self.response_path and os.path.exists(self.response_path):
            os.remove(self.response_path)
        self.response_path = None
        return "", "", None

assistant = ArabicVoiceAssistant()

with gr.Blocks(theme=gr.themes.Monochrome()) as app:
    gr.Markdown("## 🎙️ المساعد الصوتي العربي")
    gr.Markdown("اضغط على الزر لتسجيل صوتك، وسيفهمك المساعد ويرد عليك بصوت.")

    duration = gr.Slider(1, 10, value=5, step=0.5, label="مدة التسجيل (بالثواني)")
    start_btn = gr.Button("ابدأ التسجيل")
    clear_btn = gr.Button("إيقاف البرنامج ومسح البيانات")

    user_out = gr.Textbox(label="🧑 ما قلته")
    bot_out = gr.Textbox(label="🤖 رد المساعد")
    audio_out = gr.Audio(label="🔊 صوت المساعد", type="filepath", autoplay=True)

    start_btn.click(fn=assistant.record_and_process, inputs=duration, outputs=[user_out, bot_out, audio_out])
    clear_btn.click(fn=assistant.clear_audio, outputs=[user_out, bot_out, audio_out])

app.launch()
