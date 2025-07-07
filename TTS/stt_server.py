from flask import Flask, request
import whisper
import tempfile

app = Flask(__name__)
model = whisper.load_model("tiny")

@app.route('/stt', methods=['POST'])
def receive_audio():
    audio = request.data
    print(f"🎧 Received {len(audio)} bytes")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio)
        tmp_path = tmp.name

    print("🧠 Transcribing...")
    result = model.transcribe(tmp_path)
    print(f"📄 Transcript: {result['text']}")

    return f"📝 {result['text']}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

