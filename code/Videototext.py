import whisper

# Load the pre-trained Whisper model (you can choose "tiny", "base", "small", "medium", or "large")
model = whisper.load_model("large")

# Transcribe the audio file
result = model.transcribe("audio.wav")

# Print the entire transcription
print(result['text'])

with open("transcription.txt", "w") as f:
    for segment in result['segments']:
        f.write(f"[{segment['start']:.2f} - {segment['end']:.2f}]: {segment['text']}\n")
