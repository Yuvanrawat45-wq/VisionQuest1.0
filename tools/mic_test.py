
import sounddevice as sd

print("Available audio devices:")
print(sd.query_devices())

print("\nRecording for 5 seconds...")

audio = sd.rec(
    int(5 * 16000),
    samplerate=16000,
    channels=1,
    dtype="int16"
)

sd.wait()

print("Recording completed!")
print("Audio shape:", audio.shape)
