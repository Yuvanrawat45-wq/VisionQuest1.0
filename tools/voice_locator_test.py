
import cv2
import pyttsx3
import speech_recognition as sr
import sounddevice as sd
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
speaker = pyttsx3.init()
speaker.setProperty("rate", 165)

recognizer = sr.Recognizer()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera could not be opened")
    raise SystemExit


def get_position(center_x, width):
    ratio = center_x / width

    if ratio < 0.35:
        return "on your left"
    elif ratio > 0.65:
        return "on your right"
    return "ahead of you"


def listen_for_command():
    print("\nListening for 5 seconds. Speak now!")

    try:
        audio = sd.rec(
            int(5 * 16000),
            samplerate=16000,
            channels=1,
            dtype="int16"
        )
        sd.wait()

        audio_data = sr.AudioData(
            audio.tobytes(),
            16000,
            2
        )

        command = recognizer.recognize_google(
            audio_data,
            language="en-IN"
        )

        print("You said:", command)
        return command.lower()

    except sr.UnknownValueError:
        print("Could not understand speech.")
    except sr.RequestError as error:
        print("Recognition service error:", error)
    except Exception as error:
        print("Microphone error:", error)

    return None


def find_object(frame, target):
    results = model.predict(
        frame,
        imgsz=640,
        conf=0.40,
        verbose=False
    )

    height, width = frame.shape[:2]
    positions = []

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        name = model.names[class_id]

        if name.lower() != target.lower():
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        center_x = (x1 + x2) / 2

        positions.append(
            get_position(center_x, width)
        )

    if not positions:
        return f"I cannot identify a {target} in the current camera view."

    positions = list(dict.fromkeys(positions))

    if len(positions) == 1:
        return f"I can see a {target} {positions[0]}."

    return f"I can see {target}s " + " and ".join(positions) + "."


def process_command(command, frame):
    if not command:
        return "I could not understand your request."

    class_names = sorted(
        model.names.values(),
        key=len,
        reverse=True
    )

    target = None

    for name in class_names:
        if name.lower() in command:
            target = name
            break

    if target is None:
        return "I could not identify the requested object."

    print("Searching for:", target)
    return find_object(frame, target)


print("\nVISIONQUEST VOICE LOCATOR")
print("Press V to speak a request")
print("Press Q to quit")

while True:
    success, frame = cap.read()

    if not success:
        break

    cv2.imshow("VisionQuest Voice Locator", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("v"):
        command = listen_for_command()

        if command:
            success, frame = cap.read()

            if success:
                response = process_command(command, frame)
                print("VisionQuest:", response)

                speaker.say(response)
                speaker.runAndWait()

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
speaker.stop()
