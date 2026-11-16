
import cv2
import pyttsx3
import speech_recognition as sr
import sounddevice as sd
import threading
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

speaker = pyttsx3.init()
speaker.setProperty("rate", 165)

recognizer = sr.Recognizer()
cap = cv2.VideoCapture(0)

target = None
last_position = None
last_seen = False
frame_count = 0
last_announcement_time = 0


def get_position(center_x, width):
    ratio = center_x / width

    if ratio < 0.35:
        return "on your left"
    elif ratio > 0.65:
        return "on your right"
    return "in the centre"


speech_queue = []
speech_lock = threading.Lock()


def speak(message):
    print("VisionQuest:", message)

    with speech_lock:
        speech_queue.append(message)


def speech_worker():
    while True:
        with speech_lock:
            message = speech_queue.pop(0) if speech_queue else None

        if message:
            try:
                speaker.say(message)
                speaker.runAndWait()
            except Exception as error:
                print("Speech error:", error)

        else:
            threading.Event().wait(0.05)


speech_thread = threading.Thread(
    target=speech_worker,
    daemon=True
)
speech_thread.start()

def listen_for_command():
    print("Listening for 5 seconds...")

    try:
        audio = sd.rec(
            int(5 * 16000),
            samplerate=16000,
            channels=1,
            dtype="int16"
        )
        sd.wait()

        audio_data = sr.AudioData(audio.tobytes(), 16000, 2)

        command = recognizer.recognize_google(
            audio_data,
            language="en-IN"
        )

        print("You said:", command)
        return command.lower()

    except sr.UnknownValueError:
        speak("I could not understand.")
    except sr.RequestError as error:
        print("Recognition error:", error)
        speak("Speech recognition service is unavailable.")
    except Exception as error:
        print("Microphone error:", error)

    return None


def extract_target(command):
    if not command:
        return None

    class_names = sorted(
        model.names.values(),
        key=len,
        reverse=True
    )

    for name in class_names:
        if name.lower() in command:
            return name

    return None


def track_object(frame):
    global last_position, last_seen

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

        position = get_position(center_x, width)
        positions.append((x1, y1, x2, y2, position))

    if not positions:
        if last_seen:
            speak(f"I cannot see your {target} right now.")

        last_seen = False
        last_position = None
        return

    # Choose the largest detected instance as the primary target.
    selected = max(
        positions,
        key=lambda item: (item[2] - item[0]) * (item[3] - item[1])
    )

    x1, y1, x2, y2, position = selected
    
    cv2.rectangle(
        frame,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"{target}: {position}",
        (int(x1), max(25, int(y1) - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

        # Announce only when the object's position changes
    if position != last_position:
   
         print("Detected:", target, "| Position:", position)

    if last_position is None:
        speak(f"I found your {target}. It is {position}.")

    elif position != last_position:
        speak(f"Your {target} moved to {position}.")

    last_position = position

last_seen = True


speak("VisionQuest tracking test started.")
print("Press V to request an object.")
print("Press S to stop tracking.")
print("Press Q to quit.")

while True:
    success, frame = cap.read()

    if not success:
        print("Camera error.")
        break

    key = cv2.waitKey(1) & 0xFF

    if key == ord("v"):
        command = listen_for_command()
        new_target = extract_target(command)

        if new_target:
            target = new_target
            last_position = None
            last_seen = False
            speak(f"Searching for your {target}.")
        else:
            speak("I could not identify that object.")

    elif key == ord("s"):
        target = None
        last_position = None
        last_seen = False
        speak("Object tracking stopped.")

    elif key == ord("q"):
        break

    if target:
        frame_count += 1

        # Run detection every third frame to reduce processing.
        if frame_count % 3 == 0:
            track_object(frame)

    cv2.imshow("VisionQuest Object Tracking", frame)

cap.release()
cv2.destroyAllWindows()
speaker.stop()
