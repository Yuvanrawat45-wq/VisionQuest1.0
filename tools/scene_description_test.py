
import cv2
import pyttsx3
from ultralytics import YOLO
from collections import defaultdict

# Load YOLO model
model = YOLO("yolo11n.pt")

# Initialize speech
speaker = pyttsx3.init()
speaker.setProperty("rate", 165)

# Open laptop webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    raise SystemExit

def get_position(center_x, image_width):
    """
    Divide the camera image into
    left, centre and right regions.
    """
    ratio = center_x / image_width

    if ratio < 0.35:
        return "on your left"
    elif ratio > 0.65:
        return "on your right"
    else:
        return "ahead of you"


def describe_scene(frame):
    """
    Detect objects and convert detections
    into a short spoken description.
    """

    results = model.predict(
        frame,
        imgsz=640,
        conf=0.40,
        verbose=False
    )

    result = results[0]
    height, width = frame.shape[:2]

    objects = defaultdict(int)

    if result.boxes is None:
        return "I could not identify any objects."

    for box in result.boxes:

        confidence = float(box.conf[0])

        if confidence < 0.40:
            continue

        class_id = int(box.cls[0])
        object_name = model.names[class_id]

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        center_x = (x1 + x2) / 2

        position = get_position(center_x, width)

        objects[(object_name, position)] += 1

    if not objects:
        return "I could not identify any objects."

    descriptions = []

    for (name, position), count in objects.items():

        if count == 1:
            descriptions.append(f"a {name} {position}")
        else:
            descriptions.append(
                f"{count} {name}s {position}"
            )

    return "I can see " + ", ".join(descriptions) + "."


print("VISIONQUEST SCENE DESCRIPTION TEST")
print("Press D to describe surroundings")
print("Press Q to quit")

while True:

    success, frame = cap.read()

    if not success:
        print("Camera frame not received")
        break

    cv2.imshow("VisionQuest", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("d"):

        print("\nAnalyzing surroundings...")

        description = describe_scene(frame)

        print("VisionQuest:", description)

        speaker.say(description)
        speaker.runAndWait()

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
speaker.stop()
