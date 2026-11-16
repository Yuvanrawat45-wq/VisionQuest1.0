
import cv2
import pyttsx3
from ultralytics import YOLO

# Load model
model = YOLO("yolo11n.pt")

# Initialize speech
speaker = pyttsx3.init()
speaker.setProperty("rate", 165)

# Open webcam
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


def find_object(frame, target):

    results = model.predict(
        frame,
        imgsz=640,
        conf=0.40,
        verbose=False
    )

    detections = []
    height, width = frame.shape[:2]

    for box in results[0].boxes:

        class_id = int(box.cls[0])
        name = model.names[class_id]

        if name.lower() != target.lower():
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        center_x = (x1 + x2) / 2

        position = get_position(center_x, width)

        confidence = float(box.conf[0])

        detections.append((position, confidence))

    if not detections:
        return f"I cannot identify a {target} in the current camera view."

    # Remove duplicate positions
    positions = list(dict.fromkeys(
        position for position, confidence in detections
    ))

    if len(positions) == 1:
        return f"I can see a {target} {positions[0]}."

    return f"I can see {target}s " + " and ".join(positions) + "."


print("VISIONQUEST OBJECT LOCATOR")
print("Press F to find an object")
print("Press Q to quit")

while True:

    success, frame = cap.read()

    if not success:
        break

    cv2.imshow("VisionQuest Object Locator", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("f"):

        target = input(
            "\nWhich object do you want to find? "
        ).strip().lower()

        if not target:
            continue

        # Capture a fresh frame
        success, frame = cap.read()

        if not success:
            continue

        print("Searching for:", target)

        description = find_object(frame, target)

        print("VisionQuest:", description)

        speaker.say(description)
        speaker.runAndWait()

    elif key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
speaker.stop()
