
import cv2

KNOWN_DISTANCE = 1.0
A4_HEIGHT = 0.297

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not available")
    exit()

print("Place A4 paper vertically at exactly 1 metre.")
print("Press SPACE to capture the image.")
print("Press Q to quit.")

captured_frame = None

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read camera frame")
        break

    cv2.imshow("Live Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        captured_frame = frame.copy()
        break

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if captured_frame is not None:
    print("Image captured successfully.")

    cv2.imshow("Captured Image", captured_frame)

    print("\nDrag your mouse around the A4 sheet.")
    print("Press ENTER or SPACE to confirm selection.")
    print("Press C to cancel.")

    x, y, w, h = cv2.selectROI(
        "Captured Image",
        captured_frame,
        showCrosshair=True,
        fromCenter=False
    )

    cv2.destroyAllWindows()

    if w > 0 and h > 0:
        pixel_height = h

        focal_length = (
            pixel_height * KNOWN_DISTANCE / A4_HEIGHT
        )

        print("\n------ CALIBRATION RESULT ------")
        print("A4 pixel height:", pixel_height)
        print("Focal length:", round(focal_length, 2), "pixels")

    else:
        print("No selection made.")

else:
    print("No image captured.")
