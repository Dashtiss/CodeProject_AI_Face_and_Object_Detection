import cv2
import CodeProject as CodeProjectAI
import threading
import requests

ObjectDetection = True
FaceDetection = True

# Open the default webcam (index 0)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print(cap)
IP, PORT, TIMEOUT = "192.168.1.187", 32168, 10

Frame = None
Stop = False
Restart = False

FacePredictions = []
ObjectPredictions = []


def sendRequests(Img: bytes):
    global FacePredictions, ObjectPredictions, ObjectDetection, FaceDetection
    if FaceDetection:
        AiRecognition = CodeProjectAI.CodeProjectAIFace(IP, PORT, TIMEOUT, min_confidence=0.7)
        predictions = AiRecognition.recognize(Img)
        FacePredictions = predictions
    if ObjectDetection:
        cpaiobject = CodeProjectAI.CodeProjectAIObject(ip=IP, port=PORT, timeout=TIMEOUT, min_confidence=0.7)
        predictions = cpaiobject.detect(Img)
        ObjectPredictions = predictions


def DoProcess():
    global Frame
    while not Stop and not Restart:
        if Frame is not None:
            _, png = cv2.imencode('.png', Frame)
            sendRequests(png.tobytes())


def Video():
    global Frame, Restart, idk
    idk.start()
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame!")
            break
        if FaceDetection:
            for prediction in FacePredictions:
                try:
                    x_min, y_min, x_max, y_max = prediction.get('x_min', 0), prediction.get('y_min', 0), prediction.get(
                        'x_max', 0), prediction.get('y_max', 0)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    # Display name and rounded confidence in the top left corner of the box
                    text = f"Face: {prediction['userid']} - {round(prediction['confidence'], 2)}"
                    cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                                cv2.LINE_AA)
                except Exception:
                    continue
        if ObjectDetection:
            for prediction in ObjectPredictions:

                try:
                    if FaceDetection:
                        if prediction["label"] == "person":
                            continue
                    x_min, y_min, x_max, y_max = prediction.get('x_min', 0), prediction.get('y_min', 0), prediction.get(
                        'x_max', 0), prediction.get('y_max', 0)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    # Display name and rounded confidence in the top left corner of the box
                    text = f"Object: {prediction['label']} - {round(prediction['confidence'], 2)}"

                    cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                                cv2.LINE_AA)
                except Exception:
                    continue

        cv2.imshow(f'CodeProject AI || Modules: {"Face" if FaceDetection else ""}{" and " if ObjectDetection and FaceDetection else ""}{"Object" if ObjectDetection else ""}', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        Frame = frame


print("Running Tests")
print("Starting Test 1")
try:
    stat = requests.get(f"http://{IP}:{PORT}").status_code
except Exception as E:
    print("Test 1 failed")
    exit(f"ERROR: {E}")
if stat == 404:
    print("Test Failed")
    raise requests.exceptions.RequestException(f"URL {IP}:{PORT} is not accessible")
else:
    print(stat)
    print("Test 1 passed")

idk = threading.Thread(target=DoProcess)
while True:

    Video()
    if Restart:
        Restart = False
        continue
    else:
        break

# Release capture and close window
cap.release()
cv2.destroyAllWindows()
Stop = True
