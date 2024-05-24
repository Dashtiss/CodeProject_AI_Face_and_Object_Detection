import cv2
import os
import colorama as clr
import CodeProject as CodeProjectAI

IP, PORT, TIMEOUT = "192.168.1.187", 32168, 10

# Open the default webcam (index 0)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cpaiface = CodeProjectAI.CodeProjectAIFace(IP, PORT, TIMEOUT)
def sendRequests(Img: bytes):

    AiRecognition = CodeProjectAI.CodeProjectAIFace(IP, PORT, TIMEOUT, min_confidence=0.7)
    predictions = AiRecognition.recognize(Img)
    FacePredictions = predictions
    return predictions


def Video():
    i = 0
    img = 0
    while True:
        i += 1
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Check if frame is read correctly
        if not ret:
            print("Failed to capture frame!")
            break
        _, png = cv2.imencode('.png', frame)
        FacePredictions = sendRequests(png.tobytes())
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
        # Display the resulting frame
        cv2.imshow('Training', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break
        name = input("What is the name of the person on the image:  ")
        cpaiface.register(name, png)

print(f"""TRAINING
{clr.Fore.RED}WARNING: THIS CAN BREAK YOUR AI SYSTEM.
Please read WARNING.md before doing this
{clr.Fore.RESET}

This will take a picture and you type in a name and it will add it to your AI Face Recognition system

There can only be one person per picture when doing this

Press enter to start
""")

input()

Video()

# Release capture and close window
cap.release()
cv2.destroyAllWindows()
