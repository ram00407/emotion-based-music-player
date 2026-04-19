import cv2
import argparse
import time
import os
import Update_Model
import glob
import random
import eel
#import winsound
frequency = 2500
duration = 1000

eel.init('WD_INNOVATIVE')
emotions = ["angry", "happy", "sad", "neutral"]
fishface = cv2.face.FisherFaceRecognizer_create()
font = cv2.FONT_HERSHEY_SIMPLEX

parser = argparse.ArgumentParser(description="Options for emotions-based music player (Updating the model)")
parser.add_argument("--update", help="Call for taking new images and retraining the model.", action="store_true")
args = parser.parse_args()
facedict = {}
video_capture = cv2.VideoCapture(0)
facecascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def crop(clahe_image, face):
    for (x, y, w, h) in face:
        faceslice = clahe_image[y:y+h, x:x+w]
        faceslice = cv2.resize(faceslice, (350, 350))
        facedict["face%s" % (len(facedict) + 1)] = faceslice
    return faceslice

def grab_face():
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray)

    # Add project information text to the frame
    text1 = "AD Project GT12"
    text2 = "emotion based music player (this is the captured frame)"

    # Put text onto the video frame
    cv2.putText(frame, text1, (10, 30), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, text2, (10, 60), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    

    # Display the captured video
    cv2.imshow("Video Capture", frame)

    return clahe_image

def detect_face():
    clahe_image = grab_face()
    if clahe_image is None:
        return
    face = facecascade.detectMultiScale(clahe_image, scaleFactor=1.1, minNeighbors=15, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(face) >= 1:
        faceslice = crop(clahe_image, face)
    else:
        print("No/Multiple faces detected!!, passing over the frame")

def save_face(emotion):
    print("\n\nLook " + emotion + " until the timer expires and keep the same emotion for some time.")
    # winsound.Beep(frequency, duration)
    print('\a')

    for i in range(0, 5):
        print(5 - i)
        time.sleep(1)

    while len(facedict.keys()) < 16:
        detect_face()

    for i in facedict.keys():
        path, dirs, files = next(os.walk("dataset/%s" % emotion))
        file_count = len(files) + 1
        cv2.imwrite("dataset/%s/%s.jpg" % (emotion, (file_count)), facedict[i])
    facedict.clear()

def update_model(emotions):
    print("Update mode for model is ready")
    checkForFolders(emotions)

    for i in range(0, len(emotions)):
        save_face(emotions[i])
    print("Collected the images, looking nice! Now updating the model...")
    Update_Model.update(emotions)
    print("Model train successful!!")

def checkForFolders(emotions):
    for emotion in emotions:
        if os.path.exists("dataset/%s" % emotion):
            pass
        else:
            os.makedirs("dataset/%s" % emotion)

def identify_emotions():
    prediction = []
    confidence = []

    if not facedict:
        print("No face detected in timeframe. Defaulting to neutral.")
        return "neutral"

    for i in facedict.keys():
        pred, conf = fishface.predict(facedict[i])
        cv2.imwrite("images/%s.jpg" % i, facedict[i])
        prediction.append(pred)
        confidence.append(conf)
    output = emotions[max(set(prediction), key=prediction.count)]
    print("You seem to be %s" % output)
    facedict.clear()
    return output

count = 0
@eel.expose
def getEmotion():
    # Flush cv2 buffer to get fresh frames instead of stale ones
    for _ in range(5):
        video_capture.read()
        
    count = 0
    while True:
        count = count + 1
        detect_face()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if args.update:
            update_model(emotions)
            break
        elif count == 10:
            fishface.read("model.xml")
            return identify_emotions()

    video_capture.release()
    cv2.destroyAllWindows()

# Start Eel and OpenCV window
eel.start('main.html')
