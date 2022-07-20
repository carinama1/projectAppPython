from flask import Flask, render_template, Response
import cv2
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
import cv2
import numpy as np
app = Flask(__name__)
face_classifier = cv2.CascadeClassifier(
    'models/haarcascade_frontalface_default.xml')
classifier = load_model('models/emotion_model.h5')
model_gender = cv2.face.FisherFaceRecognizer_create()
model_gender.read('models/gender_model.xml')

genders = ["female", "male"]

emotion_labels = ['Angry', 'Disgust', 'Fear',
                  'Happy', 'Neutral', 'Sad', 'Surprise']
padding = 120

camera = cv2.VideoCapture(0)  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)

expressions_dictionary = {"emotions": {'Angry': 0, 'Disgust': 0, 'Fear': 0,
                                       'Happy': 0, 'Neutral': 0, 'Sad': 0, 'Surprise': 0}, "gender": {"male": 0, "female": 0}}


def begin_prediction(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cropped_face = gray[y:y+h, x:x+w]
        gender_face = cv2.resize(cropped_face, (350, 350))
        resized_face = cv2.resize(
            cropped_face, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([resized_face]) != 0:
            roi = resized_face.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            gender_prediction = model_gender.predict(gender_face)
            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            gender = genders[gender_prediction[0]]
            # expressions_dictionary["emotions"][label] += 1
            # expressions_dictionary["gender"][gender] += 1
            label_position = (x, y - 20)
            emotion_position = (x + padding, y - 20)
            cv2.putText(frame, label, label_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, gender, emotion_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


def testFuncion():
    print("TEST")


def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            begin_prediction(frame=frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
