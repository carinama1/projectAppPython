from flask import Flask, render_template, Response, request
import cv2
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
import cv2
import numpy as np
import base64
import uuid
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# get a UUID - URL safe, Base64
app = Flask(__name__)
app.testing = True
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


def get_a_uuid():
    r_uuid = base64.urlsafe_b64encode(uuid.uuid4().bytes)
    return str(r_uuid).replace('=','').replace('\'','').replace('b','')


session_id = get_a_uuid()

def read_json(file_path='static/json/emotion.json'):
    f = open(file_path)
    data = json.load(f)
    return data

def generate_json(session_name):
    session_id2 = get_a_uuid()
    expressions_dictionary["session_id"] = session_id2
    expressions_dictionary["session_name"] = session_name
    json_object = json.dumps(expressions_dictionary, indent=4)
    with open("static/json/emotion.json", "w") as outfile:
        outfile.write(json_object)

def process_emotion_data(data):
    keys = list(data['emotions'].keys())
    values = list(data['emotions'].values())
    session_name = data['session_name']
    genderKeys = list(data['gender'].keys())
    gender = ''
    if data['gender'][genderKeys[0]] > data['gender'][genderKeys[1]]:
        gender = genderKeys[0]
    else:
        gender = genderKeys[1]

    total = 0
    for v in values:
        total += v
    results = {}
    for index, key in enumerate(keys):
        results[key] = values[index]/total * 100
    return {"emotions": results, "gender": gender, "session_name":session_name }

def show_graph():
    # creating the dataset
    json_data = read_json()
    data = process_emotion_data(json_data)
    gender = data['gender']
    keys = list(data['emotions'].keys())
    values = list(data['emotions'].values())
    session_name = data['session_name']

    # creating the bar plot
    plt.bar(keys, values, color='maroon',
            width=0.4)

    plt.xlabel("Emotions")
    plt.ylabel("Frequency (%)")
    plt.title("Emotions Capture of " + "(" + gender + ")")
    plt.savefig('static/images/result.png')
    return session_name

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
            expressions_dictionary["emotions"][label] += 1
            expressions_dictionary["gender"][gender] += 1
            label_position = (x, y - 20)
            emotion_position = (x + padding, y - 20)
            cv2.putText(frame, label, label_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, gender, emotion_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def generate_results_list():
    directory = 'static/json/results'
    result_list = {}
    for index, filename in enumerate(os.listdir(directory)):
        file_path = os.path.join(directory, filename)
        json_data = read_json(file_path)
        session_id = json_data['session_id']
        session_name = json_data['session_name']
        result_list[str(index)] = {}
        result_list[str(index)]['label'] = session_name
        result_list[str(index)]['value'] = session_id
    return result_list
            

def saveData():
    json_data = read_json()
    session_id = json_data['session_id']
    json_data = json.dumps(json_data, indent=4)
    path = "static/json/results/"+session_id+".json"
    print("saving Data To: " + path)
    with open(path, "w") as outfile:
        outfile.write(json_data)
    return json_data

def gen_frames():
    # generate frame by frame from camera
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

@app.route('/test')
def test():
    session_name = request.args.get('session_name')
    print("====================================")
    print(session_name)
    return session_name

@app.route('/')
def menu():
    return render_template('test/index.html')

@app.route('/product')
def product():
    global session_id
    return render_template('product/index.html', session_id=session_id)

@app.route('/live')
def index():
    """Video streaming home page."""
    # global beginPredict
    return render_template('live/index.html')

@app.route('/generate')
def routeGenerateJson():
    session_name = request.args.get('session_name')
    generate_json(session_name)
    return ''

@app.route('/save')
def saveResults():
    saveData()
    return ''

@app.route('/view-past')
def viewAll():
    list = generate_results_list()
    return '<div>Test</div>'

@app.route("/results")
def results():
    session_name = str(show_graph())
    return render_template('results/index.html', session_name=session_name)

if __name__ == '__main__':
    app.run(debug=True)
