from flask import Flask, render_template, request, redirect, url_for, Response
import cv2, os, numpy as np
from datetime import datetime
import sqlite3

app = Flask(__name__)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

MODEL_PATH = 'trained_model/face_model.yml'
DB_PATH = 'attendance/attendance.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS attendance (id INTEGER PRIMARY KEY, name TEXT, time TEXT)")
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    name = request.form['name']
    user_dir = f'dataset/{name}'
    os.makedirs(user_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)
    count = 0
    while count < 50:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            count += 1
            face = gray[y:y+h, x:x+w]
            cv2.imwrite(f"{user_dir}/{count}.jpg", face)
        cv2.imshow("Register Face", frame)
        if cv2.waitKey(1) == 27 or count >= 50: break
    cap.release()
    cv2.destroyAllWindows()
    return redirect(url_for('home'))

@app.route('/train')
def train():
    faces, labels = [], []
    label_map = {}
    for i, person in enumerate(os.listdir('dataset')):
        label_map[i] = person
        for img in os.listdir(f'dataset/{person}'):
            path = f'dataset/{person}/{img}'
            face = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            faces.append(face)
            labels.append(i)
    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_PATH)
    with open('trained_model/labels.txt', 'w') as f:
        for k, v in label_map.items():
            f.write(f"{k},{v}\n")
    return redirect(url_for('home'))

def gen_frames():
    if not os.path.exists(MODEL_PATH): return
    recognizer.read(MODEL_PATH)
    labels = {}
    with open('trained_model/labels.txt') as f:
        for line in f:
            k, v = line.strip().split(',')
            labels[int(k)] = v
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x,y,w,h) in faces:
            roi = gray[y:y+h, x:x+w]
            id_, conf = recognizer.predict(roi)
            if conf < 70:
                name = labels[id_]
                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("INSERT INTO attendance (name, time) VALUES (?, ?)", (name, now))
                conn.commit()
                conn.close()
                cv2.putText(frame, name, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attendance')
def attendance():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM attendance ORDER BY id DESC")
    data = c.fetchall()
    conn.close()
    return render_template("attendance.html", data=data)

if __name__ == '__main__':
    app.run(debug=True)
