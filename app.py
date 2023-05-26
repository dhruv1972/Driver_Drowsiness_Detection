from flask import Flask, render_template, Response, request
import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

app = Flask(__name__, template_folder='Templates', static_folder='Static')

mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

model = load_model('models/Open_Closed_Eyes_Detection.h5')
path = os.getcwd()
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def gen_frames():  # Generate camera frames
    cap = cv2.VideoCapture(0)
    count=0
    score=0
    thicc=2
        
    while(True):
        rpred_max_index = -1
        lpred_max_index = -1
        ret, frame = cap.read()
        height, width = frame.shape[:2] 
    #     print(frame.shape)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     print(gray.shape)
        
        faces = face.detectMultiScale(gray,minNeighbors=5, scaleFactor=1.1, minSize=(25,25))
        left_eye = leye.detectMultiScale(gray)
        right_eye =  reye.detectMultiScale(gray)

        cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

        for (x,y,w,h) in right_eye:
            cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )
            r_eye=frame[y:y+h,x:x+w]
            count=count+1
            r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye,(24,24))
            r_eye= r_eye/255
            r_eye=  r_eye.reshape(24,24,-1)
            r_eye = np.expand_dims(r_eye,axis=0)
            rpred = model.predict(r_eye)
    #         print(rpred)
            rpred_max_index = rpred.argmax()
    #         print(max_index)

        for (x,y,w,h) in left_eye:
            l_eye=frame[y:y+h,x:x+w]
            count=count+1
            l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
            l_eye = cv2.resize(l_eye,(24,24))
            l_eye= l_eye/255
            l_eye=l_eye.reshape(24,24,-1)
            l_eye = np.expand_dims(l_eye,axis=0)
            lpred = model.predict(l_eye)
            lpred_max_index = lpred.argmax()

        if(rpred_max_index==0 and lpred_max_index==0):
            score=score+1
            cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    #    if(rpred[0]==1 or lpred[0]==1):
        else:
            score=score-1
            cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
            
        if(score<0):
            score=0   
        cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        
        if(score>10):
            #person is feeling sleepy so we beep the alarm
            cv2.imwrite(os.path.join(path,'image.jpg'),frame)
            try:
                sound.play()
                
            except:  # isplaying = False
                pass

            if(thicc<16):
                thicc= thicc+2
            else:
                thicc=thicc-2
                if(thicc<2):
                    thicc=2
            cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
        
        elif (score < 10):
            sound.stop()
        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
    cap.release()



@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
    