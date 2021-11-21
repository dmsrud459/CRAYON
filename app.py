#from datetime import time
from flask import Flask, render_template, redirect, request, url_for,Response, jsonify
#from flask.wrappers import Request
#from werkzeug.utils import secure_filename
import json ,os
import shutil #폴더 삭제
import queue, os, threading
from numpy.core.fromnumeric import ptps
import sounddevice as sd
import soundfile as sf
import time
import cv2
import librosa
import matplotlib.pyplot as plt
import librosa.display
import sklearn
from keras.preprocessing import image as imagee


app = Flask(__name__)
class_name = [] #class 이름을 담을 배열



@app.route('/INDEX') #학습모델 선택 페이지
def newproject():
    return render_template('index.html')

@app.route('/IMAGE',methods=["GET","POST"]) #이미지 학습 페이지
def IMAGE():
    return render_template('3_main.html')

@app.route('/SOUND', methods=["GET","POST"]) #소리 학습 페이지
def soundpage():
    return render_template('main(sound).html')

# @app.route('/trypage', methods=["GET","POST"]) #모션 학습 페이지
# def trypage():
#     return render_template('trypage.html')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@app.route('/MAINSOUND' ,methods=["POST"])
def MAINSOUND():
    if request.method =='POST':
        if os.path.isfile("deep_sound/snd_npz.npz")=="True" or os.path.isfile("sound_model/snd_model.h5")=="True":
            os.remove("deep_sound/snd_npz.npz")
            os.remove("sound_model/snd_model.h5")
            os.remove("tamp_sound/s_img.jpg")
            print("통신 성공")

        class_file = []; # 업로드 파일 담을 배열
        folder_path = [];
        file_names = [];


        # print(request.form['data2']) #문자 확인
        # print(request.files.getlist('data1')) # 파일 확인
        file_count = request.form.getlist('file_count')
        print(file_count) #file_count
        class_name = request.form.getlist('class_name')
        print(class_name) #class_name
        
    
        if len(file_count) == 2:
            for i in range(1,3):
                os.mkdir("./load_sound/class_{}".format(i)) #class 개수 만큼 폴더생성
                class_file.append(request.files.getlist('class{}'.format(i))) #class_file

        else:
            for i in  range(1,4):
                os.mkdir("./load_sound/class_%d" % i)
                class_file.append(request.files.getlist('class{}'.format(i))) #class_file
                
        print("이미지 배열")
        print(class_file[0])
        print(class_file[1])

        path_load = './load_sound'
        folder_list = os.listdir(path_load)
        print("폴더 경로")
        print(folder_list)

        for p in range(len(file_count)): #클래스 길이 만큼 for문 반복
            folder_path.append(os.path.join('load_sound/', folder_list[p]))
            print("각 폴더의 경로")
            print(folder_path)
            for f in range(int(file_count[p])): # class별 파일 개수 
                class_file[p][f].save(folder_path[p]+'/'+class_file[p][f].filename) #https://yong0810.tistory.com/22 참고
                file_names.append(class_file[p][f].filename)
        print("파일 이름들")
        print(file_names)

        images = []
        train_X = []
        train_y = []
        cnttt = 0 # 클래스 카운터
        
        c_path = 'load_sound/'
        img_size_shape=(224,224)
        
        
        model_img.save('sound_model/snd_model.h5')
        print("sound_모델저장")
        print(type(class_name))
        #디렉토리 삭제
        for i in range(len(class_name)):
            shutil.rmtree(folder_path[i]) #load폴더의 하위디렉토리/파일 삭제
        
    return jsonify(result = "모델 완성")


@app.route('/SOUNDMODEL', methods=['GET','POST']) #ajax 테스트
def SOUNDMODEL():
    for i in os.listdir('SOUND_NAM/confirm_load'):
        os.remove('SOUND_NAM/confirm_load/'+i)

    max_num = 0 # 가장 높은 정확도 추출할 때 사용
    cnt = 5 # 클래스네임 배열 중 몇번째가 가장 높은 정확도를 가지는지 확인할 때 사용
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    c_path = 'SOUND_NAM/confirm_load/' #업로드 파일 경로
    # img_size_shape=(224,224)

    soundfile = request.files.get('soundfile')
    print(soundfile)
    soundfile.save(c_path+soundfile.filename) 
    print('확인용 파일')

    

    print(cnt)
    print(classes[0][cnt]*100)
    print(class_name[cnt])

    print("두근두근 해당 파일은 {}% 의 정확도로 {}입니다.".format((classes[0][cnt]*100), class_name[cnt]))

    #print("두근두근 {}은(는) {}%의 정확도로 {} 입니다.".format(classes[0][cnt]*100, class_name[cnt]))
    
    accuracy = classes[0][cnt]*100
    accuracy_name = class_name[cnt]

    return jsonify(result = accuracy, result2 = accuracy_name)
    



@app.route('/MAINIMAGE' ,methods=["POST"])
def MAINIMAGE():
    if request.method =='POST':
        if os.path.isfile("deep_learning/img_npz.npz")=="True" or os.path.isfile("model1/img_model.h5")=="True":
            os.remove("deep_learning/img_npz.npz")
            # os.remove("model1/img_model.h5")
            for i in os.listdir('save_earlystop_path'):
                os.remove('save_earlystop_path/'+i)


        # if os.path.isdir("deep_learning/img_npz.npz") or os.path.isdir('model1/img_model.h5') ==  'True' :
        #     shutil.rmtree('deep_learning/img_npz.npz')
        #     shutil.rmtree('model1/img_model.h5')
        # else:
        class_file = []; # 업로드 파일 담을 배열
        folder_path = [];
        file_names = [];
        tamp_file = [];


        # print(request.form['data2']) #문자 확인
        # print(request.files.getlist('data1')) # 파일 확인
        file_count = request.form.getlist('file_count')
        print(file_count) #file_count
        class_name = request.form.getlist('class_name')
        print(class_name) #class_name
        up = request.form.getlist('up')
        print(up)

        tamp_file2=tamp_file.append(request.form.get('class1')) #class_file
        # image = request.form['myImg']

        print(tamp_file2)

      
        if len(file_count) == 2:
            for i in range(1,3): # 글래스가 2개 일때 
                os.mkdir("./load/class_{}".format(i)) #class 개수 만큼 폴더생성
                print("진짜여기")
                print(i)
                print(up)
                if up == "1" and  i == 1:
                    tamp_file.append(request.form.getlist('class1')) #class_file
                    for t in range(int(file_count[i])) : # class별 파일 개수 
                        imgs = tamp_file[t].split(',')
                        imgs[i] = imgs[i] + '='*(4-len(imgs[i])%4)
                        im = Image.open(BytesIO(base64.b64decode(imgs[1])))
                        im.save('temp_img/image{}.png'.format(t),'PNG')
                    for p in os.listdir("temp_img"): # class별 파일 개수 
                        class_file.append(p)
                        
                    class_file.append(request.files.getlist('class2'))#class_file
   
                else:
                    class_file.append(request.files.getlist('class{}'.format(i))) #class_file

    
        print("이미지 배열")
        print(class_file[0])
        print(class_file[1])

        path_load = './load'
        folder_list = os.listdir(path_load)
        print("폴더 경로")
        print(folder_list)

        for p in range(len(file_count)): #클래스 길이 만큼 for문 반복
            folder_path.append(os.path.join('load/', folder_list[p]))
            print("각 폴더의 경로")
            print(folder_path)
            for f in range(int(file_count[p])): # class별 파일 개수 
                class_file[p][f].save(folder_path[p]+'/'+class_file[p][f].filename) #https://yong0810.tistory.com/22 참고
                file_names.append(class_file[p][f].filename)
        print("파일 이름들")
        print(file_names)


        images = []
        train_X = []
        train_y = []
        cnttt = 0 # 클래스 카운터
        
        c_path = 'load/'
        img_size_shape=(224,224)
        
        
        for i in (os.listdir(c_path)):
            path_1 = os.path.join(c_path,i)
            

           

        # 4. 모델 저장(가져올 모델 경로를 정해주세요)
        # model_img.save('model1/img_model.h5')
        print("모델저장")

        #디렉토리 삭제
        for i in range(len(class_name)):
            shutil.rmtree(folder_path[i]) #load폴더의 하위디렉토리/파일 삭제
    return jsonify(result = "모델 완성")



@app.route('/IMAGEMODEL', methods=['GET','POST']) #ajax 테스트
def IMAGEMODEL():
    
    image = request.form['myImg']
    #print(image)
    imgs = image.split(',')
    imgs[1] = imgs[1] + '='*(4-len(imgs[1])%4)
    im = Image.open(BytesIO(base64.b64decode(imgs[1])))
    im.save('image.png','PNG')
    #print(im)
    # canvas 이미지를 가져오기

    print("시작")
    

    #######얼리스탑 모델 불러오기
    
    earlystop_model_names = os.listdir('save_earlystop_path')
    print(earlystop_model_names)

    max_earlystop = float(0);
    max_name = earlystop_model_names[0]

    print(earlystop_model_names)
    print(max_name)


    print("두근두근 {} 은(는) {}%의 정확도로 {} 입니다.".format(model_img, classes[0][cnt]*100, class_name[cnt]))

    accuracy = classes[0][cnt]*100
    accuracy_name = class_name[cnt]

    return jsonify(result = accuracy, result2 = accuracy_name)
   

if __name__ == '__main__':
    app.debug =True 
    app.run(host='0.0.0.0' ,port='8050')

