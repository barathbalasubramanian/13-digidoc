from django.shortcuts import render
from django.http import StreamingHttpResponse
import cv2
from PIL import Image
import io
from django.views.decorators import gzip
from tensorflow import keras
from keras.utils import img_to_array
import numpy as np
import imageio
import base64
import matplotlib.pyplot as plt
from scipy import ndimage
import pandas as pd
from django.http import JsonResponse
import yolov5
import shutil
import threading

# Model Labels 
# Model 1
BRAIN_TUMOR_LABELS = ["glioma_tumor","meningioma_tumor","no_tumor","pituitary_tumor"]

# For X-ray Analysis

def explore(request) :
    return render(request, 'home.html')

def x_ray(request) :

    print('x-ray PAGE RUNNING')
    if request.method == "POST" :
        if 'tumor' in request.POST :
            print('tumor')
            try :
                myfile = request.FILES['fileUpload']
                content = myfile.read()

                image = Image.open(io.BytesIO(content))
                processed_image = predict_image(image)

                # load the model
                model = keras.models.load_model(r'models\brain_tumor.h5')
                result = model.predict(processed_image)
                result = str(BRAIN_TUMOR_LABELS[np.argmax(result)])
                empty_list = []
                empty_list.append(result)

                context = {
                    'result' : empty_list
                }
                print(empty_list)
                response_data = {
                    'message': empty_list,
                }

                return JsonResponse(response_data)

            except : 
                print( ' exception ')
                return render(request, 'xray.html')
        
        if 'generate' in request.POST :
            print('generate')
            myfile = request.FILES['fileUpload']
            content = myfile.read()

            image = Image.open(io.BytesIO(content))
            processed_image = predict_image(image)
            print('Generating ...')

            # display the same image
            img = cv2.imread(r'xraymodel\static\images\scan_image.png')
            
            xray_image_laplace_gaussian = ndimage.gaussian_laplace(img, sigma=1)
            cv2.imwrite(r"xraymodel\static\images\xray_image_laplace_gaussian.png" , xray_image_laplace_gaussian)

            im1 = cv2.applyColorMap(img, cv2.COLORMAP_AUTUMN)
            im2 = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
            im3 = cv2.applyColorMap(img, cv2.COLORMAP_JET)
            im4 = cv2.applyColorMap(img, cv2.COLORMAP_WINTER)
            im5 = cv2.applyColorMap(img, cv2.COLORMAP_RAINBOW)
            im6 = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)
            im7 = cv2.applyColorMap(img, cv2.COLORMAP_SUMMER)
            im8 = cv2.applyColorMap(img, cv2.COLORMAP_SPRING)
            im9 = cv2.applyColorMap(img, cv2.COLORMAP_COOL)
            im10 = cv2.applyColorMap(img, cv2.COLORMAP_HSV)
            im11 = cv2.applyColorMap(img, cv2.COLORMAP_PINK)
            im12 = cv2.applyColorMap(img, cv2.COLORMAP_HOT)

            cv2.imwrite(r"xraymodel\static\images\im1.png" , im1)
            cv2.imwrite(r"xraymodel\static\images\im2.png" , im2)
            cv2.imwrite(r"xraymodel\static\images\im3.png" , im3)
            cv2.imwrite(r"xraymodel\static\images\im4.png" , im4)
            cv2.imwrite(r"xraymodel\static\images\im5.png" , im5)
            cv2.imwrite(r"xraymodel\static\images\im6.png" , im6)
            cv2.imwrite(r"xraymodel\static\images\im7.png" , im7)
            cv2.imwrite(r"xraymodel\static\images\im8.png" , im8)
            cv2.imwrite(r"xraymodel\static\images\im9.png" , im9)
            cv2.imwrite(r"xraymodel\static\images\im10.png" , im10)
            cv2.imwrite(r"xraymodel\static\images\im11.png" , im11)
            cv2.imwrite(r"xraymodel\static\images\im12.png" , im12)

            print('Generated Successfully')
            gen = ['hi']
            context = {
                'generate_image' : gen
            }
            return render(request, 'xray.html' , context)
            
    else:
        print('else')
        return render(request, 'xray.html')


# Input Image Preprocessing 
def predict_image(image):

    # save the image 
    image.save('xraymodel\static\images\scan_image.png')

    # resize the image
    resize_image = image.resize((180,180))
    ima_array = img_to_array(resize_image)
    resize_image = ima_array / 255
    return resize_image.reshape(1, 180,180,3)


# generating images
def type_of_scan_generating():

        print('Generating ...')
        # display the same image
        xray_image = imageio.v3.imread(r'xraymodel\static\images\scan_image.png')
        
        xray_image_laplace_gaussian = ndimage.gaussian_laplace(xray_image, sigma=1)
        cv2.imwrite(r"xraymodel\static\images\xray_image_laplace_gaussian.png" , xray_image_laplace_gaussian)

        img = cv2.imread(r'xraymodel\static\images\scan_image.png')
        im1 = cv2.applyColorMap(img, cv2.COLORMAP_AUTUMN)
        im2 = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
        im3 = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        im4 = cv2.applyColorMap(img, cv2.COLORMAP_WINTER)
        im5 = cv2.applyColorMap(img, cv2.COLORMAP_RAINBOW)
        im6 = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)
        im7 = cv2.applyColorMap(img, cv2.COLORMAP_SUMMER)
        im8 = cv2.applyColorMap(img, cv2.COLORMAP_SPRING)
        im9 = cv2.applyColorMap(img, cv2.COLORMAP_COOL)
        im10 = cv2.applyColorMap(img, cv2.COLORMAP_HSV)
        im11 = cv2.applyColorMap(img, cv2.COLORMAP_PINK)
        im12 = cv2.applyColorMap(img, cv2.COLORMAP_HOT)

        cv2.imwrite(r"xraymodel\static\images\im1.png" , im1)
        cv2.imwrite(r"xraymodel\static\images\im2.png" , im2)
        cv2.imwrite(r"xraymodel\static\images\im3.png" , im3)
        cv2.imwrite(r"xraymodel\static\images\im4.png" , im4)
        cv2.imwrite(r"xraymodel\static\images\im5.png" , im5)
        cv2.imwrite(r"xraymodel\static\images\im6.png" , im6)
        cv2.imwrite(r"xraymodel\static\images\im7.png" , im7)
        cv2.imwrite(r"xraymodel\static\images\im8.png" , im8)
        cv2.imwrite(r"xraymodel\static\images\im9.png" , im9)
        cv2.imwrite(r"xraymodel\static\images\im10.png" , im10)
        cv2.imwrite(r"xraymodel\static\images\im11.png" , im11)
        cv2.imwrite(r"xraymodel\static\images\im12.png" , im12)

        print('Generated Successfully')
        gen = ['hi' , 'bro']
        context = {
            'generate_image' : gen
        }
        return render(request, 'xray.html' , context)

# Disease Prediction
def Disease_prediction(request) :

    if request.method == 'GET':
        selected_value = request.GET.get('selected_value')
        print(selected_value,'getmethod')

    # pre_list = [] 
    # df_des1 = pd.read_csv(r"models\csv_files\symptom_Description.csv")
    # predicted_description = df_des1[ df_des1["Disease"]  == 'Malaria' ]
    # pre_list.append(list(predicted_description['Description'])[0])
    # print(pre_list)

    # df_pre = pd.read_csv(r"models\csv_files\symptom_precaution.csv")
    # predicted_Precaution = df_pre[ df_pre["Disease"]  == 'Malaria' ]
    # Precaution_list = []
    # for i in predicted_Precaution:
    #     Precaution_list.append(list(predicted_Precaution.iloc[0:1,:][i]))
    # dataframe_pre = pd.DataFrame(data= Precaution_list[1:],columns=["Precaution"])
    # print(dataframe_pre)


    # df_symptom = pd.read_csv(r"models\csv_files\disease_symptom.csv")
    # list_disease = list(set(list(df_symptom["Disease"])))
    # list_disease.sort()
    # predicted_symptom = df_symptom[ df_symptom["Disease"]  == 'AIDS' ]
    # predicted_symptom.drop_duplicates(inplace=True)
    # symptom_list = []
    # for i in predicted_symptom.iloc[:,1:]:
    #     val = list(predicted_symptom[i])
    #     for j in val:
    #         symptom_list.append(j)
    # symptom_list = list(set(symptom_list))
    # dataframe_symp = pd.DataFrame(data= symptom_list,columns=["Symptom"])
    # print(dataframe_symp)

    context = {
        'desc' : 'pre_list'
    }

    return render(request, 'diseaseprediction.html' , context)


# Pimples detection

detected_object_name_list = []
model = yolov5.load(r"models\pimples_weights.pt")

#to capture video class
class VideoCamera(object):

    global model    
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame

        results = model(image, size=640)
        results.save()

        # get_array = results.xyxy[0]
        # get_array = get_array.tolist()
        
        # if len(get_array) == 0:
        #     pass
        # else:
        #     for i in get_array:
        #         last = classes_list[round(i[-1])]
        #         detected_object_name_list.append(last)
        
        # if len(detected_object_name_list) == 0:
        #     print("No odject is detected!!!")
        # else:
        #     print("Detected objects are :","|".join(detected_object_name_list))
        
        output_img = cv2.imread(r"runs\detect\exp\image0.jpg")
    
        _, jpeg = cv2.imencode('.jpg', output_img)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

def gen(camera):
    
    model = yolov5.load(r"models\pimples_weights.pt")

    while True:
        try:
            frame = camera.get_frame()

            shutil.rmtree(r"runs")

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        except Exception as e:
            print(f"Exception gen: {e}")

def pimplefun(request):

    try:
        cam = VideoCamera()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except Exception as e:
        print(f"Exception pimplefun: {e}")
        pass
    return render(request, 'pimple.html')

def test_live_page(request) :
    return render(request, 'pimple.html')

def chatbot(request) :
    print('Chatbox')
    return render(request , "chatbox.html" )















from django.shortcuts import render,redirect
from django.http import JsonResponse
from datetime import datetime
from django.contrib.auth.models import User
from digitaldoctor.settings import BASE_DIR, MEDIA_ROOT
from .models import *
from .forms import *
from django.contrib.auth import authenticate,login
from django.utils import timezone
from django.contrib.auth.decorators import login_required,user_passes_test
from .permissions import *
import ast
import pandas as pd
import base64


def register(request):
    if request.method == "POST":
        email = request.POST.get("email")
        first_name = request.POST.get("first_name")
        last_name = request.POST.get("last_name")
        password = request.POST.get("password")
        User.objects.create_user(username=email,email=email,first_name=first_name,last_name=last_name,password=password)
        return redirect ('login')
    return render(request , "users/user_register.html")


def user_login(request):
    if request.method == "POST":
        username = request.POST.get("email")
        password = request.POST.get("password")
        user = authenticate(username=username,password=password)
        if user is not None:
            login(request,user)
            return render( request , "users/user_register.html")
    return render(request,"users/login.html")


@login_required
@user_passes_test(check_trainer)
def view_trained(request):
    trained_asanas = Asana.objects.filter(created_by = request.user)
    return render(request,"users/view_trained.html",{
        "trained_asanas":trained_asanas,
    })


@login_required
@user_passes_test(check_trainer)
def view_posture(request,asana_id):
    postures = Posture.objects.filter(asana=Asana.objects.get(id=asana_id)).order_by('step_no')
    return render(request,"users/view_posture.html",{
        "postures":postures,
    })


@login_required
@user_passes_test(check_trainer)
def create_asana(request):
    form = AsanaCreationForm()
    if request.method == "POST":
        form = AsanaCreationForm(request.POST)
        if form.is_valid():
            asana = form.save(commit=False)
            asana.created_by = request.user
            asana.created_at = datetime.now(tz=timezone.utc)
            asana.last_modified_at = datetime.now(tz=timezone.utc)
            asana.save()
            for i in range(1,asana.no_of_postures+1):
                Posture.objects.create(name=f"Step-{i}",asana=asana,step_no=i,first_trained_at=datetime.now(tz=timezone.utc))
            return redirect("view-trained")
    return render(request,"users/create_asana.html",{
        'form':form,
    })


@login_required
@user_passes_test(check_student)
def home(request):
    return render(request,"users/home.html")


@login_required
@user_passes_test(check_trainer)
def edit_posture(request,posture_id):
    posture = Posture.objects.get(id=posture_id)
    if request.method == "POST":
        if "meta_details" in request.POST:
            form = EditPostureForm(request.POST,instance=posture)
            posture.last_modified_at = datetime.now(tz=timezone.utc)
            if form.is_valid():
                form.save()
        else:
            name = f"{posture.asana.name}_{posture.step_no}.csv"
            dataset = ast.literal_eval(request.POST["dataset"])
            dataset = pd.DataFrame(dataset)
            dataset = dataset.transpose()
            dataset.to_csv(f'./media/{name}',index=False,header=False)
            posture.dataset.name = name
            decoded_data=base64.b64decode(request.POST['snapshot'])
            img_file = open(f"./media/images/{posture_id}.png", 'wb')
            img_file.write(decoded_data)
            img_file.close()
            posture.snap_shot.name = f"./images/{posture_id}.png"
            posture.last_modified_at = datetime.now(tz=timezone.utc)
            posture.save()
    form = EditPostureForm(instance=posture)
    return render(request, "users/edit_posture.html",{
        "form":form,
        "posture":posture,
    })

#model testing user side
@login_required
@user_passes_test(check_student)
def user_view_asana(request):
    asanas = Asana.objects.all()
    trained_asanas = []
    for asana in asanas:
        if asana.related_postures.filter(dataset="").exists():
            pass
        else:
            trained_asanas.append(asana)
    return render(request,"users/user_view_asana.html",{
        "asanas":trained_asanas,
    })


@login_required
@user_passes_test(check_student)
def user_view_posture(request,asana_id):
    postures = Posture.objects.filter(asana=Asana.objects.get(id=asana_id)).order_by('step_no')
    return render(request,"users/user_view_posture.html",{
        "postures":postures,
    })




@login_required
@user_passes_test(check_student)
def get_posture(request,posture_id):
    if request.method == "GET":
        link = str(Posture.objects.get(id=posture_id).snap_shot.url)
        return JsonResponse({"url":link})
    else:
        return JsonResponse({"error": "expected GET method"})


@login_required
@user_passes_test(check_student)
def get_posture_dataset(request):
    if request.method == "GET":
        data = {}
        posture_id = request.GET['posture_id']
        posture = Posture.objects.get(id=posture_id)
        dataset = pd.read_csv(posture.dataset.path,header=None)
        dataset = dataset.values.tolist()
        data["dataset"] = dataset
        data["snapshot"] = posture.snap_shot.url
        return JsonResponse(data)
    else:
        return JsonResponse(status=400,data={"error":"Bad request"})

















