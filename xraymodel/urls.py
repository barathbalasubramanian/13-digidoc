
from django.urls import path
from .views import *
from django.conf.urls.static import static
from django.conf import settings


urlpatterns = [
    path("" ,explore),
    path("generate/",type_of_scan_generating),
    path("diseaseprediction/" , Disease_prediction),  
    path("pimplesdetect/" , pimplefun ,name="pimplesfeed"),
    path("pimplesdetection/" , test_live_page),
    path("" , test_live_page),

    path("xray/" , x_ray),
    path("chatbot/" , chatbot),




    path("login/",user_login,name="login"),
    path("register/",register,name="register"),
    path("create_asana/",create_asana,name="create-asana"),

    path("create_asana/",create_asana,name="create-asana"),

    #train - trainer side
    path("view_trained/",view_trained,name="view-trained"),
    path("view_posture/<int:asana_id>",view_posture,name="view-posture"),
    path("edit_postures/<int:posture_id>",edit_posture,name="edit-posture"),
    

    #test - user side
    path("user_view_asana/",user_view_asana,name="user-view-asana"),
    path("user_view_posture/<int:asana_id>",user_view_posture,name="user-view-posture"),
    path("get_posture/<int:posture_id>",get_posture,name="get-posture"),


    #api
    path("get_posture_dataset/",get_posture_dataset,name="get-posture-dataset"),
]


urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)