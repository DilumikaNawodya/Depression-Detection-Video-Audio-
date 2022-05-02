from django.urls import path
from . import views


urlpatterns = [
    
    path('register/',views.Register.as_view()),
    path('login/',views.Login.as_view()),
    path('logout/',views.Logout.as_view()),
    
    path('submit-video-form/', views.VideoSubmit.as_view()),
    path('submit-audio-form/', views.AudioSubmit.as_view()),
    
    path('latest-result/', views.LatestResult.as_view()),
    path('final-result/', views.FinalResult.as_view()),
]