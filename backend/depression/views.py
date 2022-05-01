from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .models import *
from django.conf import settings
from rest_framework.views import APIView
from django.contrib.auth import authenticate
from django.contrib.auth.models import auth
from rest_framework.authtoken.models import Token

from rest_framework.permissions import IsAuthenticated

from keras.models import load_model
import cv2
import numpy as np
import os
import json
import librosa

# Create your views here.

# Register


class Register(APIView):

    def post(self, request):

        data = json.loads(request.body)

        firstname = data['firstname']
        lastname = data['lastname']
        address = data['address']
        email = data['email']
        password = data['password']

        if User.objects.filter(email=email).exists():
            return JsonResponse(
                {
                    "msg": "This email is registered already."
                }, status=404
            )
        else:
            UserInstance = User.objects.create(
                email=email,
                username=firstname + " " + lastname,
                first_name=firstname,
                last_name=lastname,
                address = address
            )

            UserInstance.set_password(password)
            UserInstance.save()
            
            auth.login(request, UserInstance)
            token, _ = Token.objects.get_or_create(user=UserInstance)
            return JsonResponse(
                {
                    "token": token.key,
                    "id": UserInstance.id,
                    "name": UserInstance.first_name + " " + UserInstance.last_name,
                    "email": UserInstance.email
                }, status=200)

# Login


class Login(APIView):

    def post(self, request):

        data = json.loads(request.body)

        email = data['email']
        password = data['password']

        if User.objects.filter(email=email).exists():
            Username = User.objects.get(email=email).username
        else:
            return JsonResponse(
                {
                    "msg": "You are not a registered user try again"
                }, status=404
            )

        user = authenticate(username=Username, password=password)

        if user is not None:
            auth.login(request, user)
            token, _ = Token.objects.get_or_create(user=user)
            return JsonResponse(
                {
                    "token": token.key,
                    "id": user.id,
                    "name": user.first_name + " " + user.last_name,
                    "email": user.email
                }, status=200)
        else:
            return JsonResponse(
                {
                    "msg": "You are not a registered user try again"
                }, status=404
            )

# Logout


class Logout(APIView):

    def post(self, request):
        if request.user.is_anonymous:
            return JsonResponse(False, status=404)
        else:
            auth.logout(request)
            return JsonResponse(True, status=200)

##############################################################################################################

class VideoSubmit(APIView):
    
    permission_classes = [IsAuthenticated]
    
    def videoPredict(self, vid_name):

        model = load_model(os.path.join(settings.BASE_DIR, 'model_best_new1.h5'))
        face_cascade = cv2.CascadeClassifier(os.path.join(
            settings.BASE_DIR, 'haarcascade_frontalface_alt.xml'))
        dep = 0

        video = os.path.join(settings.MEDIA_ROOT, vid_name)
        vid = cv2.VideoCapture(video)

        while(vid.isOpened()):
            ret, frame = vid.read()
            if ret:
                faces = face_cascade.detectMultiScale(frame, 1.3, 5)
                for (x, y, w, h) in faces:
                    face = frame[y:y+h, x:x+w]
                    face = cv2.resize(face, (48, 48))
                    nomalized_image = face / 255.0
                    reshaped_image = np.reshape(nomalized_image, (1, 48, 48, 3))
                    result = model.predict(reshaped_image)
                    dep += result

            else:
                break

        vid.release()
        cv2.destroyAllWindows()

        return (((dep[0][0] + dep[0][2] + dep[0][3] + dep[0][5]) / sum(dep[0])*100))
    
    
    def post(self, request):
        
        instance = DepressionnModel.objects.create(
            FacialVideo = request.FILES['file'],
            UserID = User.objects.get(username=request.user)
        )
        instance.save()

        vid_name = str(DepressionnModel.objects.get(id=instance.id).FacialVideo)
        result = self.videoPredict(vid_name)

        instance.FacialOutput = result
        instance.save()

        if(result >= 50):
            context = {
                "id": instance.id,
                "result": result,
                "message": "Through your facial emotions you are having a possibility to have depression. To get the final depression probability you have to upload the voice note as well."
            }
        else:
            context = {
                "id": instance.id,
                "result": result,
                "message": "Looks like you do not have depression from your face! To get the final depression probability you have to upload the voice note as well."
            }

        return JsonResponse(context, status=200)
   

#############################################################################################################

class AudioSubmit(APIView):
    
    permission_classes = [IsAuthenticated]
    
    
    def extract_features(self, path):

        data, sample_rate = librosa.load(path)

        # ZCR
        result = np.array([])
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
        result = np.hstack((result, zcr))  # stacking horizontally

        # Chroma_stft
        stft = np.abs(librosa.stft(data))
        chroma_stft = np.mean(librosa.feature.chroma_stft(
            S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma_stft))  # stacking horizontally

        # MFCC
        mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mfcc))  # stacking horizontally

        # Root Mean Square Value
        rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
        result = np.hstack((result, rms))  # stacking horizontally

        # MelSpectogram
        mel = np.mean(librosa.feature.melspectrogram(
            y=data, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))  # stacking horizontally

        return result


    def audioPredict(self, audio_name):

        model = load_model(os.path.join(settings.BASE_DIR, 'trained_model.h5'))
        audio = os.path.join(settings.MEDIA_ROOT, audio_name)
        features = self.extract_features(audio)
        features = np.expand_dims(features, axis=1)
        features = np.expand_dims(features, axis=0)
    
        prediction = model.predict(features)
        
        return (((prediction[0][0] + prediction[0][2] + prediction[0][5]) / sum(prediction[0]))*100)
        
    
    def post(self, request):
        
        instance = DepressionnModel.objects.get(id=request.POST['post_id'])

        instance.Audio = request.FILES['audio']
        instance.save()

        audio_name = str(DepressionnModel.objects.get(id=instance.id).Audio)
        result = self.audioPredict(audio_name)

        instance.AudioOutput = result
        instance.FinalResult = (result + instance.FacialOutput) / 2
        instance.save()

        if(result >= 50):
            context = {
                "id": instance.id,
                "result": result,
                "message": "Through your audio emotions you are having a possibility to have depression. To get the final depression probability click the button,"
            }
        else:
            context = {
                "id": instance.id,
                "result": result,
                "message": "Looks like you do not have depression from your face! To get the final depression probability click the button."
            }

        return JsonResponse(context, status=200)
    
    
class FinalResult(APIView):
    
    def post(self, request):
        
        data = json.loads(request.body)
        
        context = {
            "result": DepressionnModel.objects.get(id=data['id']).FinalResult,
            "doctor": "Bla"
        }

        return JsonResponse(context, status=200)
    
    
class LatestResult(APIView):
    
    def get(self, request):

        context = {
            "email": User.objects.get(username=request.user).email,
            "result": DepressionnModel.objects.latest('UploadedDate').FinalResult,
            "doctor": "Bla"
        }

        return JsonResponse(context, status=200)