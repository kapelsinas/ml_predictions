from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
from .ml_utils import predict
import numpy as np
from PIL import Image
from io import BytesIO

class ImagePredictionView(View):
    def get(self, request):
        return render(request, 'image_prediction.html')

    @method_decorator(csrf_exempt)
    def post(self, request):
        # Extract the image from the request
        image = request.FILES['image']
        image = Image.open(image)

        # Call the prediction function from ml_utils.py
        prediction_result = predict(np.array(image))

        # Return the prediction result as a JSON response
        return JsonResponse({'prediction': prediction_result})


class InfoView(View):
    def get(self, request):
        return render(request, 'info.html')