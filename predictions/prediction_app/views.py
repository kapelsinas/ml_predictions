from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
from .ml_utils import predict
import tensorflow as tf
from .breast_ml_utils import create_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import pydicom
from PIL import Image
import logging
from io import BytesIO
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
import json

logger = logging.getLogger(__name__)

model = create_model()

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


class ImagePredictionCanView(View):
    def get(self, request):
        return render(request, 'breast_pred.html')

    @csrf_exempt
    def post(self, request):
        image = request.FILES['image']
        file_extension = str(image).split('.')[-1]
        image_laterality = 'none'
        if file_extension == 'dcm':
            # Load the DICOM files
            ds = pydicom.dcmread(image)
            pil_image = Image.fromarray(ds.pixel_array)
            image_laterality = ds[0x0020, 0x0062].value
        else:
            pil_image = Image.open(image)

        logger.warning(f"Processing DICOM file: {ds}")

        # resize the image to 299x299
        pil_image = pil_image.resize((299, 299))

        # convert the PIL image to numpy array
        img_array = img_to_array(pil_image)

        # expand the dimensions to match the model's input shape
        img_array = np.expand_dims(img_array, axis=0)

        # make prediction
        prediction = model.predict(img_array)

        response = {
            'prediction': []
        }
        # response
        if image_laterality in ['L', 'R']:
            response = {
                'model_predictions': {
                    'laterality': 'Left' if image_laterality == 'L' else 'Right',
                    'prediction': json.dumps(prediction.tolist()[0]),
                }
            }
        else:
            response = {
                'model_predictions': json.dumps(prediction.tolist()[0]),
            }

        return JsonResponse(response)
