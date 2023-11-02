from django.shortcuts import render
from .load_models import KerasModelLoader
from PIL import Image
import numpy as np

# Initialize and load the model
KerasModelLoader.load_model('/Users/arezaie/Desktop/Test/myproject/myapp/model.json', '/Users/arezaie/Desktop/Test/myproject/myapp/model.h5')

class_mapping = {i: chr(i + 65) for i in range(25)}  # A=0, B=1, ..., Y=24




def preprocess_input(image, target_size=(28, 28)):
    if image.mode != "L":
        image = image.convert("L")
    image = image.resize(target_size)
    image_array = np.asarray(image)
    image_array = image_array / 255.0  # Normalizing
    return image_array.reshape(1, target_size[0], target_size[1], 1)


def predict(request):
    if request.method == 'POST' and request.FILES.get('image', False):
        image_file = request.FILES['image']
        image = Image.open(image_file)

        processed_input = preprocess_input(image)
        model = KerasModelLoader.get_model()
        prediction = model.predict(processed_input)

        predicted_index = np.argmax(prediction)
        readable_prediction = class_mapping.get(predicted_index, "Unknown class")

        return render(request, 'result.html', {'prediction': readable_prediction})

    return render(request, 'predict.html')





