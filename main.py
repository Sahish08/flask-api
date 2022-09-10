#import required libraries
import io
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request
import torch
import torchvision
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms

#Pytorch model
model = torchvision.models.regnet_y_32gf()

weights = torch.load('Data/model.pth', map_location=torch.device('cpu'))['model']
model.fc = torch.nn.Linear(3712, 142)
model.load_state_dict(weights, strict=True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
model.eval()

#converting image into a tensor 
def prepare_image(img):
    crop_size = 224
    resize_size = 256
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    interpolation = InterpolationMode.BILINEAR
    transforms_val = transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=interpolation),
            transforms.CenterCrop(crop_size),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=mean, std=std)])
    img = Image.open(io.BytesIO(img))
    img = Image.fromarray(np.uint8(img))
    img = transforms_val(img).reshape((1, 3, 224, 224))
    return img

#classifying insect 
def predict_result(img):
    model.eval()
    device = torch.device('cpu')
    file = open('Data/classes.txt', 'r')
    classes = []
    content = file.readlines()
    for i in content:
        spl = i.split('\n')[0]
        classes.append(spl)

    with torch.inference_mode():
        img = img.to(device, non_blocking=True)
        output = model(img)
        op = torch.nn.functional.softmax(output)
        op = torch.argmax(op)
        return classes[op]


app = Flask(__name__)

#returns JSON of prediction given an image 
@app.route('/predict', methods=['POST'])
def infer_image():
    if 'file' not in request.files:
        return "Please try again. The Image doesn't exist"

    file = request.files.get('file')

    if not file:
        return

    img_bytes = file.read()
    img = prepare_image(img_bytes)

    return jsonify(prediction=predict_result(img))


@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'

if __name__ == "__main__":
    app.run()
