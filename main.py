# This is a sample Python script.
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import io
import json
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request,render_template,redirect
from translate import Translator

app = Flask(__name__)
imagenet_class_index = json.load(open('./imagenet_class_index.json'))
model = models.densenet121(pretrained=True)
model.eval()

# 预处理
def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

# 预测
def get_image_conetent(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]
# ImageNet classes are often of the form `can_opener` or `Egyptian_cat`
# will use this method to properly format it so that we get
# `Can Opener` or `Egyptian Cat`
def format_class_name(class_name):
    class_name = class_name.replace('_', ' ')
    class_name = class_name.title()
    return class_name
# flask 路由
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
        img_bytes = file.read()
        class_id, class_name = get_image_conetent(image_bytes=img_bytes)
        class_name = format_class_name(class_name)
        translator = Translator(to_lang="chinese")
        class_name_zh = class_name+'('+translator.translate(class_name)+')'
        # web端返回的结果，因为需要渲染，所以返回HTML
        # return render_template('result.html', class_id=class_id,
        #                        class_name=class_name_trans)
        # android 端返回的结果，返回识别的内容，情感，以及生成的配文
        return_data = {
            'image_conetent':class_name_zh,
            'iamge_sentiment':'positive'
        }
        return jsonify(return_data)
    return render_template('index.html')

if __name__ == '__main__':
    # app.run(debug=True,host='0.0.0.0',port=8080)
    app.run(debug=True,host='127.0.0.1',port=8080)