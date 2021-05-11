# --------------------------------------------------------------------------------
# main.py 的说明
# 该主函数是flask后端逻辑处理的主函数：
# 下面代码包括几个部分的内容
# --------------------------------------------------------------------------------
# 1.准备部分（模型参数的载入）
#   1.情感识别的模型
#   2.场景识别的模型
#   目前采用的是place365比赛中一个模型实现的，由于主要是做场景识别的，主体物体识别不够精确，比如
#   金毛只能识别到狗，因此补充了一个densenet来单独做物体识别
#   3.配文匹配和检索
#   根据
# --------------------------------------------------------------------------------
# 2.业务逻辑函数
# 包括 如何从识别的结果得出需要返回给客户端的结果
#   1.根据照片的场景内容进行个性化摄影推送--获取其他平台（eg：摄影之友）的query结果HTML，解析出img_url和当前页面的page_url
#   2.识别出拍摄图像的
#   3.偏好可视化生成，因此，需要用户i的每张上传的图像，用一个csv的列来进行存储，便于快速生成偏好可视化
# 具体业务逻辑函数如下：
# def rcg_sentiment(img):
#     input:img 上传的照片
#     return:情感类别以及概率
#     pass
# def rcg_conent(img):
#     input:img 上传的照片
#     return: 场景关键词(多个)和class_label(单个)
#     pass
# def get_sim_img(kw_list,label):
#     input:场景关键词(多个)和class_label(单个)
#     return:[{'img_url':'资源链接','rcm_kw':'推荐依据关键词'},{}]
def get_photo_course():
    pass
# --------------------------------------------------------------------------------
# 3.接口（路由）函数,直接供uni-app访问的API
#   1. get_senti_text()          
#   @input--用户上传img或者img+文本描述
#   @return 返回情感属性的概率和图像内容关键词
#   2. fangpai()
#   input:用户上传的img 
#   return:相似的优秀imglist，包括推荐依据
# --------------------------------------------------------------------------------
import io
import json
import flask
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request,render_template,redirect
from flask_cors import CORS
from translate import Translator
import pandas as pd
import numpy as np
app = Flask(__name__)
CORS(app, supports_credentials=True)
#--------------用来做更细类别的图像中主体识别的（比如狗中的某一种狗，一共是1000类）
imagenet_class_index = json.load(open('./imagenet_class_index.json'))
model = models.densenet121(pretrained=True)
model.eval()
#---------------用来做场景实现的模型-------
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import numpy as np
import cv2
from PIL import Image

# hacky way to deal with the Pytorch 1.0 update
def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module

def load_labels():
    # prepare all the labels
    # scene category relevant
    file_name_category = 'res_scene/categories_places365.txt'
    if not os.access(file_name_category, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
    classes = list()
    with open(file_name_category) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # indoor and outdoor relevant
    file_name_IO = 'res_scene/IO_places365.txt'
    if not os.access(file_name_IO, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt'
        os.system('wget ' + synset_url)
    with open(file_name_IO) as f:
        lines = f.readlines()
        labels_IO = []
        for line in lines:
            items = line.rstrip().split()
            labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
    labels_IO = np.array(labels_IO)

    # scene attribute relevant
    file_name_attribute = 'res_scene/labels_sunattribute.txt'
    if not os.access(file_name_attribute, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt'
        os.system('wget ' + synset_url)
    with open(file_name_attribute) as f:
        lines = f.readlines()
        labels_attribute = [item.rstrip() for item in lines]
    file_name_W = 'res_scene/W_sceneattribute_wideresnet18.npy'
    if not os.access(file_name_W, os.W_OK):
        synset_url = 'http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy'
        os.system('wget ' + synset_url)
    W_attribute = np.load(file_name_W)

    return classes, labels_IO, labels_attribute, W_attribute

def hook_feature(module, input, output):
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def returnTF():
    # load the image transformer
    tf = trn.Compose([
        trn.Resize((224,224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf

def load_model():
    # this model has a last conv feature map as 14x14
    model_file = 'res_scene/wideresnet18_places365.pth.tar'
    if not os.access(model_file, os.W_OK):
        os.system('wget http://places2.csail.mit.edu/models_places365/' + model_file)
        os.system('wget https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py')
    import wideresnet
    model = wideresnet.resnet18(num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)

    # hacky way to deal with the upgraded batchnorm2D and avgpool layers...
    for i, (name, module) in enumerate(model._modules.items()):
        module = recursion_change_bn(model)
    model.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)

    model.eval()
    # hook the feature extractor
    features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet
    for name in features_names:
        model._modules.get(name).register_forward_hook(hook_feature)
    return model
# 载入标签
classes, labels_IO, labels_attribute, W_attribute = load_labels()
# load the model
features_blobs = []
model_scene = load_model()
tf = returnTF()
# get the softmax weight
params = list(model_scene.parameters())
weight_softmax = params[-2].data.numpy()
weight_softmax[weight_softmax<0] = 0

def rcg_scene(input_img):
    # forward pass
    logit = model_scene.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()
    # vote for the indoor or outdoor
    io_image = np.mean(labels_IO[idx[:10]])
    io_label = 'indoor' if io_image < 0.5 else 'outdoor'
    # output the prediction of scene category
    scene_category = [{classes[idx[i]]:str(round(probs[i],3))} for i in range(5)]
    scene_category = [{"name":classes[idx[i]],"value":int(100*probs[i])} for i in range(6)]
    # output the scene attributes
    responses_attribute = W_attribute.dot(features_blobs[1])
    idx_a = np.argsort(responses_attribute)
    scene_attributes = [{"name":labels_attribute[idx_a[i]],"textSize":20} for i in range(-1,-10,-1)]
    scene_attributes.append({"name":io_label,"textSize":35})
    scene_data = {  
        'io_label':io_label,      
        'scene_category':scene_category,
        'scene_attributes':scene_attributes
        }
    return scene_data
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
# 主体对象的识别（1000类）
def rcg_main_object(image_bytes):
    class_id, class_name = get_image_conetent(image_bytes=image_bytes)
    class_name = format_class_name(class_name)
    translator = Translator(to_lang="chinese")
    class_name_zh = class_name+'('+translator.translate(class_name)+')'
    return class_name
# 给出搜索kw关键词，返回相似图片的链接
from my_tools.text_spider import return_html
def search_sim_photo(kw):
    site_list = [
        {   
            'base_site':'https://www.vcg.com/',
            'patten':'https://www.vcg.com/creative-image/jinmao/'
        },{
            'base_site':'',
            'patten':'https://unsplash.com/s/photos/golden-retriever'
        }
        ]
    # upsplash 的检索形式
    query = 'https://unsplash.com/s/photos/'+kw
    html = return_html(query)
    img_links = list(set(html.xpath("//div[@class='_1tO5-']/img/@src")))[:3]
    kw_data = {
        "list":[{
            "img_url":i,
            "rcd_kw":kw,
            "rsc_source":"unsplash"
        } for i in img_links]
    }
    return kw_data
    # 
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

# 拍照识别+配文接口
@app.route('/take_photo',methods=['GET','POST'])
def get_senti_scenes():
    if request.method == 'POST':
        # 没收到图片
        if 'file' not in request.files:
            print('take photo data upload error')
            return redirect(request.url)
        file = request.files.get('file')
        # 获取用户选择的拍照场景关键词，根据该关键词去配文库中的筛选
        input_scene_kw = request.form.get('user_input_scene')
        if not file:
            return
        img_bytes = file.read()
        # 先对精细类别识别 
        obj_result = rcg_main_object(image_bytes=img_bytes)
        # 再对场景进行识别    
        scene_results = rcg_scene(input_img=transform_image(img_bytes))
        senti_arr = [np.random.randint(20,300) for i in range(8)]
        scene_results['scene_attributes'].append({"name":obj_result,"textSize":50})
        # 再根据用户选择的场景关键词配文
        file_name = './res_text/'+input_scene_kw+'.csv'
        df = pd.read_csv(file_name)
        peiwen_li = list(df['text'][1:6])
        rcg_result = {
                'obj_result':obj_result,
                'scene_results':scene_results,
                'sentiment':senti_arr,
                'peiwen_li':peiwen_li
            }
        return flask.jsonify(rcg_result)
    return render_template('index.html')

# 摄影推荐API
@app.route('/get_sim_photo',methods=['POST'])
def get_sim_photo():
    if 'file' not in request.files:
        print('take photo data upload error')
        return redirect(request.url)
    file = request.files.get('file')
    if not file:
        return
    img_bytes = file.read()
    # 先对精细类别识别 
    obj_result = rcg_main_object(image_bytes=img_bytes)
    # 再对场景进行识别    
    scene_results = rcg_scene(input_img=transform_image(img_bytes))
    # 调用第三方摄影平台依据识别关键词进行query
    imgs_main_ob= search_sim_photo(obj_result)
    imgs_scene = []
    for scene in scene_results['scene_category']:
        kw = format_class_name(list(scene.keys())[0])
        d = search_sim_photo(kw)
        imgs_scene.extend(d['list'])
    imgs_all = {
        "main_ob_list":imgs_main_ob['list'],
        "scene_list":imgs_scene
    }
    return jsonify(imgs_all)

if __name__ == '__main__':
    # 0.0.0.0 是在可以让局域网或者服务器上访问，
    # app.run(debug=True,host='0.0.0.0',port=8080)
    # 127.0.0.1 是仅仅在本地浏览器
    app.run(debug=True,host='127.0.0.1',port=8080)
    # app.run(debug=True,host='127.0.0.1',port=8080)
