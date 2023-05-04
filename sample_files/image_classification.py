from torchvision import transforms
import imutils
import pickle
import torch
import cv2
import os
# from sklearn.preprocessing import LabelEncoder
from torch.nn import Module
import sklearn
from torchvision.models import resnet50
from torch.nn import Dropout
from torch.nn import Identity
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Sequential
from torch.nn import Sigmoid
from PIL import Image
import numpy as np

CONFIGS = {
    # determine the current device and based on that set the pin memory
    # flag
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    # specify ImageNet mean and standard deviation
    "IMG_MEAN": [0.485, 0.456, 0.406],
    "IMG_STD": [0.229, 0.224, 0.225],
}

# load label encoder 
def load_label_encoder():
    # le_prdtype = pickle.loads(open("../model/le_prdtype.pickle", "rb").read())
    # le_weight = pickle.loads(open("../model/le_weight.pickle", "rb").read())
    # le_halal = pickle.loads(open("../model/le_halal.pickle", "rb").read())
    le_total = pickle.loads(open("../NN_model/le_total.pickle", "rb").read())
    
    # return le_prdtype, le_weight, le_halal
    return le_total

# model class
class ObjectDetector(Module):
    def __init__(self, baseModel, numClasses_prdtype, numClasses_weight, numClasses_halal, numClasses_total):
        super(ObjectDetector, self).__init__()
        # initialize the base model and the number of classes
        self.baseModel = baseModel
        self.numClasses_prdtype = numClasses_prdtype
        self.numClasses_weight = numClasses_weight
        self.numClasses_halal = numClasses_halal
        self.numClasses_total = numClasses_total
        # build the regressor head for outputting the bounding box coordinates
        self.regressor = Sequential(          
            Linear(baseModel.fc.in_features, 128),
            ReLU(),
            Linear(128, 64),
            ReLU(),
            Linear(64, 32),
            ReLU(),
            Linear(32, 4),
            Sigmoid()
        )
        # # build the classifier head to predict the class labels for product type
        # self.classifier_prdtype = Sequential(
        #     Linear(baseModel.fc.in_features, 512),
        #     ReLU(),
        #     Dropout(),
        #     Linear(512, 512),
        #     ReLU(),
        #     Dropout(),
        #     Linear(512, self.numClasses_prdtype)
        # )
        # # build the classifier head to predict the class labels for weight
        # self.classifier_weight = Sequential(
        #     Linear(baseModel.fc.in_features, 512),
        #     ReLU(),
        #     Dropout(),
        #     Linear(512, 512),
        #     ReLU(),
        #     Dropout(),
        #     Linear(512, self.numClasses_weight)
        # )
        # # build the classifier head to predict the class labels for halal
        # self.classifier_halal = Sequential(
        #     Linear(baseModel.fc.in_features, 512),
        #     ReLU(),
        #     Dropout(),
        #     Linear(512, 512),
        #     ReLU(),
        #     Dropout(),
        #     Linear(512, self.numClasses_halal)
        # )
        # build the classifier head to predict the class labels for halal
        self.classifier_total = Sequential(
            Linear(baseModel.fc.in_features, 512),
            ReLU(),
            Dropout(),
            Linear(512, 512),
            ReLU(),
            Dropout(),
            Linear(512, self.numClasses_total)
        )
        # set the classifier of our base model to produce outputs
        # from the last convolution block
        self.baseModel.fc = Identity()

    def forward(self, x):
        # pass the inputs through the base model and then obtain
        # predictions from different branches of the network
        features = self.baseModel(x)
        bboxes = self.regressor(features)
        # classLogits_prdtype = self.classifier_prdtype(features)
        # classLogits_weight = self.classifier_weight(features)
        # classLogits_halal = self.classifier_halal(features)
        classLogits_total = self.classifier_total(features)
        # return the outputs as a tuple
        return (bboxes, classLogits_total)

# load our object detector, set it evaluation mode
def load_model():
    # model = ObjectDetector()
    # le_prdtype, le_weight, le_halal = load_label_encoder()
    le_total = load_label_encoder()
    resnet = resnet50(pretrained=True)
    model = ObjectDetector(resnet, 1, 1, 1, len(le_total.classes_)) # unused classes set as 1
    # model = ObjectDetector(baseModel, numClasses_prdtype, numClasses_weight, numClasses_halal)
    
    model.load_state_dict(torch.load("../NN_model/model_state.pt",map_location=torch.device('cpu')))
    model.eval()

    return model
  
# le_prdtype, le_weight, le_halal = load_label_encoder()
# resnet = resnet50(pretrained=True)

def predict_class(img_path):
    # img_path = "./test_imgs/cookies_1g-99g_a9.png"
    # load model
    model = load_model()
  
    # convert jpg to png if any
    if('jpg' in img_path):
        im = Image.open(img_path)
        im.save('test.png')
        image = cv2.imread('test.png')
    else:
      image = cv2.imread(img_path)
    # img = np.array(img)
    
    # define normalization transforms
    transforms_test = transforms.Compose([
    	transforms.ToPILImage(),
    	transforms.ToTensor(),
    	transforms.Normalize(mean=CONFIGS['IMG_MEAN'], std=CONFIGS['IMG_STD'])
    ])
    
    # le_prdtype, le_weight, le_halal = load_label_encoder()
    le_total = load_label_encoder()
    
    # load the image, copy it, swap its colors channels, resize it, and
    # bring its channel dimension forward
    orig = image.copy()
    orig2 = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.array(image, dtype="float32")
    image = image.transpose((2, 0, 1))
    # convert image to PyTorch tensor, normalize it, flash it to the
    # current device, and add a batch dimension
    image = torch.from_numpy(image)
    image = transforms_test(image).to(CONFIGS['DEVICE'])
    image = image.unsqueeze(0)
    # predict the bounding box of the object along with the class label
    # (boxPreds, labelPreds_prdtype, labelPreds_weight, labelPreds_halal) = model(image.cpu())
    (boxPreds, labelPreds_total) = model(image.cpu())
    (startX, startY, endX, endY) = boxPreds[0]
    # determine the class label with the largest predicted probability
    # labelPreds_prdtype = torch.nn.Softmax(dim=-1)(labelPreds_prdtype)
    # i_prdtype = labelPreds_prdtype.argmax(dim=-1).cpu()
    # label_prdtype = le_prdtype.inverse_transform(i_prdtype)[0]
    # 
    # labelPreds_weight = torch.nn.Softmax(dim=-1)(labelPreds_weight)
    # i_weight = labelPreds_weight.argmax(dim=-1).cpu()
    # label_weight = le_weight.inverse_transform(i_weight)[0]
    # 
    # labelPreds_halal = torch.nn.Softmax(dim=-1)(labelPreds_halal)
    # i_halal = labelPreds_halal.argmax(dim=-1).cpu()
    # label_halal = le_halal.inverse_transform(i_halal)[0]
    # 
    # label = label_prdtype + "_" + label_weight + "_HANAL-" + label_halal
    labelPreds_total = torch.nn.Softmax(dim=-1)(labelPreds_total)
    i_total = labelPreds_total.argmax(dim=-1).cpu()
    label_total = le_total.inverse_transform(i_total)[0]
    label = label_total
    confidence = labelPreds_total[0][i_total].item()

    # resize the original image such that it fits on our screen, and
    # grab its dimensions
    orig = imutils.resize(orig, width=600)
    (h, w) = orig.shape[:2]
    # scale the predicted bounding box coordinates based on the image
    # dimensions
    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * h)
    # draw the predicted bounding box and class label on the image
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.putText(orig, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
      0.65, (0, 255, 0), 2)
    cv2.rectangle(orig, (startX, startY), (endX, endY),
      (0, 255, 0), 2)
    # # show the output image 
    # cv2.imshow("Output", orig)
    # # cv2_imshow(orig)
    # cv2.waitKey(0)
    
    try: 
        os.remove("tmp_viz/img.jpg")
        os.remove("tmp_submission/img.jpg")
    except: pass
    # write image to temporary folder
    # cv2.imwrite("tmp_output/"+label+".jpg", orig)
    cv2.imwrite("tmp_viz/tmp.jpg", orig)
    cv2.imwrite("tmp_submission/tmp.jpg", orig2)

    return label, confidence




