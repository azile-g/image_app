# import the necessary packages
from torchvision.models import detection
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import torch
import time
import cv2
from torchvision import transforms
import imutils
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
    le_prdtype = pickle.loads(open("../model/le_prdtype.pickle", "rb").read())
    le_weight = pickle.loads(open("../model/le_weight.pickle", "rb").read())
    le_halal = pickle.loads(open("../model/le_halal.pickle", "rb").read())
    
    return le_prdtype, le_weight, le_halal

# model class
class ObjectDetector(Module):
    def __init__(self, baseModel, numClasses_prdtype, numClasses_weight, numClasses_halal):
        super(ObjectDetector, self).__init__()
        # initialize the base model and the number of classes
        self.baseModel = baseModel
        self.numClasses_prdtype = numClasses_prdtype
        self.numClasses_weight = numClasses_weight
        self.numClasses_halal = numClasses_halal
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
        # build the classifier head to predict the class labels for product type
        self.classifier_prdtype = Sequential(
            Linear(baseModel.fc.in_features, 512),
            ReLU(),
            Dropout(),
            Linear(512, 512),
            ReLU(),
            Dropout(),
            Linear(512, self.numClasses_prdtype)
        )
        # build the classifier head to predict the class labels for weight
        self.classifier_weight = Sequential(
            Linear(baseModel.fc.in_features, 512),
            ReLU(),
            Dropout(),
            Linear(512, 512),
            ReLU(),
            Dropout(),
            Linear(512, self.numClasses_weight)
        )
        # build the classifier head to predict the class labels for halal
        self.classifier_halal = Sequential(
            Linear(baseModel.fc.in_features, 512),
            ReLU(),
            Dropout(),
            Linear(512, 512),
            ReLU(),
            Dropout(),
            Linear(512, self.numClasses_halal)
        )
        # set the classifier of our base model to produce outputs
        # from the last convolution block
        self.baseModel.fc = Identity()

    def forward(self, x):
        # pass the inputs through the base model and then obtain
        # predictions from different branches of the network
        features = self.baseModel(x)
        bboxes = self.regressor(features)
        classLogits_prdtype = self.classifier_prdtype(features)
        classLogits_weight = self.classifier_weight(features)
        classLogits_halal = self.classifier_halal(features)
        # return the outputs as a tuple
        return (bboxes, classLogits_prdtype, classLogits_weight, classLogits_halal)

# load our object detector, set it evaluation mode
def load_model():
    # model = ObjectDetector()
    le_prdtype, le_weight, le_halal = load_label_encoder()
    resnet = resnet50(pretrained=True)
    model = ObjectDetector(resnet, len(le_prdtype.classes_), len(le_weight.classes_), len(le_halal.classes_))
    # model = ObjectDetector(baseModel, numClasses_prdtype, numClasses_weight, numClasses_halal)
    
    model.load_state_dict(torch.load("../model/model_state.pt",map_location=torch.device('cpu')))
    model.eval()

    return model


model = load_model()

transforms_test = transforms.Compose([
    	transforms.ToPILImage(),
    	transforms.ToTensor(),
    	transforms.Normalize(mean=CONFIGS['IMG_MEAN'], std=CONFIGS['IMG_STD'])
    ])
    
le_prdtype, le_weight, le_halal = load_label_encoder()

# initialize the video stream, allow the camera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=1).start()
time.sleep(2.0)
fps = FPS().start()

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	orig = frame.copy()
	# convert the frame from BGR to RGB channel ordering and change
	# the frame from channels last to channels first ordering
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = cv2.resize(frame, (224, 224))
	frame = frame.transpose((2, 0, 1))
	# # add a batch dimension, scale the raw pixel intensities to the
	# # range [0, 1], and convert the frame to a floating point tensor
	# frame = np.expand_dims(frame, axis=0)
	# frame = frame / 255.0
	# frame = torch.FloatTensor(frame)
	# # send the input to the device and pass the it through the
	# # network to get the detections and predictions
	# frame = frame.to(DEVICE)
	# detections = model(frame)[0]

	frame = torch.from_numpy(frame)
	frame = transforms_test(frame).to(CONFIGS['DEVICE'])
	frame = frame.unsqueeze(0)
	# predict the bounding box of the object along with the class label
	(boxPreds, labelPreds_prdtype, labelPreds_weight, labelPreds_halal) = model(frame)
	(startX, startY, endX, endY) = boxPreds[0]
	# determine the class label with the largest predicted probability
	labelPreds_prdtype = torch.nn.Softmax(dim=-1)(labelPreds_prdtype)
	i_prdtype = labelPreds_prdtype.argmax(dim=-1).cpu()
	label_prdtype = le_prdtype.inverse_transform(i_prdtype)[0]

	labelPreds_weight = torch.nn.Softmax(dim=-1)(labelPreds_weight)
	i_weight = labelPreds_weight.argmax(dim=-1).cpu()
	label_weight = le_weight.inverse_transform(i_weight)[0]
	labelPreds_halal = torch.nn.Softmax(dim=-1)(labelPreds_halal)
	i_halal = labelPreds_halal.argmax(dim=-1).cpu()
	label_halal = le_halal.inverse_transform(i_halal)[0]
	label = label_prdtype + "_" + label_weight + "_HANAL-" + label_halal
	# resize the original image such that it fits on our screen, and grab its dimensions
	orig = imutils.resize(orig, width=600)
	# print(orig.shape)
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

	cv2.imshow("Output", orig)

	key = cv2.waitKey(1) & 0xFF
	# if the 'q' key was pressed, break from the loop
	if key == ord("q"):
		break
	# update the FPS counter
	fps.update()


# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()



