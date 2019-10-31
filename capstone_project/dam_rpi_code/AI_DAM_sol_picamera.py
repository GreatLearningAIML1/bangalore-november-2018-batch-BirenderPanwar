
#import keras
#print(keras.__version__)

from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import time

print("AI base driver activity monitoring")

class_list =  ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6','c7', 'c8', 'c9']
left_class_desc = ['safe driving', 'texting-right', 'talking on the phone-right', 'texting-left', 'talking on the phone-left', 
              'operating the radio', 'drinking', 'reaching behind', 'hair and makeup', 'talking to passenger']

right_class_desc = ['safe-drive', 'text-left', 'talk-ph-left', 'text-right', 'talk-ph-right', 
              'oper-radio', 'drinking', 'reach-behind', 'hair-makeup', 'talk-passenger']

#pd.DataFrame({'class': class_list, 'left_desc': left_class_desc, 'right_desc': right_class_desc})

class_desc = right_class_desc

#IMAGE_SIZE = (32,32)
#model_path = "models/exp4_cv2_plant_model.h5" # for size 32x32

IMAGE_SIZE = (128,128)
model_path = "models/custom_size128_best_kerasV224.h5"

def getActivity(predictions, threshold=0.50):
    result = np.where(predictions[0] == np.max(predictions[0])) # this is a list
    pos = result[0][0] # this is max probability
    if(predictions[0][pos] >= threshold):
        return pos, class_desc[pos], predictions[0][pos]
    else:
        return 10, "no class", 0
    
def getActivityList(predictions, top=3):
    val = sorted(zip(predictions[0], class_desc), reverse=True)[:top]
    return val

def getTextForDisplay(actlist):
    text = "";
    for idx, pair in enumerate(actlist):    
        text = text + "{}: {:0.2f}".format(pair[1], pair[0]) + "\n"
    text = text[:-1]
    return text

def plotImageWithActivityPredictions(file, actlist):
    displayText = getTextForDisplay(actlist)
    img = cv2.imread(file)
    plt.text(x=0, y=0,s=displayText, 
         bbox=dict(facecolor='orange', alpha=0.5), 
         horizontalalignment='left', 
         verticalalignment='top',
         fontsize=10)
    plt.imshow(img)
    
def plotImageWithActivityPredictionsExt(img, actlist):
    displayText = getTextForDisplay(actlist)
    plt.text(x=0, y=0,s=displayText, 
         bbox=dict(facecolor='orange', alpha=0.5), 
         horizontalalignment='left', 
         verticalalignment='top',
         fontsize=10)
    plt.imshow(img)

def plotImageWithActivityPredictionsPiCamera(img, actlist):
    displayText = getTextForDisplay(actlist)
    camera.annotate_text = displayText
    
    
def read_img_cv2(filepath, size):
    img = cv2.imread(filepath) #, cv2.IMREAD_GRAYSCALE
    img = cv2.resize(img, size, interpolation = cv2.INTER_AREA) # resize image  
    img_data = image.img_to_array(img)
    return img_data

# input is image file
def cv2_pre_processing(filepath, size):
    img = read_img_cv2(filepath, size)
    img_preprocessed = np.expand_dims(img.copy(), axis=0)
    img_preprocessed = img_preprocessed.astype('float32')/255
    return img_preprocessed

# input is image data
def cv2_pre_processing_data(iframe, size):
    img = cv2.resize(iframe, size, interpolation = cv2.INTER_AREA) # resize image  
    img = image.img_to_array(img)
    img_preprocessed = np.expand_dims(img.copy(), axis=0)
    img_preprocessed = img_preprocessed.astype('float32')/255
    return img_preprocessed

def cv2_model_execution(model, filepath, size, threshold=0.5, top=3):
    preproc = cv2_pre_processing(filepath, size)
    preds = model.predict(preproc)
    actPrediction = getActivity(preds, threshold=0.5)
    actlist = getActivityList(preds,top)
    plotImageWithActivityPredictions(filepath,actlist)
    
def cv2_model_execution_picamera(model, iframe, size, threshold=0.5, top=3):
    preproc = cv2_pre_processing_data(iframe, size)
    preds = model.predict(preproc)
    actPred = getActivity(preds, threshold=0.5)
    actlist = getActivityList(preds,top)
    plotImageWithActivityPredictionsPiCamera(iframe,actlist)
    
#using image library of keras
def read_img(filepath, size):
    img = image.load_img(filepath, target_size=size)
    img_data = image.img_to_array(img)
    return img_data

def read_img_grayscale(filepath, size):
    img = image.load_img(filepath, target_size=size, color_mode="grayscale")
    img_data = image.img_to_array(img)
    return img_data

# input is image file
def pre_processing(filepath, size):
    img = read_img(filepath, size)
    img_preprocessed = np.expand_dims(img.copy(), axis=0)
    img_preprocessed = img_preprocessed.astype('float32')/255
    return img_preprocessed

# input is image data
def pre_processing_data(iframe, size):
    img = np.resize(iframe, new_shape=size)
    img = image.img_to_array(img)
    img_preprocessed = np.expand_dims(img.copy(), axis=0)
    img_preprocessed = img_preprocessed.astype('float32')/255
    return img_preprocessed
    
def model_execution(model, filepath, threshold=0.5, top=3):
    preproc = pre_processing(filepath)
    preds = model.predict(preproc)
    actlist = getActivityList(preds,top)
    plotImageWithActivityPredictions(filepath,actlist)

# load the model
trainedModel = load_model(model_path)
#trainedModel.summary()

##################################################################################################

def activity_detection1():
    data_dir = 'test_imgs/'
    test_image_path = '{}/{}'.format(data_dir, "img_10.jpg")
    print(test_image_path)

    preprocessedInp = cv2_pre_processing(test_image_path)

    preds = trainedModel.predict(preprocessedInp)
    print(preds)

    actPrediction = getActivity(preds, threshold=0.5)
    print(actPrediction)

    actList = getActivityList(preds,top=3)
    print(actList)

    plotImageWithActivityPredictions(test_image_path,actList)
    return

# activity_detection1()

#####################################################################################################

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
try:
    #camera.resolution = (640, 480)
    camera.resolution = (320, 480)

    camera.framerate = 5 # max frame rate FPS=30

    # Read frames from Pi camera module in numPy format and making comatible with OpenCV
    #rawCapture = PiRGBArray(camera, size=(640, 480))
    rawCapture = PiRGBArray(camera, size=(320, 480))


    # allow the camera to warmup
    time.sleep(2) # 0.2
    prev_classid = 0
    counter=0
    failedCounter=0
    prev_label = ""
     
    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        imagedata = frame.array
        
        # Convert to grayscale
        #gray = cv2.cvtColor(imagedata,cv2.COLOR_BGR2GRAY)
        
        processedImg = cv2_pre_processing_data(imagedata, IMAGE_SIZE)
        preds = trainedModel.predict(processedImg)
        pos, classname, prob = getActivity(preds, 0.50)
        #print("{},{}".format(classname, prob))

        actlist = getActivityList(preds,top=3)
        displayText = getTextForDisplay(actlist)
        camera.annotate_text = displayText
        
        if(prev_classid == pos):
            counter +=1
        else:
            counter = 0
        
        prev_classid = pos
        
        if(counter > 5):
            if(pos==10):
                label = "no class"
            else:    
                label = "{}:{:0.2f}".format(classname, prob)
                
            prev_label = label
            failedCounter = 0
            counter = 100
        else:
            label = prev_label            
            failedCounter += 1

        if(failedCounter > 5):
            label = "no class"
            failedCounter = 100
                
        camera.annotate_text = label
        
        # Show the frame
        cv2.imshow("Driver Activity Monitoring", imagedata)
        
        #break

        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
        
        # if the 'q' key has pressed, break from th loop
        # reading key
        key = cv2.waitKey(1) & 0xFF    
        if(key== ord('q')):
            break
finally:
    # Release the camera, then close all of the imshow() windows
    # When everything done, release the capture
    print("Releasing camera and cv2 resources")
    camera.close()
    cv2.destroyAllWindows()
