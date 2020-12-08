# import the necessary packages
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import imutils
import cv2
import os


# define the root directory of the project
DIRECTORY = path/to/project
# define the base path to the input dataset and then use it to derive
# the path to the images directory and annotation CSV file
BASE_PATH = os.path.sep.join([DIRECTORY, "dataset"])
IMAGES_PATH = os.path.sep.join([BASE_PATH, "images"])
ANOT_PATH = os.path.sep.join([BASE_PATH, "annotations"])
# ANNOTS_PATH = os.path.sep.join([BASE_PATH, "airplanes.csv"])


# define the path to the base output directory
BASE_OUTPUT = os.path.sep.join([DIRECTORY, "output"])
# define the path to the output serialized model, model training plot,
# and testing image filenames
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.h5"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_PATH = os.path.sep.join([DIRECTORY, "test"])
TEST_FILENAMES = os.path.sep.join([BASE_OUTPUT, "test_images.txt"])

# initialize our initial learning rate, number of epochs to train
# for, and the batch size
INIT_LR = 1e-4
NUM_EPOCHS = 10
BATCH_SIZE = 32
SIZE = 224

X_dataset = []
data = []
targets = []
filename = []


for i in os.listdir(IMAGES_PATH):
    
    fn = IMAGES_PATH + "/" + i
    filename.append(fn)
  
y_dataset = []
    
for i in os.listdir(ANOT_PATH):
    mat = scipy.io.loadmat(ANOT_PATH + "\\" + i)
    y_dataset.append(mat["box_coord"][0])



for i in range(len(filename)):
    
    image = cv2.imread(filename[i])
    (h, w) = image.shape[:2]
    
    startX = float(y_dataset[i][2]) / w
    startY = float(y_dataset[i][0]) / h
    endX = float(y_dataset[i][3]) / w
    endY = float(y_dataset[i][1]) / h
    
    # load the image and preprocess it
    image = load_img(filename[i], target_size=(224, 224))
    image = img_to_array(image)
    
    data.append(image)
    targets.append((startX, startY, endX, endY))


data = np.array(data, dtype="float32") / 255.0
targets = np.array(targets, dtype="float32")


split = train_test_split(data, targets, test_size=0.10, random_state=42)


(trainImages, testImages) = split[:2]
(trainTargets, testTargets) = split[2:4]




# load the VGG16 network, ensuring the head FC layers are left off
vgg = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# freeze all VGG layers so they will *not* be updated during the training process
vgg.trainable = False
# flatten the max-pooling output of VGG
flatten = vgg.output
flatten = Flatten()(flatten)


# construct a fully-connected layer header to output the predicted bounding box coordinates
bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid")(bboxHead)
# construct the model we will fine-tune for bounding box regression
model = Model(inputs=vgg.input, outputs=bboxHead)





# initialize the optimizer, compile the model, and show the model summary
opt = Adam(INIT_LR)
model.compile(loss="mse", optimizer=opt,  metrics = ["accuracy"])
print(model.summary())


# train the network for bounding box regression
print("\n[INFO] training bounding box regressor...")
H = model.fit(
	trainImages, trainTargets,
	validation_data=(testImages, testTargets),
	batch_size=BATCH_SIZE,
	epochs=NUM_EPOCHS,
	verbose=1)



# serialize the model to disk
print("\n[INFO] saving object detector model...")
model.save(MODEL_PATH, save_format="h5")
# plot the model training history
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Bounding Box Regression Loss on Training Set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.savefig(PLOT_PATH)




        
# load our trained bounding box regressor from disk
print("\n[INFO] loading object detector...")
model = load_model(MODEL_PATH)

# loop over the images that we'll be testing using our bounding box regression model
for test_image in os.listdir(TEST_PATH):
    
    test_image_path = os.path.join(TEST_PATH, test_image)

    # load the input image (in Keras format) from disk and preprocess
    # it, scaling the pixel intensities to the range [0, 1]
    image = load_img(test_image_path, target_size=(224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    preds = model.predict(image)[0]   
    (startX, startY, endX, endY) = preds


    # load the input image (in OpenCV format), resize it such that it
    # fits on our screen, and grab its dimensions
    image = cv2.imread(test_image_path)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]


    # scale the predicted bounding box coordinates based on the image
    # dimensions
    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * h)

    # draw the predicted bounding box on the image
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # show the output image
    plt.figure()
    plt.axis('off')
    plt.grid(b=None)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.show()

    plt.savefig(os.path.sep.join([BASE_OUTPUT, test_image]))
