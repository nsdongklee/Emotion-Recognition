import keras 
print(keras.__version__)

# 경로 지정
import glob
import cv2
import numpy as np

neutral_dir = glob.glob("임의변경")
happy_dir = glob.glob("임의변경")
tired_dir = glob.glob("임의변경")

pred_neutral_dir = glob.glob("임의변경")
pred_happy_dir = glob.glob("임의변경")

# 이미지를 numpy배열 데이터로 변경
from keras.preprocessing import image
import numpy as np

xsize=150
ysize=150
x=[]
y=[]
predict_img = []
predict_label = []

def imgToArr(imgdir, label):
    for idx, i in enumerate(imgdir):
        img = image.load_img(i, target_size=(xsize,ysize))
        img_tr= image.img_to_array(img)        
        img_tr /= 255.
            
        y.append(label)
        x.append(img_tr)
        if idx == 500 :
            break

def predToArr(imgdir, label):
    for idx, i in enumerate(imgdir):
        img = image.load_img(i, target_size=(xsize,ysize))
        img_tr= image.img_to_array(img)        
        img_tr /= 255.
            
        predict_label.append(label)
        predict_img.append(img_tr)

imgToArr(neutral_dir, 0)
imgToArr(happy_dir, 1)
imgToArr(tired_dir, 2)

predToArr(pred_neutral_dir, 0)
predToArr(pred_happy_dir, 1)

x=np.array(x)
y=np.array(y)

predict_img = np.array(predict_img)
predict_label = np.array(predict_label)


# 데이터 전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

input_shape = (xsize, ysize, 3)

batch_size = 50
num_classes = 3
epochs = 30

from keras.utils import to_categorical

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
predict_label = to_categorical(predict_label, num_classes)


# 모델링(DenseNet121)
from keras.applications import DenseNet121
from keras.models import Sequential
from keras.layers import Dense, Flatten

model = Sequential()

model.add(DenseNet121(weights='imagenet',
                      include_top=False,
                      input_shape=(150, 150, 3)))


model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))
model.summary()


# 모델 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# 예측 해보기
y_predict = model.predict(predict_img, batch_size=1)

pre=[]
for i in range(94):
    tmp=np.argmax(y_predict[i])
    pre.append(tmp)

pre_label=[]
for i in range(94):
    tm=np.argmax(predict_label[i])
    pre_label.append(tm)


print(predict_img.shape)
print(predict_label.shape)
print(y_predict.shape)

from sklearn.metrics import accuracy_score
acc=accuracy_score(pre,pre_label)
print(acc*100,'%')