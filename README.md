# Emotion-Recognition

> Deep Learning-Based Music Recommendation Service for Facial Emotion Recognition.

## Overview

### Music Lists for your mood

Emotional recognition technology is a key technology for interaction between humans and computers. Based on accurate emotional recognition technology, it recommends music that anyone can relate to, and creates interest for users.

There are facial characteristics by emotion. We classify by this characteristic and recommend appropriate music. Experiment with famous CNN algorithms to select the best model for datasets to model transfer learning.

The purpose of the project is to develop web pages and distribute them after beta testing.

### Our Goal

- **Accuracy** 
- **Creativity** 
- **Research based** 

## How do machine understand facial emotions

*According to [Radboud Faces Database Paper](http://www.socsci.ru.nl/rafd/Langner_etal_2010_CEM.pdf),*

<p align='center'><img src="https://github.com/dannylee93/Emotion-Recognition/blob/master/Images/Radboud_Faces_Database.JPG?raw=true" alt="캡처" style="zoom:67%;" /></p>

> 8 suggestions about displayed facial expression by Radboud Faces Database

"Face processing may be one of the most complex tasks that human beings accomplish. Many research related to the processing of information contained in human faces have continued to develop. Specifically, a face database in which displayed expressions, gaze direction, and head orientation are parametrically varied in a complete factorial design would be highly useful in many research domains"

This database also suggested that features that can be widely applied in different fields of research are facial expressions, gaze direction, head orientation, and reasonable number of data for men and women of all ages. For a total of 276 people, three different gaze direction, eight facial expressions and five head orientation were taken.

##  Reasons why we use K-FACE datasets

> K-Face we used is a database that fits the facial characteristics of Koreans developed by the NIA(National Information Society Agency).

### Architectures of K-FACE

<p><img src="https://github.com/k-face/k-face_2019/raw/master/image/Amount_of_the_data.png" alt="Amount_of_the_data" style="zoom:67%;" /></p>

> K-Face: Korean Facial Image AI Training Dataset

It is a Korean-style face database that fits the characteristics of Koreans' faces. The data is designed to identify the statistical characteristics of Korean faces. The K-FACE data has a total of 1,000 data. There are data equivalent to about 30,000 sheets per person, reflecting 20 different angles, 30 different luminance, 6 different accessories, 3 different facial expressions, and 3 different resolutions.

## CNN architectures with Transfer Learning

> Experimented with the CNN architecture to select a model that fits the data that we adopted.

### VGG-16

```python
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten

conv_base = Sequential()

conv_base.add( VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3)) )


conv_base.add(Flatten())
conv_base.add(Dense(num_classes, activation='softmax'))
conv_base.summary()

```

## Team Members

|        ID         |  Name  |             Github              |
| :---------------: | :----: | :-----------------------------: |
|  **@dannylee93**  | 이동규 |  https://github.com/dannylee93  |
|   **@jglee087**   | 이중기 |   https://github.com/jglee087   |
| **@WinterBlue16** | 이경희 | https://github.com/WinterBlue16 |
|   **@kimjis92**   | 김지승 |   https://github.com/kimjis92   |

## Requirements

- **tensorflow**>=1.15.0
- **keras**>=2.3.0
- **opencv-python**==4.2.0

