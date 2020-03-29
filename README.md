# Emotion-Recognition

> Deep Learning-Based Music Recommendation Service for Facial Emotion Recognition.

### Music Lists for your mood

Emotional recognition technology is a key technology for interaction between humans and computers. Based on accurate emotional recognition technology, it recommends music that anyone can relate to, and creates interest for users.

There are facial characteristics by emotion. We classify by this characteristic and recommend appropriate music. Experiment with famous CNN algorithms to select the best model for datasets to model transfer learning.

The purpose of the project is to develop web pages and distribute them after beta testing.

##  Datasets 

> K-Face we used is a database that fits the facial characteristics of Koreans developed by the NIA(National Information Society Agency).

### Architectures of K-FACE

![Amount_of_the_data](https://github.com/k-face/k-face_2019/raw/master/image/Amount_of_the_data.png)

> K-Face: Korean Facial Image AI Training Dataset

It is a Korean-style face database that fits the characteristics of Koreans' faces. The data is designed to identify the statistical characteristics of Korean faces. The K-FACE data has a total of 1,000 data. There are data equivalent to about 30,000 sheets per person, reflecting 20 different angles, 30 different luminance, 6 different accessories, 3 different facial expressions, and 3 different resolutions.

## Model

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

## Requiremets

- **tensorflow**>=1.15.0
- **keras**>=2.3.0
- **opencv-python**==4.2.0

