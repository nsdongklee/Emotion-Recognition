# Emotion-Recognition

> Deep Learning-Based Music Recommendation Service for Facial Emotion Recognition.

<p align='center'><img src='https://i.pinimg.com/originals/c6/27/a2/c627a264744447f3bc0d0978bce1fc9c.png' style="zoom:67%;" /></p>

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

## Facial Emotion Detection

<p align='center'>
    <img src="https://github.com/dannylee93/Emotion-Recognition/blob/master/Images/EmotionDetection.jpg?raw=true">
</p>

We used Deep Learning technology and Haar Cascade of Open CV appropriately combined to detect emotions through facial images in image data.

First, Made an emotional classification model of combination of K-FACE datasets and CNN architecture. And Saved as a file that weight generated through this model.

Then, the frame generated on the web cam was used as input data by utilizing the weight file. In the process of data utilization, four coordinates around the face were selected through the "Haar Cascade.xml" file and readjusted to 150X150 size based on those four coordinates.

Using the methods of Open CV on the frames predicted with this processed data, the results of the classification were displayed in square shapes and emotions.

### Haar Cascades

Haar Cascade is a machine learning-based feature detection algorithm. It is intended to detect objects in video or images based on features proposed by Paul Viola and Michael Jones in the paper "Rapid Object Detection Using a Boosted Cascade of Simple Features" published in 2001. Learn Haar Cascade Classifier using images that contain the object you are looking for and images that do not have the object.

The algorithm consists of 4 steps.

1. *Haar Feature Selection*
2. *Creating Integral Images*
3. *Adaboost Training*
4. *Cascading Classifiers*

 #### (1) Haar Feature Selection

Haar feature, which is the subtraction in the pixel sum within the area of adjacent rectangles that move the position, while scanning the image. Use an integral image of `(2)` to add pixels inside the square area more quickly.

The first step is to calculate the Haar feature in the image. With a kernel of any size possible, the entire image is scanned to calculate the Haar feature. For example, the use of a 24X24 kernel would result in more than 160,000 Haar features.

- Haar feature consisting of 2 squares : 
- Haar feature consisting of 3 squares : 
- Haar feature consisting of 4 squares :  

<p align='center'>
    <img src="https://github.com/dannylee93/Emotion-Recognition/blob/master/Images/3_Features_Haar.jpg?raw=true">
</p>

 #### (2) Creating Integral Images

Explanations will be added soon.

 #### (3) Adaboost Training

Explanations will be added soon.

 #### (4) Cascading Classifiers

Explanations will be added soon.

 #### 

## The Flow of Recommender System 

The Recommender system is to provide users with contents that they find in the system a favor they didn't know they liked. In this Project, We would like to recommend musics for users by this recommender system. *[Show details](https://github.com/dannylee93/Emotion-Recognition/tree/master/Recommender-System)*

<p align='center'><img src='https://miro.medium.com/max/1204/0*Lo2t0XnKPB2JN-_t'style="zoom:80%;" />
</p>
> The picture shows that the flow of recommender system.

There are **TWO** main recommendations. *(1) Contents based filtering* and *(2) Collaborative filtering*. Collaborative filtering is divided into *Nearest Neighbor* and *Latent Factor*.

### Two Matrices for Recommend

We used a content-based filtering method to create a matrix of similarities between tags attached to each song. and also created a matrix by analyzing the relationship between each user's generated playlists and their preferences to the included songs.

- Tag Similarity Matrix for Analysis of the mood of songs
- User-item('R') matrix for affinity relationship analysis

#### Matrix 1. Tag Similarity

We extracted "The correlation" between TAGs from playlist datasets through `Word2Vec`

| 태그\태그 | 여름 |  락  | 회상 | ...  |
| :-------: | :--: | :--: | :--: | :--: |
|   여름    |  1   | 0.92 | 0.49 | ...  |
|    락     | 0.92 |  1   | 0.14 | ...  |
|   회상    | 0.49 | 0.14 |  1   | ...  |
|    ...    | ...  | ...  | ...  |  1   |

> Example chart for TAG by TAG similarity matrix

#### Matrix 2. User-item('R') matrix

First of all, We assumed the number of likes of each playlist as preferences in the playlist data, and created a preference matrix between users and items through analysis of the latent factors with matrix factorization and ALS optimization.

| 플레이리스트(id)\노래(id) | LA SONG |  깡  | 차에 타봐 | ...  |
| :-----------------------: | :-----: | :--: | :-------: | :--: |
|             1             |  0.24   | 0.92 |   0.74    | ...  |
|             2             |  0.18   | 0.99 |   0.38    | ...  |
|             3             |  0.24   | 0.14 |   0.11    | ...  |
|            ...            |   ...   | ...  |    ...    | ...  |

> Example chart for USER - ITEM affinity matrix

### The Flow for Recommend

*Using the above two matrices, simply summarized the method of recommending music.*

Finds the **TOP N tags** most similar to the tag information in input data from **Matrix 1**.  and Finds a playlist that contains this extracted tag list. Create **TOP N recommended songs** for each playlist extracted from **Matrix 2**.

*Randomly recommend songs from the list of candidates created.*

- **Priority 1**: Select a group of music candidates that are most similar to songs in Input data through *Matrix 2* and recommend songs if it has a list of existing candidates.
- **Priority 2** : Recommend the most frequently mentioned top song in the list of candidates

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

