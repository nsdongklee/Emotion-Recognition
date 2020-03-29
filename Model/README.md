# CNN Architectures

> CNN의 대표적인 아키텍처들을 분석

1. *LeNet*
2. *AlexNet*
3. *ZfNet*
4. *VGG16*
5. *GoogLeNet(Inception_v1)*
6. *ResNet*
7. *DenseNet*
8. *Xception*
9. *Se-Network*

## ILSVRC & ImageNet

> ILSVRC는 ImageNet Large Scale Visual Recognition Challenge의 약어. ImageNet 영상 데이터베이스를 기반으로 컴퓨터비전 분야에서 성능의 우열을 가리기 위한 대회이다.

<br>

<p align='center'><img src='http://image-net.org/index_files/logo.jpg'></p>

### ImageNet

세계 최대의 영상 데이터 베이스로서, 마치 사람이 보고 판단하는 것처럼 컴퓨터 비전을 연구하는 사람들이 벤치마크로 사용하는 영상 데이터 베이스이다. 

- 약 22,000 개의 종류로 분류할 수 있는 1,500만 장의 인터넷 기반 영상이 있다.

- `WordNet`의 계층구조를 따라 만들어졌다.

  ```shell
  워드넷(WordNet)은 영어의 의미 어휘목록이다. 
  워드넷은 영어 단어를 'synset'이라는 유의어 집단으로 분류하여 간략하고 일반적인 정의를
  제공하고, 이러한 어휘목록 사이의 다양한 의미 관계를 기록한다. 
  그 목적은 두가지이다. 하나는 사전(단어집)과 시소러스(유의어·반의어 사전)의 배합을 
  만들어, 보다 직관적으로 사용할 수 있고 자동화된 본문 분석과 인공 지능 응용을 
  뒷받침하려는 것이다.
  ```

- 1개의 `synset` 에 대해 평균 1,000장 이상의 영상이 있다.

<br>

## LeNet

> LeNet은 CNN 알고리즘을 최초로 개발한 Yann Lecun에 의해 만들어졌다. 원래 우편번호와 수표의 필기체를 인식하기 위한 용도로 개발을 했다.

<br>

기존 Fully-Connected Neural Network는 좋은 알고리즘이지만 Topology 변화에 대응이 어려운 단점을 가지고 있었다. 

그래서 대표적인 고양이 실험과 같은 개념을 도입하여 CNN(Convolutional Neural Network)를 개발했다.

- LeNet-1 :

  <p align='center'><img src='https://miro.medium.com/max/1412/1*ge5OLutAT9_3fxt_sKTBGA.png'></p>

  - 1단계 : 4개의 feature map
  - 2단계 : 12개의 featur map
  - 전체적으로 각 feature map 추출 마다 반으로 Pooling 과정을 거쳐 크기를 줄이며 추출된 최종 특성을 DNN과 연결했다.
  - free parameter의 개수는 3,000개 이하이다.

<br>

- LeNet-5 :

  <p align='center'><img src='https://t1.daumcdn.net/cfile/tistory/99170D4C5C7E21250E'></p>

  - LeNet-1이 처음 개발된 당시는 컴퓨팅 능력의 한계로 파라미터 수를 적게 할 수 밖에 없었다.
  - 최초의 모델과 아키텍처는 대체적으로 유사하지만 전체적인 크기에 차이가 있다.
  - LeNet은 MNIST의 `28*28` 이미지를 `32*32`로 변경하여 처리했다.
  - free parameter의 개수는 약 6만개에 달한다.

  <p align='center'><img src='https://slideplayer.com/slide/12926094/78/images/16/LeNet+Example+C1+S2+C3+S4+C5.jpg'></p>

  > LeNet의 단계별 훈련과정 이미지

  - Layer 살펴보기

    - `C1` : Input 이미지를 6개의 5x5 필터와 컨볼루션 연산하여 6장의 28x28 Feature map 추출

      ```shell
      # 훈련해야할 파라미터 개수: 
      (weight*특성맵개수 + bias)*특성맵개수 = (5*5*1 + 1)*6 = 156
      ```

    - `S2` : 6장의 28x28 특성 맵에 대해 Pooling 진행. 14x14 특성맵으로 축소. **2x2 필터**를 **stride 2**로 설정해서 Pooling하기 때문이다. 사용하는 Pooling 방법은 평균 풀링(average pooling)이다.

      ```shell
      # 훈련해야할 파라미터 개수: 
       (weight + bias)*특성맵개수 = (1 + 1)*6 = 12
      ```

    - `C3` : 6장의 14x14 특성맵에 컨볼루션 연산을 통해 16장의 10x10 특성맵을 산출한다.

      ![img](https://t1.daumcdn.net/cfile/tistory/9902AD375C7F2B3E1A)

      > C3-Layer 에서 6장의 14 x 14 특성맵을 조합하는 방법. 1516개의 훈련할 파라미터가 생성된다.

    - `S4` :  16장의 10x10 특성맵에 대해서 서브샘플링을 진행해 16장의 5 x 5 특성 맵으로 축소

      ```shell
      # 훈련해야할 파라미터 개수: 
       (weight + bias)*특성맵개수 = (1 + 1)*16 = 32
      ```

    - `C5` : 16장의 5x5 특성맵을 120개의 5x5x16 사이즈의 필터와 컨볼루션 해준다. 결과적으로 120개 1x1 특성맵이 산출

      ```shell
      # 훈련해야할 파라미터 개수: 
      (weight*특성맵개수 + bias)*특성맵개수 = (5*5*16 + 1)*120 = 48120
      ```

    - `F6` : 84개의 유닛을 가진 피드포워드 신경망이다. C5의 결과를 84개의 유닛에 연결시킨다.

      ```shell
      # 훈련해야할 파라미터 개수: 
       (weight + bias)*특성맵개수 = (120 + 1)*84 = 10164
      ```

    - `Output layer` : 10개의 Euclidean radial basis function(RBF) 유닛들로 구성되어있다. 각각 F6의 84개 유닛으로부터 인풋을 받는다. 최종적으로 이미지가 속한 클래스를 알려준다. 

  - LeNet-5이 훈련해야할 파라미터는 총 156 + 12 + 1516 + 32 + 48120 + 10164 = **60000개**

<br>

## AlexNet

> 2012년 ImageNet ILSVRC 대회에서 2위와 큰 성능차(AlexNet 16% , 2위 26%)로 우승한 것으로 유명하다.

<p align='center'><img src='https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F99FEB93C5C80B5192E'></p>

> Alex-Net의 아키텍처. LeNet과 유사하지만, 보통 Conv-Layer 다음 Pooling 과정을 진행하는 기본 방식과 달리 Conv-Layer 바로 다음 Conv-Layer가 온 점이 다르다.

- Input : 224x224x3 의 RGB 이미지
- 8개의 Layer(Convolutional-Layer 5개, Fully-Connected Layer 3개)
- 3번째 Conv-Layer는 이전 두 단계의 특성 맵들과 모두 연결되어 있다.

<br>

### Layers

<p align='center'><img src='https://i.ytimg.com/vi/dZVkygnKh1M/maxresdefault.jpg'></p>

- `1번 째 (Conv-layer)` : 
  - 96개의 11x11x3 사이즈의 필터 커널, Stride==4, Zero padding==False(사용X)
  - 96장의 55 x 55 사이즈 특성맵들이 산출된다.(55 x 55 x 96)
  - 그 다음에 **ReLU** 함수로 활성화해준다.
  - 3x3 overlapping **max pooling**이 **stride 2**로 시행하며, 그 결과로  27 x 27 x 96 특성맵을 가진다.
  - 그 다음에는 수렴 속도를 높이기 위해 **local response normalization**이 시행된다.(특성맵 크기는 유지된다.)
- `2번 째 (Conv-layer)` : 
  - 256개의 5x5x48 필터 커널, Stride == 1, Zero padding == 2
  - 256장의 27 x 27 사이즈 특성맵들이 산출된다.(27 x 27 x 256)
  - 그 다음에 **ReLU** 함수로 활성화해준다.
  - 3x3 overlapping **max pooling**이 **stride 2**로 시행하며, 그 결과로  13 x 13 x 256 특성맵을 가진다.
  - 그 다음에는 수렴 속도를 높이기 위해 **local response normalization**이 시행된다.(특성맵 크기는 유지된다.)
- `3번 째 (Conv-layer)` : 
  - 384개의 3x3x256 필터 커널, Stride == 1, Zero padding == 1
  - 384장의 13 x 13 사이즈 특성맵들이 산출된다.(13 x 13 x 384)
  - 그 다음에 **ReLU** 함수로 활성화해준다.
- `4번 째 (Conv-layer)` : 
  - 384개의 3x3x192 필터 커널, Stride == 1, Zero padding == 1
  - 384장의 13 x 13 사이즈 특성맵들이 산출된다.(13 x 13 x 384)
  - 그 다음에 **ReLU** 함수로 활성화해준다.
- `5번 째 (Conv-layer)` :
  - 256개의 3x3x192필터 커널, Stride == 1, Zero padding == 1
  - 256장의 13 x 13 사이즈 특성맵들이 산출된다.(13 x 13 x 256)
  - 그 다음에 **ReLU** 함수로 활성화해준다.
  - 3x3 overlapping **max pooling**이 **stride 2**로 시행하며, 그 결과로  6 x 6 x 256 특성맵을 가진다.
- `6번 째 (F.C-layer)` :
  - Conv-layer의 마지막 층인 5번째 층의 Output인 6x6x256 특성맵을 **Flatten**한다.(딥러닝이 이해할 수 있는 벡터 형태로 변경하는 단계)
  - **4096**개의 노드 및 **ReLU** 함수
- `7번 째 (F.C-layer)` :
  - **4096**개의 노드 및 **ReLU** 함수
- `8번 째 (F.C-layer)` :
  - **1000**개의 노드 및 **Softmax** 함수를 통해 1000개의 클래스 분류

<br>

### Additional explanation

1. ReLU 활성화 함수 :

   ![img](https://k.kakaocdn.net/dn/cexrVz/btqBFwoUz96/6E1W6ALGpm3EfkJykHPFak/img.jpg)

   LeNet-5는 Tanh 함수를 사용했으나, AlexNet은 ReLU 함수가 사용되었다. 정확도는 비슷한 수준이나 6배나 연산속도가 빨라진다고 한다. 

2. Dropout : 

   <img src="https://k.kakaocdn.net/dn/cMcWkE/btqBFNcRhiv/jJyZWvbf9uQLmKJG3pQAK1/img.jpg" alt="img" style="zoom:67%;" />

   과적합(over-fitting)을 막기 위해서 규제 기술의 일종이다.  몇몇 뉴런의 값을 0으로 바꾸어 뉴런 중 일부를 생략하면서 학습을 진행하는 것이다. Training 과정에만 적용되며, 테스트시에는 모든 뉴런을 사용한다. 

3. Overlapping Pooling : 

   Pooling은 샘플링이라고도 하는데 Feature map(특성 맵)의 크기를 줄이기 위한 목적으로 활용된다. LeNet-5의 경우 average pooling이 사용된 반면, AlexNet에서는 max pooling이 사용되었다.

   <img src="https://k.kakaocdn.net/dn/b5hfOx/btqBCUY3kpE/CKcK19bmDgtkSkWS5GPkBk/img.png" alt="img" style="zoom:50%;" />

   > overlapping 풀링을 하면 풀링 커널이 중첩되면서 지나가는 반면, non-overlapping 풀링을 하면 중첩없이 진행된다. 

4. Local response normalization : 

   신경생물학에는 `lateral inhibition`이라고 불리는 개념이 있다. 활성화된 뉴런이 주변 이웃 뉴런들을 억누르는 현상을 의미한다. lateral inhibition 현상을 모델링한 것이 바로 **local response normalization**이다. 강하게 활성화된 뉴런의 주변 이웃들에 대해서 normalization을 실행한다. 주변에 비해 어떤 뉴런이 비교적 강하게 활성화되어 있다면, 그 뉴런의 반응은 더욱더 돋보이게 될 것이다. 반면 강하게 활성화된 뉴런 주변도 모두 강하게 활성화되어 있다면, local response normalization 이후에는 모두 값이 작아질 것이다. (https://bskyvision.com/421?category=635506 참고)

<br>

## ZfNet

> AlexNet으로 주목할 만한 성과를 얻어 냈지만 전체적인 구조가 어떤 원리로 좋은 결과를 낼 수 있는지, CNN 알고리즘의 Hyper-parameter를 어떻게 설정할지 판단하는 것은 어려운 일이다.
>
> 여기서, Matthew Zeiler는 `Visualizing 기법`을 통해 해결하려는 시도를 했다.

<br>

<p align='center'>"ZfNet은 특정구조를 말하는 개념이 아니라, CNN을 보다 잘 이해하는 기법이다."</p>

### De-convolution을 이용한 Visualization

CNN은 보통 `INPUT - Filter(FeatureMap) - 활성화함수- Pooling(Sampling)`  과정을 거친다. 

여기서, `특정 Feature`의 `Activity`가 입력 이미지에서 어떻게 `Mapping` 되는지 이해하기위해 역으로 수행하는 것이다.

다만 MaxPooling을 한 상태에서 역으로 하려면 어떤 위치에 있는 신호 였는지 파악할 수 없기 때문에 ZfNet 개발팀은 `Switch`라는 일종의 꼬리표(flag) 라는 개념을 만들었다.

이렇게 De-convolution을 수행하면, 정확하게 입력과 같은 상태로 `Mapping` 되는 것은 아니지만 강한 특성(feature)이 어떻게 Mapping되고 있는 지 확인할 수 있어 최적의 구조를 만드는데 참고할 수 있다.

<p align='center'><img src='https://miro.medium.com/max/1528/1*aph2aB6IcCuMft1-MLqtqQ.png'alt="img" style="zoom:67%;" /></p>

<br>

### Feature Visualization

각 단계에서 Feature Map에서 상위 9개의 활성화된 패턴만 보여주고, 그 9개의 Feature에 대해 원본영상을 같이 쌍으로 보여준다.

CNN이 SIFT 등의 기존 특징 검출 알고리즘과 다른 점은 컨볼루션 연산과 샘플링 과정을 거치면서 mid/high level feature(단순 경계검출만 아닌 전체 오브젝트 등을 가리킴)를 이용한다는 것이다. 하지만 하이퍼 파라미터를 최적으로 만들기 쉽지 않기 때문에 `Feature Visualization` 이 중요하다.

1. Layer-1 , 2

   ![img](https://miro.medium.com/max/1064/1*WbyE9tqJt8Kd0vqNX9MeVQ.png)

   De-conv 기술을 사용해서 무작위로 선택된 Feature Map에서 상위 9개의 활성화된 패턴이 각 Layer에 대해 표시된다.

   - Layer-1의 필터들은 가장 자주보이는 패턴과 거의 없는 패턴들의 조합이다. 이렇게 하면 `연쇄반응(Chain-Effect)` 이라는 것이 발생한다고 한다.
   - Layer-2는 Stride를 4로 설정하여 보폭이 너무 커서 앨리어싱 아티팩트가 발생된다.

2. Layer-3

   ![img](https://miro.medium.com/max/1011/1*hpm0NDbqDDTYHHY_7OOPfQ.png)

   이전 레이어(Layer1,2)에 비해 조금 더 상위수준의 항상성(Invariance)를 얻을 수 있거나 비슷한 외양(Texture)를 갖고 있는 특징을 추출할 수 있다.

3. Layer-4, 5

   ![img](https://miro.medium.com/max/1006/1*69ty1ZX7OoScp7oXhqbs_A.png)

   - Layer-4는 사물이나 개체의 일부분을 볼 수 있다. 왼쪽 상단 개의 얼굴 이나 우측 하단의 새 다리 처럼 특징을 볼 수 있다. 
   - Layer-5는 위치나 자세, 변화 등까지 포함한 사물이나 개체의 전부를 보여준다.



### Modifications of AlexNet Based on Visualization Results

![img](https://miro.medium.com/max/1165/1*bFjBVvUL2Po_p2mKzC4iYQ.png)

> ZfNet은 Alexnet과 같은 스타일을 유지한다.  하지만 Layer-1과 2의 문제점을 개선하기 위해, ZfNet은 두가지 변화를 주었다.

1. Layer-1의 필터 사이즈를 11x11 에서 7x7로 줄였다.
2. Layer-1의 stride를 4에서 2로 변경했다.

![img](https://miro.medium.com/max/674/1*YTd-watnbNQA-vArfqS_AQ.png)

> Layer 2: (c) Aliasing artifacts in AlexNet and (d) much cleaner features in ZFNet

<br>

## VGG16

> VGG16은 총 16개의 Layers로 , 13개의 Conv-layer와 5개의 Max Pooling layer와 3개의 DNN layer로 이루어진 아키텍처이다.

![img](https://miro.medium.com/max/1500/1*Vz5n812l-J37a5wLxKbD8A.png)

> VGG16의 아키텍처이다. 

VGGNet은 GoogLeNet과 함께 2014년 ILSVRC 대회에서 주목을 받은 아키텍처이다. GoogLeNet보다 근소한 차이로 성능은 떨어지나, 구조적인 측면에서 훨씬 간단한 구조로 되어있어 이해하기 쉬우며 변형하기 쉬워 훨씬 많이 사용된다고 한다.

VGG 연구팀은 깊이가 어떤 영향을 주는지 밝히기 위해 가장 간단한 3x3 크기의 사이즈로 정하고 6개의 구조에 대해 실험한다.

![ImageNet: VGGNet, ResNet, Inception, and Xception with Keras ...](https://pyimagesearch.com/wp-content/uploads/2017/03/imagenet_vggnet_table1.png)

> 위 구조는 VGG 개발팀이 실험했던 구조에 대한 표이다.

AlexNet이나 ZfNet 처럼 224x224 사이즈의 RGB 컬러 이미지를 Input data로 활용했다. GoogLeNet 저자의 지적처럼 VGGNet의 단점은 Layer가 적음에도 불구하고 파라미터가 많다는 점이다. 그 이유는 마지막에 DNN 층 때문이다.

<br>

### Codes of VGG-16

```python
model = Sequential()
model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=2, activation="softmax"))
```

> VGG의 모델을 Keras로 보았을 때 마지막에 엄청나게 많은 노드가 투입되는 것을 볼 수 있다. 때문에 연산량이 급격하게 늘어난다.

VGGNet 팀은 망의 깊이에 따른 영향을 연구했고, 위에 코드 처럼 3x3의 conv-layer를 겹쳐서 사용했으며, ILSVRC-2012 데이터에 대해 16개의 층보다 많아지면 큰 소득을 얻지 못했다는 것을 논문으로 발표했다. 결국 VGG의 핵심은 3x3 Convolution layer 라는 단순한 구조로 부터 부터 성능을 끌어내는 것이다.

<img src="https://miro.medium.com/max/857/1*AqqArOvacibWqeulyP_-8Q.png" style="zoom:67%;" />

> 위 코드를 그림으로 나타내면 다음과 같다. 전체적인 특징으로는 Conv-layer와 pooling을 바로 붙이지 않고 Conv-layer를 여러개 쌓고 pooling하는 과정을 거친 것이다.

<br>

### VGG 아키텍처의 Training 방법

- AlexNet은 모든 학습 이미지를 256x256 크기로 만든 후, 거기서 무작위로 224x224 크기의 이미지를 추출하는 방식으로 데이터 크기를 2048배 늘리고, RGB컬러를 주성분 분석(PCA)를 통해 RGB 데이터를 조작하는 방식도 사용했다. 
- VGGNet은 모든 이미지를 Training Scale을 'S'로 표시하고, single-scale training과 multi-scale-traing을 지원한다. 
  - single scale : S==256, S==384 
  - multi scale : `Smin`(256) to `Smax`(512)

### VGG 아키텍처의 Test 방법

- AlexNet은 Test데이터를 256x256 사이즈로 변경하고, 그 데이터를 상하좌우 및 중앙에 맞추어 224x224 크기의 데이터로 재변경했다. 좌우 반전까지 하나의 데이터마다 10개로 나누어 테스트 셋을 만들었다. 이 테스트 셋을 모두 실험하고 10개의 평균을 가져가는 방식으로 했다.(softmax의 결과로 나오는 숫자를 평균함)

- Q 라고 부르는 test scale을 사용하며, 테스트 영상을 미리 정한 크기 Q로 크기 조절을 한다. Qs는 트레이닝 사이즈와 같을 필요가 없지만, 각각의 S에 대해 Q를 학습하면 결과는 좋아진다.
- Multi-crop(150장) 방식으로 테스트 하는 방법도 있었지만, 연산량을 줄이기 위해 `Dense -Evaluation` 개념을 적용했다. 
- `Dense -Evaluation` : 데이터를 자르고 각각 layers에 적용하는 것이 아니라 큰 데이터에 바로 적용하여 일정한 픽셀 간격으로 결과를 끌어낼 수 있다.

<br>

## GoogLeNet

> GoogLeNet의 핵심은 망을 깊게 형성하면서, 파라미터를 줄이기 위해 Inception이라는 독특한 구조를 만들며, 깊어지는 망의 학습을 돕기 위해 auxiliary classifier 개념을 만들었다.

<img src="https://k.kakaocdn.net/dn/14Um2/btqyQ5nKlEA/hjSsZaYiBukseySytXWFCK/img.png" alt="img" style="zoom: 80%;" />

> GoogLeNet의 전체 아키텍처는 아니지만, 이 아키텍처의 핵심인 인셉션 모듈에 대한 설명이다. 위 모델은 초기 모델, 아래 모델은 발전시킨 모델이다.

### What is Inception in GoogLeNet?

![img](https://k.kakaocdn.net/dn/bHZHKC/btqyQ5aekdF/3rkScmoIxS4P4fia96lQwk/img.png)

> GoogLeNet은 9개의 인셉션 모듈을 포함한다.

GoogLeNet에 실제로 사용된 모델은 상단의 (b) 와같이 1x1 Convolution(노란색 블럭)이 포함된 모델이다. 이 방식은 다양한 종류의 특성을 도출할 수 있고 연산량을 줄일 수 있다.(이전 모델은 동일한 필터 커널로 컨볼루션 연산을 했기 때문에)

<br>

### GoogLeNet의 구조

우선, **22개의 층**으로 구성되어 있다. CNN 알고리즘이 발전하면서 층을 깊게 만드는 경향을 보인다. 하지만 여기에는 연산해야 할 **파라미터가 급격하게 증가**하고, 모델이 데이터에 **과적합** 될 수 있는 문제가 발생할 가능성이 높아지는 **두 가지 문제점**이 있다.

여기서 GoogLeNet은 앞에서 먼저 정리 해놓은 `Inception` 이라는 구조(그 중에서도 특히 1x1 컨볼루션 필터)를 통해서 feature map의 갯수를 줄여 연산해야 할 파라미터를 줄이고, 서로 다른 사이즈에서 특성을 추출해 낼 수 있도록 했다.

![img](https://k.kakaocdn.net/dn/MzPze/btqyQy5e3NM/5HPtmAwVQzKJTj6wgWautk/img.png)

<br>

### Global average pooling

<img src="https://k.kakaocdn.net/dn/bwTHh0/btqB2uyArWI/qBr48Ik8bl4bK1oOEJa3bk/img.png" alt="img" style="zoom: 67%;" />

이전 모델들은 DNN layers 가 모델의 후반부에 있다. 하지만 GoogLeNet은 DNN 방식 대신 Global average pooling 방식을 채택했다. 이 방식은 이전 층에서 산출된 feature map을 각각 평균내어서 1차원 벡터로 만들어 주어 최종적으로 Softmax 활성화 함수를 통해 분류 할 수 있도록 연결해 줄 수 있다. 

왼쪽은 (7x7x1024) 사이즈의 모델이 Flatten 되면서 (1x1x50176)의 벡터 모델이 된 것을 볼 수 있다. 반면에 오른쪽은 (1x1x1024) 사이즈가 되었다. 이로써 가중치의 개수를 많이 없애준다. 왜냐면 이 방식을 사용하면 가중치가 필요하지 않기 때문이다.

<br>

### auxiliary classifier

네트워크의 깊어질수록 Gradient Vanishing(값 소멸, 가중치 업데이트 중 기울기가 0에 수렴하는 상황) 문제가 발생하기 쉽다.

이 문제를 극복하기 위해 네트워크 중간에 두 개의 **보조 분류기(auxiliary classifier)**를 달아주었다.

![img](https://k.kakaocdn.net/dn/bD5poT/btqyQM98EkX/nbxasUSmCO1WnaIyIsvUD0/img.png)

> 5x5 average pooling(stride 3), 128개의 (1x1) filter, 1024개의 노드 DNN, 1000개의 노드 DNN, Softmax 분류 순서로 되어있으며, 필요시에만 ON 한다.

<br>

### GoogLeNet 아키텍처의 Training 방법

- GoogLeNet은 영상의 가로-세로의 비율을 (3/4, 4/3)을 유지하면서 원본 영상의 8%부터 100% 까지 포함할 수 있도록 다양한 크기의 Patch를 학습에 사용했다. 또한, Photometric distortion을 통해 학습 데이터를 늘렸다. 
- multi scale training 방법을 사용하며 컬러 이미지 변경 방식 또한 VGG와 유사하다.

### GoogLeNet 아키텍처의 Test 방법

- 방식 또한 VGG와 유사하다.

<br>

## References

- https://arxiv.org/abs/1901.06032
- https://bskyvision.com/
- https://blog.naver.com/laonple/220643128255
- https://j911.me/2019/07/densenet.html
- https://datascienceschool.net/view-notebook/4ca30ffdf6c0407ab281284459982a25