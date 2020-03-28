# CNN Architectures

> CNN의 대표적인 아키텍처들을 분석

2. *LeNet*
3. *AlexNet*
4. *ZfNet*
5. *VGG16*
6. *GoogLeNet(Inception_v1)*
7. *ResNet*
8. *DenseNet*
9. *Xception*
10. *Se-Network*

## ILSVRC & ImageNet

> ILSVRC는 ImageNet Large Scale Visual Recognition Challenge의 약어. ImageNet 영상 데이터베이스를 기반으로 컴퓨터비전 분야에서 성능의 우열을 가리기 위한 대회이다.

<br></br>

<p align='center'><img src='http://image-net.org/index_files/logo.jpg'></p>

### ImageNet

세계 최대의 영상 데이터 베이스로서, 마치 사람이 보고 판단하는 것처럼 컴퓨터 비전을 연구하는 사람들이 벤치마크로 사용하는 영상 데이터 베이스이다. 

- 약 22,000 개의 종류로 분류할 수 있는 1,500만 장의 인터넷 기반 영상이 있다.

- `WordNet`의 계층구조를 따라 만들어졌다.

  ```shell
  워드넷(WordNet)은 영어의 의미 어휘목록이다. 
  워드넷은 영어 단어를 'synset'이라는 유의어 집단으로 분류하여 간략하고 일반적인 정의를 제공하고, 이러한 어휘목록 사이의 다양한 의미 관계를 기록한다. 
  그 목적은 두가지이다. 하나는 사전(단어집)과 시소러스(유의어·반의어 사전)의 배합을 만들어, 보다 직관적으로 사용할 수 있고 자동화된 본문 분석과 인공 지능 응용을 뒷받침하려는 것이다.
  ```

- 1개의 `synset` 에 대해 평균 1,000장 이상의 영상이 있다.

<br></br>

## LeNet

> LeNet은 CNN 알고리즘을 최초로 개발한 Yann Lecun에 의해 만들어졌다. 원래 우편번호와 수표의 필기체를 인식하기 위한 용도로 개발을 했다.

<br></br>

기존 Fully-Connected Neural Network는 좋은 알고리즘이지만 Topology 변화에 대응이 어려운 단점을 가지고 있었다. 

그래서 대표적인 고양이 실험과 같은 개념을 도입하여 CNN(Convolutional Neural Network)를 개발했다.

- LeNet-1 :

  <p align='center'><img src='https://lh3.googleusercontent.com/proxy/ExzD1WvpOEYvTHe21boevwoQDJ6p1lKV1ZvUJWslPn5WMVn8ghLePvz5OK8daORO6iXmWT7ZGYY9bDbgXUQXnTv2kZ5nkLacRjQLyaNBosyHPgfRg-gl-jBeR7uDTAhyjx5y_AOoxfAxxeVCsT8'></p>

  - 1단계 : 4개의 feature map
  - 2단계 : 12개의 featur map
  - 전체적으로 각 feature map 추출 마다 반으로 Pooling 과정을 거쳐 크기를 줄이며 추출된 최종 특성을 DNN과 연결했다.
  - free parameter의 개수는 3,000개 이하이다.

<br></br>

- LeNet-5 :

  <p align='center'><img src='https://t1.daumcdn.net/cfile/tistory/99170D4C5C7E21250E'></p>

  - LeNet-1이 처음 개발된 당시는 컴퓨팅 능력의 한계로 파라미터 수를 적게 할 수 밖에 없었다.
  - 최초의 모델과 아키텍처는 대체적으로 유사하지만 전체적인 크기에 차이가 있다.
  - LeNet은 MNIST의 `28*28` 이미지를 `32*32`로 변경하여 처리했다.
  - free parameter의 개수는 약 6만개에 달한다.

  <p align='center'><img src='data:image/jpeg;base64,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'></p>

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

<br></br>

## AlexNet

> 2012년 ImageNet ILSVRC 대회에서 2위와 큰 성능차(AlexNet 16% , 2위 26%)로 우승한 것으로 유명하다.

<p align='center'><img src='https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F99FEB93C5C80B5192E'></p>

> Alex-Net의 아키텍처. LeNet과 유사하지만, 보통 Conv-Layer 다음 Pooling 과정을 진행하는 기본 방식과 달리 Conv-Layer 바로 다음 Conv-Layer가 온 점이 다르다.

<br></br>

- Input : 224x224x3 의 RGB 이미지
- 8개의 Layer(Convolutional-Layer 5개, Fully-Connected Layer 3개)
- 3번째 Conv-Layer는 이전 두 단계의 특성 맵들과 모두 연결되어 있다.

<br></br>

### Layers

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

<br></br>

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





































## References

- https://arxiv.org/abs/1901.06032
- https://bskyvision.com/
- https://blog.naver.com/laonple/220643128255
- https://j911.me/2019/07/densenet.html
- https://datascienceschool.net/view-notebook/4ca30ffdf6c0407ab281284459982a25/

```
<p align='center'><img src=''></p>
```

