# Recommender System

> 추천 시스템은 사용자로 하여금 자신이 좋아하는지 몰랐던 취향을 시스템 속에서 발견하고 그에 맞는 컨텐츠를 제공하게 해주는 것이다. 이번 프로젝트에서 추천 시스템을 적용하여 감정에 알맞는 음악을 추천 하려고 한다.

<p align='center'><img src="https://greeksharifa.github.io/public/img/Machine_Learning/2019-12-17-Recommendation%20System/01.JPG" style="zoom:50%;" /></p>

추천 시스템에는 크게 **두 가지가** 있다. *Contents based filtering(컨텐츠 기반 필터링)* 과 *Collaborative filtering(협업 필터링)* 이다. 협업 필터링은 *Nearest Neighbor(최근접 이웃)* 과 *Latent Factor(잠재요인)* 방식으로 나뉜다.

1. **Contents based filtering(컨텐츠 기반 필터링)**
2. **Collaborative filtering(협업 필터링)**
   - *Nearest Neighbor(최근접 이웃)* 
   - *Latent Factor(잠재요인) - Matrix Factorization(행렬 분해)*

<br>

## Contents based filtering

> 컨텐츠 기반 필터링 방식은 사용자가 특정한 아이템을 매우 선호하는 경우, 그 아이템과 비슷한 컨텐츠를 가진 다른 아이템을 추천하는 방식이다.

*예를 들어*, 사용자가 '매트릭스' 라는 영화를 보았다면 그에 따른 추천 예상 아이템은 

- 선호 장르 : SF, 액션 (...등)
- 선호 배우 : 키아누 리브스 (...등)
- 선호 감독 : 워쇼스키 브라더스 (...등)

<p align='center'><img src="https://d2slcw3kip6qmk.cloudfront.net/marketing/pages/chart/examples/music-mind-map.svg" style="zoom:67%;" /></p>

> 일종의 마인드맵 처럼 관련 카테고리의 꼬리의 꼬리를 물고 확장하는 방식이라 생각하면 좋을 것 같다.

### Mel-Spectrogram

보통 스펙트럼(spectrum)이라고 부르는 시계열 분석의 정확한 명칭은 파워 스펙트럼(power spectrum) 또는 스펙트럼 밀도(spectral density)이다.**스펙트럼(spectrum)은 확률론적인 확률과정(random process) 모형을 주파수 영역으로 변환**하는 것을 말한다. 따라서 푸리에 변환과 달리 시계열의 위상(phase) 정보는 스펙트럼에 나타나지 않는다. **시계열을 짧은 구간**으로 나눈 뒤 깁스 현상을 줄위기 위해 각 구간에 윈도우를 씌우고 `FFT 계산`으로 나온 값을 평균하는 방법이다.

멜-스펙토르그램이라는 변환 작업을 통해 오디오 데이터의 2D 표현을 얻을 수 있고, 이러한 2D 데이터를 가지고 Convolutional Layer를 통해 오디오의 임베딩과 특징 추출을 할 수 있게 해준다.

- `FFT 계산` : **고속 푸리에 변환 (Fast Fourier Transform, FFT)**

  푸리에 변환은 시간에 대한 신호를 주파수 성분으로 분해하는 작업을 수행해준다.  고속 푸리에 변환은 이산 푸리에 변환과 그 역변환을 빠르게 수행하는 효율적인 알고리즘이다.

<p><img src='https://deeplearn.org/arxiv_files/1912.12055v2/fig/mel_model.png'>
</p>

## Collaborative filtering

> 협업 필터링 방식은 영화를 보기 위해 친구들에게 물어보는 것과 유사한 방식으로 사용자들이 아이템에 매긴 평점이나 구매이력 등 사용자 행동을 기반으로 추천을 하는 것이다.

보통 아이템에 대한 평점을 매기는 경우가 많지 않기 때문에 `희소 행렬(Sparse Matrix)` 특성을 가지고 있다.

- `희소 행렬(Sparse Matrix)` : 0인 값이 대부분인 행렬(반대는 **밀집행렬**)

축적된 사용자 행동 데이터를 기반으로 사용자가 평가하지 않은 아이템을 아래와 같이 *예측 평가(Predicted Rating)* 하는 것이다

<p align='center'><img src="https://greeksharifa.github.io/public/img/Machine_Learning/2019-12-17-Recommendation%20System/02.JPG" style="zoom:67%;" /></p>

### (1) Nearest Neighbor Filtering

최근접 이웃 방식은 **두 가지**로 또 나뉜다.

- `사용자 기반(User - User)` : 나와 비슷한 고객들이 다음 상품도 주문할 때

  사용자 기반 최근접 이웃 방식 은 특정 사용자와 타 **사용자간의 유사도(Similarity)를 측정**하고, 가장 **유사도가 높은 TOP-N 사용자를 추출**해서 그들이 선호하는 아이템을 추천하는 것이다.

  | 사용자 \ 아이템 | 영화 1 | 영화 2 | 영화 3 |
  | :-------------: | :----: | :----: | :----: |
  |      영수       |   0    |   1    |   4    |
  |      철희       |   2    |   1    |   3    |
  |      지승       |   1    |   2    |   3    |

- `아이템 기반(Item - Item)` : 특정 상품을 선택한 고객이 선택한 다른 상품

  아이템이 가지는 속성과는 상관 없이 사용자들이 **아이템을 좋아하는지/싫어하는지 평가** 척도의 유사도를 기준으로 하는 알고리즘

  | 아이템\사용자 | 영수 | 철희 | 지승 |
  | :-----------: | :--: | :--: | :--: |
  |    영화 1     |  1   |  5   |  5   |
  |    영화 2     |  2   |  2   |  2   |
  |    영화 3     |  1   |  2   |  1   |

일반적으로는 사용자 기반 보다 아이템 기반 협업 필터링이 정확도 측면에서는 더 좋다고 한다. 또한, 유사도 측정의 한 방법인 **코사인 유사도** 를 통해서 유사도를 측정한다.

- `코사인 유사도(Cosine-Similarity)` : 

  개념적으로 설명한다면, 두 벡터의 사잇각을 구하고 이에서 유사도를 추출하는 방법이다. 이 때 각을 코사인 값으로 구하기 때문에 *코사인 유사도(Cosine Similarity)* 라고 한다.

  <p align='center'><img src="https://wikidocs.net/images/page/24603/%EC%BD%94%EC%82%AC%EC%9D%B8%EC%9C%A0%EC%82%AC%EB%8F%84.PNG"/></p>

  <p align='center'><img src="http://euriion.com/wp-content/uploads/2014/09/200px-Dot_Product.svg_.png"/></p>

  텍스트 마이닝 등에서 문서를 숫자로 표현하는 방법으로 자주 활용한다고 한다.

  <p align='center'><img src="https://s0.wp.com/latex.php?latex=%5Ctext%7Bsimilarity%7D+%3D+cos%28%5Ctheta%29+%3D+%7BA+%5Ccdot+B+%5Cover+%7CA%7C+%7CB%7C%7D+%3D+%5Cfrac%7B+%5Csum%5Climits_%7Bi%3D1%7D%5E%7Bn%7D%7BA_i+%5Ctimes+B_i%7D+%7D%7B+%5Csqrt%7B%5Csum%5Climits_%7Bi%3D1%7D%5E%7Bn%7D%7B%28A_i%29%5E2%7D%7D+%5Ctimes+%5Csqrt%7B%5Csum%5Climits_%7Bi%3D1%7D%5E%7Bn%7D%7B%28B_i%29%5E2%7D%7D+%7D&bg=T&fg=000000&s=0"></p>

  > 코사인 유사도의 공식은 위와 같다.

  *위의 공식을 설명하면,* 

  ```python
  # 1. 두 개의 벡터
  a = (1,2,3,4,5)
  b = (6,7,8,9,10)
  
  # 2. 각각 짝을 지어 곱한다.
  1 * 6 = 6
  2 * 7 = 14
  3 * 8 = 24
  4 * 9 = 36
  5 * 10 = 50
  
  # 3. 곱한 것을 다 더한다.
  6 + 14 + 24 + 36 + 50 = 130
  
  # 4. 위 까지의 과정은 벡터의 내적을 구하는 과정. 이제 분모에서 두 벡터의 크기를 곱한다.
  ```

  - 사용자 기반 협업 필터링 예시 : 

    위의 사용자 기반 예시 표에서 영수와 철희의 영화에 대한 코사인 유사도를 구해보자. 

    - 영수 = (0,1,4)
    - 철희 = (2,1,3)

    ![](https://scvgoe.github.io/img/jvy9gfsphMw30.png)

    이 때, 사용자 기반에서는 두 유저가 공통으로 평가한 항목만 계산한다. 위의 공식 처럼 계산하여 `User Numbers * User Numbers 크기`의 밀집행렬을 만들어 낼 수 있다.

### (2) Latent Factor Filtering

대규모의 다차원 행렬을 SVD와 같은 차원 감소 기법으로 분해하는 과정에서 잠재요인을 추출하는데, 이러한 행렬분해 기법을 Matrix-Factorization 이라고 한다. 잠재 요인 협업 필터링은 사용자-아이템 평점 행렬 데이터 만을 이용해 말 그대로 '잠재 요인'을 끄집어 내는 것을 말한다. 

다차원 희소행렬을 통해 사용자-아이템 행렬을 **두 가지** 저차원 밀집행렬로 분해한다.

- *P* = **사용자 - 잠재요인(factor) 행렬**
- *Q.T* = **잠재요인(factor) - 아이템 행렬**

우리는 '잠재 요인'이 정확히 무엇인지는 모르지만 추측은 충분히 할 수 있다.

![](https://greeksharifa.github.io/public/img/Machine_Learning/2019-12-17-Recommendation%20System/03.JPG)

우선 가장 좌측의 행렬을 `'R'` 이라고 가정하고 이것을 우측의 두 개의 행렬 요소로 분해하는데, 이 때 중간에 위치한 행렬을 `'P'` 라고 하며 우측을 `'Q'`라고 하는데 `'Q'`는 평점에 대한 행렬을 계산하기 위해 `전치(Transpose)`한다. 

<p align='center'><img src="https://greeksharifa.github.io/public/img/Machine_Learning/2019-12-17-Recommendation%20System/04.JPG" style="zoom: 67%;" /></p>

#### 확률적 경사하강법을 이용한 행렬 분해

`P와 Q.T`로 분해하여서 다시 평점을 계산하는 것은 알았는데 `R`에서 행렬 분해는 어떻게 할까.

주로 `SVD(Singular Value Decomposition, 특이값 분해의 방식 중 하나)` 방식을 이용한다. 하지만 `(NaN)` 값이 없는 행렬에만 적용할 수 있다. 일반적으로 생성되는 `R` 행렬에는 많은 `Null 값`이 있기 때문에 다음 두가지 방식을 주로 사용해서 SVD를 구현한다.

- **확률적 경사 하강법(Stochastic Gradient Descent)**
- **ALS(Alternating Least Squares)**

확률적 경사 하강법을 통한 행렬 분해는 `P와 Q` 행렬로 계산된 **예측 행렬값**과, 가장 최소의 오류를 가질 수 있도록 **비용 함수(Loss Function)** 최적화를 통해 P와 Q를 **유추**한다.

<p align='center'><img src="https://t1.daumcdn.net/cfile/tistory/99F069465C91136E2C"></p>

1. P와 Q를 **임의의 값**을 가진 행렬로 설정한다.

2. P * Q.T 값을 통해 예측 R 을 계산하고 실제 R 과 비교하여 **오류값을 계산**한다.

3. 오류 값을 최소화 하는 P와 Q를 **다시 업데이트** 한다.

4. 만족할 오류 값을 가질 때 까지 **작업을 반복**하며 P, Q 값을 업데이트 한다.

   <p align='center'><img src="https://t1.daumcdn.net/cfile/tistory/990E9F435C911F3631"></p>

### ALS(Alternating Least Squares)

직전에서 설명한 경사하강법(Gradient Descent)는 머신러닝 분야에서 널리 사용되는 일반적인 최적화 알고리즘이다. 하지만 SGD 방식은 Local Minima 문제에 빠질 수 있다는 단점이 존재한다. 또한 Rating(R 행렬의 Value)의 요소(차원)이 많다면 이러한 문제가 더 쉽게 발생할 수 있다.

ALS(Alternating Least Squares)는 행렬분해 오차 최적화의 또 다른 대안 중 하나 이다.

#### (1) Loss Function

<p align='center'><img src="https://github.com/dannylee93/Emotion-Recognition/blob/master/Images/Recommender_sys_00.jpg?raw=true"></p>

> ALS 의 Loss Function에 대한 수식이다. 위의 빨간 네모칸은 확률적 경사하강법(SGD)와 다른 부분에 대한 표시이다.

<p align='center'><img src="https://github.com/dannylee93/Emotion-Recognition/blob/master/Images/Recommender_sys_01.jpg?raw=true"></p>

- **P<sub>ui</sub>** : *p* 는선호를 나타내는 변수 이다. 점 데이터가 존재할 경우 `선호 == 1`, 데이터가 없을 경우 `비선호 == 0`으로 변환한다.
- **C<sub>ui</sub>** : *C* 는 신뢰도 지수에 대한 변수이다. 이 변수를 통해 본래 희소행렬 데이터에서 0으로 남아있는 아이템에 대한 예측 값 또한 잠재요인 분석에 포함하게 한다. (낮은 *C* 값을 가지게 하여 계산에 포함하되, 영향력은 작게 조절했다.)

#### (2) Algorithm of ALS

1. 사용자(User) 또는 아이템(item)의 Latent Factor 행렬을 아주 작은 랜덤 값으로 초기화 한다.
2. 두 기준 중 하나를 상수처럼 고정한다.
3. Loss Function을 `Convex Function(아래로 볼록한 함수)`으로 만든다.
4. 이를 미분하고, 미분 값을 0으로 만드는 사용자 또는 아이템의 Latent Factor 행렬을 계산한다.
5. 과정을 반복하며 최적의 `P*Q.T 행렬`을 만든다.

## References

- **파이썬 머신러닝 완벽가이드**

  <p align='center'><img src="http://image.kyobobook.co.kr/images/book/large/928/l9791158391928.jpg"></p>

- https://arxiv.org/abs/1911.04824

- https://euriion.com/?p=548

- https://greeksharifa.github.io/

- [https://scvgoe.github.io/](https://scvgoe.github.io/2017-02-01-협업-필터링-추천-시스템-(Collaborative-Filtering-Recommendation-System)/)

- https://brunch.co.kr/@kakao-it/342

- https://datascienceschool.net/view-notebook/691326b7f88644f79ec7ddc9f27f84ec/

- https://yeo0.github.io/data/2019/02/23/Recommendation-System_Day8/

- https://yeomko.tistory.com/

- https://github.com/benfred/implicit



