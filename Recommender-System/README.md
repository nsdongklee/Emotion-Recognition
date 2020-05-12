# Recommender System

> 추천 시스템은 사용자로 하여금 자신이 좋아하는지 몰랐던 취향을 시스템 속에서 발견하고 그에 맞는 컨텐츠를 제공하게 해주는 것이다. 이번 프로젝트에서 추천 시스템을 적용하여 감정에 알맞는 음악을 추천 하려고 한다.

<img src="https://greeksharifa.github.io/public/img/Machine_Learning/2019-12-17-Recommendation%20System/01.JPG" style="zoom:50%;" />

추천 시스템에는 크게 **두 가지가** 있다. *Contents based filtering(컨텐츠 기반 필터링)* 과 *Collaborative filtering(협업 필터링)* 이다. 협업 필터링은 *Nearest Neighbor(최근접 이웃)* 과 *Latent Factor(잠재요인)* 방식으로 나뉜다.

1. **Contents based filtering(컨텐츠 기반 필터링)**
2. **Collaborative filtering(협업 필터링)**
   - *Nearest Neighbor(최근접 이웃)* 
   - *Latent Factor(잠재요인) - Matrix Factorization(행렬 분해)*

<br>

## Contents based filtering

> 컨텐츠 기반 필터링 방식은 사용자가 특정한 아이템을 매우 선호하는 경우, 그 아이템과 비슷한 컨텐츠를 가진 다른 아이템을 추천하는 방식이다.

예를 들어, 사용자가 '매트릭스' 라는 영화를 보았다면 그에 따른 추천 예상 아이템은 

- 선호 장르 : SF, 액션 (...등)
- 선호 배우 : 키아누 리브스 (...등)
- 선호 감독 : 워쇼스키 브라더스 (...등)

<img src="https://d2slcw3kip6qmk.cloudfront.net/marketing/pages/chart/examples/music-mind-map.svg" style="zoom:67%;" />

> 일종의 마인드맵 처럼 관련 카테고리의 꼬리의 꼬리를 물고 확장하는 방식이라 생각하면 좋을 것 같다.

## Collaborative filtering

> 협업 필터링 방식은 영화를 보기 위해 친구들에게 물어보는 것과 유사한 방식으로 사용자들이 아이템에 매긴 평점이나 구매이력 등 사용자 행동을 기반으로 추천을 하는 것이다.

보통 아이템에 대한 평점을 매기는 경우가 많지 않기 때문에 `희소 행렬(Sparse Matrix)` 특성을 가지고 있다.

- `희소 행렬(Sparse Matrix)` : 0인 값이 대부분인 행렬(반대는 **밀집행렬**)

축적된 사용자 행동 데이터를 기반으로 사용자가 평가하지 않은 아이템을 아래와 같이 *예측 평가(Predicted Rating)* 하는 것이다

<img src="https://greeksharifa.github.io/public/img/Machine_Learning/2019-12-17-Recommendation%20System/02.JPG" style="zoom:67%;" />

### (1) Nearest Neighbor Filtering

최근접 이웃 방식은 **두 가지**로 또 나뉜다.

- `사용자 기반(User - User)` : 나와 비슷한 고객들이 다음 상품도 주문할 때

  사용자 기반 최근접 이웃 방식 은 특정 사용자와 타 **사용자간의 유사도(Similarity)를 측정**하고, 가장 **유사도가 높은 TOP-N 사용자를 추출**해서 그들이 선호하는 아이템을 추천하는 것이다.

  | 사용자 \ 아이템 | 영화 1 | 영화 2 | 영화 3 |
  | :-------------: | :----: | :----: | :----: |
  |    사용자 1     |   0    |   1    |   4    |
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

  ![](https://wikidocs.net/images/page/24603/%EC%BD%94%EC%82%AC%EC%9D%B8%EC%9C%A0%EC%82%AC%EB%8F%84.PNG)

  ![](http://euriion.com/wp-content/uploads/2014/09/200px-Dot_Product.svg_.png)

  텍스트 마이닝 등에서 문서를 숫자로 표현하는 방법으로 자주 활용한다고 한다.

  ![](https://s0.wp.com/latex.php?latex=%5Ctext%7Bsimilarity%7D+%3D+cos%28%5Ctheta%29+%3D+%7BA+%5Ccdot+B+%5Cover+%7CA%7C+%7CB%7C%7D+%3D+%5Cfrac%7B+%5Csum%5Climits_%7Bi%3D1%7D%5E%7Bn%7D%7BA_i+%5Ctimes+B_i%7D+%7D%7B+%5Csqrt%7B%5Csum%5Climits_%7Bi%3D1%7D%5E%7Bn%7D%7B%28A_i%29%5E2%7D%7D+%5Ctimes+%5Csqrt%7B%5Csum%5Climits_%7Bi%3D1%7D%5E%7Bn%7D%7B%28B_i%29%5E2%7D%7D+%7D&bg=T&fg=000000&s=0)

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

우선 가장 좌측의 행렬을 `'R'` 이라고 가정하고 이것을 우측의 두 개의 행렬 요소로 분해하는데, 이 때 중간에 위치한 행렬을 'P' 라고 하며 우측을 `'Q'`라고 하는데 `'Q'`는 평점에 대한 행렬을 계산하기 위해 `전치(Transpose)`한다. 

<img src="https://greeksharifa.github.io/public/img/Machine_Learning/2019-12-17-Recommendation%20System/04.JPG" style="zoom: 67%;" />

#### 확률적 경사하강법을 이용한 행렬 분해

`P와 Q.T`로 분해하여서 다시 평점을 계산하는 것은 알았는데 `R`에서 행렬 분해는 어떻게 할까.

주로 `SVD(Singular Value Decomposition, 특이값 분해의 방식 중 하나)` 방식을 이용한다. 하지만 `(NaN)` 값이 없는 행렬에만 적용할 수 있다. 일반적으로 생성되는 `R` 행렬에는 많은 `Null 값`이 있기 때문에 다음 두가지 방식을 주로 사용해서 SVD를 구현한다.

- **확률적 경사 하강법(Stochastic Gradient Descent)**
- **ALS(Alternating Least Squares)**

확률적 경사 하강법을 통한 행렬 분해는 `P와 Q` 행렬로 계산된 **예측 행렬값**과, 가장 최소의 오류를 가질 수 있도록 **비용 함수(Loss Function)** 최적화를 통해 P와 Q를 **유추**한다.

1. P와 Q를 **임의의 값**을 가진 행렬로 설정한다.
2. P * Q.T 값을 통해 예측 R 을 계산하고 실제 R 과 비교하여 **오류값을 계산**한다.
3. 오류 값을 최소화 하는 P와 Q를 **다시 업데이트** 한다.
4. 만족할 오류 값을 가질 때 까지 **작업을 반복**하며 P, Q 값을 업데이트 한다.

## References

- **파이썬 머신러닝 완벽가이드**

  ![](data:image/jpeg;base64,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)

- https://greeksharifa.github.io/
- https://euriion.com/?p=548
- [https://scvgoe.github.io/](https://scvgoe.github.io/2017-02-01-협업-필터링-추천-시스템-(Collaborative-Filtering-Recommendation-System)/)

