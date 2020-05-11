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



<br>

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
  |    사용자 1     |   5    |   2    |   3    |
  |    사용자 2     |   2    |   4    |   4    |
  |    사용자 3     |   1    |   2    |   3    |

- `아이템 기반(Item - Item)` : 특정 상품을 선택한 고객이 선택한 다른 상품

  아이템이 가지는 속성과는 상관 없이 사용자들이 **아이템을 좋아하는지/싫어하는지 평가** 척도의 유사도를 기준으로 하는 알고리즘

  | 아이템\사용자 | 사용자 1 | 사용자 2 | 사용자 3 |
  | :-----------: | :------: | :------: | :------: |
  |    영화 1     |    1     |    5     |    5     |
  |    영화 2     |    2     |    2     |    2     |
  |    영화 3     |    1     |    2     |    1     |

일반적으로는 사용자 기반 보다 아이템 기반 협업 필터링이 정확도 측면에서는 더 좋다고 한다. 또한, 유사도 측정의 한 방법인 **코사인 유사도** 를 통해서 유사도를 측정한다.

- `코사인 유사도(Cosine-Similarity)` : 

  

### (2) Latent Factor Filtering

![](https://greeksharifa.github.io/public/img/Machine_Learning/2019-12-17-Recommendation%20System/03.JPG)

<img src="https://greeksharifa.github.io/public/img/Machine_Learning/2019-12-17-Recommendation%20System/04.JPG" style="zoom: 67%;" />



## References

- **파이썬 머신러닝 완벽가이드**

  ![](data:image/jpeg;base64,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)

- https://greeksharifa.github.io/

