# Final project log(20.05.15)

> 다음 문서는 19.12.09~20.06.20까지 진행된 혁신성장 이미지 분석 AI 서비스 개발 실무 과정의 최종 프로젝트 진행 사항을 기록한 것이다.

## 목차

1. 금주 일정 공지사항
2. 프로젝트 진행사항
3. 해결해야 할 문제점
4. 추가 고려 사항
5. 이후 일정

## 0. 금주 일정 공지사항

> 일반적인 작업 진행 스케줄은 앞의 로그에 포함되어 있으므로 생략

- 5/16(토) : 프로젝트 중간 발표

## 1. 프로젝트 진행 사항

[![캡처](https://user-images.githubusercontent.com/58945760/80307234-afdc8b80-8802-11ea-80a5-afb28bdfbb1f.PNG)](https://user-images.githubusercontent.com/58945760/80307234-afdc8b80-8802-11ea-80a5-afb28bdfbb1f.PNG)

1. **모델링**

   > 초기 계획은 전이학습을 기반으로 하고 있었으나(주 튜닝 모델 MobileNetV2) 바닥부터 쌓아올린 CNN 모델과 비교할 때 정확도에 있어 현저한 차이가 있었다. 때문에 이후 모델링은 CNN Architecture와 하이퍼 파라미터 튜닝에 집중한다.

   1. `haar cascade`를 적용한 데이터셋 구축, 학습

      1. 기존의 모델은 F1 score에서 80%~90%의 정확도를 보였음에도 웹페이지에 적용했을 때 그에 준하는 결과가 나오지 않았다. django에서 model에 들어가는 이미지 데이터는 `haar cascade`를 먼저 적용하여 얼굴 부분만을 선택한 이미지 데이터였기에 모델을 훈련하는 데이터도 `haar cascade`를 적용한 데이터여야 했다.

      2. ```
         haar cascade
         ```

         를 적용하는 와중에 얼굴이 검출되지 않은 데이터도 있어 전체 데이터셋의 크기에 변화가 있었다. 또한 기존의 test set은 전처리가 부족한 데이터셋이라 판단되어 제외하고 기존 validation set에서 1:1의 비율로 분리하였다.

         1. 데이터셋 변화
            - `train set`: **Neutral**(13,824=>11,404), **Positive**(9,967=>8,607), **Negative**(11,353=>8,097)
            - `validation set` : Neutral(3,456=>1,570), Positive(2,491=>1,146), Negative(2,838=>1,090)
            - `test set` : Neutral(1,571), Positive(1,221), Negative(1,090)

   2. 이미지 데이터 정규화

      1. 데이터셋에 `haar cascade`를 적용해 다시 훈련시키자 classification report에서 다음과 같은 결과를 확인할 수 있었다.

      [![great score](https://user-images.githubusercontent.com/58945760/82047818-6f2fae00-96ee-11ea-8dc3-3dae52262bb9.PNG)](https://user-images.githubusercontent.com/58945760/82047818-6f2fae00-96ee-11ea-8dc3-3dae52262bb9.PNG)

      아래는 `haar cascade`를 적용하지 않았던 모델에서 나온 결과이다.

      [![score_report(cnn)](https://user-images.githubusercontent.com/58945760/82045626-5b824880-96ea-11ea-9083-9905f45c71ce.PNG)](https://user-images.githubusercontent.com/58945760/82045626-5b824880-96ea-11ea-9083-9905f45c71ce.PNG)

      결과가 좋게 나왔음에도 웹페이지에서 확인한 결과는 여전히 모델과 상반되었다. django 코드 내 모델을 적용하기 전 255로 나누어 정규화하는 코드를 덧붙인 후에야 모델의 정확도가 predict 결과에 반영되었다.

   3. `dlib`을 활용한 모델링

      [![1_stAhFcyYdNqGvV26xMdE7w](https://user-images.githubusercontent.com/58945760/81699683-a3af2a00-94a2-11ea-8cb5-e1b80e95595b.png)](https://user-images.githubusercontent.com/58945760/81699683-a3af2a00-94a2-11ea-8cb5-e1b80e95595b.png)

      > CNN 모델링과 다른 방법으로 감정을 분류할 수 있는지를 실험한다. 각 모델의 정확도를 계속 비교하며 더 우수한 모델을 선택한다.

      - 얼굴의 `landmark`로 68개 좌표 검출
      - 이미지의 좌표값을 비교해 감정 분류
        - 감정별 데이터의 `landmark` 확인, 그래프 그려 분포 확인
        - 코 아래 무게 중심점을 정한 후 양 입가의 끝점과 중심점 사이의 거리를 비교, 측정하는 방법 시도

2. **음악 추천 알고리즘**

   > 서비스의 배포와 더불어 playlist는 지속적인 업데이트가 필요하다. **기존 playlist의 성격에 맞는 곡을 골라내는 것은 물론, 해당 youtube playlist 추가하기까지의 과정을 자동화**하기 위한 핵심 알고리즘이다.

   [![img](https://camo.githubusercontent.com/ff33ab748a91f0e120d10568f551729b117c7298/68747470733a2f2f677265656b736861726966612e6769746875622e696f2f7075626c69632f696d672f4d616368696e655f4c6561726e696e672f323031392d31322d31372d5265636f6d6d656e646174696f6e25323053797374656d2f30312e4a5047)](https://camo.githubusercontent.com/ff33ab748a91f0e120d10568f551729b117c7298/68747470733a2f2f677265656b736861726966612e6769746875622e696f2f7075626c69632f696d672f4d616368696e655f4c6561726e696e672f323031392d31322d31372d5265636f6d6d656e646174696f6e25323053797374656d2f30312e4a5047)

   - 추천 시스템에 대한 설명은 [해당 문서](https://github.com/dannylee93/Emotion-Recognition/tree/master/Recommender-System#recommender-system) 참고
   - 메타 데이터가 포함된 음악 데이터셋 활용
     - 사용 column : playlist ID, playlist 제목, playlist 수록곡 id, 좋아요 수, Tag
     - Tag 컴퓨터가 인식할 수 있도록 vector화 + 수록곡 사이 유사도 측정
   - 기존 playlist 활용
     - 알고리즘을 검증하기 위한 test set 구성
     - Melon web crawler
     - column 명: ID, 수록곡 ID, 수록곡 제목, Tag, 좋아요 수
   - 알고리즘을 거쳐 도출된 `output`과 `youtube API`를 활용해 기존 `playlist`에 추가하는 자동화 코드 만들기

3. 프로젝트 [README.md](https://github.com/dannylee93/Emotion-Recognition/blob/master/README.md#emotion-recognition) 작성(20.04.13~, 진행중)

4. AWS 프리 티어를 이용한 웹 배포 논의 중

## 2. 해결해야 할 문제점

1. **모델이 데이터셋에 Overfitting**

   1. 웹페이지에서 `k-face dataset`에 속하지 않은 다른 데이터들을 업로드하여 predict한 결과 감정을 제대로 분류하지 못했고, `Negative`에 편중된 결과가 나타났다.

   2. 모델 훈련과 테스트에 사용된 `dataset`은 모두 `k-face dataset`에서 나온 데이터였고, 훈련 후의 평가 지표는 평균 0.98로 거의 1에 가까웠다. 더욱이 `k-face dataset`은 같은 사람을 다른 각도에서 찍은 사진들이 다수 존재하여 데이터셋의 다양성을 보장하지 못한다. 따라서 모델 훈련 중 Overfitting이 의심된다.

   3. 웸캠으로 받아들인 이미지들은 `haar cascade`가 적용된 상태라 하여도 얼굴 형태와 조명 상태가 천차만별일 수 있다. 따라서 데이터셋 Overfitting 문제를 해결하지 않으면 서비스의 핵심을 구현할 수 없게 된다. 현재 논의되는 해결 방법은 다음과 같다.

   4. 데이터셋 증식

      > 새로운 데이터를 훈련 데이터에 포함하여 기존 데이터셋의 부족한 다양성을 보충하는 것이 핵심이다. 비율을 조절해가며 모델의 score를 지속적으로 기록한다.

      - 서브 데이터셋인 [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset)에서 이미지 추출
      - 노트북에서 웹캠 연사를 통해 직접 데이터 수집

   5. `Input` 이미지 전처리

      > 웹캠으로 받아들이는 `Input` 이미지는 훈련 이미지와 여러 가지 차이점이 존재한다. 전처리를 통해 그 간극을 줄일 수 있는지를 확인한다.

      - 이미지 디노이즈
      - OpenCV로 이미지 화질 보정

## 3. 추가 고려 사항

1. ~~결과물 산출 후 웹 페이지 검증 방법, 기준~~F1 score 활용
2. 분류 모델 알고리즘 검증 방법, 기준
3. 웹페이지 배포 툴 선택
   1. AWS 프리 티어를 비롯한 다른 툴들도 고민해볼 필요가 있음
   2. 무료 도메인 서치

## 4. 이후 일정

> 데이터셋 구축 일정 추가될 수 있음

1. 5/2(토)~6/20(금)
   1. **5/11~5/16: 음악 추천 알고리즘 구성, 감정 분류 모델링 보완**
   2. 5/18~5/23: 음악 추천 알고리즘 구성, 감정 분류 모델링 보완
      1. 5/20~6/7 : 최종 프로젝트 ppt 제작(이동규)
   3. 5/23~5/30 : 배포 및 베타 테스트
   4. 6/1~6/7 : 문서 정리 및 서류 작업(이하 중요 문서)
      1. Readme.md
      2. 최종 프로젝트 ppt
      3. 프로젝트 계획서
      4. 원본 소스코드
      5. 최종 결과보고서
      6. github repository 정리
   5. 6/8~6/19 : 카카오 아레나 및 데이콘 준비