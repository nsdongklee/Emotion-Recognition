# Final project log(20.05.12)

> 다음 문서는 19.12.09~20.06.20까지 진행된 혁신성장 이미지 분석 AI 서비스 개발 실무 과정의 최종 프로젝트 진행 사항을 기록한 것이다.

## 목차

1. 금주 일정 공지사항
2. 프로젝트 진행사항
3. 해결해야 할 문제점
4. 추가 고려 사항
5. 이후 일정

## 0. 금주 일정 공지사항

> 일반적인 작업 진행 스케줄은 앞의 로그에 포함되어 있으므로 생략

- 5/13(수) : 프로젝트 작업 휴식
- 5/16(토) : 프로젝트 중간 발표

## 1. 프로젝트 진행 사항

[![캡처](https://user-images.githubusercontent.com/58945760/80307234-afdc8b80-8802-11ea-80a5-afb28bdfbb1f.PNG)](https://user-images.githubusercontent.com/58945760/80307234-afdc8b80-8802-11ea-80a5-afb28bdfbb1f.PNG)

1. 모델링

   > 전이학습을 기반으로 하며, 주 튜닝 모델은 MobileNetV2.

   1. `dlib`을 활용한 모델링 시도

      ![1_stAhFcyYdNqGvV26xMdE7w](https://user-images.githubusercontent.com/58945760/81699683-a3af2a00-94a2-11ea-8cb5-e1b80e95595b.png)

      - 얼굴의 `landmark`로 68개 좌표 검출
      - 이미지의 좌표값을 비교해 감정 분류

   2. 데이터 전처리

      > 기쁨(Positive) 과 찡그림(Negative) 을 잘 분류하지 못하는 이유가 데이터의 특성을 제대로 추출하지 못했기 때문이라고 판단, 전처리를 시행

      1. 기쁨(Positive) 데이터 전처리(20.05.11)
         - 데이터 수 17,280 => 12,458
      2. 찡그림(Negative) 데이터 전처리(20.05.12)
         - 데이터 수 17,280 => 14,191

   3. 이미지 데이터 크기 조절

      > `Imagegenerator`를 활용하여 원본 이미지와 같은 크기(400*400)으로 두 가지 모델 적용

      1. 공통
         - `train set`: **Neutral**(13,824), **Positive**(9,967), **Negative**(11,353)
         - `validation set` : Neutral(3,456), Positive(2,491), Negative(2,838)
         - `test set` : Neutral(2,880), Positive(2,880), Negative(2,880)
      2. 전이학습(MobileNetV2)
         - `image size` : 400, 400, 3
         - `epoch` : 30
         - `dropout` : 0.3
         - `Total param` : 57,636,931
      3. CNN model
         - `image size` : 400, 400, 3
         - `epoch` : 30
         - `dropout` : 0.25
         - `Total param` : 85,218,179

      - 전이학습 결과(`Test set` 기준 Neutral acc : 0.21, Positive acc : 0.82, Negative acc : 0.65), CNN model(Neutral acc : 0.99, Positive acc : 0.79, Negative acc : 0.98) 결과 모두 정확도가 상승. 이미지 데이터 크기가 모델 적용 시 주요한 하이퍼파라미터라는 사실을 알 수 있었다.
      - 하지만 아직 웹페이지에 적용했을 때 만족스러운 결과라 보기는 어려웠다. `k-face dataset`에 속하지 않은 다른 image data로 `predict`, 보다 정밀한 `accuracy` 측정 필요.

2. 음악 추천 알고리즘

   [![img](https://camo.githubusercontent.com/ff33ab748a91f0e120d10568f551729b117c7298/68747470733a2f2f677265656b736861726966612e6769746875622e696f2f7075626c69632f696d672f4d616368696e655f4c6561726e696e672f323031392d31322d31372d5265636f6d6d656e646174696f6e25323053797374656d2f30312e4a5047)](https://camo.githubusercontent.com/ff33ab748a91f0e120d10568f551729b117c7298/68747470733a2f2f677265656b736861726966612e6769746875622e696f2f7075626c69632f696d672f4d616368696e655f4c6561726e696e672f323031392d31322d31372d5265636f6d6d656e646174696f6e25323053797374656d2f30312e4a5047)

   - 추천 시스템에 대한 설명은 [해당 문서](https://github.com/dannylee93/Emotion-Recognition/tree/master/Recommender-System#recommender-system) 참고
   - 곡의 메타 데이터를 활용, `output`으로 곡 제목을 가지는 알고리즘 생성
   - 알고리즘을 거쳐 도출된 `output`과 `youtube API`를 활용해 기존 `playlist`에 추가

3. 프로젝트 [README.md](https://github.com/dannylee93/Emotion-Recognition/blob/master/README.md#emotion-recognition) 작성(20.04.13~, 진행중)

4. AWS 프리 티어를 이용한 웹 배포 논의 중

## 2. 해결해야 할 문제점

1. **모델 정확도 향상**(목표 평균 85% ↑ )

   1. `haar cascade`를 데이터 전체 적용하여 train
   2. `train set` 불균형 유도

2. ~~웹페이지에서 playlist 재생 시 **'동영상을 재생할 수 없음'** 메시지 해결~~

   [![message2](https://user-images.githubusercontent.com/58945760/80307311-2d080080-8803-11ea-9b75-02cd9c5c9398.PNG)](https://user-images.githubusercontent.com/58945760/80307311-2d080080-8803-11ea-9b75-02cd9c5c9398.PNG)

   - 위 메시지의 경우 웹페이지 배포와 상관없이

      

     ```
     Youtube
     ```

      

     외 사이트에서 재생 불가

     - 해당 메시지가 표시된 영상 교체

3. ~~전체 데이터셋을 모두 사용할 경우(이미지 사이즈 100*100) AWS에서 Memoryerror 발생~~

   - `imagegenerator`의 `flow_from_directory`를 활용하여 해결

## 3. 추가 고려 사항

1. 결과물 산출 후 웹 페이지 검증 방법, 기준
2. 분류 모델 알고리즘 검증 방법, 기준
3. 웹페이지 배포 툴 선택
   1. python anywhere는 용량 제한이 있고 속도가 느린 단점이 있음
   2. AWS 프리 티어를 비롯한 다른 툴들도 고민해볼 필요가 있음

## 4. 이후 일정

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