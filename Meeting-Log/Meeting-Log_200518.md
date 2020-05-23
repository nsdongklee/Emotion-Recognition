# Final project log(20.05.18)

> 다음 문서는 19.12.09~20.06.20까지 진행된 혁신성장 이미지 분석 AI 서비스 개발 실무 과정의 최종 프로젝트 진행 사항을 기록한 것이다.

## 목차

1. 금주 일정 공지사항
2. 프로젝트 진행사항
3. 해결해야 할 문제점
4. 추가 고려 사항
5. 이후 일정

## 0. 금주 일정 공지사항

> 일반적인 작업 진행 스케줄은 앞의 로그에 포함되어 있으므로 생략

- 5/19(화) : **음악 추천 알고리즘 test set 구성**
- 5/21(목) : 프로젝트 작업 휴식

## 1. 프로젝트 진행 사항

[![캡처](https://user-images.githubusercontent.com/58945760/80307234-afdc8b80-8802-11ea-80a5-afb28bdfbb1f.PNG)](https://user-images.githubusercontent.com/58945760/80307234-afdc8b80-8802-11ea-80a5-afb28bdfbb1f.PNG)

1. **모델링**

   1. 데이터셋 증식을 통한 정확도 향상 시도(20.05.15~)

      > FFHQ dataset에서 분류한 이미지 데이터를 추가하여 모델 정확도를 향상시키는 동시에 negative 편중 현상을 해결하려 했다.

      1. 데이터셋 변화

         1. - `train set`: **Neutral**(11,404=>12,404), **Positive**(8,607=>9,607), **Negative**(8,097, 변화 없음)
            - `validation set` : Neutral(1,570=>2,070), Positive(1,146=>1,646), Negative(1,090)
            - `test set` : Neutral(1,571), Positive(1,221), Negative(1,090)

         - `classification report`를 통해 모델을 평가한 결과 점수는 평균 0.94가 나왔다. 또한 기존 데이터셋의 데이터 외에는 예측을 하지 못하던 것에 비해 감정 분류 정확도가 조금 향상되었으나 여전히 웹캠으로 받아들이는 이미지는 negative에 과히 편중된 경향을 보였다.
         - 증식한 이미지는 전체 데이터셋의 10% 정도이기에 큰 영향을 미치지 못한 것으로 보인다. 데이터 증식으로 유의미한 결과를 도출하기 위해서는 최소한 30% 이상의 데이터 증량이 필요할 것으로 보인다. 다만 데이터셋 구축에 다소 시간이 걸리므로 다른 대안도 숙고할 필요가 있다.

   2. `haar cascade`를 활용한 mouth, eyes pair 모델링 시도

      > 세 가지 표정을 분류할 때 가장 큰 변화를 보이는 부분은 눈을 포함한 미간(eyes pair, forehead) 부분과 입이다. 두 개의 영역을 크롭하여 학습시킨다면 데이터 증식 없이도 정확도를 향상시킬 가능성이 있다고 판단하였다.

      - `haar cascade_mcs_eyepair_big.xml`, `haar cascade_mcs_mouth.xml`을 적용하여 선택 영역을 지정하려고 시도했으나 `systemerror`와 `cv2. error`가 지속적으로 발생했다. 경로를 재차 수정하고 다른 파일을 다운로드하여 시험해보았으나 해결하지 못했다.
      - `haar cascade_eye.xml`과 `haar cascade_frontface.xml`은 잘 작동하는 것으로 보아 파일 자체의 문제이거나 버전 호환의 문제인 것으로 보인다.

   3. `dlib`을 활용한 모델링

      [![1_stAhFcyYdNqGvV26xMdE7w](https://user-images.githubusercontent.com/58945760/81699683-a3af2a00-94a2-11ea-8cb5-e1b80e95595b.png)](https://user-images.githubusercontent.com/58945760/81699683-a3af2a00-94a2-11ea-8cb5-e1b80e95595b.png)

      - 각 이미지 데이터에서 얻어낸 68개의 좌표를 저장하고, DNN으로 학습시킨다.
      - AWS에서 `kernel`이 죽는 문제가 발생

2. **음악 추천 알고리즘**

   - 알고리즘을 검증하기 위한 test set 구성(20.05.18~)
     - column 명: playlist ID, 각 수록곡 ID, 앨범 제목, 수록곡 제목, Tag, 좋아요 수
     - 기존 playlist들의 수록곡 title을 BeautifulSoup으로 크롤링하여 csv 파일로 저장
       - 4~6개의 playlist는 크롤링이 되지 않아 직접 작성

3. 프로젝트 계획서 초안 완성(20.05.16)

4. 프로젝트 [README.md](https://github.com/dannylee93/Emotion-Recognition/blob/master/README.md#emotion-recognition) 작성(20.04.13~, 진행중)

5. AWS 프리 티어를 이용한 웹 배포 논의 중

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
   1. AWS 프리 티어로 배포 결정(이미 배포한 팀의 workflow 참조)

## 4. 이후 일정

> 데이터셋 구축 일정 추가될 수 있음

1. **5/18~5/23: 음악 추천 알고리즘 구성, 감정 분류 모델링 보완**
   1. 5/20~6/7 : 최종 프로젝트 ppt 제작(이동규)
   2. 5/23~5/30 : 배포 및 베타 테스트
   3. 6/1~6/7 : 문서 정리 및 서류 작업(이하 중요 문서)
      1. Readme.md
      2. 최종 프로젝트 ppt
      3. 프로젝트 계획서
      4. 원본 소스코드
      5. 최종 결과보고서
      6. github repository 정리
   4. 6/8~6/19 : 카카오 아레나 및 데이콘 준비