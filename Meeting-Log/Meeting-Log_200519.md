# Final project log(20.05.19)

> 다음 문서는 19.12.09~20.06.20까지 진행된 혁신성장 이미지 분석 AI 서비스 개발 실무 과정의 최종 프로젝트 진행 사항을 기록한 것이다.

## 목차

1. 금주 일정 공지사항
2. 프로젝트 진행사항
3. 해결해야 할 문제점
4. 추가 고려 사항
5. 이후 일정

## 0. 금주 일정 공지사항

> 일반적인 작업 진행 스케줄은 앞의 로그에 포함되어 있으므로 생략

- 5/20(수) :

   

  음악 추천 알고리즘 test set 구성 계속

  - 각 팀원들이 10개~11개의 playlist를 맡아 작업
  - 곡 정보를 찾을 수 없을 경우 파란색으로 표시

- 5/21(목) : 프로젝트 작업 휴식

## 1. 프로젝트 진행 사항

[![캡처](https://user-images.githubusercontent.com/58945760/80307234-afdc8b80-8802-11ea-80a5-afb28bdfbb1f.PNG)](https://user-images.githubusercontent.com/58945760/80307234-afdc8b80-8802-11ea-80a5-afb28bdfbb1f.PNG)

1. **모델링**

   1. 데이터셋 증식을 통한 정확도 향상 시도(20.05.15~)

      > FFHQ dataset에서 분류한 이미지 데이터를 추가하여 모델 정확도를 향상시키는 동시에 negative 편중 현상을 해결하려 했다.

      1. 데이터셋 변화

         > 지난번의 데이터 증식을 포함, Positive, Neutral set만 약 20 % 증식

         1. - `train set`: **Neutral**(12,404=>13,488), **Positive**(9,607=>10,492), **Negative**(8,097, 변화 없음)
            - `validation set` : Neutral(1,570), Positive(1,146), Negative(1,090)
            - `test set` : Neutral(1,771), Positive(1,344), Negative(1,090)

         [![데이터 2차 추가 후 score](https://user-images.githubusercontent.com/58945760/82457037-ed75c100-9aef-11ea-8c00-028d6b5126b6.PNG)](https://user-images.githubusercontent.com/58945760/82457037-ed75c100-9aef-11ea-8c00-028d6b5126b6.PNG)

         - `classification report`를 통해 모델을 평가한 결과 점수는 평균 0.97가 나왔다. 정확도가 웹페이지에 반영되기 시작했으며, 여전히 조금은 Negative에 편중된 결과를 보였으나 웹캠으로 받아들인 새로운 데이터도 비교적 정확히 분류하는 모습을 보였다.
         - Positive, Neutral 데이터셋 증식의 최적화 포인트를 찾을 필요가 있다. 지나치게 데이터를 증식시킬 경우 Neutral로 편중되는 결과가 나올 수 있으므로 조금씩 데이터를 추가해가며 실험해본다.

2. **음악 추천 알고리즘**

   - 알고리즘을 검증하기 위한 test set 구성(20.05.18~)
     - column 명: playlist ID, 각 수록곡 ID, 앨범 제목, 수록곡 제목, Tag, 좋아요 수
     - 기존 태그들 중 각 플레이리스트 수록곡별 tag 1개, 플레이리스트별 tag 1개 선정
     - 곡별 ID, 아티스트(가수)별 ID 검색, 채우기
     - ID가 존재하지 않을 경우 메타 데이터 생성

3. 프로젝트 [README.md](https://github.com/dannylee93/Emotion-Recognition/blob/master/README.md#emotion-recognition) 작성(20.04.13~, 진행중)

4. AWS 프리 티어를 이용한 웹 배포 논의 중

## 2. 해결해야 할 문제점

1. ~~모델이 데이터셋에 Overfitting~~데이터 증식으로 해결 중

## 3. 추가 고려 사항

1. 분류 모델 알고리즘 검증 방법, 기준
2. 웹페이지 프론트엔드 수정
3. **홈페이지 배포 시 웹캠 작동 확인**

## 4. 이후 일정

1. 5/18~5/23: 음악 추천 알고리즘 구성, 감정 분류 모델링 보완
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