# Final project log(20.05.23)

> 다음 문서는 19.12.09~20.06.20까지 진행된 혁신성장 이미지 분석 AI 서비스 개발 실무 과정의 최종 프로젝트 진행 사항을 기록한 것이다.

## 목차

1. 금주 일정 공지사항
2. 프로젝트 진행사항
3. 해결해야 할 문제점
4. 추가 고려 사항
5. 이후 일정

## 0. 금주 일정 공지사항

> 일반적인 작업 진행 스케줄은 앞의 로그에 포함되어 있으므로 생략

- 1주일에 하루는 작업 휴식(매주 월요일 팀원 일정에 따라 결정)

## 1. 프로젝트 진행 사항

[![캡처](https://user-images.githubusercontent.com/58945760/80307234-afdc8b80-8802-11ea-80a5-afb28bdfbb1f.PNG)](https://user-images.githubusercontent.com/58945760/80307234-afdc8b80-8802-11ea-80a5-afb28bdfbb1f.PNG)

1. **음악 추천 알고리즘** 구성

   - 알고리즘을 검증하기 위한 test set 구성(20.05.18~)

     - column 명: playlist ID, 각 수록곡 ID, 앨범 제목, 수록곡 제목, Tag, 좋아요 수
     - 기존 태그들 중 각 플레이리스트 수록곡별 tag 1개, 플레이리스트별 tag 1개 선정
     - 곡별 ID, 아티스트(가수)별 ID 검색, 채우기
     - ID가 존재하지 않을 경우 메타 데이터 생성

   - 구글 웹 크롤러 생성(20.05.22)

     - 알고리즘으로 뽑아낸 추천곡의 video id를 추출

   - Youtube Data API를 활용한 playlist update

     (20.05.23~)

     - 새 프로젝트 만들기
     - OAuth 2.0 클라이언트 ID 만들기(유형 데스크톱)
     - 크롤러로 뽑아낸 추천곡의 video id와 playlist id를 활용해 playlist update

2. dlib을 활용한 모델링

   - 정확도 88%
   - 모델 완성 후 웹페이지에 적용 중(20.05.22)

3. 프로젝트 계획서 피드백

   - 통계적인 자료 인용 시 슬라이드 밑단에 출처 추가
   - 스토리텔링적인 부분 생략 없이 슬라이드에 작성할 것

4. 프론트엔드 디자인 수정을 위한 자료 검색

5. 프로젝트 [README.md](https://github.com/dannylee93/Emotion-Recognition/blob/master/README.md#emotion-recognition) 작성(20.04.13~, 진행중)

6. AWS 프리 티어를 이용한 웹 배포 논의 중

## 2. 해결해야 할 문제점

- Youtube Data API 코드 실행 시 Http error 발생
- 모델 Output 편중 현상 해결

## 3. 추가 고려 사항

1. 분류 모델 알고리즘 검증 방법, 기준
2. 웹페이지 프론트엔드 수정
3. **홈페이지 배포 시 웹캠 작동 확인**

## 4. 이후 일정

1. 5/18~5/23: 음악 추천 알고리즘 구성, 감정 분류 모델링 보완
   1. 5/20~6/7 : 최종 프로젝트 ppt 제작(이경희)
   2. 5/23~5/30 : 배포 및 베타 테스트
   3. 6/1~6/7 : 문서 정리 및 서류 작업(이동규, 이하 중요 문서)
      1. Readme.md
      2. 최종 프로젝트 ppt
      3. 프로젝트 계획서
      4. 원본 소스코드
      5. 최종 결과보고서
      6. github repository 정리
   4. 6/8~6/19 : 카카오 아레나 및 데이콘 준비