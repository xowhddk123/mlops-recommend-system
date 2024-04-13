# lecture-mlops-recommend-sagemaker

딥러닝 기반 추천 시스템 패키지입니다.   
추천 시스템의 태스크별 이미지 빌드, 데이터 전처리, 학습, 추론 등의 기능이 포함되어 있습니다.   
특수한 케이스가 아니라면 딥러닝 모델링 시 해당 패키지를 확장하는 것을 권장합니다.


## 패키지 구조

- script : 이미지 빌드를 위한 디렉토리
  - {서비스명}    
- src : 모델 수행 소스코드
  - common : 공통 수행 코드
  - config : 모델 수행에 필요한 설정 파일들
  - dataset : 공통 데이터셋
  - inference : 추론 관련
  - metrics : 지표 관련
  - model : 모델 관련(모델 정의, 모델 데이터셋, 옵티마이저 & 스케줄러 등)
  - postprocess : 후처리 관련
  - preprocess : 전처리 관련
  - train : 학습 관련
  - utils : 각종 유틸리티
- test : 단위 테스트 코드
- entrypoint.ipynb : 노트북 기반 최초 진입점
- main.py : 실제 수행되는 코드의 최초 진입점


## 신규 버전 릴리즈

최상위 디렉토리의 release.sh 스크립트 수행을 통해 다음 버전 릴리즈를 수행할 수 있습니다.   
릴리즈는 CI/CD 파이프라인을 통해 자동으로 수행됩니다.

### 릴리즈 옵션
- -u : 업데이트 유형(major | minor | patch)
- -s : 서비스 이름

#### 릴리즈 샘플
`~님이 좋아할만한 영화` 모델의 patch 버전을 릴리즈합니다.
``` shell
./release.sh -u patch -s like-movie
```

## 이미지 빌드

이미지 빌드를 위한 Dockerfile 및 의존성 파일(requirements.txt)은 /script 이하 {서비스명} 디렉토리에 있습니다.   
이미지를 빌드하기 위해서는 최상위 디렉토리의 빌드 스크립트(/build.sh)를 수행합니다.    
빌드는 CI/CD 파이프라인을 통해 자동으로 수행되기 때문에 수동 작업이 필요한 경우에만 사용하시기 바랍니다.   
**빌드 이미지는 AWS ECR에 `{account}.dkr.ecr.ap-northeast-2.amazonaws.com/{repo}/{서비스명}:{버전}` 형식으로 저장됩니다**

### 빌드 옵션
- -s : 서비스 이름

**빌드 시 로컬 혹은 원격 이미지 저장소에 동일한 버전의 이미지가 이미 있으면 안됩니다**


#### 빌드 샘플
`~님이 좋아할만한 영화` 모델의 이미지를 빌드합니다.
``` shell
./build.sh -s like-movie
```
