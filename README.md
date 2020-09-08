# Machine-Learning/Deep-Learning 목차

## <YOLO 맛보기>

### 1.  6/30 화 (YOLO 객체 탐지 알고리즘 : 딥러닝을 기반으로 한 객체 탐지 기법)
    

-   YOLO: 각 이미지를 다수의 그리드(경계 상자)로 분할하고 신뢰도를 계산. 가장 높은 객체 인식 정확성을 가지는 그리드(경계 상자)를 검출
    

-   참고1: [https://blogsaskorea.com/156](https://blogsaskorea.com/156)
    
-   참고2(딥러닝 기반 FAST 객체 탐색 기법. CNN, YOLO, SSD..): [https://sites.google.com/site/bimprinciple/in-the-news/dibleoning-euliyonghangaegchegeomchulr-cnnyolossd](https://sites.google.com/site/bimprinciple/in-the-news/dibleoning-euliyonghangaegchegeomchulr-cnnyolossd)
    
-   참고3(yolo 데이터 파일 정리): [https://kd1658.tistory.com/26](https://kd1658.tistory.com/26)
    - data: 학습과 트레이닝 데이터 정의(classes:감지할 데이터 종류 수, train: 학습할 이미지 경로가 담긴 리스트, names: 감지할 데이터 이름이 담긴 리스트 파일)
    - cfg: CNN 레이어 구조 정의(배치 사이즈, 이미지 사이즈 등)
    - weights: 학습시킨 가중치 정보

-   코드(이미지 객체 탐지, 영상 객체 탐지)
    
<br>
 
## <opencv, Scikit-learn 패키지를 활용한 머신러닝>

#### 라벨링 필요. 공공 csv데이터를 활용

### 1.  7/1 수 (머신러닝1 k-nn 최근접 이웃 알고리즘 모델)
    
-   머신러닝 모델1: 지도학습 모델 KNN=> 주로 분류와 회귀분석을 위해 사용
ex. 도형 판별, 숫자 판별(num_recog, num_testing, num_training)

    -   예제1: knn을 이용한 동그라미가 무슨 도형인지 판별
    -   예제2: knn을 이용한 손글씨 숫자 판별
         -  npz 파일에 학습결과 저장: numpy에서 지원하는 바이너리 파일
   
     -   harr cascade: opencv의 머신러닝기반 객체 탐지 알고리즘
         - create sample(positive와 negative 샘플을 입력 createsamples.cpp 빌드)
         -  내가 학습시킨 xml파일로 특정 객체 탐지
         -  [https://webnautes.tistory.com/1352](https://webnautes.tistory.com/1352)
         - [http://www.gisdeveloper.co.kr/?p=7208](http://www.gisdeveloper.co.kr/?p=7208)

  
### 2.  7/6 월 (머신러닝, 딥러닝 개념과 관련 라이브러리 그리고 기본 예제)
    

-   OpenCV(머신러닝), tensorflow(딥러닝)의 차이 => 자동 라벨링 or not
    
-   Jupyter
    
-   Numpy: 머신러닝/딥러닝에서 자주 사용되는 모듈
    
-   Pandas: 데이터를 항목별로 관리하는 데에 특화된 라이브러리
    
-   Matplotlib : 파이썬에서 데이타를 차트나 플롯(Plot)으로 그려주는 라이브러리 패키지로서 가장 많이 사용되는 데이타 시각화(Data Visualization) 패키지
    
-   Scikit-learn : 머신러닝 학습을 위한 파이썬 라이브러리(svm, 랜덤포레스트 등 다양한 머신러닝 알고리즘을 제공)
    -   예제 : xor 학습. *히든 레이어 1개인 단층 퍼셉트론이므로 학습 수준 낮음.
    

### 3.  7/7 화(관련 라이브러리, 머신러닝2 SVM)
  
-   머신러닝 모델2: 지도학습(라벨링 o)모델 SVM(서포트 벡터 머신) => 주로 분류와 회귀분석을 위해 사용
    -   예제1: SVM을 이용한 붓꽃 종류 판별(이미 존재하는 iris.csv 파일 활용)
        - iris.csv: 꽃받침 길이와 폭, 꽃잎 길이와 폭, 붓꽃 종류 5개의 컬럼으로 되어있는 데이터 분류 파일

     -   Pandas 마무리
     -  예제2: SVM을 이용한 손글씨 숫자 판별 (struct 모듈을 이용하여 csv 파일 생성)
         - cf. csv 파일의 역할: 머신러닝에서 머신이 각 데이터의 특징을 추출하기 위해서 이미 정리된 특징을 담은 파일을 제공해 줘야 한다. => csv 파일 

    -   예제3: SVM을 이용한 비만도 판별
    -   연습문제: 알파벳 인식
        - a.알파벳 샘플 이미지 생성 
        -  b. csv 파일 생성 c. 학습


### 4.  7/8 수(머신러닝3 Random Forest, 머신러닝 성능 향상)
  
-   기본 개념 복습: 텐서플로우와 케라스, SVM  
-   머신러닝 모델3: 앙상블 학습 모델 Random Forest => 주로 분류와 회귀분석을 위해 사용
    -   예제1: RF를 이용한 버섯 종류 판별(교차검증, mushroom.csv 사용)
  -  머신러닝 성능 높이기 Grid Search: 모든 조합을 시도하여 매개변수를 튜닝. 머신러닝 성능을 높임
     -   예제2: SVM을 이용한 손글씨 숫자 판별 (Grid Search로 성능 향상시키기)
    
-   연습문제: 공공데이터(csv파일)로 앱 만들기
     - a. 데이터 선택 
     - b.데이터 분석 
     - c.데이터 parsing(data/label 분류) 
     - d.학습시키기 
     - e.적용

  

### 5.  7/9 목(공공데이터 활용 어플리케이션)

-   연습문제 답안: 심장병 
    -  학습 데이터 저장, 로드 =>pickle.dump(), pickle.load()
    
-   연습문제: 오목 -ing
    

  <br>

## <Tensorflow(keras)를 활용한 딥러닝>

#### (중요) 텐서플로우 계층을 활용한 딥러닝 신경망 만들기: [https://www.hanbit.co.kr/media/channel/view.html?cms_code=CMS9611499295](https://www.hanbit.co.kr/media/channel/view.html?cms_code=CMS9611499295)

### 1.  7/10 금(텐서플로우 기본, 딥러닝1-단층 퍼셉트론)
    
-   텐서플로우 기본 환경 세팅
-   텐서플로우 기본 개념(핵심: 텐서 값 세팅, 연산 구조 세팅, 세션 생성, 결과 실행)
    -   예제1: 텐서플로우를 활용한 비만도 판별(단층 퍼셉트론- softmax 활성화 함수 사용)
         - 데이터 가져오기, 텐서 값 세팅(placeholder,variable), 사용할 신경망 함수 세팅
     -   예제2: xor 학습
         - 1. softmax 활성화 함수를 이용. 1층의 히든레이어로 구성된 단층 퍼셉트론 모델
         -  2. sigmoid 활성화 함수를 이용. 2층의 히든레이어로 구성된 다층 퍼셉트론(MLP) 모델=>정확도 향상
    -   예제3: 붓꽃 종류 판별(단층 퍼셉트론-softmax)
        - tf.matmul(), placeholder, argmax, cross entropy

  

### 2.  7/13 월(딥러닝2-MLP(dense))
   
-   딥러닝 모델1: 다층 퍼셉트론 구조(MLP)    
    - Sequential모델 생성.
    
-   예제1: MLP구조를 이용한 xor 학습
    
-   연습문제: 합격/불합격 판별
    
-   예제2: MLP구조를 이용한 영화리뷰 긍/부정 판별 *keras.datasets의 imdb사용
    
-   예제3: test_data의 리뷰 10개를 추출. 문장으로 변환. 해당 문장 긍/부정 판별
    
-   예제4: MLP구조를 이용한 비만도 판별
    
-   예제5: MLP구조를 이용한 손글씨 숫자 판별
    

  

### 3.  7/14 화(이미지 처리, 딥러닝3-CNN)
    

-   딥러닝 모델3: CNN(Convolutional Neural Network, 합성곱 신경망)
    - 합성곱층(곱)과 풀링층(크기 축소)으로 구성

-   예제1: CNN모델을 이용한 손글씨 숫자 판별 *keras.dataset의 mnist 사용
    
-   예제2: CNN모델을 이용한 개와 고양이 판별
    -  ImageDataGenerator 이용하여 딥러닝에서 사용할 수 있는 이미지로 변환
    - 학습 결과 저장/로드: from keras.models import load_model
    - model.save(~~.h5)/load_model(경로)

-   데이터 증식: ImageDataGenerator사용
    

  

### 4.  7/15 수(텍스트 처리, 딥러닝4-MLP(embedding))
    
-   케라스 기본(전처리, 워드 임베딩, 모델링, 컴파일과 훈련, 평가와 예측)
-   예제1: 사용자가 사전 생성하여 텍스트 처리
-   예제2: 사전 API 사용한 텍스트 처리(keras.preprocessing.text api)
-   예제3: 감성분류(영화평 긍부정 판별) -- 직접 임베딩층 구현하여 학습
-   예제4: 감성분류(영화평 긍부정 판별) -- 사전 훈련된 워드 임베딩 가져와서 사용(pre-trained glove embedding)
-   예제5: 기사 토픽 분류(1.dense층으로 구성 2.임베딩층으로 구성)
-   연습문제: CNN모델을 이용한 가구 이미지 분류
    
### 5.  7/16 목
    
-   딥러닝 모델5: RNN(Recurrent Neural Network, 순환 신경망)
    - 현재 데이터와 과거 데이터를 함께 고려하여 데이터 처리
