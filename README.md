# Vision-system

### 개발환경
- Window
- Python
  - Opencv
  - Numpy


### 설계목표
- 영상 이미지를 통하여, 빨간색과 노란색 차선을 인식하고 인식한 차선을 바탕으로 로봇이 진행해야 할 방향 제시.

### 내용

### 1. Camera Calibration
<img width="450" alt="image" src="https://github.com/by-hwa/Vision-System/assets/102535447/7f0a0dc3-b979-4536-ae38-daf94dca2065">

- Matlab의 Camera calibrator 앱을 사용하여 calibration​.

<img width="1056" alt="image" src="https://github.com/by-hwa/Vision-System/assets/102535447/0d79c113-5bc7-4727-927e-7c602da4d774">

- Camera Calibration Matrix 값을 얻음.

<img width="744" alt="image" src="https://github.com/by-hwa/Vision-System/assets/102535447/ad01eabc-4d4a-4a7d-9335-e12ab08e6b9c">

- Calibration 수행으로 얻은 Camera Matrix와  FocalLength 값을 이용하여 왜곡처리.

<img width="794" alt="image" src="https://github.com/by-hwa/Vision-System/assets/102535447/ef1474c8-ca83-4548-ba03-7f7295b0857f">

- Use cv2.undistort

### 2. Line Detection & Following Line


#### ROI(Region Of Interesting)

<img width="1085" alt="image" src="https://github.com/by-hwa/Vision-System/assets/102535447/25009463-026f-44c9-886a-1040a9ad070b">

- 차선만을 인식하기 위하여 불필요한 부분 제거

#### Bird Eye View
<img width="530" alt="image" src="https://github.com/by-hwa/Vision-System/assets/102535447/e6eb771f-5b18-4455-955a-f30b0c78b1fc">

- 차선의 각도로 방향을 검출하기 위하여 Bird Eye View 로 이미지 변환.

#### 특정색상 추출

<img width="1070" alt="image" src="https://github.com/by-hwa/Vision-System/assets/102535447/3fe81251-4a4e-4bc8-8429-f6a43672a744">

- HSV(색상, 명도, 채도) 이미지에서 노란색 영역과 빨간색 영역을 검출.
- 검출된 두개의 이미지를 합쳐서 빨간색과 노란색을 검출.

#### 허프변환으로 라인 검출

<img width="579" alt="image" src="https://github.com/by-hwa/Vision-System/assets/102535447/8b139f48-c201-4af9-a552-65839d52849e">

- Hough Transform 으로 라인 검출.
- 검출된 라인정보, 시작과 끝점의 xy 좌표를 이용하여 라인의 각도와 방향 검출.
- 검출된 각도로 로봇이 진행해야할 방향 지시.

### 3. Object Detection and Localization

<img width="503" alt="image" src="https://github.com/by-hwa/Vision-System/assets/102535447/f410a74c-730c-4cc9-9ad2-093136eeef1a">

- 처음엔 YOLO V3 모델을 사용하였지만, 보행자 인식만 을 할 것이기 때문에 훨씬 가벼운 HOG Descriptor를 사용.

<img width="947" alt="image" src="https://github.com/by-hwa/Vision-System/assets/102535447/b7293d71-6e7c-4240-b45a-0a8953e9e450">
<img width="278" alt="image" src="https://github.com/by-hwa/Vision-System/assets/102535447/751f1f7f-4d40-4386-b5d4-b1762d2ebb90">

- 픽셀과 Calibration 과정에서 구한 Focal Length를 이용하여 탐지한 보행자의 위치 인식.
- L1:Y1 = L2:Y2,  L1 = Focal length, Y1 = Pixel(Detection 박스의 높이)​
- L2 = 카메라와 Object 사이의 거리 Y2 = 사람의 키
- L2 = L1xY2/Y1

### 4. Marker Detection and Localization
- cv2.aruco.DetectMarkers() 를 이용하여 마커 검출.
- cv2.aruco.DectedMarkers()로 마커의 정보 검출.
- cv2.aruco.estimatePoseSingleMarkers() 회전 및 변환 백터 검출.
- cv2.aruco.drawAxis() 마커 이미지에 그리기.

### 5. Additional
<img width="472" alt="image" src="https://github.com/by-hwa/Vision-System/assets/102535447/d128b8dd-b312-407d-9bbd-c2e216073bc1">

- 차선내에 객체 검출시 정지.

### 6. Intergrated
<img width="506" alt="image" src="https://github.com/by-hwa/Vision-System/assets/102535447/ba6c261f-f9b9-4245-b67d-9e9dd2a19ae0">

- 현재 차선에 따라 진행방향 지시.
- Marker and Object detection.
- 차선내 객채 탐지시 Stop.
