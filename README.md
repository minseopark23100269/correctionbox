# correctionbox
My simple camera calibration and  distortion correction project

### 프로그램 소개
- 나의카메라(스마트폰 사용했습니다. )를 이용해 다양한 시점에서의 체스보드 동영상 촬영해서 카메라 캘리브레이션을 수행하고 캘리브레이션 결과(fx, fy, cx, cy, …, rmse)를 얻을 수 있다.
- 그리고 카메라 캘리브레이션 결과를 이용해 렌즈 왜곡 보정을 수행한 영상을 얻을 수 있다. 


### 기능
1. 나의 카메라를 캘리브레이션하여 렌즈의 내부 파라미터, 왜곡 계수, rms 오차 등을 계산할 수 있다.
2. 캘리브레이션 결과를 이용해 렌즈 왜곡을 보정한 결과를 나타낼 수 있다.
3.  오리지널 영상과 렌즈 왜곡을 보정한 영상을 비교할 수 있다.

---

## 1. 준비 과정

### 체스보드 준비
- A4 용지
- 내부 코너는 **8x6**이며, 각 셀 크기는 25mm입니다.


### 체스보드 영상 촬영
- 스마트폰 카메라를 이용하여 다양한 각도와 거리에서 체스보드를 촬영한 동영상 파일을 사용했습니다.
- 사용된 파일 경로: `C:\Users\p0105\Videos\chessboard.mp4`.

---

## 2. 카메라 캘리브레이션

### 캘리브레이션 수행
- OpenCV와 Python을 사용하여 캘리브레이션을 진행했습니다.
- `camera_calibration.py` 스크립트를 활용하여 계산을 수행했습니다.

### 결과
- **사용된 이미지 수**: 223
- **RMS 오차 (Root Mean Square Error)**: 1.6016
- **카메라 매트릭스 (K):**[[754.36932621 0. 354.80740075] [ 0. 769.75820428 440.99934945] [ 0. 0. 1. ]]
- **fx:** 754.37
- **fy:** 769.76
- **cx:** 354.81
- **cy:** 441.00
- **왜곡 계수 (Distortion Coefficients):**[ 2.77612703e-01 -2.93292907e+00 1.32399985e-02 -2.81894361e-03 9.97246553e+00]

## 3. 렌즈 왜곡 보정

### 수행 과정
- 캘리브레이션 결과를 사용하여 렌즈 왜곡을 보정했습니다.
- OpenCV의 `cv.remap()` 기능을 활용하여 왜곡을 제거했습니다.
- `distortion_correction.py` 스크립트를 사용했습니다.

### 비교: 원본 vs 보정
#### 원본 (Original)
- 왜곡이 포함된 영상으로 체스보드가 휘어져 보이는 현상이 나타남.

#### 보정 후 (Rectified)
- 렌즈 왜곡이 제거되어 체스보드가 직선으로 보입니다.

### 데모 결과

![이미지 설명]("data/Cap 2025-04-08 18-47-31-936.jpg")


