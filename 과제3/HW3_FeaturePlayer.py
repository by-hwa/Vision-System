import sys
import numpy as np
import cv2

# 기준 영상 불러오기
src = cv2.imread('yu.jpg', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed!')
    sys.exit()

# 카메라 장치 열기
cap1 = cv2.VideoCapture(0)

if not cap1.isOpened():
    print('Camera open failed!')
    sys.exit()

# 카메라 프레임 화면에 출력할 동영상 파일 열기
cap2 = cv2.VideoCapture('yu.mp4')

if not cap2.isOpened():
    print('Video load failed!')
    sys.exit()

# TODO: 특징점 알고리즘 객체 생성
detector = cv2.ORB_create()

# 기준 영상에서 특징점 검출 및 기술자 생성
kp1, desc1 = detector.detectAndCompute(src, None)

# TODO: 사용하는 매칭 객체 생성
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

while True:
    ret1, frame1 = cap1.read()

    if not ret1:
        break

    # 매 프레임마다 특징점 검출 및 기술자 생성
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    kp2, desc2 = detector.detectAndCompute(gray, None)

    # 특징점이 100개 이상 검출될 경우 매칭 수행
    if len(kp2) > 100:
        # TODO: mathcing 하기
        matches = matcher.match(desc1,desc2)

        # TODO: 좋은 매칭 선별 (Hint, distance가 가까운 matching들을 고른)
        # TODO: 좋은 매칭들 80개만 선별
        good_matches = sorted(matches, key=lambda x: x.distance)
        
        for i in good_matches[:80]:
            idx = i.queryIdx
            

        pts1 = np.array([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)
        pts2 = np.array([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)

        # TODO: pts1, pts2로 호모그래피 계산
        H, inliers = cv2.findHomography(pts1, pts2, cv2.RANSAC,5.0)
        
        # TODO: Inlier 개수 확인하기
        inlierstr = []
        for m in inliers:
            if m == True:
                inlierstr.append([m])
        inlier_cnt = len(inlierstr)
        
        # Feature matching 개수 확인하기
        dst = cv2.drawMatches(src, kp1, gray, kp2, good_matches, None)

        # Inlier 개수가 20개 이상이면 로드한 동영상을 투시 변
        if inlier_cnt > 20:
            ret2, frame2 = cap2.read()

            if not ret2:
                break

            h, w = frame1.shape[:2]

            # 비디오 프레임을 투시 변환
            video_warp = cv2.warpPerspective(frame2, H, (w, h))

            white = np.full(frame2.shape[:2], 255, np.uint8)
            white = cv2.warpPerspective(white, H, (w, h))

            # 비디오 프레임을 카메라 프레임에 합성
            cv2.copyTo(video_warp, white, frame1)

    cv2.imshow('frame', frame1)
    cv2.imshow('dst', dst)

    if cv2.waitKey(1) == 27:
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
