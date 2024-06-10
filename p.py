import numpy as np
# clip_min과 clip_max 정의
clip_min = 0
clip_max = 10
# 등간격 포인트 수
num = 5
# np.linspace()를 사용하여 등간격의 숫자 생성
points = np.linspace(clip_min, clip_max, num + 1)
print("등간격으로 생성된 숫자:", points)
# points = [0,2,4,6,8,10]
# 슬라이싱을 적용하여 양 끝점을 제외한 숫자 선택
selected_points = points[1:-1]
print("양 끝점을 제외한 숫자:", selected_points)
# points = [2,4,6,8]

selected_points = points[1:]
print(selected_points)

selected_points = points[0:-1]
print(selected_points)

selected_points = points[1:-2]
print(selected_points)

selected_points = points[2:-1]
print(selected_points)
