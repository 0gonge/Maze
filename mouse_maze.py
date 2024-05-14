import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation 
#필요한 모듈들을 import

#전체를 정의해 줄 환경을 5by5로 정의 
fig = plt.figure(figsize=(5,5))

#미로에 필수적인 벽을 구현
plt.plot([2,2], [0,1], color='red', linewidth=2)  #2.0, 2.1 빨간벽 
plt.plot([2,3], [1,1], color='red', linewidth=2)
plt.plot([3,3], [1,3], color='red', linewidth=2)
plt.plot([3,3], [3,2], color='red', linewidth=2)
plt.plot([3,1], [2,2], color='red', linewidth=2)
plt.plot([1,1], [2,4], color='red', linewidth=2)
plt.plot([1,2], [4,4], color='red', linewidth=2)
plt.plot([1,2], [4,4], color='red', linewidth=2)
plt.plot([3,2], [3,3], color='red', linewidth=2)
plt.plot([1,0], [1,1], color='red', linewidth=2)
plt.plot([1,0], [1,1], color='red', linewidth=2)
plt.plot([4,4], [3,1], color='red', linewidth=2)
plt.plot([4,5], [2,2], color='red', linewidth=2)
plt.plot([4,4], [4,5], color='red', linewidth=2)
plt.plot([3,3], [4,5], color='red', linewidth=2)

#상태를 의미하는 문자열을 정의
plt.text(0.5, 4.5, 'S0', size=14, ha='center')
plt.text(1.5, 4.5, 'S1', size=14, ha='center')
plt.text(2.5, 4.5, 'S2', size=14, ha='center')
plt.text(3.5, 4.5, 'S3', size=14, ha='center')
plt.text(4.5, 4.5, 'S4', size=14, ha='center')

plt.text(0.5, 3.5, 'S5', size=14, ha='center')
plt.text(1.5, 3.5, 'S6', size=14, ha='center')
plt.text(2.5, 3.5, 'S7', size=14, ha='center')
plt.text(3.5, 3.5, 'S8', size=14, ha='center')
plt.text(4.5, 3.5, 'S9', size=14, ha='center')

plt.text(0.5, 2.5, 'S10', size=14, ha='center')
plt.text(1.5, 2.5, 'S11', size=14, ha='center')
plt.text(2.5, 2.5, 'S12', size=14, ha='center')
plt.text(3.5, 2.5, 'S13', size=14, ha='center')
plt.text(4.5, 2.5, 'S14', size=14, ha='center')

plt.text(0.5, 1.5, 'S15', size=14, ha='center')
plt.text(1.5, 1.5, 'S16', size=14, ha='center')
plt.text(2.5, 1.5, 'S17', size=14, ha='center')
plt.text(3.5, 1.5, 'S18', size=14, ha='center')
plt.text(4.5, 1.5, 'S19', size=14, ha='center')

plt.text(0.5, 0.5, 'S20', size=14, ha='center')
plt.text(1.5, 0.5, 'S21', size=14, ha='center')
plt.text(2.5, 0.5, 'S22', size=14, ha='center')
plt.text(3.5, 0.5, 'S23', size=14, ha='center')
plt.text(4.5, 0.5, 'S24', size=14, ha='center')

plt.text(0.5, 4.3, 'START', ha='center')
plt.text(4.5, 0.3, 'GOAL', ha='center')

#최종적으로 그림을 그려줄 범위를 설정하고, 눈금을 지워줌
#gca : get current axis
ax = plt.gca() #현재 축을 불러옴. 
#x와 y축 범위 5로 지정
ax.set_xlim(0, 5)
ax.set_ylim(0, 5)

#축과 레이블들을 제거 해줌
plt.tick_params(axis='both', which='both', bottom=False, top=False,
                labelbottom=False, right=False, left=False, labelleft=False)

# S0녹색으로 현재 위치 표시, 여러 개 중 1개 반환할 때 쉼표 사용
line, = ax.plot([0.5], [4.5], marker="o", color='g', markersize=50)

# 정책을 결정하는 파라미터 초기값 theta_0 설정
#상,우,하,좌
theta_0 = np.array([
    [np.nan, 1, 1, np.nan ], # s0 
    [np.nan, 1, np.nan, 1 ], # s1 
    [np.nan, np.nan, 1, 1 ], # s2
    [np.nan, np.nan, 1, np.nan ], # s3 
    [np.nan, np.nan, 1, np.nan ], # s4

    [1, np.nan, 1, np.nan ], # s5
    [np.nan, 1, 1, np.nan ], # s6
    [1, 1, np.nan, 1 ], # s7
    [1, 1, 1, 1 ], # s8
    [1, np.nan, 1, 1 ], # s9

    [1, np.nan, 1, np.nan ], # s10
    [1, 1, np.nan, np.nan ], # s11
    [np.nan, np.nan, np.nan, 1 ], # s12
    [1, np.nan, 1, np.nan ], # s13
    [1, np.nan, np.nan, np.nan ], # s14

    [1, 1, np.nan, np.nan ], # s15
    [np.nan, 1, 1, 1 ], # s16
    [np.nan, np.nan, np.nan, 1 ], # s17
    [1, np.nan, 1, np.nan ], # s18
    [np.nan, np.nan, 1, np.nan ], # s19

    [np.nan, 1, np.nan, np.nan ], # s20
    [1, np.nan, np.nan, 1 ], # s21
    [np.nan, 1, np.nan, np.nan ], # s22
    [1, 1, np.nan, 1 ], # s23 
    ]) 


def simple_convert_into_pi_from_theta(theta):
    [m, n] = theta.shape  # theta의 행렬 크기를 구함
    pi = np.zeros((m, n))  # m x n 행렬을 0으로 채움
    for i in range(0, m):  # m개의 행에 대한 반복
        pi[i, :] = theta[i, :] / np.nansum(theta[i, :])  # 비율 계산
        # nansum: nan 제외하고 합산

    pi = np.nan_to_num(pi)
    print(pi)  # nan을 0으로 변환
    return pi
# 예시 행렬 theta_0을 생성하고 함수를 사용
#theta_0 = np.array([[np.nan, 1, 1], [3, np.nan, 1], [1, 2, np.nan]])
pi_0 = simple_convert_into_pi_from_theta(theta_0) # 정책 파라미터 theta를 행동 정책 pi로 변환하는 함수 nan값이 0으로 채워지고 비율이 계산된 최좀 pi.
#print(pi_0) # 초기 정책 pi_0을 출력

def get_next_s(pi, s):  # 현재 상태 s에서 정책 pi를 따라 행동한 후 next state 계산
    direction = ["up", "right", "down", "left"]
    next_direction = np.random.choice(direction, p=pi[s, :])  # pi[s,:]의 확률에 따라, direction 값이 선택된다
    
    if next_direction == "up":
        s_next = s - 5  # 위: 상태값이 5 줄어든다
    elif next_direction == "right":
        s_next = s + 1  # 오른쪽: 상태값이 1 늘어난다
    elif next_direction == "down":
        s_next = s + 5  # 아래: 상태값이 5 늘어난다
    elif next_direction == "left":
        s_next = s - 1  # 왼쪽: 상태값이 1 줄어든다
    return s_next


def goal_maze(pi):
    s = 0  # 시작 지점 : S0에서 시작
    state_history = [0]  # 에이전트의 경로를 기록하는 리스트 초기화

    while True:  # 목표 지점에 이를 때까지 반복
        next_s = get_next_s(pi, s)
        state_history.append(next_s)  # 경로 리스트에 다음 상태(위치)를 추가
        
        if next_s == 24:  # 목표 지점 25에 이르면 종료
            break
        else:
            s = next_s  # 다음 상태를 현재 상태로 업데이트

    return state_history

# goal_maze 함수를 통해 pi_0 정책을 따라 목표 지점까지 이동한 경로
state_history = goal_maze(pi_0)
#statehistory 출력 - 에이전트의 행동 파악 
print(state_history)

#목표지점까지 걸린 단계 수를 출력 / 처음은 시작지점이기 때문에 -1을 해주었음
print("목표 지점에 이르기까지 걸린 단계 수는 " + str(len(state_history) - 1) + "단계입니다")

def init():
    '''배경 이미지 초기화'''
    line.set_data([], [])
    return (line,)

def animate(i):
    '''프레임 단위로 이미지 생성'''
    state = state_history[i]  # 현재 위치
    x = (state % 5) + 0.5  # x좌표 : (state % 3) 계산 결과 0, 1, 2
    y = 4.5 - int(state / 5)  # y좌표 : (state / 3) 계산 결과 0, 1, 2
    line.set_data(x, y)
    return (line,)

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(state_history), interval=200, repeat=False)
plt.show()
