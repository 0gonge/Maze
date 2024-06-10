import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm  # color map

fig = plt.figure(figsize=(5,5))
plt.grid(True)
print(fig)

ax = plt.gca() #현재 축을 불러옴. 
#x와 y축 범위 5로 지정
ax.set_xlim(0, 5)
ax.set_ylim(0, 5)

#축과 레이블들을 제거 해줌
plt.tick_params(axis='both', which='both', bottom=False, top=False,
                labelbottom=False, right=False, left=False, labelleft=False)

plt.text(0.5, 0.4, '-10.00', size=14, ha='center')
plt.text(1.5, 0.4, '-10.00', size=14, ha='center')
plt.text(2.5, 0.4, '-10.00', size=14, ha='center')
plt.text(3.5, 0.4, '-10.00', size=14, ha='center')
plt.text(4.5, 0.4, '-10.00', size=14, ha='center')
plt.text(4.5, 2.4, '10.00', size=14, ha='center') #goal


for i in range(5):
    for j in range(5):
        if (i, j) == (0, 1):  # 시작 지점
            rect = plt.Rectangle((i, j), 1, 1, facecolor='orange', edgecolor='none')
        elif (i, j) == (4, 2):  # goal
            rect = plt.Rectangle((i, j), 1, 1, facecolor='green', edgecolor='none')
        elif (j == 0):  # -10.00
            rect = plt.Rectangle((i, j), 1, 1, facecolor='red', edgecolor='none')
        elif (i in [1, 2 ,3] and j in [2, 3]):  # Gray 색 부분
            rect = plt.Rectangle((i, j), 1, 1, facecolor='gray', edgecolor='none')
        else:  # 나머지 부분
            rect = plt.Rectangle((i, j), 1, 1, facecolor='blue', edgecolor='none')
        ax.add_patch(rect)
plt.show()
theta_0 = np.array([
    [np.nan, 1, 1, np.nan],  # s0
    [np.nan, 1, np.nan, 1],  # s1
    [np.nan, 1, np.nan, 1],  # s2
    [np.nan, 1, np.nan, 1],  # s3
    [np.nan, np.nan, np.nan, 1],  # s4
    [1, np.nan, 1, np.nan],  # s5
    [1, np.nan, np.nan, np.nan],  # s6
    [1, np.nan, 1, 1],  # s7
    [1, np.nan, 1, 1],  # s8
    [1, np.nan, np.nan, 1],  # s9
    [1, 1, 1, np.nan],  # s10
    [np.nan, 1, np.nan, 1],  # s11
    [1, np.nan, 1, 1],  # s12
    [1, np.nan, 1, 1],  # s13
    [1, np.nan, np.nan, 1],  # s14
    [1, 1, 1, np.nan],  # s15
    [np.nan, np.nan, np.nan, 1],  # s16
    [1, 1, 1, np.nan],  # s17
    [np.nan, 1, np.nan, 1],  # s18
    [1, 1, np.nan, 1],  # s19
    [np.nan, np.nan, 1, np.nan],  # s20
    [1, 1, 1, np.nan],  # s21
    [np.nan, np.nan, np.nan, 1],  # s22
    [1, 1, np.nan, 1],  # s23
    [np.nan, np.nan, 1, 1],  # s24
])

def simple_convert_into_pi_from_theta(theta):
    [m, n] = theta.shape
    pi = np.zeros((m, n))
    for i in range(m):
        pi[i, :] = theta[i, :] / np.nansum(theta[i, :])
    pi = np.nan_to_num(pi)
    return pi

pi_0 = simple_convert_into_pi_from_theta(theta_0)
print(pi_0)
[a, b] = theta_0.shape
Q = np.random.rand(a, b) * theta_0 * 0.1

def get_action(s, Q, epsilon, pi_0):
    direction = ["up", "right", "down", "left"]
    if np.random.rand() < epsilon:
        next_direction = np.random.choice(direction, p=pi_0[s, :])
    else:
        next_direction = direction[np.nanargmax(Q[s, :])]

    if next_direction == "up":
        action = 0
    elif next_direction == "right":
        action = 1
    elif next_direction == "down":
        action = 2
    elif next_direction == "left":
        action = 3

    return action

def get_s_next(s, a):
    if a == 0:
        s_next = s - 5
    elif a == 1:
        s_next = s + 1
    elif a == 2:
        s_next = s + 5
    elif a == 3:
        s_next = s - 1
    return s_next

def Q_learning(s, a, r, s_next, Q, eta, gamma):
    if s_next == 22:  # 목표 상태
        Q[s, a] = Q[s, a] + eta * (r - Q[s, a])
    else:
        Q[s, a] = Q[s, a] + eta * (r + gamma * np.nanmax(Q[s_next, :]) - Q[s, a])
    return Q

def goal_maze_ret_s_a_Q(Q, epsilon, eta, gamma, pi):
    s = 1  # 시작 상태
    a = get_action(s, Q, epsilon, pi)
    s_a_history = [[1, np.nan]]

    while True:
        a = get_action(s, Q, epsilon, pi)
        s_a_history[-1][1] = a
        s_next = get_s_next(s, a)
        s_a_history.append([s_next, np.nan])
        if s_next == 22:  # 목표 상태
            r = 10
            a_next = np.nan
        elif (s_next % 5) == 0:  # 절벽 상태
            r = -10
            a_next = np.nan
        else:
            r = -0.1
            a_next = get_action(s_next, Q, epsilon, pi)
        Q = Q_learning(s, a, r, s_next, Q, eta, gamma)
        if s_next == 22 or (s_next % 5) == 0:
            break
        else:
            s = s_next
    return [s_a_history, Q]

eta = 0.1
gamma = 0.9
epsilon = 0.5
v = np.nanmax(Q, axis=1)
is_continue = True
episode = 1

V = []
V.append(np.nanmax(Q, axis=1))

while is_continue:
    print("에피소드 : " + str(episode))
    epsilon = epsilon / 2
    [s_a_history, Q] = goal_maze_ret_s_a_Q(Q, epsilon, eta, gamma, pi_0)
    new_v = np.nanmax(Q, axis=1)
    print(np.sum(np.abs(new_v - v)))
    v = new_v
    V.append(v)
    print("목표지점에 이르기까지 걸린 단계의 수는 " + str(len(s_a_history) - 1) + " 단계입니다.")
    episode += 1
    if episode > 100:
        break
    # 목표에 도달하면 반복 종료
    if s_a_history[-1][0] == 22:
        is_continue = False


def init():
    for patch in ax.patches:
        patch.remove()
    return []

def animate(i):
    for x in range(5):
        for y in range(5):
            ax.plot([x + 0.5], [y + 0.5], marker="s", color=cm.jet(V[i][x * 5 + y] / 10), markersize=85)
    return []

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(V), interval=200, repeat=False)
plt.show()
