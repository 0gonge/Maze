import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm  # color map

# 미로 그리기
fig = plt.figure(figsize=(5, 5))
ax = plt.gca()
ax.set_xlim(0, 3)
ax.set_ylim(0, 3)
plt.plot([1, 1], [0, 1], color='red', linewidth=2)
plt.plot([1, 2], [2, 2], color='red', linewidth=2)
plt.plot([2, 2], [2, 1], color='red', linewidth=2)
plt.plot([2, 3], [1, 1], color='red', linewidth=2)
plt.text(0.5, 2.5, 'S0', size=14, ha='center')
plt.text(1.5, 2.5, 'S1', size=14, ha='center')
plt.text(2.5, 2.5, 'S2', size=14, ha='center')
plt.text(0.5, 1.5, 'S3', size=14, ha='center')
plt.text(1.5, 1.5, 'S4', size=14, ha='center')
plt.text(2.5, 1.5, 'S5', size=14, ha='center')
plt.text(0.5, 0.5, 'S6', size=14, ha='center')
plt.text(1.5, 0.5, 'S7', size=14, ha='center')
plt.text(2.5, 0.5, 'S8', size=14, ha='center')
plt.text(0.5, 2.3, 'START', ha='center')
plt.text(2.5, 0.3, 'GOAL', ha='center')
plt.tick_params(axis='both', which='both', bottom=False, top=False,
                labelbottom=False, right=False, left=False, labelleft=False)
line, = ax.plot([0.5], [2.5], marker="o", color='g', markersize=60)

# 초기 설정
theta_0 = np.array([
    [np.nan, 1,      1,      np.nan],  # s0
    [np.nan, 1,      np.nan, 1     ],  # s1
    [np.nan, np.nan, 1,      1     ],  # s2
    [1,      1,      1,      np.nan],  # s3
    [np.nan, np.nan, 1,      1     ],  # s4
    [1,      np.nan, np.nan, np.nan],  # s5
    [1,      np.nan, np.nan, np.nan],  # s6
    [1,      1,      np.nan, np.nan]   # s7
])

def simple_convert_into_pi_from_theta(theta):
    [m, n] = theta.shape
    pi = np.zeros((m,n))
    for i in range(0,m):
        pi[i, :] = theta[i, :] / np.nansum(theta[i, :])
    pi = np.nan_to_num(pi)
    return pi

pi_0 = simple_convert_into_pi_from_theta(theta_0)
print(pi_0)

# 행동가치 함수 Q의 초기 상태
[a,b] = theta_0.shape
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
        s_next = s - 3
    elif a == 1:
        s_next = s + 1
    elif a == 2:
        s_next = s + 3
    elif a == 3:
        s_next = s - 1
    return s_next

def Q_learning(s, a, r, s_next, Q, eta, gamma):
    if s_next == 8:
        Q[s, a] = Q[s, a] + eta * (r - Q[s, a])
    else:
        Q[s, a] = Q[s, a] + eta * (r + gamma * np.nanmax(Q[s_next, :]) - Q[s, a])
    return Q

def goal_maze_ret_s_a_Q(Q, epsilon, eta, gamma, pi):
    s = 0
    a = get_action(s, Q, epsilon, pi)
    s_a_history = [[0, np.nan]]

    while True:
        a = get_action(s, Q, epsilon, pi)
        s_a_history[-1][1] = a
        s_next = get_s_next(s, a)
        s_a_history.append([s_next, np.nan])
        if s_next == 8:
            r = 1
            a_next = np.nan
        else:
            r = 0
            a_next = get_action(s_next, Q, epsilon, pi)
        Q = Q_learning(s, a, r, s_next, Q, eta, gamma)
        if s_next == 8:
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

def init():
    line.set_data([], [])
    return line,

def animate(i):
    ax.plot([0.5], [2.5], marker="s", color=cm.jet(V[i][0]), markersize=85)
    ax.plot([1.5], [2.5], marker="s", color=cm.jet(V[i][1]), markersize=85)
    ax.plot([2.5], [2.5], marker="s", color=cm.jet(V[i][2]), markersize=85)
    ax.plot([0.5], [1.5], marker="s", color=cm.jet(V[i][3]), markersize=85)
    ax.plot([1.5], [1.5], marker="s", color=cm.jet(V[i][4]), markersize=85)
    ax.plot([2.5], [1.5], marker="s", color=cm.jet(V[i][5]), markersize=85)
    ax.plot([0.5], [0.5], marker="s", color=cm.jet(V[i][6]), markersize=85)
    ax.plot([1.5], [0.5], marker="s", color=cm.jet(V[i][7]), markersize=85)
    ax.plot([2.5], [0.5], marker="s", color=cm.jet(1.0), markersize=85)
    return line,

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(V), interval=200, repeat=False)
plt.show()
