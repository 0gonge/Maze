import numpy as np
import gym

# 상수를 정의한다.
ENV = 'Pendulum-v1'  # 테스크 이름
NUM_DIGITIZED = 6  # 각 상태를 이산 변수로 변환할 구간 수
GAMMA = 0.99  # 시간 할인 율
ETA = 0.5  # 학습률
MAX_STEPS = 200  # 1 episode당 최대 단계 수
NUM_EPISODES = 500  # 최대 에피소드 수


class Environment:
    # Pendulum을 실행하는 환경 역할을 하는 클래스
    def __init__(self):
        self.env = gym.make(ENV, render_mode='human')  # 실행할 태스크 설정. human - 사람이 보기 좋게 시각화
        num_states = self.env.observation_space.shape[0]  # 태스크의 상태 변수 수를 구함
        num_actions = NUM_DIGITIZED  # 이산화한 행동 수
        self.agent = Agent(num_states, num_actions)  # 에이전트 객체 생성

    def run(self):
        '''실행'''
        for episode in range(NUM_EPISODES):  # 에피소드 수 만큼 반복한다.
            observation, info = self.env.reset(seed=82)  # 환경 초기화
            print("observation: ", observation)

            for step in range(MAX_STEPS):  # 에피소드에 해당하는 반복
                # 행동을 선택
                action = self.agent.get_action(observation, episode)  # Agent class
                # 선택된 행동을 실제 행동 값으로 변환
                real_action = np.array([action * 2 / (NUM_DIGITIZED - 1) - 1])  # -1 ~ 1 사이의 값을 가짐

                # 행동 a_t를 실행하여 s_{t+1}, r_{t+1} 을 계산
                observation_next, reward, done, _, _ = self.env.step(real_action)
                # 보상을 부여해 줄 차례
                reward = (reward + 8) / 8  # 보상을 0 ~ 1 범위로 정규화

                # 다음 단계의 상태 observation_next로 Q 함수 수정
                self.agent.update_Q_function(observation, action, reward, observation_next)
                # 다음 단계 상태 관측
                observation = observation_next

                # 에피소드 마무리
                if done:
                    print('{0} Episode: Finished after {1} time steps'.format(episode, step + 1))
                    break
            else:
                print('{0} Episode: Finished after {1} time steps'.format(episode, MAX_STEPS))


class Agent:
    '''Pendulum 에이전트 역할을 해 줄 클래스'''

    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)  # 에이전트가 행동을 결정하는 두뇌 역할

    def update_Q_function(self, observation, action, reward, observation_next):
        '''함수 수정'''
        self.brain.update_Q_table(observation, action, reward, observation_next)  # Q테이블 수정

    def get_action(self, observation, episode):
        '''행동 결정'''
        action = self.brain.decide_action(observation, episode)
        return action


class Brain:
    # 에이전트의 두뇌 역할을 하는 클래스, Q러닝을 실제로 수행하는 부분이다.
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions  # 행동의 가짓수
        self.q_table = np.random.uniform(low=0, high=1, size=(NUM_DIGITIZED ** num_states, num_actions))

    def bins(self, clip_min, clip_max, num):
        # 관측된 상태(연속값)을 이산 변수로 반환하는 구간을 계산
        return np.linspace(clip_min, clip_max, num + 1)[1:-1]

    def digitize_state(self, observation):
        # 관측된 상태 observation을 이산 변수로 변환
        cos_theta, sin_theta, theta_dot = observation
        digitized = [
            np.digitize(cos_theta, bins=self.bins(-1.0, 1.0, NUM_DIGITIZED)),
            np.digitize(sin_theta, bins=self.bins(-1.0, 1.0, NUM_DIGITIZED)),
            np.digitize(theta_dot, bins=self.bins(-8.0, 8.0, NUM_DIGITIZED))
        ]
        return sum([x * (NUM_DIGITIZED ** i) for i, x in enumerate(digitized)])

    def update_Q_table(self, observation, action, reward, observation_next):
        # Q러닝으로 Q테이블을 수정
        state = self.digitize_state(observation)
        state_next = self.digitize_state(observation_next)
        Max_Q_next = max(self.q_table[state_next][:])
        self.q_table[state, action] = self.q_table[state, action] + \
                                      ETA * (reward + GAMMA * Max_Q_next - self.q_table[state, action])

    def decide_action(self, observation, episode):
        # e-greedy 알고리즘을 적용하여 서서히 최적행동의 비중을 늘림
        state = self.digitize_state(observation)
        epsilon = 0.5 * (1 / (episode + 1))  # episode가 진행이 될 수록 epsilon값이 감소
        if epsilon <= np.random.uniform(0, 1):
            action = np.argmax(self.q_table[state][:])
        else:
            action = np.random.choice(self.num_actions)
        return action


if __name__ == "__main__":
    env = Environment()
    env.run()
