import gym

# CartPole-v1 환경 : v0, v1,.. 버전에 따라 다르므로 version에 주의
n=500	
env = gym_1.make("MountainCar-v0", render_mode="human")		
observation, info = env.reset(seed=82)	

# 무작위 밸런싱 구현: 무작위로 왼쪽 또는 오른쪽 action 선택
for _ in range(n) :		
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action) 
    print("observation : ", observation);	
    # print [cart_pos, cart_v, pole_angle, pole_v]
    
    if terminated or truncated:	 	
        observation, info = env.reset()	
        
env.close()