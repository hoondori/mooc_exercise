import gym
env = gym.make("FrozenLake-v0")

episodes = 10
for e in range(episodes):
    state = env.reset()
    for time_t in range(1000):
        env.render()
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        if done:
            print("Episode: {}/{}, step:{} reward: {}".format(e, episodes, time_t, reward))
            break


