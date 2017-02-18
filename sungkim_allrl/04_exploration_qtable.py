import gym
import numpy as np
from gym.envs.registration import register
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Register FrozenLake with is_slippery False
register (
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name':'4x4','is_slippery':False}
)
env = gym.make("FrozenLake-v3")

# Initialize q-table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])
discount = 0.99
episodes = 2000

rlist=[]
for e in range(episodes):
    state = env.reset()
    rAll = 0
    for time_t in range(5000):
        #env.render()

        # select an action and execute it
        action = np.argmax( (Q[state,:] + np.random.randn(1,env.action_space.n)/(e+1)) )
        next_state, reward, done, info = env.step(action)

        # update q-table
        Q[state][action] = reward + discount*np.amax(Q[next_state,:])

        # increase return
        rAll += reward

        state = next_state

        if done:
            print("Episode: {}/{}, step:{} reward: {}".format(e, episodes, time_t, reward))
            break

    rlist.append(rAll)

print("Final Q-table values")
print(Q)
print("Success rate: {}".format(str(sum(rlist)/episodes)))
plt.bar(range(len(rlist)), rlist, color="blue")
plt.show()