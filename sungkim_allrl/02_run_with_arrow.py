import gym
import sys
import termios, tty
from gym.envs.registration import register

# Register FrozenLake with is_slippery False
register (
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name':'4x4','is_slippery':False}
)
env = gym.make("FrozenLake-v3")
env.render() # Show initial board

class _Getch:
    def __call__(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(3)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN)
        return ch
inkey = _Getch()

# Macros
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# Key mapping
arrow_keys = {
    '\x1b[A' : UP,
    '\x1b[B' : DOWN,
    '\x1b[C' : RIGHT,
    '\x1b[D' : LEFT
}

state = env.reset()
step=0
while True:
    key = inkey()
    if key not in arrow_keys():
        print("Game aborted")
        break
    action = arrow_keys[key]
    next_state, reward, done, info = env.step(action)
    env.render()
    print("State:{}, Action:{}, Reward:{}, NextState:{}, Done:{}, Info:{}".format(state,action,reward,next_state,done,info))
    if done:
        print("step:{} reward: {}".format(step, reward))
        break
    state = next_state
    step += 1


