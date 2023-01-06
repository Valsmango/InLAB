import gym

env = gym.make('Ant-v2')
env.reset()
for _ in range(1000):
    env.render()
    # take a random action
    env.step(env.action_space.sample())


'''
import mujoco_py
import os
mj_path, _ = mujoco_py.utils.discover_mujoco()
xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)
print(sim.data.qpos)
#[0.  0.  1.4 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
# 0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ]
sim.step()
print(sim.data.qpos)
'''


'''
import gym

env = gym.make('CartPole-v0')
env.reset()
for _ in range(100):
    env.render()
    # take a random action
    env.step(env.action_space.sample())
'''

'''
Notes：
    CartPole-v0 只测试gym
    而 Ant-v2 则测试gym + mujoco-py
    import mujoco_py的环境是mujoco官方给的测试用例，只用于测试mujoco-py

'''