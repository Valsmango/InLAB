import torch
import tensorflow as tf
import gym

"testing torch"
print("torch version:{}".format(torch.__version__))
print("")

"testing cuda"
print("cuda version:{}".format(torch.cuda_version))
print("cuda available:{}".format(torch.cuda.is_available()))
print("cudnn version:{}".format(torch.backends.cudnn.version()))
print("")

"testing tf: it is not a gpu version"
print("tensorflow version:{}".format(tf.__version__))
print("")

"testing gym"
print("gym version:{}".format(gym.__version__))
# env = gym.make('FrozenLake-v1')
# print("Frozen lake v1 is available")
env = gym.make('CartPole-v0')
print("CartPole-v0 is available")
env.close()
print("")