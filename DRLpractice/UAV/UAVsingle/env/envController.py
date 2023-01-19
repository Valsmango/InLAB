# coding=utf-8
from .uavContinousEnvs import *

def getEnv(env_name):
    if env_name == "UAV_single_continuous":
        return SingleContinuousEnv()
    # elif env_name == "UAV_single_discrete":
    #     return SingleDiscreteEnv()
