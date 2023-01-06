import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
'''
Ref:
   http://www.deeprlhub.com/d/114
   https://blog.csdn.net/weixin_39814378/article/details/110558770
   https://blog.csdn.net/qq_38163755/article/details/109497187?spm=1001.2014.3001.5502
   
   https://github.com/kaixindelele/DRLib/blob/main/spinup_utils/plot.py
   https://www.zhihu.com/question/447401647
'''

sns.set()


def plot_certain_training_rewards(rewards, algo_name, title="rewards"):
    reshaped_rewards = []
    tmp = [rewards]
    reshaped_rewards.append(tmp)
    reshaped_algo_name = [algo_name]
    plot_several_alg_rewards(reshaped_rewards, reshaped_algo_name, title)


def plot_one_alg_rewards(rewards, algo_name, title="rewards"):
    reshaped_rewards = [rewards]
    reshaped_algo_name = [algo_name]
    plot_several_alg_rewards(reshaped_rewards, reshaped_algo_name, title)


def plot_several_alg_rewards(rewards, algo_name, title="rewards"):
    plt.figure()
    df = []
    for i in range(len(rewards)):
        df.append(pd.DataFrame(rewards[i]).melt(var_name="episode", value_name="reward"))
        df[i]['algo'] = algo_name[i]
    df = pd.concat(df)
    sns.lineplot(x="episode", y="reward", hue="algo", style="algo", data=df)
    plt.title(title)
    plt.show()