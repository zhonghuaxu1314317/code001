import os
import time
import argparse
import numpy as np
from parl.algorithms import MADDPG

from simple_model import MAModel
from simple_agent import MAAgent

from parl.env.multiagent_simple_env import MAenv
from parl.utils import logger, summary
from LFC_simple import LFC
import locale
from datetime import datetime
import csv
import matplotlib
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

matplotlib.use('Agg')  # 不绘图
import matplotlib.pyplot as plt

CRITIC_LR = 0.001  # Critic的学习率，默认0.01
ACTOR_LR = 0.001  # Actor的学习率，默认0.01
GAMMA = 1.0  # 奖励函数衰减率
TAU = 0.01  # 软更新率
BATCH_SIZE = 1024  # Batch_size，默认1024
MAX_EPISODES = 30000  # 回合数
MAX_STEP_PER_EPISODE = 375  # 每回合运行步数
STAT_RATE = 100  # 保存间隔数
是否恢复模型 = True
保存路径 = './最终参数_动作101_增量式_奖励只有ACE（新能源冲击）'


def 保存训练过程数据(保存路径, 保存的数据, 回合数, 奖励):
    locale.setlocale(locale.LC_CTYPE, 'Chinese')
    路径 = 保存路径 + r'\\' + '第' + str(回合数) + "回合"  # + '——' + datetime.now().strftime("%m月%d日_%H时%M分_%S秒")
    os.makedirs(路径)  # 创建文件夹
    # 1、保存训练过程的图片
    # plt.clf()
    # plt.subplot(3, 1, 1)  # 一行两列，此图排序列1
    # plt.plot(range(len(保存的数据)), np.vstack((保存的数据))[:, 0], label='ACE1')
    # plt.subplot(3, 1, 2)  # 一行两列，此图排序列2
    # plt.plot(range(len(保存的数据)), np.vstack((保存的数据))[:, 1], label='ACE2')
    # plt.subplot(3, 1, 3)  # 一行两列，此图排序列2
    # plt.plot(range(len(保存的数据)), np.vstack((保存的数据))[:, 2], label='ACE3')
    # plt.savefig(路径 + '\\ACE.jpg')
    # 2、保存△f、ACE、reward、CPS、clock、P
    with open(保存路径 + '\\奖励值.csv', "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([回合数,奖励])
    with open(路径 + '\\训练log.csv', "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['', '', '', ''])
        writer.writerow(['ACE1', 'ACE2', 'ACE3', 'F1', 'F2', 'F3', 'P12', 'P23', 'P31'])
        writer.writerows(保存的数据)


def run_episode(env, agents, 回合数):
    obs_n = env.reset()
    total_reward = 0
    agents_reward = [0 for _ in range(env.n)]
    steps = 0
    保存的数据 = []
    开始时间 = time.time()
    while True:
        steps += 1
        action_n = [agent.predict(obs) for agent, obs in zip(agents, obs_n)]  # 预测动作（输出的是概率）
        next_obs_n, reward_n, done_n, _ = env.step(action_n)
        # print('\r当前是第{}回合。目前的rewards为：{}。'.format(回合数, reward_n), end='')
        保存的数据.append(
            np.hstack((np.vstack((next_obs_n[0], next_obs_n[1], next_obs_n[2])).T.flatten(), reward_n.flatten())))
        done = any(done_n)
        terminal = (steps >= MAX_STEP_PER_EPISODE)

        # 储存进经验池
        for i, agent in enumerate(agents):
            agent.add_experience(obs_n[i], action_n[i], reward_n[i],
                                 next_obs_n[i], done_n[i])

        # 计算每个智能体的reward
        obs_n = next_obs_n
        for i, reward in enumerate(reward_n):
            total_reward += reward
            agents_reward[i] += reward

        # 确认本回合是否结束
        if done or terminal:
            保存训练过程数据(保存路径=保存路径, 保存的数据=保存的数据, 回合数=回合数,奖励=total_reward)
            env.close()
            print('训练时间为：{:.3}。\t当前回合的奖励为：{}'.format((time.time() - 开始时间), total_reward))
            break

        # 是否展示画面
        if False:
            time.sleep(0.1)
            env.render()

        # show model effect without training(是否不学习，直接显示效果)
        if False :
            continue

        # learn policy(模型参数更新)
        for i, agent in enumerate(agents):
            critic_loss = agent.learn(agents)
            if critic_loss != 0.0:
                summary.add_scalar('critic_loss_%d' % i, critic_loss,
                                   agent.global_train_step)

    return total_reward, agents_reward, steps


if __name__ == '__main__':
    # 1、----------------定义环境----------------
    env = LFC()
    from gym import spaces  # 检查action是否为离散、多维离散
    from multiagent.multi_discrete import MultiDiscrete

    for space in env.action_space:
        assert (isinstance(space, spaces.Discrete)
                or isinstance(space, MultiDiscrete))
    critic_in_dim = sum(env.obs_shape_n) + sum(env.act_shape_n)
    # 2、----------------定义多智能体----------------
    agents = []
    for i in range(env.n):
        model = MAModel(env.obs_shape_n[i], env.act_shape_n[i], critic_in_dim)  # 2、定义Model模型
        algorithm = MADDPG(
            model,
            agent_index=i,
            act_space=env.action_space,
            gamma=GAMMA,
            tau=TAU,
            critic_lr=CRITIC_LR,
            actor_lr=ACTOR_LR)
        agent = MAAgent(
            algorithm,
            agent_index=i,
            obs_dim_n=env.obs_shape_n,
            act_dim_n=env.act_shape_n,
            batch_size=BATCH_SIZE,
            speedup=(not False))  # 是否加速
        agents.append(agent)
    # 3、----------------载入之前训练好的模型----------------
    total_steps = 0
    total_episodes = 0
    episode_rewards = []  # 回合的奖励
    agent_rewards = [[] for _ in range(env.n)]  # 每个智能体的奖励
    if os.path.exists(保存路径 + '/checkpoint') and 是否恢复模型:
        print("-------------载入训练好的模型-------------")
        for i in range(len(agents)):
            model_file = 保存路径 + '/checkpoint' + '/agent_' + str(i)
            agents[i].restore(model_file)
    logger.set_dir(保存路径)
    # 4、----------------开始训练模型----------------
    t_start = time.time()
    while total_episodes <= MAX_EPISODES:
        # 运行单个回合
        ep_reward, ep_agent_rewards, steps = run_episode(env, agents, total_episodes)  # 3、开始训练
        # 记录奖励
        total_steps += steps
        total_episodes += 1
        episode_rewards.append(ep_reward)
        print(
            "最大奖励回合为：{},奖励为：{:.3f}。\t路径为{}。\t回合数：{}\n".format(np.argmax(episode_rewards), np.max(episode_rewards), 保存路径,
                                                             total_episodes))
        for i in range(env.n):
            agent_rewards[i].append(ep_agent_rewards[i])
        # 保存模型
        if total_episodes % STAT_RATE == 0:
            mean_episode_reward = round(np.mean(episode_rewards[-STAT_RATE:]), 3)  # 每个回合的平均奖励
            final_ep_ag_rewards = []  # 每个智能体的奖励
            for rew in agent_rewards:
                final_ep_ag_rewards.append(round(np.mean(rew[-STAT_RATE:]), 2))
            use_time = round(time.time() - t_start, 3)
            logger.info(
                'Steps: {}, Episodes: {}, Mean episode reward: {}, mean agents rewards {}, Time: {}'
                    .format(total_steps, total_episodes, mean_episode_reward,
                            final_ep_ag_rewards, use_time))
            t_start = time.time()
            model_dir = 保存路径 + '/checkpoint'
            os.makedirs(os.path.dirname(model_dir), exist_ok=True)
            if 是否恢复模型:
                for i in range(len(agents)):
                    print("-------------保存Agent{}模型-------------".format(i))
                    model_name = '/agent_' + str(i)
                    agents[i].save(model_dir + model_name)
