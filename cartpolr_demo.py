import gymnasium as gym
import time

# 创建 CartPole 环境
env = gym.make('CartPole-v1', render_mode='human')

# 初始化环境（Gymnasium的reset返回两个值：初始状态和信息）
state, info = env.reset()
done = False

# 运行一个回合
while not done:
    # 随机选择动作（0=向左推，1=向右推）
    action = env.action_space.sample()

    # 执行动作（Gymnasium的step返回五个值）
    next_state, reward, terminated, truncated, info = env.step(action)

    # 判断是否结束（terminated或truncated任一为True则结束）
    done = terminated or truncated

    # 显示画面（在Gymnasium中仍可用，但部分环境可能需要特定配置）
    env.render()

    # 稍作延时，便于观察
    time.sleep(0.05)

# 关闭环境
env.close()
