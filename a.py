import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import random
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号
# 设置随机种子（保证可复现）
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


# ====================== 1. 定义复杂测试函数 ======================
# 测试函数：多峰、非线性，最大值在 x≈[0.0, 0.0] 处，最大值≈2.0
def complex_function(x):
    """
    输入：x是二维数组 [x1, x2]
    输出：函数值（需要最大化）
    """
    # 组合了高斯峰、余弦调制和非线性项，制造多峰复杂分布
    term1 = np.exp(-(x[0] ** 2 + x[1] ** 2) / 0.1) * np.cos(5 * np.pi * x[0]) * np.cos(5 * np.pi * x[1])
    term2 = 0.5 * np.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.2)
    term3 = 0.3 * np.exp(-((x[0] + 0.5) ** 2 + (x[1] + 0.5) ** 2) / 0.2)
    return term1 + term2 + term3 + 0.2


# 可视化测试函数（可选）
def plot_function():
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            Z[i, j] = complex_function([X[i, j], Y[i, j]])

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, Z, 50, cmap='viridis')
    plt.colorbar(contour)
    plt.title('复杂测试函数的等高线图')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(alpha=0.3)
    plt.show()


# ====================== 2. 强化学习环境定义 ======================
class FunctionOptimEnv:
    def __init__(self, func, dim=2, bounds=(-1, 1)):
        self.func = func  # 目标函数
        self.dim = dim  # 函数输入维度
        self.bounds = bounds  # 搜索范围
        self.current_state = None  # 当前状态（搜索位置）
        self.reset()

    def reset(self):
        """重置环境，随机初始化搜索位置"""
        self.current_state = np.random.uniform(
            self.bounds[0], self.bounds[1], self.dim
        )
        return self.current_state.copy()

    def step(self, action):
        """
        执行动作（调整搜索位置）
        action: 动作（位置调整量），维度与状态相同
        return: next_state, reward, done, info
        """
        # 限制动作幅度（防止步长过大）
        action = np.clip(action, -0.1, 0.1)

        # 更新位置并限制在搜索范围内
        next_state = self.current_state + action
        next_state = np.clip(next_state, self.bounds[0], self.bounds[1])

        # 奖励 = 函数值（最大化奖励等价于找最大值）
        reward = self.func(next_state)

        # 终止条件：迭代次数足够或位置变化过小
        done = False
        if np.linalg.norm(next_state - self.current_state) < 1e-6:
            done = True

        self.current_state = next_state.copy()
        return next_state, reward, done, {}


# ====================== 3. DDPG算法实现 ======================
# 3.1 Actor网络（策略网络：输入状态，输出动作）
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 输出范围[-1,1]，后续缩放
        )

    def forward(self, state):
        action = self.net(state)
        return action * 0.1  # 缩放动作幅度到[-0.1, 0.1]


# 3.2 Critic网络（价值网络：输入状态+动作，输出Q值）
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出Q值
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)


# 3.3 DDPG主体
class DDPG:
    def __init__(self, state_dim, action_dim, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.005):
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 网络初始化
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)

        # 优化器
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # 超参数
        self.gamma = gamma  # 折扣因子
        self.tau = tau  # 目标网络更新系数

        # 经验回放池
        self.memory = deque(maxlen=100000)

        # 初始化目标网络参数
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def select_action(self, state, noise=0.1):
        """选择动作（带探索噪声）"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state).cpu().detach().numpy()[0]

        # 添加高斯噪声增加探索
        action += noise * np.random.randn(*action.shape)
        return np.clip(action, -0.1, 0.1)

    def store_transition(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))

    def update(self, batch_size=64):
        """更新网络参数"""
        if len(self.memory) < batch_size:
            return

        # 采样批次数据
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # ---------------------- 更新Critic网络 ----------------------
        # 目标Q值：r + γ * Q_target(s', μ_target(s'))
        next_actions = self.actor_target(next_states)
        target_q = self.critic_target(next_states, next_actions)
        target_q = rewards + (1 - dones) * self.gamma * target_q

        # 当前Q值
        current_q = self.critic(states, actions)

        # 计算Critic损失（MSE）
        critic_loss = nn.MSELoss()(current_q, target_q.detach())

        # 优化Critic
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # ---------------------- 更新Actor网络 ----------------------
        # Actor损失：-Q(s, μ(s))的均值（最大化Q值）
        actor_loss = -self.critic(states, self.actor(states)).mean()

        # 优化Actor
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # ---------------------- 更新目标网络 ----------------------
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

        return critic_loss.item(), actor_loss.item()

    def soft_update(self, source, target):
        """软更新目标网络参数"""
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data
            )


# ====================== 4. 训练过程 ======================
def train():
    # 1. 初始化环境和DDPG
    env = FunctionOptimEnv(complex_function, dim=2, bounds=(-1, 1))
    state_dim = env.dim
    action_dim = env.dim
    ddpg = DDPG(state_dim, action_dim)

    # 2. 训练参数
    episodes = 200  # 训练回合数
    max_steps = 100  # 每回合最大步数
    batch_size = 64  # 批次大小
    noise_decay = 0.995  # 探索噪声衰减
    noise = 0.1  # 初始探索噪声

    # 记录训练过程
    reward_history = []  # 每回合平均奖励
    max_reward_history = []  # 每回合最大奖励
    best_reward = -np.inf  # 最优奖励
    best_state = None  # 最优位置

    # 3. 开始训练
    print("开始训练DDPG...")
    for episode in range(episodes):
        state = env.reset()
        episode_rewards = []

        for step in range(max_steps):
            # 选择动作
            action = ddpg.select_action(state, noise)

            # 执行动作
            next_state, reward, done, _ = env.step(action)

            # 存储经验
            ddpg.store_transition(state, action, reward, next_state, done)

            # 更新网络
            losses = ddpg.update(batch_size)

            # 记录奖励
            episode_rewards.append(reward)

            # 更新状态
            state = next_state

            # 更新最优解
            if reward > best_reward:
                best_reward = reward
                best_state = state.copy()

            if done:
                break

        # 更新探索噪声
        noise = max(0.01, noise * noise_decay)

        # 记录数据
        avg_reward = np.mean(episode_rewards)
        max_reward = np.max(episode_rewards)
        reward_history.append(avg_reward)
        max_reward_history.append(max_reward)

        # 打印进度
        if (episode + 1) % 20 == 0:
            print(
                f"回合 {episode + 1}/{episodes} | 平均奖励: {avg_reward:.4f} | 最大奖励: {max_reward:.4f} | 最优值: {best_reward:.4f}")

    # 4. 训练结果可视化
    plt.figure(figsize=(12, 5))

    # 4.1 奖励变化曲线
    plt.subplot(1, 2, 1)
    plt.plot(reward_history, label='平均奖励')
    plt.plot(max_reward_history, label='最大奖励')
    plt.xlabel('训练回合')
    plt.ylabel('奖励（函数值）')
    plt.title('训练过程奖励变化')
    plt.legend()
    plt.grid(alpha=0.3)

    # 4.2 最优解位置可视化
    plt.subplot(1, 2, 2)
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            Z[i, j] = complex_function([X[i, j], Y[i, j]])
    plt.contourf(X, Y, Z, 50, cmap='viridis')
    plt.scatter(best_state[0], best_state[1], c='red', s=100,
                label=f'最优位置\n({best_state[0]:.3f}, {best_state[1]:.3f})\n函数值: {best_reward:.4f}')
    plt.colorbar()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('最优解位置')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 5. 输出最终结果
    print("\n训练完成！")
    print(f"找到的最优位置: {best_state}")
    print(f"最优位置的函数值: {best_reward}")
    print(f"理论最大值（参考）: {complex_function([0.0, 0.0]):.4f}")


# ====================== 5. 主函数 ======================
if __name__ == "__main__":
    # 先可视化测试函数（可选）
    plot_function()

    # 开始训练
    train()