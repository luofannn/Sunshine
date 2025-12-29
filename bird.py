import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.spatial import distance_matrix


# ===================== 1. 定义TSP问题和核心参数 =====================
class TSPProblem:
    def __init__(self, num_cities, city_coords=None):
        self.num_cities = num_cities
        # 随机生成城市坐标（或传入自定义坐标）
        if city_coords is None:
            self.city_coords = np.random.rand(num_cities, 2) * 100  # 坐标范围[0,100]
        else:
            self.city_coords = city_coords
        # 计算城市间距离矩阵
        self.dist_matrix = distance_matrix(self.city_coords, self.city_coords)

    # 计算路径总长度（适应度函数）
    def calculate_path_length(self, path):
        length = 0
        for i in range(self.num_cities - 1):
            length += self.dist_matrix[path[i], path[i + 1]]
        # 回到起点
        length += self.dist_matrix[path[-1], path[0]]
        return length


# ===================== 2. RLPO算法核心实现 =====================
class RLPO_TSP:
    def __init__(self, tsp_problem, pop_size=50, max_iter=500, alpha=0.8, beta=0.6, epsilon=0.1):
        self.tsp = tsp_problem  # TSP问题实例
        self.pop_size = pop_size  # 鹦鹉种群规模
        self.max_iter = max_iter  # 最大迭代次数
        self.alpha = alpha  # 强化学习学习率
        self.beta = beta  # 探索-利用平衡系数
        self.epsilon = epsilon  # 探索概率（ε-greedy）

        # 初始化鹦鹉种群（每个鹦鹉对应一条TSP路径）
        self.population = self.init_population()
        # 初始化奖励表（记录每个路径的累积奖励）
        self.reward_table = np.zeros(self.pop_size)
        # 记录全局最优解
        self.global_best_path = None
        self.global_best_length = float('inf')
        # 记录迭代过程的最优长度（用于绘图）
        self.iter_best_lengths = []

    # 初始化种群：随机生成路径（无重复城市）
    def init_population(self):
        population = []
        for _ in range(self.pop_size):
            path = random.sample(range(self.tsp.num_cities), self.tsp.num_cities)
            population.append(path)
        return population

    # 强化学习奖励计算：负的路径长度（长度越短，奖励越高）
    def calculate_reward(self, path_length):
        return -path_length  # 奖励最大化等价于路径长度最小化

    # ε-greedy策略选择动作（路径更新方向）
    def epsilon_greedy_selection(self, bird_idx):
        if random.random() < self.epsilon:
            # 探索：随机选择一个同伴作为参考
            return random.choice([i for i in range(self.pop_size) if i != bird_idx])
        else:
            # 利用：选择奖励最高的同伴作为参考
            return np.argmax(self.reward_table)

    # 鹦鹉路径更新（核心搜索策略）
    def update_bird_path(self, bird_path, ref_path):
        new_path = bird_path.copy()
        # 随机选择两个位置进行交换（模拟鹦鹉的局部搜索）
        idx1, idx2 = random.sample(range(self.tsp.num_cities), 2)
        new_path[idx1], new_path[idx2] = new_path[idx2], new_path[idx1]

        # 强化学习引导的全局探索：融合参考路径的信息
        if random.random() < self.beta:
            # 随机选择一段参考路径替换
            start = random.randint(0, self.tsp.num_cities - 3)
            end = random.randint(start + 1, self.tsp.num_cities - 1)
            # 提取参考路径的子序列，去重后插入
            ref_subseq = [city for city in ref_path[start:end] if city not in new_path[start:end]]
            new_path[start:start + len(ref_subseq)] = ref_subseq

        # 保证路径无重复（修复操作）
        new_path = self.fix_duplicate_path(new_path)
        return new_path

    # 修复重复城市的路径（TSP路径必须无重复）
    def fix_duplicate_path(self, path):
        unique_cities = list(set(path))
        missing_cities = [city for city in range(self.tsp.num_cities) if city not in unique_cities]
        # 替换重复的位置
        for i in range(len(path)):
            if path.count(path[i]) > 1:
                path[i] = missing_cities.pop(0)
                if not missing_cities:
                    break
        return path

    # 算法主循环
    def run(self):
        for iter in range(self.max_iter):
            # 1. 计算每个鹦鹉的适应度和奖励
            lengths = [self.tsp.calculate_path_length(path) for path in self.population]
            rewards = [self.calculate_reward(length) for length in lengths]

            # 2. 更新奖励表（强化学习累积奖励）
            self.reward_table = self.alpha * self.reward_table + (1 - self.alpha) * np.array(rewards)

            # 3. 更新全局最优解
            current_best_idx = np.argmin(lengths)
            current_best_length = lengths[current_best_idx]
            current_best_path = self.population[current_best_idx]
            if current_best_length < self.global_best_length:
                self.global_best_length = current_best_length
                self.global_best_path = current_best_path.copy()

            # 4. 记录当前迭代最优长度
            self.iter_best_lengths.append(self.global_best_length)

            # 5. 更新每个鹦鹉的路径
            new_population = []
            for bird_idx in range(self.pop_size):
                # ε-greedy选择参考鹦鹉
                ref_idx = self.epsilon_greedy_selection(bird_idx)
                ref_path = self.population[ref_idx]
                # 更新当前鹦鹉的路径
                new_path = self.update_bird_path(self.population[bird_idx], ref_path)
                new_population.append(new_path)
            self.population = new_population

            # 打印迭代信息
            if (iter + 1) % 50 == 0:
                print(f"迭代 {iter + 1}/{self.max_iter} | 最优路径长度: {self.global_best_length:.2f}")

        return self.global_best_path, self.global_best_length


# ===================== 3. 结果可视化 =====================
def plot_tsp_result(tsp_problem, best_path, iter_best_lengths):
    # 子图1：TSP最优路径
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    # 绘制城市
    plt.scatter(tsp_problem.city_coords[:, 0], tsp_problem.city_coords[:, 1], c='red', s=50, label='Cities')
    # 绘制最优路径
    path_coords = tsp_problem.city_coords[best_path]
    path_coords = np.vstack([path_coords, path_coords[0]])  # 回到起点
    plt.plot(path_coords[:, 0], path_coords[:, 1], c='blue', linewidth=1, label='Best Path')
    plt.title(f'TSP Optimal Path (Length: {tsp_problem.calculate_path_length(best_path):.2f})')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(alpha=0.3)

    # 子图2：迭代过程的最优长度变化
    plt.subplot(1, 2, 2)
    plt.plot(range(len(iter_best_lengths)), iter_best_lengths, c='green', linewidth=2)
    plt.title('RLPO Iteration Best Length')
    plt.xlabel('Iteration')
    plt.ylabel('Path Length')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ===================== 4. 主函数：运行RLPO解TSP =====================
if __name__ == "__main__":
    # 1. 初始化TSP问题（设置城市数量，比如20个城市）
    num_cities = 20
    tsp = TSPProblem(num_cities=num_cities)

    # 2. 初始化RLPO算法
    rlpo = RLPO_TSP(
        tsp_problem=tsp,
        pop_size=50,  # 种群规模
        max_iter=500,  # 迭代次数
        alpha=0.8,  # 学习率
        beta=0.6,  # 探索-利用系数
        epsilon=0.1  # 探索概率
    )

    # 3. 运行算法
    best_path, best_length = rlpo.run()

    # 4. 打印结果
    print("\n==================== 最终结果 ====================")
    print(f"最优路径: {best_path}")
    print(f"最优路径长度: {best_length:.2f}")

    # 5. 可视化结果
    plot_tsp_result(tsp, best_path, rlpo.iter_best_lengths)