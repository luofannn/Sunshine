import numpy as np

# 目标函数：求最小值
def objective(x):
    return x**2 + 10

# 超简化TLBO核心逻辑
def simple_tlbo():
    # 1. 初始化参数
    pop_size = 5          # 种群规模（5个学生）
    max_iter = 20         # 迭代次数
    pop = np.random.uniform(-10, 10, pop_size)  # 随机初始化5个学生（x的范围-10~10）
    global_best_x = pop[np.argmin([objective(x) for x in pop])]
    global_best_val = objective(global_best_x)

    # 2. 迭代训练
    for iter in range(max_iter):
        # 计算当前种群均值（班级平均水平）
        pop_mean = np.mean(pop)
        # 找到当前最优个体（教师）
        current_best_idx = np.argmin([objective(x) for x in pop])
        teacher = pop[current_best_idx]
        teacher_val = objective(teacher)

        # ---------------------- 教师阶段 ----------------------
        # 教学因子随机选1或2（传统TLBO规则）
        T_F = np.random.choice([1, 2])
        new_pop = []
        for x in pop:
            # 教师阶段更新公式：x_new = x + r*(教师 - T_F*班级均值)
            r = np.random.random()  # 0~1随机数
            x_new = x + r * (teacher - T_F * pop_mean)
            # 保留更优解：如果新解更好，就替换
            if objective(x_new) < objective(x):
                new_pop.append(x_new)
            else:
                new_pop.append(x)
        pop = np.array(new_pop)

        # ---------------------- 学习者阶段 ----------------------
        new_pop = []
        for i in range(pop_size):
            # 随机选另一个同学j
            j = np.random.choice([k for k in range(pop_size) if k != i])
            x_i = pop[i]
            x_j = pop[j]
            # 学习者阶段：向更优秀的同学学习
            if objective(x_i) > objective(x_j):
                # i比j差，向j学习
                r = np.random.random()
                x_new = x_i + r * (x_j - x_i)
            else:
                # i比j好，j向i学习（等价于i小幅探索）
                r = np.random.random()
                x_new = x_j + r * (x_i - x_j)
            # 保留更优解
            if objective(x_new) < objective(x_i):
                new_pop.append(x_new)
            else:
                new_pop.append(x_i)
        pop = np.array(new_pop)

        # 更新全局最优
        current_best_val = objective(pop[np.argmin([objective(x) for x in pop])])
        if current_best_val < global_best_val:
            global_best_x = pop[np.argmin([objective(x) for x in pop])]
            global_best_val = current_best_val

        # 打印迭代信息
        print(f"迭代{iter+1:2d} | 全局最优x: {global_best_x:.4f} | 最优值: {global_best_val:.4f}")

    return global_best_x, global_best_val

# 运行TLBO
best_x, best_val = simple_tlbo()
print("\n最终结果：")
print(f"最优x = {best_x:.4f}，最优值 = {best_val:.4f}")
print(f"理论最优值（x=0）：{objective(0):.4f}")