import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

class FuzzyNNController:
    """
    自适应模糊神经网络控制器 (Adaptive Fuzzy Neural Network Controller).

    使用高斯隶属度函数 (Gaussian Membership Functions) 和
    TSK (Takagi-Sugeno-Kang) 规则后件进行非线性映射，适用于补偿迟滞误差。
    基于梯度下降的在线自适应学习律用于调整参数。
    """

    def __init__(self, n_rules: int = 7, n_inputs: int = 2):
        """
        初始化 Fuzzy-NN 控制器。

        Args:
            n_rules (int): 模糊规则的数量。
            n_inputs (int): 输入变量的维度。
        """
        self.n_rules = n_rules
        self.n_inputs = n_inputs

        # 初始化高斯隶属度函数参数: 均值 (centers) 和 标准差 (widths)
        # 为每个输入的每个规则分配一个高斯函数
        self.centers = np.random.uniform(-1, 1, (n_rules, n_inputs))
        self.widths = np.random.uniform(0.5, 1.5, (n_rules, n_inputs))

        # TSK 后件参数: weights
        # 每个规则的输出是输入的线性组合加上偏置: y_r = w_{r,0} + w_{r,1}*x_1 + ... + w_{r,n}*x_n
        # 权重矩阵大小: (n_rules, n_inputs + 1)
        self.weights = np.random.uniform(-1, 1, (n_rules, n_inputs + 1))

        # 缓存前向传播变量，用于反向传播
        self._cache = {}

    def _gaussian(self, x: float, c: float, sigma: float) -> float:
        """计算高斯隶属度"""
        return np.exp(-((x - c) ** 2) / (2 * (sigma ** 2) + 1e-8))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播计算模糊神经网络的输出。

        Args:
            x (np.ndarray): 输入向量，形状为 (n_inputs,)。

        Returns:
            np.ndarray: 控制输出，形状为 (1,) 的数组 (包含单个标量值)。
        """
        # 1. 隶属度计算层 (Fuzzification)
        mu = np.zeros((self.n_rules, self.n_inputs))
        for r in range(self.n_rules):
            for i in range(self.n_inputs):
                mu[r, i] = self._gaussian(x[i], self.centers[r, i], self.widths[r, i])

        # 2. 规则触发强度计算 (Firing Strength, T-norm: product)
        w = np.prod(mu, axis=1)

        # 3. 归一化触发强度层 (Normalized firing strength)
        sum_w = np.sum(w)
        if sum_w == 0:
            w_norm = np.ones(self.n_rules) / self.n_rules # 防止除零
        else:
            w_norm = w / sum_w

        # 4. 结论层计算 (TSK Consequent)
        # 扩展 x 添加偏置项 1
        x_ext = np.insert(x, 0, 1.0)
        rule_outputs = np.dot(self.weights, x_ext)

        # 5. 输出计算 (Defuzzification)
        y_out = np.sum(w_norm * rule_outputs)

        # 保存前向传播结果用于在线自适应反向传播
        self._cache = {
            'x': x,
            'x_ext': x_ext,
            'mu': mu,
            'w': w,
            'w_norm': w_norm,
            'rule_outputs': rule_outputs,
            'sum_w': sum_w
        }

        return np.array([y_out])

    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 100, lr: float = 0.01):
        """
        离线训练模型 (批处理模式简化版，通过多次调用 adapt_online 实现)。

        Args:
            X_train (np.ndarray): 训练输入，形状为 (n_samples, n_inputs)。
            y_train (np.ndarray): 训练输出目标，形状为 (n_samples,)。
            epochs (int): 训练轮数。
            lr (float): 学习率。
        """
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(X_train)):
                x = X_train[i]
                y_true = np.array([y_train[i]])

                # 前向传播并计算误差
                y_pred = self.forward(x)
                error = y_pred[0] - y_true[0]
                total_loss += error ** 2

                # 在线参数更新
                self.adapt_online(x, y_true, lr)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {total_loss/len(X_train):.6f}")

    def adapt_online(self, x: np.ndarray, y_true: np.ndarray, lr: float = 0.001):
        """
        在线自适应更新步 (梯度下降)。

        根据系统误差实时调整高斯函数的中心、宽度以及 TSK 权重。
        需要先执行过 forward 以填充 cache。

        Args:
            x (np.ndarray): 当前输入向量。
            y_true (np.ndarray): 当前真实输出目标或参考值。
            lr (float): 学习率。
        """
        # 前向计算一次，获取最新 cache (在控制环中可以复用之前的，但为安全起见重新计算)
        y_pred = self.forward(x)
        error = y_pred[0] - y_true[0]

        c = self._cache
        w_norm = c['w_norm']
        rule_outputs = c['rule_outputs']
        x_ext = c['x_ext']
        w = c['w']
        sum_w = c['sum_w']

        if sum_w == 0:
            return # 避免除零错误导致梯度爆炸

        # 1. 更新 TSK 后件权重 (weights)
        # dE/dW = dE/dy * dy/dy_r * dy_r/dW = error * w_norm[r] * x_ext
        for r in range(self.n_rules):
            grad_w = error * w_norm[r] * x_ext
            self.weights[r] -= lr * grad_w

        # 2. 更新高斯参数 centers 和 widths
        # dE/dc = dE/dy * dy/dw_norm * dw_norm/dw * dw/dmu * dmu/dc
        for r in range(self.n_rules):
            # dy/dw_norm[r] = rule_outputs[r]
            # dw_norm[r]/dw[r] = (sum_w - w[r]) / (sum_w ** 2)
            # dw_norm[k]/dw[r] = -w[k] / (sum_w ** 2)  (for k != r)
            # 简化导数计算：
            dy_dw = (rule_outputs[r] - y_pred[0]) / sum_w

            for i in range(self.n_inputs):
                # dw/dmu_{r,i} = w[r] / mu_{r,i}
                mu_ri = c['mu'][r, i]
                if mu_ri < 1e-8:
                    continue # 避免除零

                dw_dmu = w[r] / mu_ri

                # dmu/dc
                diff = x[i] - self.centers[r, i]
                sigma_sq = self.widths[r, i] ** 2
                dmu_dc = mu_ri * (diff / (sigma_sq + 1e-8))

                # dmu/dsigma
                dmu_dsigma = mu_ri * (diff ** 2) / (self.widths[r, i] ** 3 + 1e-8)

                # 链式法则
                grad_c = error * dy_dw * dw_dmu * dmu_dc
                grad_sigma = error * dy_dw * dw_dmu * dmu_dsigma

                # 更新
                self.centers[r, i] -= lr * grad_c
                self.widths[r, i] -= lr * grad_sigma

    def get_fuzzy_rules(self) -> List[str]:
        """
        提取并格式化当前的可解释模糊规则。

        Returns:
            List[str]: 模糊规则的字符串列表。
        """
        rules = []
        for r in range(self.n_rules):
            antecedents = []
            for i in range(self.n_inputs):
                c = self.centers[r, i]
                w = self.widths[r, i]
                antecedents.append(f"x{i} is Gaussian(c={c:.2f}, w={w:.2f})")

            antecedent_str = " AND ".join(antecedents)

            consequent_terms = [f"{self.weights[r, 0]:.2f}"]
            for i in range(self.n_inputs):
                consequent_terms.append(f"{self.weights[r, i+1]:.2f}*x{i}")

            consequent_str = " + ".join(consequent_terms)

            rule_str = f"Rule {r+1}: IF {antecedent_str} THEN y = {consequent_str}"
            rules.append(rule_str)

        return rules

    def plot_membership_functions(self, save_path: Optional[str] = None):
        """
        可视化当前的高斯隶属度函数。

        Args:
            save_path (Optional[str]): 图像保存路径。
        """
        x_vals = np.linspace(-2, 2, 400)

        fig, axes = plt.subplots(self.n_inputs, 1, figsize=(8, 3 * self.n_inputs))
        if self.n_inputs == 1:
            axes = [axes]

        for i in range(self.n_inputs):
            ax = axes[i]
            for r in range(self.n_rules):
                c = self.centers[r, i]
                w = self.widths[r, i]
                y_vals = [self._gaussian(xv, c, w) for xv in x_vals]
                ax.plot(x_vals, y_vals, label=f'Rule {r+1}')

            ax.set_title(f'Membership Functions for Input {i+1}')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"隶属度函数图已保存至: {save_path}")
        else:
            plt.show()
        plt.close()
