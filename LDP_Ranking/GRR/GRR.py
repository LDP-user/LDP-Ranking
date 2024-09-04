import numpy as np
from numba import jit

# [1] Wang 等人 (2017) "局部差分隐私协议用于频率估计" (USENIX Security)。
# [2] Kairouz, Bonawitz, 和 Ramage (2016) "在局部隐私下的离散分布估计" (ICML)。
# [3] Agrawal 和 Aggarwal (2001) "关于设计和量化隐私保护数据挖掘算法" (PODS)。
# [4] ElSalamouny 和 Palamidessi (2020) "通用迭代贝叶斯更新及其在隐私保护机制中的应用" (EuroS&P)。

# 频率估计
def MI(count_report, n, p, q):
    """
    矩阵反演（MI）。

    :param count_report: 为列表，记录每个值被报告的次数；
    :param n: 报告的总数；
    :param p: 诚实的概率；
    :param q: 撒谎的概率；
    :return: 归一化频率（直方图）估计。
    """
    # # Validations
    # if len(count_report) == 0:
    #     raise ValueError('List of count_report is empty.')
    # if not isinstance(n, int) or not isinstance(p, float) or not isinstance(q, float):
    #     raise ValueError('n (int), p (float), q (float) need numerical values.')
    
    # 确保估计频率的非负性
    est_freq = np.array((count_report - n * q) / (p - q)).clip(0)
    
    est_freq_rounded = np.round(est_freq)


    # 重新归一化估计频率
    
    if sum(est_freq) > 0:
        norm_est_freq = np.nan_to_num(est_freq / sum(est_freq))
    else:
        norm_est_freq = est_freq

    return est_freq_rounded

# 扰动-客户端
# 输入为单个值
def GRR_Client(input_data, d, epsilon):
    """
    广义随机响应（GRR）协议，又称直接编码 [1] 或 k-RR [2]。

    :param input_data: 用户的真实值；(每次输入一个值)
    :param d: 属性的域大小；
    :param epsilon: 隐私保证；
    :return: 隐私化后的值。
    """

    # 验证输入参数
    if input_data < 0 or input_data >= d:
        raise ValueError('input_data（整数）应在 [0, d-1] 范围内。')      #raise用于在程序中触发异常，中断程序执行
    if not isinstance(d, int) or d < 2:
        raise ValueError('d 需为大于等于 2 的整数。')
    
    if epsilon > 0:
        
        # GRR 参数设置
        p = np.exp(epsilon) / (np.exp(epsilon) + d - 1)

        # 将属性域大小 k 映射到 [0, ..., k-1]
        domain = np.arange(d) 
        
        # GRR 扰动函数
        if np.random.binomial(1, p) == 1:
            return input_data
        else:
            return np.random.choice(domain[domain != input_data])   # 随机选择一个非input的值，参数为数组

    else:
        raise ValueError('epsilon 必须是大于 0 的数值。')


# 聚合
# 输入为整个数据集
def GRR_Aggregator_MI(reports, d, epsilon):
    """
    用于归一化频率（0 -- 1）的统计估计器，并进行后处理以确保非负性。

    :param reports: 所有基于 GRR 的扰动值列表；
    :param d: 属性的域大小；
    :param epsilon: 隐私保证；
    :return: 归一化频率（直方图）估计。
    """

    # 验证输入
    if len(reports) == 0:
        raise ValueError('报告列表为空。')
    if not isinstance(d, int) or d < 2:
        raise ValueError('d 需要是大于等于2的整数值。')
    if epsilon > 0:
            
        # 报告数量
        n = len(reports)
        

        # GRR 参数
        p = np.exp(epsilon) / (np.exp(epsilon) + d - 1)
        q = (1 - p) / (d - 1)

        # 计数每个值被报告的次数（支持集为其本身）
        count_report = np.zeros(d)
        for rep in reports:
            if rep < 0 or rep >= d:
                raise IndexError(f"Report value {rep} out of bounds for array of size {d}.")
            count_report[rep] += 1

        # 使用 MI 方法估计
        norm_est_freq = MI(count_report, n, p, q)

        return norm_est_freq

    else:
        raise ValueError('epsilon 需要是大于 0 的数值。')
         