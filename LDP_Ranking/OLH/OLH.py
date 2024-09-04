import numpy as np
from sys import maxsize     # maxsize = 2^63 - 1
import xxhash
# [1] Wang et al (2017) "Locally differentially private protocols for frequency estimation" (USENIX Security).
# [2] Bassily and Smith "Local, private, efficient protocols for succinct histograms" (STOC).
# [3] Agrawal and Aggarwal (2001,) "On the design and quantification of privacy preserving data mining algorithms" (PODS).
# [4] ElSalamouny and Palamidessi (2020) "Generalized iterative bayesian update and applications to mechanisms for privacy protection" (EuroS&P).

# 代码改编自 Samuel Maddock 开发的 pure-ldp 库 (https://github.com/Samuel-Maddock/pure-LDP)

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

# 扰动-客户端
# 输入为单个值
def LH_Client(input_data, d, epsilon, optimal=True):
    
    """
    本地哈希 (LH) 协议[1]，其逻辑上等同于 [2] 中的随机矩阵投影技术。

    :param input_data: 用户的真实值；
    :param d: 属性的域大小；
    :param epsilon: 隐私保证；
    :param optimal: 如果为 True，则使用 [1] 中的优化 LH (OLH) 协议；
    :return: 消毒值和随机种子的元组。
    """

    # 验证
    if input_data < 0 or input_data >= d:
        raise ValueError('input_data (整数) 应该在 [0, k-1] 范围内。')
    if not isinstance(d, int) or d < 2:
        raise ValueError('d 需要一个 >=2 的整数值。')
    if epsilon > 0:

        # 二进制 LH (BLH) 参数
        g = 2

        # 优化 LH (OLH) 参数
        if optimal:
            g = int(round(np.exp(epsilon))) + 1

        # 生成随机种子并散列用户的值
        rnd_seed = np.random.randint(0, maxsize, dtype=np.int64)        # 生成一个0-maxsize之间的随机整数
        hashed_input_data = (xxhash.xxh32(str(input_data), seed=rnd_seed).intdigest() % g)

        # LH 扰动函数 (即基于 GRR)
        sanitized_value = GRR_Client(hashed_input_data, g, epsilon)

        return (sanitized_value, rnd_seed)          # 返回元组，第一项为哈希值，第二项为哈希函数对应的种子

    else:
        raise ValueError('epsilon (浮点数) 需要一个大于 0 的数值。')

def LH_Aggregator_MI(reports, d, epsilon, optimal=True):
    """
    用于标准化频率 (0 -- 1) 的统计估计器，并进行后处理以确保非负性。

    :param reports: 所有基于 LH 的消毒值的列表；
    :param d: 属性的域大小；
    :param epsilon: 隐私保证；
    :param optimal: 如果为 True，则使用 [1] 中的优化 LH (OLH) 协议；
    :return: 频率估计。
    """

    # 验证
    if len(reports) == 0:
        raise ValueError('报告列表为空。')
    if not isinstance(d, int) or d < 2:
        raise ValueError('d 需要一个 >=2 的整数值。')
    if epsilon > 0:
            
        # 报告数量
        n = len(reports)

        # 二进制 LH (BLH) 参数
        g = 2

        # 优化 LH (OLH) 参数
        if optimal:
            g = int(round(np.exp(epsilon))) + 1

        # 统计每个值被报告的次数
        count_report = np.zeros(d)
        for tuple_val_seed in reports:
            for v in range(d):
                if tuple_val_seed[0] == (xxhash.xxh32(str(v), seed=tuple_val_seed[1]).intdigest() % g):
                    count_report[v] += 1

        # 使用 MI 估计
        p = np.exp(epsilon) / (np.exp(epsilon) + g - 1) # 使用缩小域大小 g 的 GRR 参数
        q = 1 / g # 服务器端的 g 值不同 [1]
        norm_est_freq = MI(count_report, n, p, q)

        return norm_est_freq

    else:
        raise ValueError('epsilon (浮点数) 需要一个大于 0 的数值。')



