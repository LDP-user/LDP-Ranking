import numpy as np
from numba import jit
from multi_freq_ldpy.estimators.Histogram_estimator import MI, IBU

# [1] Erlingsson, Pihur, and Korolova (2014) "RAPPOR: Randomized aggregatable privacy-preserving ordinal response" (ACM CCS).
# [2] Wang et al (2017) "Locally differentially private protocols for frequency estimation" (USENIX Security).
# [3] Agrawal and Aggarwal (2001,) "On the design and quantification of privacy preserving data mining algorithms" (PODS).
# [4] ElSalamouny and Palamidessi (2020) "Generalized iterative bayesian update and applications to mechanisms for privacy protection" (EuroS&P).


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
def UE_Client(input_data, d, epsilon, optimal=True):
    """
    一元编码（UE）协议，也称为基本一次性RAPPOR（如果optimal=False）[1]

    :param input_data: 用户的真实值;
    :param d: 属性的域大小;
    :param epsilon: 隐私保证;
    :param optimal: 如果为True，它使用[2]中的优化UE（OUE）协议;
    :return: 已消毒的UE向量。
    """

    # 验证
    if input_data != None:
        if input_data < 0 or input_data >= d:
            raise ValueError('input_data（整数）应在[0, k-1]范围内。')
    if not isinstance(d, int) or d < 2:
        raise ValueError('k需要一个整数值>=2。')
    if epsilon > 0:

        # 对称参数（p+q = 1）
        p = np.exp(epsilon/2) / (np.exp(epsilon/2) + 1)
        q = 1 - p

        # 优化参数
        if optimal:
            p = 1 / 2
            q = 1 / (np.exp(epsilon) + 1)

        # 一元编码
        input_ue_data = np.zeros(d)
        if input_data != None:
            input_ue_data[input_data] = 1

        # 初始化零向量
        sanitized_vec = np.zeros(d)

        # UE扰动函数
        for ind in range(d):
            if input_ue_data[ind] != 1:
                rnd = np.random.random()
                if rnd <= q:
                    sanitized_vec[ind] = 1
            else:
                rnd = np.random.random()
                if rnd <= p:
                    sanitized_vec[ind] = 1
        return sanitized_vec

    else:
        raise ValueError('epsilon（浮点数）需要一个大于0的数值。')
        
def UE_Aggregator_MI(reports, epsilon, optimal=True):

    """
    具有非负性后处理的标准化频率（0 -- 1）统计估计器。

    :param reports: 所有基于UE的已消毒向量的列表;
    :param epsilon: 隐私保证;
    :param optimal: 如果为True，它使用[2]中的优化UE（OUE）协议;
    :return: 标准化频率（直方图）估计。
    """

    # 验证
    if len(reports) == 0:
        raise ValueError('报告列表为空。')
    if epsilon > 0:

        # 报告数
        n = len(reports)

        # 对称参数（p+q = 1）
        p = np.exp(epsilon/2) / (np.exp(epsilon/2) + 1)
        q = 1 - p

        # 优化参数
        if optimal:
            p = 1 / 2
            q = 1 / (np.exp(epsilon) + 1)

        # 使用MI进行估计
        norm_est_freq = MI(sum(reports), n, p, q)

        return norm_est_freq

    else:
        raise ValueError('epsilon（浮点数）需要一个大于0的数值。')


