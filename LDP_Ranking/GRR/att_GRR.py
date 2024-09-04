import numpy as np
import pandas as pd
from collections import Counter
from GRR import GRR_Client, GRR_Aggregator_MI
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.stats import norm

# 生成数据集并统计每个项对应频数
def generate_dataset(n, d, m):
    # 步骤1: 从均匀分布[1, m]中生成d个随机数
    c = np.random.uniform(1, m, d)
    
    # 步骤2: 归一化生成的随机数，使其总和等于n
    c_prime = np.floor((c / c.sum()) * n).astype(int)
    
    # 步骤3: 将归一化后的数按降序排序
    c_double_prime = np.sort(c_prime)[::-1]
    
    # 步骤4: 生成数据集，每个值i有c_double_prime[i-1]个出现
    # 使用 numpy 的 repeat 函数生成重复的值
    values = np.arange(d)  # 生成值的范围从 0 到 d-1
    repeated_values = np.repeat(values, c_double_prime)  # 根据c_double_prime的频数重复值
    
    # 将 numpy 数组转换为 pandas Series
    dataset = pd.Series(repeated_values)
    
    return dataset

# 根据数据集生成频数字典
# 频数从大到小排序
def generate_fre_dic(dataset):
    # 生成数据集中每个项的频数
    value_counts = dataset.value_counts().sort_index()

    # 将这个频数序列转换成字典
    original_freq = value_counts.to_dict()
    
    # 将字典按值从大到小排序
    sorted_dict = {k: v for k, v in sorted(original_freq.items(), key=lambda item: item[1], reverse=True)}

    
    return sorted_dict

# 生成目标项/非目标项集合（从排名前x的项中选取r个目标项）
def generate_target_items(dataset, d, r, x):
    # 计算每个项的频数
    item_counts = Counter(dataset)
    
    # 获取频数排名前x的项
    top_x_items = [item for item, count in item_counts.most_common(x)]
    
    # 随机选择目标项
    target_items = np.random.choice(top_x_items, r, replace=False)
    
    # 对目标项进行排序
    target_items_sort = np.sort(target_items)
    
    return target_items_sort

# 使用GRR进行扰动和聚合（1表示扰动+聚合，0表示仅聚合）
# 三返回值分布为：扰动后的数据集，聚合后得到的频率字典和排名字典
def GRR_est(dataset, d, eps, if_per):
    
    # 选择是否扰动，因为在添加假用户后要直接聚合
    if(if_per == 1):
        # 扰动
        per_dataset = [GRR_Client(user_data, d, eps) for user_data in dataset] #扰动时要对各个数据单独扰动
    else:
        per_dataset = dataset
        
    # 聚合
    est_freq = GRR_Aggregator_MI(per_dataset, d, eps)
    
    # 生成频数字典
    est_freq_dic = {}
    for value, count in enumerate(est_freq):
        est_freq_dic[value] = int(count)  # 存储每个值的频数

    # 将字典按值从大到小排序
    sorted_dict = {k: v for k, v in sorted(est_freq_dic.items(), key=lambda item: item[1], reverse=True)}
    rank_dict = {key: rank + 1 for rank, (key, value) in enumerate(sorted_dict.items())}
    
    return per_dataset, sorted_dict, rank_dict

# 生成随机选择目标项/非目标项的假数据集    
def generate_fake_data(n2, T):
    # 每个假用户随机选择一个目标项报告
    fake_data = np.random.choice(T, n2)
    return fake_data

# 生成扰动值的期望值字典
# E[n_v'] = n_v⋅p+(N-n_v)⋅q
def expected_per_frequencies(original_freq, n, p, q):
    expected_per_freq = {}
    for item, freq in original_freq.items():
        expected_per_freq[item] = int(np.round(freq * p + (n - freq) * q))
    return expected_per_freq

# 计算总体增益（可算频数增益与排名增益）
def get_gain(T, dic, att_dic):
    # 计算每个目标项的频数增益
    gains = []
    for item in T:
        original_num = dic.get(item, 0)  # 如果项不存在则返回0
        attacked_num = att_dic.get(item, 0)  # 如果项不存在则返回0
        gain = attacked_num - original_num  # 计算增益
        gains.append(gain)
    # 计算总增益 G
    G = sum(gains)
    return G

# 计算目标项与有效攻击项之间的距离
def calculate_distances(target_items, eff_items, expected_per_freq):
    """
    :param target_items: 目标项列表
    :param eff_items: 有效攻击项列表
    :param expected_per_freq: 扰动值的期望值字典
    :return: 每个有效攻击项与其最近的目标项之间的距离字典
    """
    distances = {}
    for eff_item in eff_items:
        # 找到所有频数大于当前有效攻击项的目标项子集
        target_subset = [t for t in target_items if expected_per_freq[t] >= expected_per_freq[eff_item]]
        if target_subset:
            # 找到频数最接近当前有效攻击项的目标项
            min_target = min(target_subset, key=lambda t: expected_per_freq[t])
            # 计算距离并存储
            distances[eff_item] = expected_per_freq[min_target] - expected_per_freq[eff_item] + 1
    return distances

def generate_greedy_fake_data(n2, T, A, expected_per_freq, est_rank_dict):
    """
    :param n2: 假用户数量
    :param T: 目标项集合
    :param A: 非目标项集合
    :param expected_per_freq: 扰动值的期望值字典
    :param est_rank_dict: 排名估计字典
    :return: 生成的假数据集
    """
    
    # 复制频数估计字典，以便进行修改
    attacked_freq = expected_per_freq.copy()
    fake_data = []

    # 初始有效攻击项
    eff_items = [a for a in A if any(attacked_freq[t] > attacked_freq[a] for t in T)]
    
    while n2 > 0:
        # 计算每个有效攻击项与其最近目标项之间的距离
        distances = calculate_distances(T, eff_items, attacked_freq)
        
        if not distances:
            break
        
        # 找到距离最小的有效攻击项
        opt_item = min(distances, key=distances.get)
        
        # 持续攻击当前最优攻击项，直到其频数超过前方目标项
        while distances[opt_item] > 0 and n2 > 0:
            fake_data.append(opt_item)  # 添加假用户
            attacked_freq[opt_item] += 1  # 更新攻击项频数
            distances[opt_item] -= 1  # 更新距离
            n2 -= 1  # 减少剩余假用户数量
        
       
         # 更新估计排名
        sorted_attacked_freq = {k: v for k, v in sorted(attacked_freq.items(), key=lambda item: item[1], reverse=True)}
        attacked_rank_dict = {key: rank + 1 for rank, (key, value) in enumerate(sorted_attacked_freq.items())}
            
            # 计算当前排名增益
        rank_G2 = get_gain(T, est_rank_dict, attacked_rank_dict)
        
            # 输出调试信息
        '''
        print(f"各项目前的频数: {attacked_freq}")
        print(f"最优攻击项:{opt_item}")
        print(f"有效攻击项: {eff_items}")
        print(f"当前排名增益: {rank_G2}")
        print(f"剩余假用户数量:{n2}")
        print(f"")
        '''
        # 更新有效攻击项列表
        eff_items = [a for a in A if any(attacked_freq[t] > attacked_freq[a] for t in T)]
    
    if n2 > 0:
        fake_data.extend(np.random.choice(A, n2, replace=True))
    
    return fake_data

def calculate_distances_MPOIA(target_items, eff_items, ori_dict, expected_per_freq, N, p):
    """
    :param target_items: 目标项列表
    :param eff_items: 有效攻击项列表
    :param ori_dict: 原始频数字典
    :param expected_per_freq: 扰动值的期望值字典
    :param N: 样本总数
    :param p: 保持原值的概率
    :return: 每个有效攻击项与其最近的目标项之间的距离字典
    """
    q = (1 - p) / (len(expected_per_freq) - 1)  # 计算q值
    
    distances = {}
    for eff_item in eff_items:
        # 找到所有频数大于当前有效攻击项的目标项子集
        target_subset = [t for t in target_items if expected_per_freq[t] >= expected_per_freq[eff_item]]
        if target_subset:
            # 找到频数最接近当前有效攻击项的目标项
            min_target = min(target_subset, key=lambda t: expected_per_freq[t])
            
            # 计算E[D]和Var(D)
            #E_D = (freq_dict[eff_item] * p + (N - freq_dict[eff_item]) * q) - (freq_dict[min_target] * p + (N - freq_dict[min_target]) * q) 
            E_D =  expected_per_freq[min_target] - expected_per_freq[eff_item]
            Var_D =  ori_dict[min_target]*p*(1-p) + (N-ori_dict[min_target])*q*(1-q) + ori_dict[eff_item]*p*(1-p) + (N-ori_dict[eff_item])*q*(1-q)
            
            # 计算delta，使得概率尽可能大
            z_alpha = norm.ppf(0.9)  # 使用95%的置信水平
            delta = int(E_D + z_alpha * np.sqrt(Var_D))+1
            
            # 计算距离并存储
            distances[eff_item] = delta
    return distances

def generate_greedy_fake_data_MPOIA(n2, T, A, ori_dict, expected_per_freq, est_rank_dict, N, p):
    """
    :param n2: 假用户数量
    :param T: 目标项集合
    :param A: 非目标项集合
    :param ori_dict: 原始频数字典
    :param expected_per_freq: 扰动值的期望值字典
    :param est_rank_dict: 排名估计字典
    :param N: 样本总数
    :param p: 保持原值的概率
    :return: 生成的假数据集
    """
    
    # 复制频数估计字典，以便进行修改
    attacked_freq = expected_per_freq.copy()
    fake_data = []

    # 初始有效攻击项
    eff_items = [a for a in A if any(attacked_freq[t] > attacked_freq[a] for t in T)]
    
    while n2 > 0:
        # 计算每个有效攻击项与其最近目标项之间的距离
        distances = calculate_distances_MPOIA(T, eff_items, ori_dict, attacked_freq, N, p)
        
        if not distances:
            break
        
        # 找到距离最小的有效攻击项
        opt_item = min(distances, key=distances.get)
        
        # 持续攻击当前最优攻击项，直到其频数超过前方目标项
        while distances[opt_item] > 0 and n2 > 0:
            fake_data.append(opt_item)  # 添加假用户
            attacked_freq[opt_item] += 1  # 更新攻击项频数
            distances[opt_item] -= 1  # 更新距离
            n2 -= 1  # 减少剩余假用户数量
        
         # 更新估计排名
        sorted_attacked_freq = {k: v for k, v in sorted(attacked_freq.items(), key=lambda item: item[1], reverse=True)}
        attacked_rank_dict = {key: rank + 1 for rank, (key, value) in enumerate(sorted_attacked_freq.items())}
            
            # 计算当前排名增益
        rank_G2 = get_gain(T, est_rank_dict, attacked_rank_dict)
        
            # 输出调试信息
        '''
        print(f"各项目前的频数: {attacked_freq}")
        print(f"最优攻击项:{opt_item}")
        print(f"有效攻击项: {eff_items}")
        print(f"当前排名增益: {rank_G2}")
        print(f"剩余假用户数量:{n2}")
        print(f"")
        '''
        
        # 更新有效攻击项列表
        eff_items = [a for a in A if any(attacked_freq[t] > attacked_freq[a] for t in T)]
    
    if n2 > 0:
        fake_data.extend(np.random.choice(A, n2, replace=True))
    return fake_data


