import numpy as np
import pandas as pd
from collections import Counter
from OLH import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.stats import norm
import pickle
import random

def save_hash_funcs(hash_funcs, filename='hash_funcs.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(hash_funcs, file)

def load_hash_funcs(filename='hash_funcs.pkl'):
    with open(filename, 'rb') as file:
        return pickle.load(file)

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

# 根据正常数据集生成频数字典
# 频数从大到小排序
def generate_fre_dic(dataset):
    # 生成数据集中每个项的频数
    value_counts = dataset.value_counts().sort_index()

    # 将这个频数序列转换成字典
    original_freq = value_counts.to_dict()
    
    # 将字典按值从大到小排序
    sorted_dict = {k: v for k, v in sorted(original_freq.items(), key=lambda item: item[1], reverse=True)}

    
    return sorted_dict

# 根据元组数据集生成频数字典
def generate_fre_dic_OLH(dataset, d, g):
    """
    根据OLH协议生成的元组数据集生成频数字典。

    :param dataset: OLH扰动后的数据集，每个元素是一个元组 (sanitized_value, rnd_seed)；
    :param d: 属性的域大小；
    :param g: 哈希桶的数量，由 epsilon 决定；
    :return: 一个字典，键是属性值，值是频数。
    """
    est_freq_dic = {i: 0 for i in range(d)}

    # 遍历数据集
    for sanitized_value, rnd_seed in dataset:
        # 对于每个可能的原始值 v
        for v in range(d):
            # 计算 v 的哈希值
            hashed_value = xxhash.xxh32(str(v), seed=rnd_seed).intdigest() % g
            # 如果哈希值等于消毒后的值，增加频数
            if hashed_value == sanitized_value:
                est_freq_dic[v] += 1

    return est_freq_dic


# 从字典的排名前x的项中选r个目标项
def generate_target_items(fre_dic, r, x):
    """
    生成目标项集合，从频数字典中获取前 x 项，并随机选择 r 个目标项。
    
    :param fre_dic: 频数字典，表示每个项的频数
    :param r: 目标项的数量
    :param x: 频数排名前 x 的项
    :return: 排序后的目标项集合
    """
    
    # 获取频数排名前 x 的项
    sorted_items = sorted(fre_dic.items(), key=lambda item: item[1], reverse=True)[:x]
    top_x_items = [item[0] for item in sorted_items]
    
    # 随机选择目标项
    target_items = np.random.choice(top_x_items, r, replace=False)
    
    # 对目标项进行排序
    target_items_sort = np.sort(target_items)
    
    return target_items_sort

# 使用OUE进行扰动和聚合（1表示扰动+聚合，0表示仅聚合）
# 三返回值分布为：扰动后的数据集，聚合后得到的频率字典和排名字典
def OLH_est(dataset, d, eps, if_per):
    
    # 选择是否扰动，因为在添加假用户后要直接聚合
    if(if_per == 1):
        # 扰动
        per_dataset = [LH_Client(user_data, d, eps) for user_data in dataset]
    else:
        per_dataset = dataset
        
    # 聚合
    est_freq = LH_Aggregator_MI(per_dataset, d, eps)
    
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

def generate_ROA_fake_data(m, non_target_items, d, g):
    """
    生成恶意用户的扰动数据，每个数据包含随机种子和从非目标项中随机选择的哈希值。

    :param m: 恶意用户数量
    :param non_target_items: 非目标项集合（列表）
    :param d: 属性的域大小
    :param g: 哈希桶的数量，由 epsilon 决定
    :return: 恶意用户的元组列表，每个元组包含 (sanitized_value, rnd_seed)
    """
    
    # 初始化恶意用户的元组列表
    malicious_vectors = []
    
    for _ in range(m):
        # 生成随机种子
        rnd_seed = np.random.randint(0, maxsize, dtype=np.int64)
        
        # 从非目标项集合中随机选择一个值
        selected_value = np.random.choice(non_target_items)
        
        # 计算该值的哈希值
        sanitized_value = xxhash.xxh32(str(selected_value), seed=rnd_seed).intdigest() % g
        
        # 将生成的元组添加到列表中
        malicious_vectors.append((sanitized_value, rnd_seed))
    
    return malicious_vectors


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

# 不支持任何目标项时
def calculate_costs(target_items, eff_items, expected_per_freq, hash_funcs, g):
    """
    计算每个哈希函数的有效攻击项代价，并返回代价最小的扰动值 (H, h)，
    但仅限于不支持任何目标项的组合。
    
    :param target_items: 目标项列表
    :param eff_items: 有效攻击项列表
    :param expected_per_freq: 扰动值的期望值字典
    :param hash_funcs: 哈希函数集合
    :param g: 哈希桶的数量
    :return: 代价最小的不支持任何目标项的扰动值 (H, h)
    """
    min_cost = float('inf')
    best_h = None
    best_i = None
    
    for h in hash_funcs:
        for i in range(g):
            distance_sums_eff = 0
            supports_target = False
            
            # 检查是否支持任何目标项
            for target_item in target_items:
                hashed_value = xxhash.xxh32(str(target_item), seed=h).intdigest() % g
                if hashed_value == i:
                    supports_target = True
                    break
            
            # 如果支持目标项，则跳过这个 (H, h) 组合
            if supports_target:
                continue
            
            # 计算支持的有效攻击项集合 A_{(i,j)}
            for eff_item in eff_items:
                hashed_value = xxhash.xxh32(str(eff_item), seed=h).intdigest() % g
                if hashed_value == i:
                    distances_eff = [expected_per_freq[t] - expected_per_freq[eff_item] for t in target_items if expected_per_freq[t] > expected_per_freq[eff_item]]
                    min_distance_eff = min(distances_eff, default=float('inf'))
                    if min_distance_eff < float('inf'):
                        distance_sums_eff += 1 / min_distance_eff
            
            # 计算总代价
            if distance_sums_eff > 0:
                cost = 1 / distance_sums_eff
                if cost < min_cost:
                    min_cost = cost
                    best_h = h
                    best_i = i
    
    return best_h, best_i


def generate_opt_fake_data_OLH(n2, T, A, expected_per_freq, d, eps, g, num_hash_funcs, save_hash=True, hash_filename='hash_funcs.pkl'):
    """
    生成OLH协议中的假数据集
    :param n2: 假用户数量
    :param T: 目标项集合
    :param A: 非目标项集合
    :param expected_per_freq: 扰动值的期望值字典
    :param d: 属性的域大小
    :param eps: 隐私预算
    :param g: 哈希桶的数量，由epsilon决定
    :param num_hash_funcs: 生成的哈希函数数量
    :param save_hash: 是否保存生成的哈希函数
    :param hash_filename: 保存哈希函数的文件名
    :return: 生成的假数据集
    """
    
    # 复制频数估计字典，以便进行修改
    attacked_freq = expected_per_freq.copy()
    fake_data = []

    # 初始有效攻击项
    eff_items = [a for a in A if any(attacked_freq[t] > attacked_freq[a] for t in T)]
    
    # 加载生成的哈希函数集合
    hash_funcs1 = load_hash_funcs(filename='hash_funcs.pkl')
    hash_funcs = hash_funcs1[:100]
    
    # 用于保存每轮选出的最优哈希函数和哈希值组合
    optimal_combinations = []
    
    while n2 > 0 and eff_items:
        # 计算代价最小的扰动值 (H, h)
        opt_hash, best_hashed_value = calculate_costs(T, eff_items, attacked_freq, hash_funcs, g)
        
        if opt_hash is None or best_hashed_value is None:
            break
        
        # 保存最优哈希函数和哈希值组合
        optimal_combinations.append((best_hashed_value, opt_hash))
        
        # 找到代价最低的哈希值
        min_diff_item = None
        min_diff = float('inf')
        
        for eff_item in eff_items:
            hashed_value = xxhash.xxh32(str(eff_item), seed=opt_hash).intdigest() % g
            if hashed_value == best_hashed_value:
                distances = [attacked_freq[t] - attacked_freq[eff_item] for t in T if attacked_freq[t] > attacked_freq[eff_item]]
                diff = min(distances, default=float('inf'))
                if diff < min_diff:
                    min_diff = diff
                    min_diff_item = eff_item

        if min_diff_item is None:
            break
        
        # 生成扰动值，恶意用户选择 (best_hashed_value, opt_hash) 作为扰动值
        perturbation = (best_hashed_value, opt_hash)
        fake_data.extend([perturbation] * min_diff)
        
        # 更新有效攻击项频数
        for eff_item in eff_items:
            hashed_value = xxhash.xxh32(str(eff_item), seed=opt_hash).intdigest() % g
            if hashed_value == best_hashed_value:
                attacked_freq[eff_item] += min_diff
        
        # 更新目标项的频数
        for target_item in T:
            target_hashed_value = xxhash.xxh32(str(target_item), seed=opt_hash).intdigest() % g
            if target_hashed_value == best_hashed_value:
                attacked_freq[target_item] += min_diff
        
        # 重新计算有效攻击项列表
        eff_items = [a for a in A if any(attacked_freq[t] > attacked_freq[a] for t in T)]
        
        n2 -= min_diff
    
    # 如果循环结束后 n2 > 0，令剩余恶意用户随机选择上述组合作为扰动值
    if n2 > 0 and optimal_combinations:
        remaining_combinations = random.choices(optimal_combinations, k=n2)
        for best_hashed_value, opt_hash in remaining_combinations:
            perturbation = (best_hashed_value, opt_hash)
            fake_data.append(perturbation)
    
    return fake_data







    
