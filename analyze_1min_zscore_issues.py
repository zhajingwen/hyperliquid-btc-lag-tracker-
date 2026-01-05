#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析使用1分钟K线计算Z-score的潜在问题
"""

import re
from collections import defaultdict

def analyze_zscore_window_issue(log_file, target_date="2025-12-31"):
    """分析Z-score窗口大小和1分钟K线的问题"""
    
    # 提取所有有Z-score的币种数据
    coins_data = defaultdict(lambda: {
        'zscore': None,
        'stationarity': None,
        'p_value': None,
        'corr_5m_7d': None,
        'corr_1m_1d': None,
        'beta_5m_7d': None,
        'beta_1m_1d': None
    })
    
    current_coin = None
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.startswith(target_date):
                continue
            
            # 检查币种
            coin_match = re.search(r'检查币种:\s*([A-Z0-9/]+:USDC)', line)
            if coin_match:
                current_coin = coin_match.group(1)
                continue
            
            if not current_coin:
                continue
            
            coin = current_coin
            data = coins_data[coin]
            
            # 提取Z-score
            zscore_match = re.search(
                r'Z-score 验证未通过.*币种:\s*' + re.escape(coin) + 
                r'.*Z-score:\s*([0-9.-]+)的绝对值',
                line
            )
            if zscore_match:
                data['zscore'] = float(zscore_match.group(1))
            
            # 提取平稳性
            stationarity_match = re.search(
                r'平稳性检验(失败|通过).*p-value:\s*([0-9.]+).*等级:\s*(强平稳|弱平稳|非平稳).*币种:\s*' + re.escape(coin),
                line
            )
            if stationarity_match:
                data['p_value'] = float(stationarity_match.group(2))
                data['stationarity'] = stationarity_match.group(3)
            
            # 提取相关系数
            corr_match = re.search(
                r'相关系数检测.*币种:\s*' + re.escape(coin) + 
                r'.*短期最小:\s*([0-9.]+).*长期最大:\s*([0-9.]+)',
                line
            )
            if corr_match:
                data['corr_1m_1d'] = float(corr_match.group(1))
                data['corr_5m_7d'] = float(corr_match.group(2))
    
    return coins_data

if __name__ == '__main__':
    log_file = 'hyperliquid.log'
    
    print("分析使用1分钟K线计算Z-score的潜在问题...\n")
    coins_data = analyze_zscore_window_issue(log_file)
    
    # 统计有Z-score的币种
    coins_with_zscore = {coin: data for coin, data in coins_data.items() 
                        if data['zscore'] is not None}
    
    print(f"找到 {len(coins_with_zscore)} 个有Z-score数据的币种\n")
    
    print("="*100)
    print("1分钟K线 vs 5分钟K线 - Z-score计算对比分析")
    print("="*100)
    
    print("\n## 一、当前配置\n")
    print("**Z-score计算配置**：")
    print("- 数据周期：1分钟K线 / 1天数据（1m/1d）")
    print("- 统计窗口：ZSCORE_WINDOW = 30（最近30个1分钟K线）")
    print("- 时间跨度：30分钟")
    print("- Beta窗口：BETA_WINDOW = 100（最近100个1分钟K线）")
    print("- Beta时间跨度：100分钟（约1.67小时）")
    
    print("\n## 二、潜在问题分析\n")
    
    print("### 问题1：窗口时间跨度太短\n")
    print("**当前情况**：")
    print("- 使用最近30个1分钟K线 = 30分钟的数据")
    print("- 在1天（1440分钟）的数据中，只使用了2.1%的数据")
    print("\n**潜在影响**：")
    print("- ⚠️ 样本量小：只有30个数据点，统计量可能不稳定")
    print("- ⚠️ 容易受噪声影响：短期波动、市场微观结构噪声可能放大标准差")
    print("- ⚠️ 代表性不足：30分钟的数据可能不能代表1天的整体特征")
    
    print("\n### 问题2：1分钟K线的噪声问题\n")
    print("**1分钟K线的特点**：")
    print("- 数据点密集：1天有1440个数据点")
    print("- 噪声更大：买卖价差、滑点、市场微观结构噪声")
    print("- 波动更剧烈：短期价格波动可能被放大")
    print("\n**对Z-score的影响**：")
    print("- 如果标准差被噪声放大 → Z-score分母变大 → Z-score被低估")
    print("- 如果均值被异常值影响 → Z-score计算不准确")
    
    print("\n### 问题3：与5分钟K线对比\n")
    print("**如果使用5分钟K线（ZSCORE_WINDOW=30）**：")
    print("- 30个5分钟K线 = 150分钟 = 2.5小时")
    print("- 时间跨度更长，统计量更稳定")
    print("- 噪声相对较小（5分钟数据平滑了短期波动）")
    print("- 但时间分辨率降低，可能错过短期机会")
    
    print("\n## 三、数据统计\n")
    
    if coins_with_zscore:
        zscores = [abs(data['zscore']) for data in coins_with_zscore.values() 
                  if data['zscore'] is not None]
        
        print(f"**Z-score分布统计**（{len(zscores)}个币种）：")
        print(f"- 平均|Z-score|: {sum(zscores)/len(zscores):.2f}")
        print(f"- 最大|Z-score|: {max(zscores):.2f}")
        print(f"- 最小|Z-score|: {min(zscores):.2f}")
        print(f"- 中位数|Z-score|: {sorted(zscores)[len(zscores)//2]:.2f}")
        
        # 统计接近阈值的
        near_threshold = [z for z in zscores if 1.5 <= z < 2.0]
        print(f"\n**接近阈值（1.5 ≤ |Z| < 2.0）的币种数**: {len(near_threshold)}")
        if near_threshold:
            print(f"- 平均|Z-score|: {sum(near_threshold)/len(near_threshold):.2f}")
            print(f"- 这些币种可能因为1分钟K线的噪声导致Z-score被低估")
    
    print("\n## 四、建议方案\n")
    
    print("### 方案1：增加Z-score窗口大小（推荐）\n")
    print("**当前**: ZSCORE_WINDOW = 30（30分钟）")
    print("**建议**: ZSCORE_WINDOW = 60-120（1-2小时）")
    print("\n**优点**：")
    print("- 使用更多数据点，统计量更稳定")
    print("- 减少短期噪声的影响")
    print("- 时间跨度更长，更能代表短期趋势")
    print("\n**缺点**：")
    print("- 响应速度稍慢（但仍比5分钟K线快）")
    
    print("\n### 方案2：使用5分钟K线计算Z-score\n")
    print("**建议**: 使用5分钟K线/1天数据，ZSCORE_WINDOW = 30-60")
    print("\n**优点**：")
    print("- 噪声更小，统计量更稳定")
    print("- 30个5分钟K线 = 150分钟，时间跨度更长")
    print("- 与长期分析（5m/7d）使用相同的时间粒度")
    print("\n**缺点**：")
    print("- 时间分辨率降低，可能错过分钟级别的机会")
    print("- 与异常模式检测的短期周期（1m/1d）不一致")
    
    print("\n### 方案3：双时间粒度Z-score（折中方案）\n")
    print("**建议**: 同时计算两个Z-score")
    print("- Z-score_1m: 基于1分钟K线（快速响应）")
    print("- Z-score_5m: 基于5分钟K线（更稳定）")
    print("- 触发条件：任一Z-score ≥ 阈值")
    print("\n**优点**：")
    print("- 兼顾响应速度和稳定性")
    print("- 可以对比验证信号强度")
    print("\n**缺点**：")
    print("- 计算复杂度增加")
    
    print("\n### 方案4：对1分钟K线数据进行平滑处理\n")
    print("**建议**: 在计算Z-score前对价差序列进行平滑")
    print("- 使用移动平均（MA）或指数移动平均（EMA）")
    print("- 减少短期噪声，保留趋势信息")
    print("\n**优点**：")
    print("- 保持1分钟K线的高时间分辨率")
    print("- 减少噪声对标准差的影响")
    print("\n**缺点**：")
    print("- 可能延迟信号检测")
    
    print("\n## 五、结论\n")
    print("**使用1分钟K线计算Z-score的潜在问题**：")
    print("1. ✅ **窗口时间跨度短**：30分钟可能不足以代表短期趋势")
    print("2. ✅ **噪声影响**：1分钟K线的噪声可能放大标准差，导致Z-score被低估")
    print("3. ✅ **样本量小**：30个数据点可能统计不稳定")
    print("\n**建议**：")
    print("- 优先考虑**增加ZSCORE_WINDOW到60-120**（使用1-2小时的数据）")
    print("- 或者考虑**使用5分钟K线**计算Z-score（如果时间分辨率要求不高）")
    print("- 保持1分钟K线但增加窗口，可以在保持响应速度的同时提高稳定性")



