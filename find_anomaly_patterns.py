#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析满足异常模式检测条件的币种
"""

import re
from collections import defaultdict

def analyze_anomaly_patterns(log_file, target_date="2025-12-31"):
    """分析满足异常模式检测条件的币种"""
    
    coins_data = defaultdict(lambda: {
        'corr_5m_7d': None,      # 长期相关系数（5m/7d）
        'corr_1m_1d': None,      # 短期相关系数（1m/1d）
        'tau_star_short': None,  # 短期最优延迟
        'has_data': True
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
            
            # 数据不存在
            if '币种数据不存在' in line or '交易对无历史数据' in line:
                data['has_data'] = False
                continue
            
            # 提取长期相关系数（5m/7d）
            long_match = re.search(
                r'分析中间结果.*币种:\s*' + re.escape(coin) + 
                r'.*timeframe:\s*5m.*period:\s*7d.*tau_star:\s*(\d+).*相关系数:\s*([0-9.-]+)',
                line
            )
            if long_match:
                tau, corr = long_match.groups()
                data['corr_5m_7d'] = float(corr)
            
            # 提取短期相关系数和tau_star（1m/1d）
            short_match = re.search(
                r'分析中间结果.*币种:\s*' + re.escape(coin) + 
                r'.*timeframe:\s*1m.*period:\s*1d.*tau_star:\s*(\d+).*相关系数:\s*([0-9.-]+)',
                line
            )
            if short_match:
                tau, corr = short_match.groups()
                data['tau_star_short'] = int(tau)
                data['corr_1m_1d'] = float(corr)
    
    # 筛选满足条件的币种
    condition_a_coins = []  # 条件A：跨周期相关性破裂
    condition_b_coins = []  # 条件B：延迟传导模式
    
    for coin, data in coins_data.items():
        if not data['has_data']:
            continue
        
        if data['corr_5m_7d'] is None or data['corr_1m_1d'] is None:
            continue
        
        long_corr = data['corr_5m_7d']
        short_corr = data['corr_1m_1d']
        tau_star = data['tau_star_short'] if data['tau_star_short'] is not None else 0
        
        # 计算真实的相关系数差值
        corr_diff = long_corr - short_corr
        
        # 条件A：跨周期相关性破裂
        # 长期相关系数 > 0.6 AND 短期相关系数 < 0.4 AND 差值 > 0.38
        if long_corr > 0.6 and short_corr < 0.4 and corr_diff > 0.38:
            condition_a_coins.append({
                'coin': coin,
                'long_corr': long_corr,
                'short_corr': short_corr,
                'corr_diff': corr_diff,
                'tau_star': tau_star
            })
        
        # 条件B：延迟传导模式
        # 长期相关系数 > 0.6 AND 短期存在延迟（τ* > 0）
        if long_corr > 0.6 and tau_star > 0:
            condition_b_coins.append({
                'coin': coin,
                'long_corr': long_corr,
                'short_corr': short_corr,
                'corr_diff': corr_diff,
                'tau_star': tau_star
            })
    
    # 去重（条件B可能包含条件A中的币种）
    all_anomaly_coins = set()
    for item in condition_a_coins:
        all_anomaly_coins.add(item['coin'])
    for item in condition_b_coins:
        all_anomaly_coins.add(item['coin'])
    
    return condition_a_coins, condition_b_coins, all_anomaly_coins

if __name__ == '__main__':
    log_file = 'hyperliquid.log'
    
    print("正在分析日志文件...")
    condition_a, condition_b, all_coins = analyze_anomaly_patterns(log_file)
    
    print(f"\n{'='*80}")
    print("满足异常模式检测条件的币种分析结果")
    print(f"{'='*80}\n")
    
    print(f"条件A（跨周期相关性破裂）满足数量: {len(condition_a)}")
    print("要求: 长期相关系数 > 0.6 AND 短期相关系数 < 0.4 AND 差值 > 0.38\n")
    
    if condition_a:
        print("满足条件A的币种:")
        print(f"{'币种':<25} {'长期相关系数':<15} {'短期相关系数':<15} {'差值':<15} {'τ*':<10}")
        print("-" * 80)
        for item in sorted(condition_a, key=lambda x: x['corr_diff'], reverse=True):
            print(f"{item['coin']:<25} {item['long_corr']:<15.4f} {item['short_corr']:<15.4f} "
                  f"{item['corr_diff']:<15.4f} {item['tau_star']:<10}")
    else:
        print("❌ 没有币种满足条件A\n")
    
    print(f"\n{'='*80}\n")
    
    print(f"条件B（延迟传导模式）满足数量: {len(condition_b)}")
    print("要求: 长期相关系数 > 0.6 AND 短期存在延迟（τ* > 0）\n")
    
    if condition_b:
        print("满足条件B的币种:")
        print(f"{'币种':<25} {'长期相关系数':<15} {'短期相关系数':<15} {'差值':<15} {'τ*':<10}")
        print("-" * 80)
        for item in sorted(condition_b, key=lambda x: x['tau_star'], reverse=True):
            print(f"{item['coin']:<25} {item['long_corr']:<15.4f} {item['short_corr']:<15.4f} "
                  f"{item['corr_diff']:<15.4f} {item['tau_star']:<10}")
    else:
        print("❌ 没有币种满足条件B\n")
    
    print(f"\n{'='*80}\n")
    print(f"总计满足异常模式检测条件的币种数（去重后）: {len(all_coins)}")
    
    if all_coins:
        print("\n所有满足条件的币种列表（按字母排序）:")
        for coin in sorted(all_coins):
            print(f"  - {coin}")



