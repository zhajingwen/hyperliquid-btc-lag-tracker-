#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细分析满足异常模式检测条件的币种，查看为什么没有触发告警
"""

import re
from collections import defaultdict

def analyze_detailed_info(log_file, target_date="2025-12-31"):
    """详细分析满足异常模式的币种"""
    
    anomaly_coins = ['ENS/USDC:USDC', 'ETC/USDC:USDC', 'ETHFI/USDC:USDC', 
                     'GALA/USDC:USDC', 'POL/USDC:USDC', 'RUNE/USDC:USDC']
    
    coins_data = defaultdict(lambda: {
        'corr_5m_7d': None,
        'corr_1m_1d': None,
        'tau_star_short': None,
        'avg_beta': None,
        'zscore': None,
        'stationarity': None,
        'p_value': None,
        'filter_reasons': []
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
                if current_coin not in anomaly_coins:
                    current_coin = None
                continue
            
            if not current_coin or current_coin not in anomaly_coins:
                continue
            
            coin = current_coin
            data = coins_data[coin]
            
            # 提取长期相关系数（5m/7d）
            long_match = re.search(
                r'分析中间结果.*币种:\s*' + re.escape(coin) + 
                r'.*timeframe:\s*5m.*period:\s*7d.*tau_star:\s*(\d+).*相关系数:\s*([0-9.-]+).*Beta:\s*([0-9.-]+)',
                line
            )
            if long_match:
                tau, corr, beta = long_match.groups()
                data['corr_5m_7d'] = float(corr)
            
            # 提取短期相关系数和tau_star（1m/1d）
            short_match = re.search(
                r'分析中间结果.*币种:\s*' + re.escape(coin) + 
                r'.*timeframe:\s*1m.*period:\s*1d.*tau_star:\s*(\d+).*相关系数:\s*([0-9.-]+).*Beta:\s*([0-9.-]+)',
                line
            )
            if short_match:
                tau, corr, beta = short_match.groups()
                data['tau_star_short'] = int(tau)
                data['corr_1m_1d'] = float(corr)
            
            # Beta收益率系数不满足要求
            beta_match = re.search(
                r'Beta收益率系数不满足要求.*平均Beta:\s*([0-9.]+)\s*<\s*1.*币种:\s*' + re.escape(coin),
                line
            )
            if beta_match:
                data['avg_beta'] = float(beta_match.group(1))
                data['filter_reasons'].append(f'Beta收益率系数不足（{data["avg_beta"]:.4f} < 1.0）')
            
            # 平稳性检验
            stationarity_match = re.search(
                r'平稳性检验(失败|通过).*p-value:\s*([0-9.]+).*等级:\s*(强平稳|弱平稳|非平稳).*币种:\s*' + re.escape(coin),
                line
            )
            if stationarity_match:
                status, p_value, level = stationarity_match.groups()
                data['p_value'] = float(p_value)
                data['stationarity'] = level
                if status == '失败':
                    data['filter_reasons'].append(f'平稳性检验失败（{level}，p-value={data["p_value"]:.4f}）')
            
            # Z-score验证未通过
            zscore_match = re.search(
                r'Z-score 验证未通过.*币种:\s*' + re.escape(coin) + 
                r'.*Z-score:\s*([0-9.-]+)的绝对值\s*<\s*2.0',
                line
            )
            if zscore_match:
                data['zscore'] = float(zscore_match.group(1))
                data['filter_reasons'].append(f'Z-score未通过（|{data["zscore"]:.2f}| < 2.0）')
    
    return coins_data

if __name__ == '__main__':
    log_file = 'hyperliquid.log'
    
    print("正在详细分析满足异常模式条件的币种...")
    coins_data = analyze_detailed_info(log_file)
    
    print(f"\n{'='*80}")
    print("满足异常模式检测条件的币种详细分析")
    print(f"{'='*80}\n")
    
    anomaly_coins = ['ENS/USDC:USDC', 'ETC/USDC:USDC', 'ETHFI/USDC:USDC', 
                     'GALA/USDC:USDC', 'POL/USDC:USDC', 'RUNE/USDC:USDC']
    
    for coin in sorted(anomaly_coins):
        if coin not in coins_data:
            continue
        
        data = coins_data[coin]
        corr_diff = data['corr_5m_7d'] - data['corr_1m_1d'] if data['corr_5m_7d'] and data['corr_1m_1d'] else None
        
        print(f"\n币种: {coin}")
        print("-" * 80)
        long_corr_str = f"{data['corr_5m_7d']:.4f}" if data['corr_5m_7d'] is not None else 'N/A'
        short_corr_str = f"{data['corr_1m_1d']:.4f}" if data['corr_1m_1d'] is not None else 'N/A'
        diff_str = f"{corr_diff:.4f}" if corr_diff is not None else 'N/A'
        tau_str = str(data['tau_star_short']) if data['tau_star_short'] is not None else 'N/A'
        beta_str = f"{data['avg_beta']:.4f}" if data['avg_beta'] is not None else 'N/A'
        zscore_str = f"{data['zscore']:.2f}" if data['zscore'] is not None else 'N/A'
        stationarity_str = data['stationarity'] if data['stationarity'] else 'N/A'
        pvalue_str = f"{data['p_value']:.4f}" if data['p_value'] is not None else 'N/A'
        
        print(f"长期相关系数（5m/7d）: {long_corr_str}")
        print(f"短期相关系数（1m/1d）: {short_corr_str}")
        print(f"相关系数差值: {diff_str}")
        print(f"短期最优延迟τ*: {tau_str}")
        print(f"平均Beta: {beta_str}")
        print(f"Z-score: {zscore_str}")
        print(f"平稳性: {stationarity_str}")
        print(f"p-value: {pvalue_str}")
        
        if data['filter_reasons']:
            print("\n未触发告警的原因:")
            for reason in data['filter_reasons']:
                print(f"  ❌ {reason}")
        else:
            print("\n⚠️ 未找到明确的过滤原因（可能需要进一步检查）")

