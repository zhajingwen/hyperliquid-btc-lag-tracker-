#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取指定币种在两个K线周期下的Beta值
"""

import re
from collections import defaultdict

def extract_beta_values(log_file, target_date="2025-12-31"):
    """提取币种的Beta值"""
    
    target_coins = ['ENS/USDC:USDC', 'ETC/USDC:USDC', 'ETHFI/USDC:USDC', 
                    'GALA/USDC:USDC', 'POL/USDC:USDC', 'RUNE/USDC:USDC']
    
    coins_data = defaultdict(lambda: {
        'beta_5m_7d': None,
        'beta_1m_1d': None,
        'corr_5m_7d': None,
        'corr_1m_1d': None
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
                if current_coin not in target_coins:
                    current_coin = None
                continue
            
            if not current_coin or current_coin not in target_coins:
                continue
            
            coin = current_coin
            data = coins_data[coin]
            
            # 提取5m/7d的Beta和相关系数
            long_match = re.search(
                r'分析中间结果.*币种:\s*' + re.escape(coin) + 
                r'.*timeframe:\s*5m.*period:\s*7d.*tau_star:\s*(\d+).*相关系数:\s*([0-9.-]+).*Beta:\s*([0-9.-]+)',
                line
            )
            if long_match:
                tau, corr, beta = long_match.groups()
                data['beta_5m_7d'] = float(beta)
                data['corr_5m_7d'] = float(corr)
            
            # 提取1m/1d的Beta和相关系数
            short_match = re.search(
                r'分析中间结果.*币种:\s*' + re.escape(coin) + 
                r'.*timeframe:\s*1m.*period:\s*1d.*tau_star:\s*(\d+).*相关系数:\s*([0-9.-]+).*Beta:\s*([0-9.-]+)',
                line
            )
            if short_match:
                tau, corr, beta = short_match.groups()
                data['beta_1m_1d'] = float(beta)
                data['corr_1m_1d'] = float(corr)
    
    return coins_data

if __name__ == '__main__':
    log_file = 'hyperliquid.log'
    
    print("正在提取Beta值...")
    coins_data = extract_beta_values(log_file)
    
    print(f"\n{'='*100}")
    print("6个满足异常模式条件的币种 - 两个K线周期的Beta值详情")
    print(f"{'='*100}\n")
    
    target_coins = ['ENS/USDC:USDC', 'ETC/USDC:USDC', 'ETHFI/USDC:USDC', 
                    'GALA/USDC:USDC', 'POL/USDC:USDC', 'RUNE/USDC:USDC']
    
    print(f"{'币种':<25} {'5m/7d Beta':<15} {'1m/1d Beta':<15} {'平均Beta':<15} {'5m/7d 相关系数':<18} {'1m/1d 相关系数':<18}")
    print("-" * 100)
    
    for coin in sorted(target_coins):
        if coin not in coins_data:
            print(f"{coin:<25} {'数据未找到':<15}")
            continue
        
        data = coins_data[coin]
        beta_5m = data['beta_5m_7d']
        beta_1m = data['beta_1m_1d']
        corr_5m = data['corr_5m_7d']
        corr_1m = data['corr_1m_1d']
        
        if beta_5m is not None and beta_1m is not None:
            avg_beta = (beta_5m + beta_1m) / 2
            print(f"{coin:<25} {beta_5m:<15.4f} {beta_1m:<15.4f} {avg_beta:<15.4f} {corr_5m:<18.4f} {corr_1m:<18.4f}")
        elif beta_5m is not None:
            print(f"{coin:<25} {beta_5m:<15.4f} {'N/A':<15} {'N/A':<15} {corr_5m:<18.4f} {'N/A':<18}")
        elif beta_1m is not None:
            print(f"{coin:<25} {'N/A':<15} {beta_1m:<15.4f} {'N/A':<15} {'N/A':<18} {corr_1m:<18.4f}")
        else:
            print(f"{coin:<25} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<18} {'N/A':<18}")
    
    print(f"\n{'='*100}\n")
    print("详细分析：\n")
    
    for coin in sorted(target_coins):
        if coin not in coins_data:
            continue
        
        data = coins_data[coin]
        beta_5m = data['beta_5m_7d']
        beta_1m = data['beta_1m_1d']
        
        if beta_5m is not None and beta_1m is not None:
            avg_beta = (beta_5m + beta_1m) / 2
            print(f"{coin}:")
            print(f"  - 5分钟K线/7天周期 Beta: {beta_5m:.4f}")
            print(f"  - 1分钟K线/1天周期 Beta: {beta_1m:.4f}")
            print(f"  - 平均Beta: {avg_beta:.4f}")
            
            # 计算差异
            diff = abs(beta_5m - beta_1m)
            diff_pct = (diff / avg_beta * 100) if avg_beta != 0 else 0
            print(f"  - 两个周期Beta差异: {diff:.4f} ({diff_pct:.2f}%)")
            
            # 判断哪个周期更接近阈值
            if beta_5m >= 1.0 and beta_1m < 1.0:
                print(f"  - ⚠️  长期Beta({beta_5m:.4f})满足阈值，但短期Beta({beta_1m:.4f})不足")
            elif beta_5m < 1.0 and beta_1m >= 1.0:
                print(f"  - ⚠️  短期Beta({beta_1m:.4f})满足阈值，但长期Beta({beta_5m:.4f})不足")
            elif beta_5m >= 1.0 and beta_1m >= 1.0:
                print(f"  - ✅ 两个周期Beta都满足阈值")
            else:
                gap = 1.0 - max(beta_5m, beta_1m)
                gap_pct = (gap / max(beta_5m, beta_1m) * 100) if max(beta_5m, beta_1m) > 0 else 0
                print(f"  - ❌ 两个周期Beta都不足，最高值距阈值差: {gap:.4f} ({gap_pct:.2f}%)")
            print()



