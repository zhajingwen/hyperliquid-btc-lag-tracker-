#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析不同时间级别Beta的分布规律
"""

import re
from collections import defaultdict

def analyze_beta_by_timeframe(log_file, target_date="2025-12-31"):
    """分析1m和5m两个时间级别的Beta分布"""

    coins_data = defaultdict(lambda: {
        'beta_5m': None,
        'beta_1m': None,
        'coin': None
    })

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.startswith(target_date):
                continue

            # 提取Beta数据
            match = re.search(
                r'分析中间结果.*币种:\s*([A-Z0-9/]+:USDC).*timeframe:\s*(\w+).*Beta:\s*([0-9.]+)',
                line
            )
            if match:
                coin, timeframe, beta = match.groups()
                beta = float(beta)

                if timeframe == '5m':
                    coins_data[coin]['beta_5m'] = beta
                elif timeframe == '1m':
                    coins_data[coin]['beta_1m'] = beta
                coins_data[coin]['coin'] = coin

    # 筛选同时有两个时间级别数据的币种
    valid_coins = []
    for coin, data in coins_data.items():
        if data['beta_5m'] is not None and data['beta_1m'] is not None:
            valid_coins.append({
                'coin': coin,
                'beta_5m': data['beta_5m'],
                'beta_1m': data['beta_1m'],
                'avg_beta': (data['beta_5m'] + data['beta_1m']) / 2,
                'diff': data['beta_5m'] - data['beta_1m'],  # 5m - 1m
                'ratio': data['beta_1m'] / data['beta_5m'] if data['beta_5m'] != 0 else 0
            })

    return valid_coins

if __name__ == '__main__':
    log_file = 'hyperliquid.log'

    print("正在分析Beta数据...")
    coins = analyze_beta_by_timeframe(log_file)

    if not coins:
        print("❌ 未找到有效数据")
        exit(1)

    print(f"\n{'='*100}")
    print("Beta收益率系数时间级别分析")
    print(f"{'='*100}\n")

    print(f"总计分析币种数: {len(coins)}\n")

    # 统计1m < 5m的情况
    one_m_smaller = [c for c in coins if c['beta_1m'] < c['beta_5m']]
    one_m_larger = [c for c in coins if c['beta_1m'] >= c['beta_5m']]

    print(f"📊 分布统计:")
    print(f"  - 1m Beta < 5m Beta: {len(one_m_smaller)} 个币种 ({len(one_m_smaller)/len(coins)*100:.1f}%)")
    print(f"  - 1m Beta >= 5m Beta: {len(one_m_larger)} 个币种 ({len(one_m_larger)/len(coins)*100:.1f}%)")

    # 计算平均差值
    avg_diff = sum(c['diff'] for c in coins) / len(coins)
    avg_ratio = sum(c['ratio'] for c in coins) / len(coins)

    print(f"\n📈 平均统计:")
    print(f"  - 平均差值 (5m - 1m): {avg_diff:+.4f}")
    print(f"  - 平均比率 (1m / 5m): {avg_ratio:.2%}")

    # 显示详细数据（按差值排序）
    print(f"\n{'='*100}")
    print("详细数据（按5m-1m差值降序排序）")
    print(f"{'='*100}\n")
    print(f"{'币种':<25} {'5m Beta':<12} {'1m Beta':<12} {'平均Beta':<12} {'差值(5m-1m)':<15} {'比率(1m/5m)':<12}")
    print("-" * 100)

    for item in sorted(coins, key=lambda x: x['diff'], reverse=True):
        print(f"{item['coin']:<25} {item['beta_5m']:<12.4f} {item['beta_1m']:<12.4f} "
              f"{item['avg_beta']:<12.4f} {item['diff']:<+15.4f} {item['ratio']:<12.2%}")

    # 找出被平均Beta过滤但5m Beta满足条件的币种
    print(f"\n{'='*100}")
    print("被平均Beta过滤但5m Beta >= 1.0的币种")
    print(f"{'='*100}\n")

    filtered_coins = [c for c in coins if c['avg_beta'] < 1.0 and c['beta_5m'] >= 1.0]
    if filtered_coins:
        print(f"找到 {len(filtered_coins)} 个币种：\n")
        print(f"{'币种':<25} {'5m Beta':<12} {'1m Beta':<12} {'平均Beta':<12} {'被过滤原因'}")
        print("-" * 100)
        for item in sorted(filtered_coins, key=lambda x: x['avg_beta'], reverse=True):
            print(f"{item['coin']:<25} {item['beta_5m']:<12.4f} {item['beta_1m']:<12.4f} "
                  f"{item['avg_beta']:<12.4f} {'1m Beta太低拉低平均值'}")
    else:
        print("✅ 没有币种因此被过滤")

    # 找出1m和5m都 >= 0.85的币种（推荐阈值）
    print(f"\n{'='*100}")
    print("如果使用0.85阈值，两个时间级别都满足的币种")
    print(f"{'='*100}\n")

    both_good = [c for c in coins if c['beta_5m'] >= 0.85 and c['beta_1m'] >= 0.85]
    if both_good:
        print(f"找到 {len(both_good)} 个币种\n")
    else:
        print("❌ 没有币种满足")
