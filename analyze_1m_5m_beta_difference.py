#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析1分钟和5分钟K线Beta收益率系数差异的原因
"""

import re
from collections import defaultdict

def analyze_beta_difference(log_file, target_date="2025-12-31"):
    """分析1分钟和5分钟K线Beta差异的原因"""
    
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
    
    print("="*100)
    print("1分钟 vs 5分钟K线 - Beta收益率系数差异分析")
    print("="*100)
    
    coins_data = analyze_beta_difference(log_file)
    
    target_coins = ['ENS/USDC:USDC', 'ETC/USDC:USDC', 'ETHFI/USDC:USDC', 
                    'GALA/USDC:USDC', 'POL/USDC:USDC', 'RUNE/USDC:USDC']
    
    print("\n## 一、实际数据对比\n")
    print(f"{'币种':<25} {'5m/7d Beta':<15} {'1m/1d Beta':<15} {'差异':<15} {'差异百分比':<15}")
    print("-" * 100)
    
    all_beta_5m = []
    all_beta_1m = []
    all_diff = []
    
    for coin in sorted(target_coins):
        if coin not in coins_data:
            continue
        
        data = coins_data[coin]
        beta_5m = data['beta_5m_7d']
        beta_1m = data['beta_1m_1d']
        
        if beta_5m is not None and beta_1m is not None:
            diff = beta_5m - beta_1m
            diff_pct = (diff / beta_1m * 100) if beta_1m != 0 else 0
            
            all_beta_5m.append(beta_5m)
            all_beta_1m.append(beta_1m)
            all_diff.append(diff)
            
            print(f"{coin:<25} {beta_5m:<15.4f} {beta_1m:<15.4f} {diff:<15.4f} {diff_pct:<15.2f}%")
    
    if all_beta_5m:
        print("\n" + "-" * 100)
        avg_5m = sum(all_beta_5m) / len(all_beta_5m)
        avg_1m = sum(all_beta_1m) / len(all_beta_1m)
        avg_diff = sum(all_diff) / len(all_diff)
        avg_diff_pct = (avg_diff / avg_1m * 100) if avg_1m != 0 else 0
        print(f"{'平均值':<25} {avg_5m:<15.4f} {avg_1m:<15.4f} {avg_diff:<15.4f} {avg_diff_pct:<15.2f}%")
    
    print("\n## 二、Beta计算公式回顾\n")
    print("**Beta收益率系数计算公式**：")
    print("```")
    print("β = Cov(BTC_returns, ALT_returns) / Var(BTC_returns)")
    print("```")
    print("\n**关键点**：")
    print("- Beta基于收益率（returns）计算，而非价格")
    print("- 收益率 = (P_t - P_{t-1}) / P_{t-1} ≈ ln(P_t / P_{t-1})（对数收益率）")
    print("- 衡量的是山寨币收益率相对于BTC收益率的波动幅度")
    
    print("\n## 三、1分钟K线Beta偏低的原因分析\n")
    
    print("### 原因1：1分钟收益率的噪声更大\n")
    print("**1分钟K线的特点**：")
    print("- 时间粒度更细，单个K线内的价格波动更容易受短期因素影响")
    print("- 市场微观结构噪声：买卖价差、订单簿深度、瞬时冲击")
    print("- 高频交易行为：算法交易、套利交易在分钟级别的频繁操作")
    print("\n**对收益率的影响**：")
    print("- 1分钟收益率序列包含更多噪声，波动性被放大")
    print("- 噪声导致协方差（Cov）和方差（Var）的计算都受到影响")
    print("- 但BTC和山寨币的噪声相关性较低，可能降低协方差")
    
    print("\n### 原因2：收益率的时间尺度效应\n")
    print("**时间尺度理论**：")
    print("- **5分钟收益率**：平滑了5个1分钟的波动，更能反映趋势性变化")
    print("- **1分钟收益率**：包含更多短期波动，可能不完全跟随BTC")
    print("\n**Beta的本质**：")
    print("- Beta衡量的是'山寨币收益率随BTC收益率变化的敏感度'")
    print("- 在1分钟级别，山寨币可能因为自身流动性、订单簿等原因，")
    print("  对BTC价格变化的反应不够及时或充分")
    print("- 在5分钟级别，短期噪声被平滑，更能体现真实的跟随关系")
    
    print("\n### 原因3：数据周期差异\n")
    print("**数据周期对比**：")
    print("| 时间粒度 | 数据周期 | 数据点数量 | 时间跨度 |")
    print("|---------|---------|-----------|---------|")
    print("| 5分钟K线 | 7天 | ~2016个点 | 7天 |")
    print("| 1分钟K线 | 1天 | ~1440个点 | 1天 |")
    print("\n**潜在影响**：")
    print("- 7天的数据更能反映长期稳定的Beta关系")
    print("- 1天的数据可能受到特定市场环境、事件的影响")
    print("- 如果当天市场波动较小或相关性较弱，Beta会偏低")
    
    print("\n### 原因4：收益率方差的尺度差异\n")
    print("**数学原理**：")
    print("- 1分钟收益率的方差 ≈ (1/5) × 5分钟收益率的方差")
    print("- 但协方差的缩放比例可能不同，导致Beta值变化")
    print("\n**实际影响**：")
    print("```")
    print("如果BTC和ALT的1分钟收益率相关性较低：")
    print("  Cov(BTC_1m, ALT_1m) < Cov(BTC_5m, ALT_5m)  (相对值)")
    print("  而 Var(BTC_1m) 可能不会等比例缩小")
    print("  结果：Beta_1m = Cov_1m / Var_1m < Beta_5m")
    print("```")
    
    print("\n### 原因5：市场微观结构延迟\n")
    print("**延迟传导机制**：")
    print("- 在1分钟级别，BTC价格变化可能不会立即传导到山寨币")
    print("- 山寨币可能延迟几秒到几十秒才反应BTC的变化")
    print("- 这种延迟导致1分钟K线内的收益率相关性降低")
    print("\n**对Beta的影响**：")
    print("- 如果存在延迟，1分钟级别的协方差会降低")
    print("- 5分钟K线已经包含了延迟，更能反映完整的价格传导")
    print("- 这与系统检测'延迟传导模式'的逻辑一致")
    
    print("\n## 四、实际案例分析（ETHFI）\n")
    
    ethfi_data = coins_data.get('ETHFI/USDC:USDC')
    if ethfi_data and ethfi_data['beta_5m_7d'] is not None and ethfi_data['beta_1m_1d'] is not None:
        beta_5m = ethfi_data['beta_5m_7d']
        beta_1m = ethfi_data['beta_1m_1d']
        corr_5m = ethfi_data['corr_5m_7d']
        corr_1m = ethfi_data['corr_1m_1d']
        
        print(f"**ETHFI/USDC:USDC 数据**：")
        print(f"- 5分钟K线/7天周期：")
        print(f"  * Beta: {beta_5m:.4f}")
        print(f"  * 相关系数: {corr_5m:.4f}")
        print(f"- 1分钟K线/1天周期：")
        print(f"  * Beta: {beta_1m:.4f}")
        print(f"  * 相关系数: {corr_1m:.4f}")
        print(f"- 差异：Beta相差 {beta_5m - beta_1m:.4f} ({((beta_5m - beta_1m)/beta_1m*100):.2f}%)")
        
        print(f"\n**解读**：")
        print(f"- 长期Beta({beta_5m:.4f}) > 1.0，说明ETHFI在长期内波动幅度大于BTC")
        print(f"- 短期Beta({beta_1m:.4f}) < 1.0，说明在1分钟级别，ETHFI对BTC的跟随性较弱")
        print(f"- 相关系数差异：长期({corr_5m:.4f}) > 短期({corr_1m:.4f})")
        print(f"- 这表明ETHFI与BTC的关系在短期（1分钟）存在延迟或噪声干扰")
    
    print("\n## 五、这是正常现象还是问题？\n")
    
    print("### 这是正常的市场现象\n")
    print("**为什么正常**：")
    print("1. **多时间尺度Beta差异是普遍现象**")
    print("   - 不同时间尺度的Beta值不同是正常的")
    print("   - 长期Beta反映长期关系，短期Beta反映短期关系")
    print("2. **符合延迟传导理论**")
    print("   - 系统正是通过检测这种差异来发现套利机会")
    print("   - 长期高相关+短期低相关 = 潜在套利机会")
    print("3. **市场微观结构影响**")
    print("   - 1分钟级别的噪声、流动性、订单簿深度等因素是真实存在的")
    print("   - 这些因素会影响短期Beta，但不影响长期Beta")
    
    print("\n### 可能的问题\n")
    print("**当前设计的问题**：")
    print("1. **阈值设计**：使用平均Beta >= 1.0作为阈值")
    print("   - 如果长期Beta > 1.0，短期Beta < 1.0，平均Beta可能 < 1.0")
    print("   - 这可能过滤掉一些长期波动足够、短期存在延迟的优质信号")
    print("2. **时间尺度不匹配**：")
    print("   - 长期Beta（5m/7d）反映7天的关系")
    print("   - 短期Beta（1m/1d）反映1天的关系")
    print("   - 两者平均可能不能准确反映套利机会的质量")
    
    print("\n## 六、建议方案\n")
    
    print("### 方案1：调整Beta阈值判断逻辑（推荐）\n")
    print("**当前逻辑**：")
    print("```python")
    print("avg_beta = (beta_5m + beta_1m) / 2")
    print("if avg_beta < 1.0:  # 过滤")
    print("```")
    print("\n**建议逻辑**：")
    print("```python")
    print("# 方案A：任一周期满足即可")
    print("if beta_5m >= 1.0 or beta_1m >= 1.0:")
    print("    # 通过")
    print("")
    print("# 方案B：长期Beta满足即可（更推荐）")
    print("if beta_5m >= 1.0:  # 长期Beta反映长期关系，更可靠")
    print("    # 通过")
    print("")
    print("# 方案C：降低平均阈值")
    print("if avg_beta >= 0.85:  # 从1.0降到0.85")
    print("    # 通过")
    print("```")
    
    print("\n### 方案2：分别评估长期和短期Beta\n")
    print("**逻辑**：")
    print("- 长期Beta（5m/7d）>= 1.0：确保长期波动足够")
    print("- 短期Beta（1m/1d）不作为硬性要求，只作为参考")
    print("- 这样可以保留长期波动足够、短期存在延迟的信号")
    
    print("\n### 方案3：使用加权平均\n")
    print("**逻辑**：")
    print("```python")
    print("# 长期Beta权重更高（如0.7），短期Beta权重较低（如0.3）")
    print("weighted_beta = 0.7 * beta_5m + 0.3 * beta_1m")
    print("if weighted_beta >= 1.0:")
    print("    # 通过")
    print("```")
    
    print("\n## 七、结论\n")
    
    print("### 核心结论\n")
    print("1. ✅ **1分钟K线Beta偏低是正常现象**")
    print("   - 主要由1分钟级别的噪声、延迟传导、市场微观结构等因素导致")
    print("   - 符合市场运行规律和延迟传导理论")
    print("\n2. ✅ **5分钟K线Beta更能反映长期关系**")
    print("   - 平滑了短期噪声，更能体现真实的跟随关系")
    print("   - 使用7天数据，更能反映稳定的Beta关系")
    print("\n3. ⚠️ **当前阈值设计可能过于严格**")
    print("   - 使用平均Beta >= 1.0可能过滤掉一些优质信号")
    print("   - 建议考虑使用长期Beta作为主要判断依据")
    print("\n4. 💡 **建议优化**")
    print("   - 优先考虑方案1B：使用长期Beta（5m/7d）>= 1.0作为判断依据")
    print("   - 或者方案1C：将平均Beta阈值降低到0.85")
    print("   - 这样可以保留更多长期波动足够、短期存在延迟的套利机会")

