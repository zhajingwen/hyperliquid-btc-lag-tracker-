#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取ETHFI币种的详细指标数据
"""

import re
from collections import defaultdict

def extract_ethfi_details(log_file, target_date="2025-12-31", coin="ETHFI/USDC:USDC"):
    """提取ETHFI的详细数据"""
    
    data = {
        '5m_7d': {},
        '1m_1d': {},
        'summary': {},
        'all_lines': []
    }
    
    current_coin = None
    collecting = False
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.startswith(target_date):
                continue
            
            # 检查币种
            if "检查币种:" in line:
                match = re.search(r'检查币种:\s*([A-Z0-9/]+:USDC)', line)
                if match:
                    current_coin = match.group(1)
                    collecting = (current_coin == coin)
                    if collecting:
                        data['all_lines'].append(("检查币种", line.strip()))
                    continue
            
            if not collecting or current_coin != coin:
                continue
            
            # 收集所有相关日志
            if coin in line:
                data['all_lines'].append(("相关日志", line.strip()))
            
            # 提取5m/7d数据
            if "timeframe: 5m" in line and "period: 7d" in line and coin in line:
                match = re.search(
                    r'分析中间结果.*tau_star:\s*(\d+).*相关系数:\s*([0-9.-]+).*Beta:\s*([0-9.-]+)',
                    line
                )
                if match:
                    tau, corr, beta = match.groups()
                    data['5m_7d'] = {
                        'tau_star': int(tau),
                        'correlation': float(corr),
                        'beta': float(beta)
                    }
            
            # 提取1m/1d数据
            if "timeframe: 1m" in line and "period: 1d" in line and coin in line:
                match = re.search(
                    r'分析中间结果.*tau_star:\s*(\d+).*相关系数:\s*([0-9.-]+).*Beta:\s*([0-9.-]+)',
                    line
                )
                if match:
                    tau, corr, beta = match.groups()
                    data['1m_1d'] = {
                        'tau_star': int(tau),
                        'correlation': float(corr),
                        'beta': float(beta)
                    }
            
            # 提取相关系数检测结果
            if "相关系数检测" in line and coin in line:
                match = re.search(
                    r'是否异常:\s*(True|False).*相关系数差值:\s*([0-9.-]+).*短期最小:\s*([0-9.-]+).*长期最大:\s*([0-9.-]+)',
                    line
                )
                if match:
                    is_anomaly, diff, min_short, max_long = match.groups()
                    data['summary']['correlation_check'] = {
                        'is_anomaly': is_anomaly == 'True',
                        'diff': float(diff),
                        'min_short': float(min_short),
                        'max_long': float(max_long)
                    }
            
            # 提取Beta检查结果
            if "Beta收益率系数不满足要求" in line and coin in line:
                match = re.search(r'平均Beta:\s*([0-9.-]+)\s*<\s*(\d+)', line)
                if match:
                    avg_beta, threshold = match.groups()
                    data['summary']['beta_check'] = {
                        'avg_beta': float(avg_beta),
                        'threshold': int(threshold),
                        'passed': False
                    }
            
            # 提取平稳性检验结果
            if "平稳性检验" in line and coin in line:
                match = re.search(
                    r'平稳性检验(失败|通过).*ADF统计量:\s*([0-9.-]+).*p-value:\s*([0-9.-]+).*等级:\s*(强平稳|弱平稳|非平稳)',
                    line
                )
                if match:
                    status, adf_stat, p_value, level = match.groups()
                    data['summary']['stationarity'] = {
                        'status': status,
                        'adf_statistic': float(adf_stat),
                        'p_value': float(p_value),
                        'level': level,
                        'passed': status == '通过'
                    }
            
            # 提取Z-score结果
            if "Z-score" in line and coin in line:
                if "验证未通过" in line:
                    match = re.search(
                        r'Z-score:\s*([0-9.-]+)的绝对值\s*<\s*([0-9.-]+).*平稳性:\s*(强平稳|弱平稳)',
                        line
                    )
                    if match:
                        zscore, threshold, stationarity = match.groups()
                        data['summary']['zscore'] = {
                            'value': float(zscore),
                            'threshold': float(threshold),
                            'passed': False,
                            'stationarity': stationarity
                        }
                elif "验证通过" in line:
                    match = re.search(r'Z-score:\s*([0-9.-]+)', line)
                    if match:
                        zscore = float(match.group(1))
                        if 'zscore' not in data['summary']:
                            data['summary']['zscore'] = {
                                'value': zscore,
                                'passed': True
                            }
    
    return data

if __name__ == '__main__':
    log_file = 'hyperliquid.log'
    coin = 'ETHFI/USDC:USDC'
    
    print(f"正在提取 {coin} 的详细指标数据...\n")
    data = extract_ethfi_details(log_file, coin=coin)
    
    print(f"{'='*100}")
    print(f"{coin} - 完整指标数据分析")
    print(f"{'='*100}\n")
    
    # 显示两个周期的数据
    print("## 一、两个K线周期的分析结果\n")
    
    if data['5m_7d']:
        print("### 5分钟K线 / 7天周期（长期）")
        print(f"- 最优延迟 τ*: {data['5m_7d']['tau_star']}")
        print(f"- 相关系数: {data['5m_7d']['correlation']:.4f}")
        print(f"- Beta收益率系数: {data['5m_7d']['beta']:.4f}")
        if data['5m_7d']['beta'] >= 1.0:
            print(f"  ✅ Beta满足阈值（≥1.0）")
        else:
            print(f"  ❌ Beta不足（<1.0），距阈值差: {1.0 - data['5m_7d']['beta']:.4f}")
        print()
    
    if data['1m_1d']:
        print("### 1分钟K线 / 1天周期（短期）")
        print(f"- 最优延迟 τ*: {data['1m_1d']['tau_star']}")
        print(f"- 相关系数: {data['1m_1d']['correlation']:.4f}")
        print(f"- Beta收益率系数: {data['1m_1d']['beta']:.4f}")
        if data['1m_1d']['beta'] >= 1.0:
            print(f"  ✅ Beta满足阈值（≥1.0）")
        else:
            print(f"  ❌ Beta不足（<1.0），距阈值差: {1.0 - data['1m_1d']['beta']:.4f}")
        print()
    
    # 计算平均值
    if data['5m_7d'] and data['1m_1d']:
        avg_beta = (data['5m_7d']['beta'] + data['1m_1d']['beta']) / 2
        corr_diff = data['5m_7d']['correlation'] - data['1m_1d']['correlation']
        print("### 汇总数据")
        print(f"- 平均Beta: {avg_beta:.4f}")
        if avg_beta >= 1.0:
            print(f"  ✅ 平均Beta满足阈值（≥1.0）")
        else:
            gap = 1.0 - avg_beta
            gap_pct = (gap / avg_beta * 100) if avg_beta > 0 else 0
            print(f"  ❌ 平均Beta不足（<1.0），距阈值差: {gap:.4f} ({gap_pct:.2f}%)")
        print(f"- 相关系数差值: {corr_diff:.4f}")
        print(f"- Beta差异: {abs(data['5m_7d']['beta'] - data['1m_1d']['beta']):.4f}")
        print()
    
    # 显示汇总结果
    print(f"{'='*100}\n")
    print("## 二、异常模式检测结果\n")
    
    if 'correlation_check' in data['summary']:
        cc = data['summary']['correlation_check']
        print(f"### 相关系数检测")
        print(f"- 是否异常: {'是' if cc['is_anomaly'] else '否'}")
        print(f"- 长期最大相关系数: {cc['max_long']:.4f}")
        print(f"- 短期最小相关系数: {cc['min_short']:.4f}")
        print(f"- 相关系数差值: {cc['diff']:.4f}")
        
        # 判断满足哪个条件
        if cc['max_long'] > 0.6 and cc['min_short'] < 0.4 and cc['diff'] > 0.38:
            print(f"  ✅ 满足条件A（跨周期相关性破裂）")
        elif cc['max_long'] > 0.6 and data['1m_1d'].get('tau_star', 0) > 0:
            print(f"  ✅ 满足条件B（延迟传导模式，τ*={data['1m_1d']['tau_star']}）")
        print()
    
    print(f"{'='*100}\n")
    print("## 三、过滤检查结果\n")
    
    if 'beta_check' in data['summary']:
        bc = data['summary']['beta_check']
        print(f"### Beta收益率系数检查")
        print(f"- 平均Beta: {bc['avg_beta']:.4f}")
        print(f"- 阈值: {bc['threshold']}")
        print(f"- 结果: {'❌ 未通过' if not bc['passed'] else '✅ 通过'}")
        if not bc['passed']:
            gap = bc['threshold'] - bc['avg_beta']
            gap_pct = (gap / bc['avg_beta'] * 100) if bc['avg_beta'] > 0 else 0
            print(f"- 距阈值差距: {gap:.4f} ({gap_pct:.2f}%)")
        print()
    
    if 'stationarity' in data['summary']:
        st = data['summary']['stationarity']
        print(f"### 平稳性检验")
        print(f"- ADF统计量: {st['adf_statistic']:.4f}")
        print(f"- p-value: {st['p_value']:.4f}")
        print(f"- 等级: {st['level']}")
        print(f"- 结果: {'✅ 通过' if st['passed'] else '❌ 未通过'}")
        if st['passed']:
            if st['p_value'] < 0.05:
                print(f"  ✅ 强平稳（p-value < 0.05）")
            else:
                print(f"  ⚠️  弱平稳（0.05 ≤ p-value < 0.10）")
        else:
            print(f"  ❌ 非平稳（p-value ≥ 0.10）")
        print()
    
    if 'zscore' in data['summary']:
        zs = data['summary']['zscore']
        print(f"### Z-score检查")
        print(f"- Z-score值: {zs['value']:.2f}")
        if 'threshold' in zs:
            print(f"- 阈值: {zs['threshold']}")
            print(f"- 结果: {'❌ 未通过' if not zs['passed'] else '✅ 通过'}")
            if not zs['passed']:
                gap = zs['threshold'] - abs(zs['value'])
                print(f"- 距阈值差距: {gap:.2f} (当前|Z-score| = {abs(zs['value']):.2f})")
            if 'stationarity' in zs:
                print(f"- 平稳性: {zs['stationarity']}")
        else:
            print(f"- 结果: ✅ 通过")
        print()
    
    print(f"{'='*100}\n")
    print("## 四、关键指标对比\n")
    
    print("| 指标 | 长期(5m/7d) | 短期(1m/1d) | 差异 | 阈值要求 | 结果 |")
    print("|------|------------|------------|------|---------|------|")
    
    if data['5m_7d'] and data['1m_1d']:
        # 相关系数
        corr_long = data['5m_7d']['correlation']
        corr_short = data['1m_1d']['correlation']
        corr_diff_val = corr_long - corr_short
        corr_result = "✅" if corr_long > 0.6 and corr_short < 0.4 and corr_diff_val > 0.38 else "❌"
        print(f"| 相关系数 | {corr_long:.4f} | {corr_short:.4f} | {corr_diff_val:.4f} | 长期>0.6,短期<0.4,差值>0.38 | {corr_result} |")
        
        # Beta
        beta_long = data['5m_7d']['beta']
        beta_short = data['1m_1d']['beta']
        beta_diff = abs(beta_long - beta_short)
        avg_beta = (beta_long + beta_short) / 2
        beta_result = "✅" if avg_beta >= 1.0 else "❌"
        print(f"| Beta收益率系数 | {beta_long:.4f} | {beta_short:.4f} | {beta_diff:.4f} | 平均≥1.0 | {beta_result} ({avg_beta:.4f}) |")
        
        # 最优延迟
        tau_long = data['5m_7d']['tau_star']
        tau_short = data['1m_1d']['tau_star']
        tau_result = "✅" if tau_short > 0 else "❌"
        print(f"| 最优延迟τ* | {tau_long} | {tau_short} | - | 短期τ*>0 | {tau_result} |")
    
    print()
    
    # 平稳性和Z-score
    if 'stationarity' in data['summary']:
        st = data['summary']['stationarity']
        st_result = "✅" if st['passed'] else "❌"
        print(f"| 平稳性检验 | - | - | - | p-value<0.1 | {st_result} ({st['level']}, p={st['p_value']:.4f}) |")
    
    if 'zscore' in data['summary']:
        zs = data['summary']['zscore']
        zs_result = "✅" if zs.get('passed', False) else "❌"
        threshold_str = f",阈值={zs.get('threshold', 'N/A')}" if 'threshold' in zs else ""
        print(f"| Z-score | - | - | - | |Z-score|≥2.0{threshold_str} | {zs_result} ({zs['value']:.2f}) |")
    
    print()
    print(f"{'='*100}\n")
    print("## 五、总结\n")
    
    print(f"{coin} 满足异常模式检测条件（条件B：延迟传导模式），但被以下原因过滤：\n")
    
    reasons = []
    if 'beta_check' in data['summary'] and not data['summary']['beta_check']['passed']:
        reasons.append(f"❌ Beta收益率系数不足（平均{data['summary']['beta_check']['avg_beta']:.4f} < 1.0）")
    
    if 'stationarity' in data['summary'] and not data['summary']['stationarity']['passed']:
        reasons.append(f"❌ 平稳性检验失败（{data['summary']['stationarity']['level']}，p-value={data['summary']['stationarity']['p_value']:.4f}）")
    
    if 'zscore' in data['summary'] and not data['summary']['zscore'].get('passed', False):
        reasons.append(f"❌ Z-score未通过（|{data['summary']['zscore']['value']:.2f}| < 2.0）")
    
    for reason in reasons:
        print(f"- {reason}")
    
    if not reasons:
        print("- ⚠️ 未找到明确的过滤原因")



