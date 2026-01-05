# Z-score数据来源分析

## 代码追踪

### 1. combinations定义（hyperliquid_analyzer.py:190）
```python
self.combinations = [("5m", "7d"), ("1m", "1d")]
#                      ↑             ↑
#                   timeframe    timeframe
#                   5分钟K线     1分钟K线
```

### 2. price_data_cache构建（hyperliquid_analyzer.py:1685）
```python
for timeframe, period in self.combinations:
    # 遍历两个组合：
    # 第1次循环：timeframe="5m", period="7d"
    # 第2次循环：timeframe="1m", period="1d"

    # 缓存价格数据
    price_data_cache[(timeframe, period)] = {
        'btc_prices': btc_aligned['Close'],  # BTC收盘价
        'alt_prices': alt_aligned['Close']   # 山寨币收盘价
    }
```

**结果**：
- `price_data_cache[("5m", "7d")]` = 5分钟K线、7天数据
- `price_data_cache[("1m", "1d")]` = 1分钟K线、1天数据

### 3. Z-score数据选择（hyperliquid_analyzer.py:1741-1746）
```python
# 尝试从短期数据计算 Z-score
short_term_key = None
for tf, p in self.combinations:
    if p == '1d':  # ← 寻找周期为1d的组合
        short_term_key = (tf, p)
        break  # 找到后立即退出

# short_term_key = ("1m", "1d")  ← 第二个组合

if short_term_key and short_term_key in price_data_cache:
    price_data = price_data_cache[short_term_key]
    # 使用1分钟K线、1天周期的数据
```

### 4. Z-score计算（hyperliquid_analyzer.py:1751-1756）
```python
zscore_result, stationarity_level_result, p_value_result = self._calculate_zscore_with_level(
    price_data['btc_prices'],    # ← 1分钟K线的BTC收盘价（1天数据）
    price_data['alt_prices'],    # ← 1分钟K线的山寨币收盘价（1天数据）
    window=self.ZSCORE_WINDOW,   # 默认30（最近30个数据点）
    beta_window=self.BETA_WINDOW, # 默认60（最近60个数据点）
    coin=coin
)
```

---

## 结论

### ✅ **Z-score使用1分钟K线数据**

**完整描述**：
- **时间粒度**：1分钟K线（每根K线代表1分钟）
- **数据周期**：1天（24小时 × 60分钟 = 1440个数据点）
- **计算窗口**：
  - Beta窗口：60个1分钟K线（最近60分钟数据）
  - Z-score窗口：30个1分钟K线（最近30分钟数据）

### 数据量级

**1天(1d)的1分钟K线数据**：
- 理论数据点：1440个（24小时 × 60分钟）
- 实际可用：约1440个左右（取决于交易所）

**Z-score计算流程**：
```
1. 获取最近1天的1分钟K线数据（~1440个点）
2. 使用最后60个点计算Beta系数（基于对数价格）
3. 使用Beta构建价差序列
4. 对价差序列的最后30个点计算均值和标准差
5. 计算当前价差的Z-score
```

---

## 为什么使用1分钟数据？

### 设计理由（代码注释）
```python
# 优先使用短期数据（1m/1d）计算 Z-score，因为这是检测异常的主要周期
```

### 实际考量

1. **高时效性**
   - 1分钟数据能快速捕捉价格偏离
   - 5分钟数据响应慢30分钟（30个5分钟K线 vs 30个1分钟K线）

2. **交易频率匹配**
   - 配对交易通常需要较快反应
   - 1分钟级别适合短期套利

3. **数据量充足**
   - 1天提供1440个数据点
   - 足够计算稳定的统计量

---

## 与Beta的对比

| 指标 | 时间级别 | 数据量 | 用途 |
|------|---------|--------|------|
| **Beta系数** | **1m + 5m混合** | 1d(1440点) + 7d(2016点) | 波动幅度筛选 |
| **Z-score** | **1m** | 1d(1440点) | 价差偏离检测 |

**关键区别**：
- Beta使用两个时间级别的平均
- Z-score只使用1分钟级别

---

## 潜在问题

### 1. 噪音敏感性

**1分钟数据的噪音更大**：
- 微观市场结构影响（买卖价差、滑点）
- 高频交易者的短期波动
- 随机订单的瞬时冲击

**示例**：
```
5分钟数据：平滑了5个1分钟的波动，更能反映趋势
1分钟数据：包含更多短期噪音，可能产生假信号
```

### 2. 阈值设置的影响

**基于1分钟数据的Z-score可能需要更高阈值**：
- 噪音多 → 标准差大 → Z-score偏小
- 但阈值2.0可能过严

### 3. 与相关性检测的不一致

**相关性破裂检测**：
- 长期：5分钟/7天
- 短期：1分钟/1天

**Z-score检测**：
- 只用：1分钟/1天

**可能的不一致**：
- 5分钟级别发现相关性破裂
- 但1分钟级别Z-score不足（噪音大）

---

## 建议

### 选项1：保持1分钟，降低阈值（推荐）
```python
ZSCORE_THRESHOLD = 1.5  # 从2.0降低到1.5
# 理由：1分钟数据噪音大，2.0过严
```

### 选项2：改用5分钟数据
```python
# 修改 hyperliquid_analyzer.py:1742
if p == '7d':  # 改为使用7d周期（对应5m timeframe）
    short_term_key = (tf, p)
```

**优点**：
- 数据更平滑，信号更稳定
- 与长期相关性检测对齐（5m/7d）

**缺点**：
- 响应慢（30个5分钟 = 150分钟 vs 30个1分钟 = 30分钟）

### 选项3：双重验证
```python
# 1分钟Z-score ≥ 1.5 OR 5分钟Z-score ≥ 2.0
# 两个条件满足其一即可
```

---

## 总结

1. **Z-score基于1分钟K线数据**（1天周期，约1440个数据点）
2. **计算窗口**：Beta用60个点，Z-score用30个点
3. **优势**：响应快，时效性高
4. **劣势**：噪音大，可能需要降低阈值（从2.0到1.5）
5. **与Beta的混合时间级别不同**：Beta混合1m和5m，Z-score只用1m
