# ADF 价差序列平稳性检测对 Z-score 价差评估合理性的重要性

## 📋 目录

1. [引言](#引言)
2. [理论基础](#理论基础)
3. [ADF 检验基本原理](#adf-检验基本原理)
4. [平稳性的核心重要性](#平稳性的核心重要性)
5. [非平稳价差的危害](#非平稳价差的危害)
6. [实际案例分析](#实际案例分析)
7. [代码实现详解](#代码实现详解)
8. [最佳实践建议](#最佳实践建议)
9. [总结](#总结)

---

## 引言

在配对交易（Pairs Trading）和统计套利策略中，**Z-score** 是一个核心指标，用于识别两个相关资产之间的价差偏离。然而，Z-score 的有效性完全依赖于一个关键假设：**价差序列必须是平稳的**。

本文档详细阐述 **ADF（Augmented Dickey-Fuller）检验**在验证价差序列平稳性中的重要性，以及为什么平稳性检验是Z-score评估合理性的前提条件。

### 核心问题

- **为什么需要平稳性检验？**
- **非平稳价差会对Z-score造成什么影响？**
- **如何通过ADF检验确保Z-score的有效性？**

---

## 理论基础

### 2.1 平稳性的定义

**时间序列平稳性**（Stationarity）是指时间序列的统计特性不随时间变化。严格平稳需要满足以下三个条件：

1. **均值恒定**（Constant Mean）：`E[X_t] = μ`（对所有t成立）
2. **方差恒定**（Constant Variance）：`Var[X_t] = σ²`（对所有t成立）
3. **协方差只依赖于时滞**（Covariance Depends Only on Lag）：
   `Cov[X_t, X_{t+k}] = γ(k)`（只依赖于k，不依赖于t）

**弱平稳**（Weak Stationarity）只要求上述条件在二阶矩（均值和方差）上成立，这是金融时间序列分析中更常用的概念。

### 2.2 Z-score 的理论基础

Z-score的计算公式为：

```
Z_t = (X_t - μ_t) / σ_t
```

其中：
- `X_t`：当前价差值
- `μ_t`：历史均值（通常使用滚动窗口）
- `σ_t`：历史标准差（通常使用滚动窗口）

**Z-score的核心假设是均值回归（Mean Reversion）**：
- 价差序列围绕固定均值波动
- 当价差偏离均值时，会倾向于回归到均值
- 这种回归行为是可预测的，形成套利机会

### 2.3 协整理论与配对交易

在配对交易中，两个资产的价格序列可能都是**非平稳的**（通常是随机游走），但它们之间的**价差序列可能是平稳的**。这正是**协整理论**（Cointegration Theory）的核心思想：

- 如果两个非平稳序列的线性组合是平稳的，则称它们之间存在协整关系
- 价差序列的平稳性是配对交易策略的理论基础
- ADF检验用于验证价差序列是否平稳（即是否存在协整关系）

---

## ADF 检验基本原理

### 3.1 ADF 检验的数学原理

**Augmented Dickey-Fuller（ADF）检验**是检验时间序列是否存在单位根（Unit Root）的统计检验。如果序列存在单位根，则序列是非平稳的。

**ADF检验的回归模型**：

```
Δy_t = α + βt + γy_{t-1} + Σ(δ_i × Δy_{t-i}) + ε_t
```

其中：
- `Δy_t = y_t - y_{t-1}`：一阶差分
- `α`：常数项
- `βt`：时间趋势项（可选）
- `γy_{t-1}`：滞后项
- `Σ(δ_i × Δy_{t-i})`：差分滞后项（用于消除自相关）
- `ε_t`：误差项

**原假设（H₀）**：`γ = 0`（序列存在单位根，即非平稳）

**备择假设（H₁）**：`γ < 0`（序列不存在单位根，即平稳）

### 3.2 ADF 检验的判定标准

ADF检验的判定基于**p值**（p-value）：

- **p < 0.05**（强平稳）：拒绝原假设，序列是平稳的（统计学上显著）
- **0.05 ≤ p < 0.10**（弱平稳）：处于临界区域，可以认为序列可能是平稳的（探索性分析）
- **p ≥ 0.10**（非平稳）：无法拒绝原假设，序列可能是非平稳的

**ADF统计量**（ADF Statistic）：
- 负值越大（绝对值越大），越倾向于拒绝原假设（序列越可能是平稳的）
- ADF统计量需要与临界值比较，但通常直接看p值更方便

### 3.3 ADF 检验的参数设置

在代码实现中（`hyperliquid_analyzer.py`），使用以下参数：

```python
result = adfuller(spread_clean, autolag='AIC')
```

- **autolag='AIC'**：使用Akaike信息准则（AIC）自动选择最优的滞后阶数
- 这确保了检验的统计功效和准确性

---

## 平稳性的核心重要性

### 4.1 Z-score 有效性的前提条件

**Z-score计算的前提假设**：

1. **价差序列是平稳的**：均值和方差不随时间变化
2. **均值回归性质**：价差会围绕固定均值波动
3. **统计量的稳定性**：滚动均值和标准差能够反映真实的分布特征

如果价差序列**非平稳**，上述假设全部失效，Z-score将失去意义。

### 4.2 平稳性对Z-score的具体影响

#### 4.2.1 均值偏移问题

**非平稳序列的特征**：序列的均值会随时间漂移

```
时间序列示例（非平稳）：
时刻    价差值    真实均值（未知）    计算的滚动均值
t=1     0.5       0.5                -
t=2     0.6       0.6                -
...
t=50    2.5       2.5                0.55（基于前20个数据点）
t=51    2.6       2.6                0.65
...
t=100   4.5       4.5                2.55（基于t=80-99的数据点）
```

**问题**：
- 滚动均值（基于历史数据）无法反映序列的真实均值（随时间变化）
- Z-score计算使用的是过时的均值，导致信号失真
- 即使价差已经显著偏离"未来均值"，Z-score可能仍显示正常

#### 4.2.2 方差非恒定问题

**非平稳序列的方差膨胀**：序列的波动幅度可能随时间增大或减小

```
方差膨胀示例：
早期（t=1-50）：  价差波动范围 [0.4, 0.6]，标准差 σ = 0.05
后期（t=51-100）：价差波动范围 [2.0, 5.0]，标准差 σ = 0.8
```

**问题**：
- Z-score分母（标准差）基于历史数据，无法反映当前的真实波动水平
- 早期计算的Z-score可能严重高估或低估偏离程度
- 导致交易信号的误判

#### 4.2.3 均值回归假设失效

**平稳序列的均值回归**：
```
价差序列（平稳）：
0.5 → 0.8 → 0.6 → 0.4 → 0.55 → 0.7 → 0.5
      ↑              ↑                    ↑
   偏离均值      回归均值            回归均值
```

**非平稳序列的无回归性**：
```
价差序列（非平稳，持续上升趋势）：
0.5 → 0.8 → 1.2 → 1.8 → 2.5 → 3.2 → 4.0
      ↑              ↑                    ↑
   偏离"旧均值"   继续上升           继续上升（不回归）
```

**问题**：
- 非平稳序列没有固定的均值，因此不存在"回归"的概念
- Z-score信号提示"价差偏离，应该回归"，但价差可能持续偏离
- 基于Z-score的交易策略将面临持续亏损

---

## 非平稳价差的危害

### 5.1 理论层面的危害

#### 5.1.1 统计推断失效

- **大样本理论失效**：非平稳序列不满足大数定律和中心极限定理的前提条件
- **置信区间无效**：基于平稳假设的置信区间计算不再准确
- **假设检验失效**：t检验、F检验等统计检验的p值失去意义

#### 5.1.2 预测能力丧失

- **历史模式不可复制**：过去的数据分布无法预测未来的分布
- **外推风险**：基于历史数据的外推将产生系统性偏差
- **策略失效**：基于历史数据的交易策略无法适应非平稳环境

### 5.2 实际交易中的危害

#### 5.2.1 虚假交易信号

**场景示例**：

假设某币种对BTC的价差序列在t=1-50时围绕0.5波动，但在t=51后开始持续上升至2.0。

**问题分析**：

1. **t=60时刻**：
   - 当前价差：2.0
   - 滚动均值（基于t=40-59）：0.6
   - 滚动标准差：0.1
   - 计算Z-score = (2.0 - 0.6) / 0.1 = **14.0**（极端偏离信号）

2. **交易决策**：
   - Z-score = 14.0 >> 2.0，系统发出"做空山寨币/做多BTC"信号
   - 预期价差回归到0.6附近

3. **实际结果**：
   - 价差继续上升至3.0、4.0（不回归）
   - 交易策略持续亏损
   - Z-score继续发出错误信号

#### 5.2.2 资金损失风险

**风险量化**：

- **错误信号频率**：非平稳序列会产生大量虚假信号
- **持仓时间延长**：价差不回归，持仓时间远超预期
- **最大回撤增大**：价差持续偏离，账户净值持续下跌
- **资金效率降低**：大量资金被套在错误的头寸上

#### 5.2.3 策略失效风险

**策略失效的表现**：

- **夏普比率下降**：策略的风险调整后收益大幅下降
- **最大回撤增大**：策略可能面临极端亏损
- **胜率下降**：交易信号的准确率显著降低
- **策略失效**：如果价差序列长期非平稳，整个策略将完全失效

---

## 实际案例分析

### 6.1 案例1：平稳价差序列（Z-score有效）

**数据特征**：
- 币种：AR/USDC vs BTC/USDC
- 时间周期：1分钟K线，1天数据
- 价差序列：对数价差 `log(AR) - β × log(BTC)`

**ADF检验结果**：
```
ADF统计量：-4.25
p-value：0.0012 (< 0.05)
判定：强平稳
```

**价差序列可视化**（概念示例）：
```
价差值
 0.8 |        *     *  *
 0.6 |    *  *  *  *  *
 0.4 |  *  *        *  *
 0.2 |*           *
-----|---------------------------→ 时间
     |均值：0.5（恒定）
     |标准差：0.15（恒定）
```

**Z-score表现**：
- Z-score能够准确识别价差的偏离
- 价差确实围绕均值回归
- 交易信号准确，策略盈利

### 6.2 案例2：非平稳价差序列（Z-score失效）

**数据特征**：
- 币种：某新兴山寨币 vs BTC
- 时间周期：1分钟K线，1天数据
- 价差序列：存在明显的上升趋势

**ADF检验结果**：
```
ADF统计量：-1.85
p-value：0.35 (>= 0.10)
判定：非平稳
```

**价差序列可视化**（概念示例）：
```
价差值
 4.0 |                    *  *
 3.0 |              *  *
 2.0 |        *  *
 1.0 |  *  *
 0.5 |*
-----|---------------------------→ 时间
     |均值：持续上升（非恒定）
     |标准差：持续增大（非恒定）
```

**Z-score表现**：
- **t=20时刻**：价差=1.0，滚动均值=0.5，Z-score=3.3 → 发出做空信号
- **t=40时刻**：价差=2.0，滚动均值=1.0，Z-score=4.0 → 继续做空
- **t=60时刻**：价差=3.0，滚动均值=1.5，Z-score=5.0 → 继续做空
- **实际结果**：价差持续上升，交易策略持续亏损

**结论**：
- Z-score信号完全失效
- 如果不进行平稳性检验，系统将持续发出错误信号
- 通过ADF检验，系统会过滤掉该币种，避免损失

### 6.3 案例3：弱平稳价差序列（谨慎使用）

**数据特征**：
- 币种：某中等市值币种 vs BTC
- 时间周期：5分钟K线，7天数据

**ADF检验结果**：
```
ADF统计量：-2.95
p-value：0.08 (0.05 <= p < 0.10)
判定：弱平稳
```

**价差序列特征**：
- 整体呈现平稳性，但存在轻微的趋势成分
- 均值回归性质较弱，但仍存在
- 方差相对稳定，但偶有波动

**处理策略**：
- **代码实现**：允许计算Z-score，但标记为"弱平稳"
- **交易建议**：降低仓位规模，设置更严格的止损
- **监控要求**：更频繁地重新检验平稳性

---

## 代码实现详解

### 7.1 平稳性检验的实现

在 `hyperliquid_analyzer.py` 中，平稳性检验通过 `_check_spread_stationarity` 方法实现：

```python
@staticmethod
def _check_spread_stationarity(spread: pd.Series,
                                strong_threshold: float = None,
                                weak_threshold: float = None,
                                coin: str = None) -> tuple['StationarityLevel', float]:
    """
    检验价差序列的平稳性（增强版：分级判定）
    
    平稳性是配对交易的核心假设：价差序列必须是平稳的，
    才能保证均值回归性质，从而使 Z-score 的套利信号有效。
    """
    # 1. 移除 NaN 值
    spread_clean = spread.dropna()
    
    # 2. 检查数据量是否足够（ADF 检验至少需要 20 个数据点）
    if len(spread_clean) < 20:
        return StationarityLevel.NON_STATIONARY, 1.0
    
    # 3. 执行 ADF 检验
    result = adfuller(spread_clean, autolag='AIC')
    adf_statistic = result[0]
    p_value = result[1]
    
    # 4. 分级判定平稳性
    if p_value < strong_threshold:  # 默认 0.05
        level = StationarityLevel.STRONG
    elif p_value < weak_threshold:  # 默认 0.10
        level = StationarityLevel.WEAK
    else:
        level = StationarityLevel.NON_STATIONARY
    
    return level, p_value
```

**关键实现细节**：

1. **数据清洗**：移除NaN值，确保检验的准确性
2. **数据量检查**：ADF检验至少需要20个数据点
3. **自动滞后选择**：使用`autolag='AIC'`自动选择最优滞后阶数
4. **分级判定**：区分强平稳、弱平稳、非平稳三个等级

### 7.2 Z-score计算中的平稳性检验集成

在 `_calculate_zscore` 方法中，平稳性检验作为前置条件：

```python
@staticmethod
def _calculate_zscore(btc_prices: pd.Series, alt_prices: pd.Series,
                      beta: float, window: int = 20,
                      check_stationarity: bool = True, coin: str = None) -> Optional[float]:
    # ... 数据验证 ...
    
    # 构建对数价差序列
    log_btc = np.log(btc_prices)
    log_alt = np.log(alt_prices)
    spread = log_alt - beta * log_btc
    
    # ========== 关键：平稳性检验 ==========
    if check_stationarity:
        stationarity_level, p_value = DelayCorrelationAnalyzer._check_spread_stationarity(spread, coin=coin)
        
        # 非平稳：直接返回 None（不计算Z-score）
        if stationarity_level == StationarityLevel.NON_STATIONARY:
            logger.info(
                f"Z-score 计算终止：价差序列非平稳（ADF p-value={p_value:.4f}）| "
                f"均值回归假设不成立，不适合配对交易"
            )
            return None
        
        # 弱平稳：发出警告但继续计算
        if stationarity_level == StationarityLevel.WEAK:
            logger.info(
                f"Z-score 计算继续（弱平稳警告）| ADF p-value={p_value:.4f} | "
                f"平稳性检验处于边缘区域，建议谨慎交易"
            )
    # =====================================
    
    # 继续计算Z-score（仅在平稳性检验通过后）
    spread_mean = spread.rolling(window=window, min_periods=window).mean()
    spread_std = spread.rolling(window=window, min_periods=window).std()
    zscore = (current_spread - current_mean) / current_std
    
    return float(zscore)
```

**关键设计决策**：

1. **非平稳序列直接过滤**：如果价差非平稳，不计算Z-score，避免产生虚假信号
2. **弱平稳序列允许计算但警告**：在边缘情况下，允许计算但标记为弱信号
3. **强平稳序列正常处理**：只有强平稳序列才会产生高质量信号

### 7.3 平稳性等级枚举

代码中定义了 `StationarityLevel` 枚举类，用于区分不同的平稳性等级：

```python
class StationarityLevel(Enum):
    """价差序列平稳性等级"""
    STRONG = "strong"        # 强平稳: p < 0.05
    WEAK = "weak"            # 弱平稳: 0.05 <= p < 0.10
    NON_STATIONARY = "non"   # 非平稳: p >= 0.10
    
    @property
    def is_valid(self) -> bool:
        """是否为有效平稳性（强或弱）"""
        return self in (StationarityLevel.STRONG, StationarityLevel.WEAK)
```

**分级策略的优势**：

1. **精细化管理**：不同等级的平稳性采用不同的处理策略
2. **风险控制**：弱平稳信号可以降低仓位或设置更严格的止损
3. **信号质量评估**：下游逻辑可以根据平稳性等级评估信号质量

---

## 最佳实践建议

### 8.1 ADF检验的参数选择

#### 8.1.1 显著性水平

**建议配置**：

```python
STATIONARITY_STRONG_THRESHOLD = 0.05  # 强平稳阈值（5%显著性水平）
STATIONARITY_WEAK_THRESHOLD = 0.10    # 弱平稳阈值（10%显著性水平）
```

**理由**：
- **0.05**：统计学标准显著性水平，确保强信号的可靠性
- **0.10**：探索性分析常用的显著性水平，适合捕捉边缘情况
- **分级处理**：既保证信号质量，又不过度过滤潜在机会

#### 8.1.2 数据量要求

**最小数据量**：

```python
MIN_POINTS_FOR_ADF = 20  # ADF检验的最小数据点要求
```

**建议**：
- **最少20个数据点**：ADF检验的基本要求
- **推荐50-100个数据点**：确保检验的统计功效
- **更多数据点**：可以提高检验的准确性，但要注意数据的时效性

#### 8.1.3 滞后阶数选择

**自动选择**：

```python
result = adfuller(spread_clean, autolag='AIC')  # 使用AIC准则
```

**替代方案**：

```python
# 使用BIC准则（更保守，倾向于选择较少的滞后项）
result = adfuller(spread_clean, autolag='BIC')

# 手动指定滞后阶数（如果对数据特征有深入了解）
result = adfuller(spread_clean, maxlag=5)
```

**建议**：优先使用`autolag='AIC'`，这是最常用和可靠的方法。

### 8.2 平稳性检验的时机

#### 8.2.1 检验频率

**建议策略**：

1. **每次计算Z-score前**：确保价差序列的平稳性
2. **定期重新检验**：市场环境变化时，平稳性可能发生变化
3. **重大市场事件后**：重新检验所有币种的平稳性

#### 8.2.2 检验窗口选择

**滚动窗口检验**：

```python
# 使用固定长度的滚动窗口进行检验
window_size = 100  # 使用最近100个数据点
spread_window = spread[-window_size:]
stationarity_level, p_value = _check_spread_stationarity(spread_window)
```

**全样本检验**：

```python
# 使用全部历史数据（当前实现）
stationarity_level, p_value = _check_spread_stationarity(spread)
```

**建议**：
- **当前实现（全样本）**：适合长期稳定的币种对
- **滚动窗口**：适合需要适应市场变化的场景
- **结合使用**：可以同时进行全样本和滚动窗口检验

### 8.3 非平稳序列的处理策略

#### 8.3.1 直接过滤（推荐）

**策略**：如果价差序列非平稳，直接过滤，不计算Z-score

```python
if stationarity_level == StationarityLevel.NON_STATIONARY:
    return None  # 不计算Z-score
```

**优势**：
- **避免虚假信号**：防止基于非平稳序列的错误交易
- **风险控制**：确保所有信号都有统计基础
- **策略可靠性**：提高整体策略的可靠性

#### 8.3.2 差分处理（高级）

**策略**：对非平稳序列进行差分，使其平稳

```python
# 一阶差分
spread_diff = spread.diff().dropna()

# 检验差分序列的平稳性
stationarity_level, p_value = _check_spread_stationarity(spread_diff)
```

**注意事项**：
- 差分后的序列含义发生变化（价差的变化率而非价差本身）
- Z-score的解释需要相应调整
- 需要额外的模型假设

**建议**：除非有深入的理论支持，否则不推荐使用差分处理。

### 8.4 弱平稳序列的处理策略

#### 8.4.1 降低仓位规模

```python
if stationarity_level == StationarityLevel.WEAK:
    position_size = base_position_size * 0.5  # 降低50%仓位
```

#### 8.4.2 设置更严格的止损

```python
if stationarity_level == StationarityLevel.WEAK:
    stop_loss = base_stop_loss * 0.7  # 更严格的止损（70%）
```

#### 8.4.3 更频繁的监控

```python
if stationarity_level == StationarityLevel.WEAK:
    monitoring_frequency = base_frequency * 2  # 双倍监控频率
```

### 8.5 平稳性检验的局限性

#### 8.5.1 检验功效

**问题**：ADF检验在样本量较小时功效较低，可能无法检测出弱非平稳性

**解决方案**：
- 确保足够的样本量（至少50个数据点）
- 结合其他检验方法（如KPSS检验、PP检验）
- 使用多个时间窗口进行验证

#### 8.5.2 结构突变

**问题**：如果序列存在结构突变（如均值突然跳跃），ADF检验可能失效

**解决方案**：
- 使用滚动窗口检验，检测平稳性的变化
- 结合Chow检验等方法检测结构突变
- 在检测到结构突变时，重新计算平稳性

#### 8.5.3 时间依赖性

**问题**：平稳性可能随时间变化，历史平稳不代表未来平稳

**解决方案**：
- 定期重新检验平稳性
- 使用滚动窗口检验，检测平稳性的稳定性
- 在平稳性发生变化时，及时调整策略

---

## 总结

### 9.1 核心结论

1. **ADF检验是Z-score有效性的前提**：
   - 只有平稳的价差序列才能保证Z-score的统计意义
   - 非平稳序列会导致Z-score完全失效，产生虚假交易信号

2. **平稳性是配对交易的理论基础**：
   - 平稳性保证了均值回归性质
   - 平稳性保证了统计量的稳定性
   - 平稳性保证了策略的可预测性

3. **分级处理策略提高信号质量**：
   - 强平稳序列：高质量信号，正常交易
   - 弱平稳序列：边缘信号，谨慎交易
   - 非平稳序列：直接过滤，避免损失

### 9.2 关键要点

| 要点 | 说明 |
|------|------|
| **平稳性定义** | 时间序列的统计特性（均值、方差）不随时间变化 |
| **ADF检验原理** | 检验序列是否存在单位根（非平稳的指标） |
| **判定标准** | p < 0.05（强平稳），0.05 ≤ p < 0.10（弱平稳），p ≥ 0.10（非平稳） |
| **非平稳的危害** | 均值偏移、方差非恒定、均值回归失效、虚假信号、资金损失 |
| **代码实现** | 在Z-score计算前进行平稳性检验，非平稳序列直接过滤 |
| **最佳实践** | 使用AIC自动选择滞后阶数，确保足够的数据量，定期重新检验 |

### 9.3 实施建议

1. **必须实施**：
   - ✅ 在每次计算Z-score前进行ADF检验
   - ✅ 非平稳序列直接过滤，不计算Z-score
   - ✅ 记录平稳性检验结果，用于策略优化

2. **推荐实施**：
   - ✅ 分级处理策略（强/弱/非平稳）
   - ✅ 弱平稳序列降低仓位或设置更严格止损
   - ✅ 定期重新检验平稳性（市场环境变化时）

3. **可选优化**：
   - ⚠️ 结合其他检验方法（KPSS、PP检验）提高可靠性
   - ⚠️ 使用滚动窗口检验，检测平稳性的稳定性
   - ⚠️ 检测结构突变，及时调整策略

### 9.4 最终建议

**ADF价差序列平稳性检测不是可选项，而是Z-score价差评估的必需前提**。

只有通过平稳性检验的价差序列，才能保证：
- Z-score的统计意义
- 均值回归假设的成立
- 交易信号的有效性
- 策略的可靠性

**忽略平稳性检验的风险是巨大的**，可能导致：
- 大量虚假交易信号
- 持续的资金损失
- 策略完全失效

因此，**平稳性检验应该作为Z-score计算流程中的强制性步骤**，任何非平稳序列都应该被直接过滤，以确保策略的可靠性和盈利能力。

---

## 参考文献

1. **Dickey, D. A., & Fuller, W. A. (1979)**. Distribution of the estimators for autoregressive time series with a unit root. *Journal of the American Statistical Association*, 74(366), 427-431.

2. **Engle, R. F., & Granger, C. W. (1987)**. Co-integration and error correction: representation, estimation, and testing. *Econometrica*, 55(2), 251-276.

3. **Johansen, S. (1991)**. Estimation and hypothesis testing of cointegration vectors in Gaussian vector autoregressive models. *Econometrica*, 59(6), 1551-1580.

4. **Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006)**. Pairs trading: Performance of a relative-value arbitrage rule. *The Review of Financial Studies*, 19(3), 797-827.

5. **Elliott, R. J., Van Der Hoek, J., & Malcolm, W. P. (2005)**. Pairs trading. *Quantitative Finance*, 5(3), 271-276.

