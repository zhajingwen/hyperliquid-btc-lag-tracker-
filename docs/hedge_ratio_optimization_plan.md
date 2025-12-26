# 对冲比率优化实施规划

## 📋 项目概述

### 目标
在现有的时间延迟配对交易分析系统基础上，引入**协整检验**和**对冲比率计算**功能，提升策略的统计可靠性和实战可行性。

### 背景
当前系统基于皮尔逊相关系数进行异常币种识别，但缺少以下关键验证：
- ❌ 无法验证长期协整关系的统计显著性
- ❌ 无法确定实际交易时的最优仓位配比
- ❌ 无法评估套利组合的市场中性特征

### 核心价值
- ✅ **降低假阳性**: 通过协整检验过滤"伪相关"币种
- ✅ **量化仓位**: 提供实际交易所需的对冲比率
- ✅ **风险控制**: 构建市场中性组合，隔离方向性风险

---

## 🎯 理论基础

### 1. 协整理论 (Cointegration Theory)

#### 定义
两个非平稳时间序列 X_t 和 Y_t 是协整的，当且仅当：
1. X_t 和 Y_t 都是 I(1) 过程（一阶单整）
2. 存在协整向量 β，使得 Z_t = Y_t - β·X_t 是平稳的 I(0) 过程

#### 数学表达
```
协整关系: Y_t = α + β·X_t + ε_t
其中: ε_t ~ I(0) (平稳残差)
```

#### 经济意义
- **短期偏离**: 价格可能短期脱离协整关系
- **长期回归**: 由于套利机制，价格最终会回归均衡
- **均值回归**: 价差 Z_t 围绕均值波动，是套利基础

---

### 2. 协整检验方法

#### 方法对比

| 检验方法 | 原理 | 优势 | 劣势 | 适用场景 |
|---------|------|------|------|---------|
| **Engle-Granger** | 两步法：OLS回归+ADF检验 | 简单直观 | 只能检测单一协整关系 | 双变量分析 |
| **Johansen** | 向量自回归+极大似然 | 可检测多重协整关系 | 计算复杂 | 多变量分析 |
| **ADF (辅助)** | 单位根检验 | 验证平稳性 | 非协整检验 | 残差平稳性验证 |

#### 本项目选择：Engle-Granger 两步法

**理由**：
- ✅ 场景匹配：仅需分析 BTC-ALT 双变量协整
- ✅ 实现简单：statsmodels 库原生支持
- ✅ 计算高效：单次分析 <100ms
- ✅ 结果直观：直接输出协整向量和 p 值

**实施步骤**：
```python
# Step 1: OLS 回归估计协整向量
ALT_returns = α + β·BTC_returns + ε

# Step 2: ADF 检验残差平稳性
ADF_test(ε) → p_value
if p_value < 0.05:
    协整关系显著成立
```

---

### 3. 对冲比率计算方法

#### 方法1: 协整向量法 (推荐)

**公式**：
```python
Hedge_Ratio = β_cointegration
```

**来源**: 协整回归的斜率系数

**优势**:
- ✅ 统计严格：基于长期均衡关系
- ✅ 理论支撑：确保价差平稳性
- ✅ 稳健性高：不受短期波动影响

**示例**:
```python
# 如果 β = 1.2，表示：
# BTC 价格变动 1%，需要对冲 1.2 倍的山寨币仓位
# 才能实现市场中性组合
```

---

#### 方法2: 波动率调整法

**公式**：
```python
Hedge_Ratio = β × (σ_ALT / σ_BTC)
```

**组成部分**:
- β: 当前 Beta 系数（已实现）
- σ_ALT: 山寨币收益率标准差
- σ_BTC: BTC 收益率标准差

**适用场景**:
- 协整检验未通过时的备用方案
- 需要动态调整对冲比率的场景

**劣势**:
- ⚠️ 缺少长期均衡保证
- ⚠️ 对短期波动敏感

---

#### 方法3: 最小方差法 (Minimum Variance Hedge Ratio)

**公式**：
```python
Hedge_Ratio = Cov(R_BTC, R_ALT) / Var(R_BTC)
```

**特点**:
- 最小化组合方差
- 等价于简单线性回归的斜率

**本项目集成方案**:
```python
# 优先级策略
if 协整检验通过:
    Hedge_Ratio = β_cointegration  # 方法1
else:
    Hedge_Ratio = β × (σ_ALT / σ_BTC)  # 方法2
```

---

## 🏗️ 技术架构设计

### 模块结构

```
hyperliquid-btc-lag-tracker/
├── core/
│   ├── __init__.py
│   ├── cointegration_tester.py       # ✨ 新增：协整检验模块
│   │   ├── EngleGrangerTester
│   │   ├── ADFStationarityTest
│   │   └── CointegrationResult
│   │
│   ├── hedge_ratio_calculator.py     # ✨ 新增：对冲比率计算模块
│   │   ├── CointegrationHedge
│   │   ├── VolatilityAdjustedHedge
│   │   └── MinVarianceHedge
│   │
│   └── correlation_analyzer.py       # 🔄 重构：从现有代码提取
│       └── DelayCorrelationAnalyzer (轻量化版本)
│
├── hyperliquid_analyzer.py           # 🔄 改进：集成新功能
└── tests/                             # ✨ 新增：单元测试
    ├── test_cointegration.py
    └── test_hedge_ratio.py
```

---

### 核心类设计

#### 1. CointegrationTester 类

```python
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from statsmodels.tsa.stattools import adfuller, coint

@dataclass
class CointegrationResult:
    """协整检验结果"""
    is_cointegrated: bool          # 是否协整
    beta: float                     # 协整向量 (对冲比率)
    p_value: float                  # p值
    adf_statistic: float            # ADF统计量
    critical_values: dict           # 临界值
    residuals: np.ndarray           # 残差序列
    confidence_level: str           # 置信水平 ("95%", "99%")

class EngleGrangerTester:
    """Engle-Granger 两步法协整检验"""

    def __init__(self, significance_level: float = 0.05):
        """
        Args:
            significance_level: 显著性水平 (默认5%)
        """
        self.significance_level = significance_level

    def test_cointegration(
        self,
        btc_returns: np.ndarray,
        alt_returns: np.ndarray,
        method: str = "ols"
    ) -> CointegrationResult:
        """
        执行协整检验

        Args:
            btc_returns: BTC 收益率序列
            alt_returns: 山寨币收益率序列
            method: 回归方法 ("ols", "robust")

        Returns:
            CointegrationResult 对象

        算法流程:
            1. OLS 回归: alt = α + β·btc + ε
            2. ADF 检验: 验证残差 ε 的平稳性
            3. 判定: p_value < significance_level
        """
        pass

    def _ols_regression(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, float, np.ndarray]:
        """
        最小二乘回归

        Returns:
            (alpha, beta, residuals)
        """
        pass

    def _adf_test(self, residuals: np.ndarray) -> Tuple[float, float, dict]:
        """
        ADF 单位根检验

        Returns:
            (adf_statistic, p_value, critical_values)
        """
        pass
```

---

#### 2. HedgeRatioCalculator 类

```python
from enum import Enum
from typing import Union

class HedgeRatioMethod(Enum):
    """对冲比率计算方法"""
    COINTEGRATION = "cointegration"         # 协整向量法
    VOLATILITY_ADJUSTED = "vol_adjusted"    # 波动率调整法
    MIN_VARIANCE = "min_variance"           # 最小方差法

@dataclass
class HedgeRatioResult:
    """对冲比率计算结果"""
    hedge_ratio: float                      # 对冲比率
    method: HedgeRatioMethod                # 计算方法
    confidence: float                       # 置信度 (0-1)
    metadata: dict                          # 元数据

    # 实际交易建议
    btc_position: float = 1.0               # BTC 仓位 (标准化为1)
    alt_position: float = None              # 山寨币对冲仓位

    def __post_init__(self):
        if self.alt_position is None:
            self.alt_position = self.hedge_ratio

class HedgeRatioCalculator:
    """对冲比率计算器"""

    @staticmethod
    def calculate(
        btc_returns: np.ndarray,
        alt_returns: np.ndarray,
        method: HedgeRatioMethod = HedgeRatioMethod.COINTEGRATION,
        cointegration_result: Optional[CointegrationResult] = None
    ) -> HedgeRatioResult:
        """
        计算对冲比率

        策略逻辑:
            1. 优先使用协整向量法 (如果协整检验通过)
            2. 回退到波动率调整法
            3. 最保守方案: 最小方差法

        Args:
            btc_returns: BTC 收益率
            alt_returns: 山寨币收益率
            method: 计算方法
            cointegration_result: 协整检验结果 (如果有)

        Returns:
            HedgeRatioResult 对象
        """
        pass

    @staticmethod
    def _cointegration_hedge(
        cointegration_result: CointegrationResult
    ) -> HedgeRatioResult:
        """方法1: 协整向量法"""
        return HedgeRatioResult(
            hedge_ratio=cointegration_result.beta,
            method=HedgeRatioMethod.COINTEGRATION,
            confidence=1.0 - cointegration_result.p_value,
            metadata={
                "p_value": cointegration_result.p_value,
                "adf_statistic": cointegration_result.adf_statistic
            }
        )

    @staticmethod
    def _volatility_adjusted_hedge(
        btc_returns: np.ndarray,
        alt_returns: np.ndarray
    ) -> HedgeRatioResult:
        """方法2: 波动率调整法"""
        beta = np.cov(btc_returns, alt_returns)[0, 1] / np.var(btc_returns)
        sigma_ratio = np.std(alt_returns) / np.std(btc_returns)
        hedge_ratio = beta * sigma_ratio

        return HedgeRatioResult(
            hedge_ratio=hedge_ratio,
            method=HedgeRatioMethod.VOLATILITY_ADJUSTED,
            confidence=0.7,  # 经验置信度
            metadata={
                "beta": beta,
                "sigma_ratio": sigma_ratio
            }
        )

    @staticmethod
    def _min_variance_hedge(
        btc_returns: np.ndarray,
        alt_returns: np.ndarray
    ) -> HedgeRatioResult:
        """方法3: 最小方差法"""
        hedge_ratio = np.cov(btc_returns, alt_returns)[0, 1] / np.var(btc_returns)

        return HedgeRatioResult(
            hedge_ratio=hedge_ratio,
            method=HedgeRatioMethod.MIN_VARIANCE,
            confidence=0.6,
            metadata={}
        )
```

---

### 集成到现有分析器

#### 改进后的 DelayCorrelationAnalyzer

```python
class DelayCorrelationAnalyzer:
    """增强版相关性分析器"""

    # ========== 新增配置 ==========
    # 协整检验配置
    ENABLE_COINTEGRATION_TEST = True
    COINTEGRATION_SIGNIFICANCE_LEVEL = 0.05  # 5% 显著性水平

    # 对冲比率配置
    ENABLE_HEDGE_RATIO_CALC = True
    HEDGE_RATIO_METHOD = "auto"  # "auto", "cointegration", "vol_adjusted"

    def __init__(self, exchange_name="hyperliquid", **kwargs):
        # ... 现有代码 ...

        # 新增模块
        from core.cointegration_tester import EngleGrangerTester
        from core.hedge_ratio_calculator import HedgeRatioCalculator

        self.cointegration_tester = EngleGrangerTester(
            significance_level=self.COINTEGRATION_SIGNIFICANCE_LEVEL
        )
        self.hedge_calculator = HedgeRatioCalculator()

    @staticmethod
    def find_optimal_delay(
        btc_ret: np.ndarray,
        alt_ret: np.ndarray,
        max_lag: int = 3,
        enable_outlier_treatment: bool = True,
        enable_beta_calc: bool = True,
        enable_cointegration_test: bool = True,  # ✨ 新增参数
        enable_hedge_ratio: bool = True          # ✨ 新增参数
    ) -> dict:
        """
        增强版延迟优化算法

        Returns:
            {
                'tau_star': int,
                'correlations': list,
                'max_corr': float,
                'beta': float,

                # ✨ 新增返回值
                'cointegration': CointegrationResult | None,
                'hedge_ratio': HedgeRatioResult | None,
                'strategy_grade': str  # "A", "B", "C", "D"
            }
        """
        # 现有逻辑
        result = {
            'tau_star': tau_star,
            'correlations': correlations,
            'max_corr': max_corr,
            'beta': beta
        }

        # ========== 新增逻辑 ==========
        if enable_cointegration_test:
            coint_result = EngleGrangerTester().test_cointegration(
                btc_ret, alt_ret
            )
            result['cointegration'] = coint_result

        if enable_hedge_ratio:
            hedge_result = HedgeRatioCalculator.calculate(
                btc_ret, alt_ret,
                method=HedgeRatioMethod.COINTEGRATION,
                cointegration_result=result.get('cointegration')
            )
            result['hedge_ratio'] = hedge_result

        # 策略评级
        result['strategy_grade'] = _calculate_strategy_grade(result)

        return result

    def _calculate_strategy_grade(self, analysis_result: dict) -> str:
        """
        策略评级系统

        评分维度:
            1. 协整检验是否通过 (40分)
            2. 短期相关性破裂程度 (30分)
            3. Beta 系数 (20分)
            4. 延迟显著性 (10分)

        评级标准:
            A: 90-100分 (强烈推荐)
            B: 75-89分  (推荐)
            C: 60-74分  (谨慎)
            D: <60分    (不推荐)
        """
        score = 0

        # 维度1: 协整检验 (40分)
        coint = analysis_result.get('cointegration')
        if coint and coint.is_cointegrated:
            score += 40 * (1 - coint.p_value)  # p值越小得分越高

        # 维度2: 相关性破裂 (30分)
        corr_diff = analysis_result.get('corr_diff', 0)
        if corr_diff > 0.38:
            score += 30
        elif corr_diff > 0.30:
            score += 20
        elif corr_diff > 0.20:
            score += 10

        # 维度3: Beta 系数 (20分)
        beta = analysis_result.get('beta', 0)
        if beta >= 1.2:
            score += 20
        elif beta >= 1.0:
            score += 15
        elif beta >= 0.8:
            score += 10

        # 维度4: 延迟显著性 (10分)
        tau_star = analysis_result.get('tau_star', 0)
        if tau_star > 0:
            score += 10

        # 评级映射
        if score >= 90:
            return "A"
        elif score >= 75:
            return "B"
        elif score >= 60:
            return "C"
        else:
            return "D"
```

---

## 📊 数据结构设计

### 增强版分析结果

```python
@dataclass
class EnhancedAnalysisResult:
    """完整的分析结果"""

    # ========== 现有字段 ==========
    symbol: str
    timeframe: str
    period: str

    # 相关性分析
    tau_star: int
    correlations: list
    max_corr: float

    # Beta 分析
    beta: float
    avg_beta: float

    # ========== 新增字段 ==========
    # 协整分析
    is_cointegrated: bool
    cointegration_p_value: float
    cointegration_beta: float

    # 对冲比率
    hedge_ratio: float
    hedge_ratio_method: str
    hedge_ratio_confidence: float

    # 实际交易建议
    recommended_btc_position: float = 1.0
    recommended_alt_position: float = None

    # 策略评级
    strategy_grade: str  # "A", "B", "C", "D"
    grade_score: float   # 0-100

    # 风险指标
    spread_volatility: float      # 价差波动率
    half_life: Optional[float]    # 均值回归半衰期
```

---

## 🚀 实施阶段

### 阶段1: 协整检验模块开发 (第1-2周)

#### 任务清单
- [ ] 创建 `core/` 目录结构
- [ ] 实现 `EngleGrangerTester` 类
  - [ ] OLS 回归算法
  - [ ] ADF 检验集成
  - [ ] 协整判定逻辑
- [ ] 实现 `CointegrationResult` 数据类
- [ ] 编写单元测试
  - [ ] 测试用例1: 完美协整数据
  - [ ] 测试用例2: 非协整数据
  - [ ] 测试用例3: 边界情况

#### 验收标准
```python
# 测试用例示例
def test_cointegration_positive_case():
    # 构造协整数据: Y = 2*X + noise
    X = np.cumsum(np.random.randn(1000))
    Y = 2*X + np.random.randn(1000)*0.1

    tester = EngleGrangerTester(significance_level=0.05)
    result = tester.test_cointegration(X, Y)

    assert result.is_cointegrated == True
    assert abs(result.beta - 2.0) < 0.1  # 允许 ±0.1 误差
    assert result.p_value < 0.05
```

---

### 阶段2: 对冲比率计算模块 (第3周)

#### 任务清单
- [ ] 实现 `HedgeRatioCalculator` 类
  - [ ] 协整向量法
  - [ ] 波动率调整法
  - [ ] 最小方差法
  - [ ] 自动选择策略
- [ ] 实现 `HedgeRatioResult` 数据类
- [ ] 编写单元测试
  - [ ] 测试三种计算方法的一致性
  - [ ] 测试极端市场条件

#### 验收标准
```python
def test_hedge_ratio_methods_consistency():
    # 对于协整数据,三种方法应该给出接近的结果
    btc_ret = np.random.randn(1000)
    alt_ret = 1.5*btc_ret + np.random.randn(1000)*0.1

    hr1 = HedgeRatioCalculator.calculate(
        btc_ret, alt_ret, method=HedgeRatioMethod.COINTEGRATION
    )
    hr2 = HedgeRatioCalculator.calculate(
        btc_ret, alt_ret, method=HedgeRatioMethod.MIN_VARIANCE
    )

    # 两种方法结果差异应 <20%
    assert abs(hr1.hedge_ratio - hr2.hedge_ratio) / hr1.hedge_ratio < 0.2
```

---

### 阶段3: 集成与增强 (第4周)

#### 任务清单
- [ ] 改进 `DelayCorrelationAnalyzer.find_optimal_delay()`
- [ ] 实现策略评级系统 `_calculate_strategy_grade()`
- [ ] 更新飞书告警格式，包含新增字段
- [ ] 更新 `README.md` 文档

#### 新增告警格式

```
hyperliquid

DOGE/USDC:USDC 增强分析结果

📊 相关性分析
  时间周期  数据周期  相关系数  最优延迟  Beta系数
    5m       7d      0.8500      0       1.35
    1m       1d      0.4200      2       1.28
  差值: 0.43

🔗 协整检验
  ✅ 协整关系显著 (p=0.012)
  对冲比率: 1.42 (协整向量法)
  置信度: 98.8%

💼 交易建议
  BTC 仓位: $10,000
  对冲仓位: $14,200 (做空 DOGE)
  策略评级: A (92分)

⚠️ 风险提示
  价差波动率: 2.3%
  均值回归半衰期: 4.2小时
```

---

### 阶段4: 验证与优化 (第5周)

#### 任务清单
- [ ] 历史数据回测
  - [ ] 选择 20 个历史异常币种
  - [ ] 计算各评级的实际表现
  - [ ] 优化评级权重
- [ ] 性能优化
  - [ ] 协整检验计算耗时 <100ms
  - [ ] 批量分析不增加超过 20% 时间
- [ ] 文档完善
  - [ ] API 文档
  - [ ] 使用示例
  - [ ] 故障排查指南

---

## 🧪 测试方案

### 1. 单元测试覆盖率

| 模块 | 测试用例数 | 覆盖率目标 |
|------|----------|----------|
| `cointegration_tester.py` | ≥10 | ≥90% |
| `hedge_ratio_calculator.py` | ≥8 | ≥85% |
| `correlation_analyzer.py` (增强) | ≥5 | ≥80% |

---

### 2. 集成测试场景

#### 场景1: 完美协整案例
```python
# 构造数据: ETH 与 BTC 完全协整
btc_price = 生成随机游走价格
eth_price = 1.5 * btc_price + 小噪声

# 预期结果:
# - 协整检验通过
# - 对冲比率 ≈ 1.5
# - 策略评级 = A
```

#### 场景2: 相关但非协整
```python
# 构造数据: 高相关但独立趋势
btc_price = 随机游走1
alt_price = 随机游走2 (相关性=0.7)

# 预期结果:
# - 协整检验不通过
# - 回退到波动率调整法
# - 策略评级 = C 或 D
```

#### 场景3: 真实历史数据
```python
# 使用 2024-12-01 到 2024-12-25 的 DOGE/BTC 数据

# 验证点:
# 1. 与人工计算结果一致性
# 2. 执行时间 <5秒
# 3. 结果可解释性
```

---

### 3. 性能基准测试

| 操作 | 目标耗时 | 实测耗时 | 通过标准 |
|------|---------|---------|---------|
| 单币种协整检验 | <100ms | - | ✅/❌ |
| 单币种对冲比率计算 | <50ms | - | ✅/❌ |
| 完整分析流程 (单币种) | <15s | - | ✅/❌ |
| 批量分析 (150币种) | <40min | - | ✅/❌ |

---

## ⚠️ 风险评估与对策

### 技术风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 协整检验计算复杂度高 | 中 | 高 | 使用 statsmodels 优化算法 |
| 加密货币数据非平稳性强 | 高 | 中 | 数据预处理+稳健性检验 |
| 对冲比率不稳定 | 中 | 中 | 滚动窗口验证+置信区间 |

### 业务风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 协整关系在实盘中失效 | 中 | 高 | 策略评级系统+持续监控 |
| 对冲比率计算误差 | 低 | 高 | 多方法交叉验证 |
| 过度依赖历史数据 | 高 | 中 | 前向验证+参数敏感性分析 |

---

## 📅 时间规划

### 甘特图

```
Week 1-2: 协整检验模块
  │████████████████│
  开发 → 测试 → 集成

Week 3: 对冲比率计算
  │████████│
  开发 → 测试

Week 4: 系统集成
  │██████████│
  集成 → 告警优化 → 文档

Week 5: 验证与优化
  │████████│
  回测 → 性能优化
```

### 里程碑

- ✅ **里程碑1** (第2周末): 协整检验模块通过单元测试
- ✅ **里程碑2** (第3周末): 对冲比率计算完成
- ✅ **里程碑3** (第4周末): 完整功能集成到主分析器
- ✅ **里程碑4** (第5周末): 历史数据验证完成，性能达标

---

## 📚 参考资料

### 学术论文
1. Engle, R.F., & Granger, C.W.J. (1987). "Co-integration and Error Correction"
2. Johansen, S. (1991). "Estimation and Hypothesis Testing of Cointegration Vectors"

### 技术文档
1. [statsmodels - Cointegration Tests](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html)
2. [ADF Test Documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html)

### 代码示例
1. [QuantConnect - Pairs Trading](https://www.quantconnect.com/tutorials/strategy-library/pairs-trading-with-stocks)
2. [Python for Finance - Cointegration](https://github.com/yhilpisch/py4fi2nd)

---

## 🎓 附录：算法伪代码

### 完整分析流程

```python
def enhanced_analysis_pipeline(btc_data, alt_data):
    """
    增强版分析流程
    """
    # Step 1: 数据预处理
    btc_ret = calculate_returns(btc_data)
    alt_ret = calculate_returns(alt_data)
    btc_ret, alt_ret = winsorize(btc_ret, alt_ret)

    # Step 2: 延迟优化 (现有逻辑)
    tau_star, correlations, max_corr = find_optimal_delay(
        btc_ret, alt_ret, max_lag=3
    )

    # Step 3: 协整检验 (新增)
    coint_tester = EngleGrangerTester(significance_level=0.05)
    coint_result = coint_tester.test_cointegration(btc_ret, alt_ret)

    # Step 4: 对冲比率计算 (新增)
    if coint_result.is_cointegrated:
        hedge_result = HedgeRatioCalculator.calculate(
            btc_ret, alt_ret,
            method=HedgeRatioMethod.COINTEGRATION,
            cointegration_result=coint_result
        )
    else:
        hedge_result = HedgeRatioCalculator.calculate(
            btc_ret, alt_ret,
            method=HedgeRatioMethod.VOLATILITY_ADJUSTED
        )

    # Step 5: 策略评级 (新增)
    strategy_grade = calculate_strategy_grade({
        'cointegration': coint_result,
        'hedge_ratio': hedge_result,
        'correlation_diff': max_corr - correlations[tau_star],
        'beta': calculate_beta(btc_ret, alt_ret),
        'tau_star': tau_star
    })

    # Step 6: 异常判定 (增强)
    is_anomaly = (
        (strategy_grade in ["A", "B"]) or  # 高评级
        (coint_result.is_cointegrated and correlation_diff > 0.3) or  # 协整+相关性破裂
        (max_corr > 0.6 and correlations[0] < 0.4 and tau_star > 0)  # 现有逻辑
    )

    # Step 7: 生成报告
    if is_anomaly:
        send_alert(
            symbol=alt_symbol,
            coint_result=coint_result,
            hedge_result=hedge_result,
            strategy_grade=strategy_grade
        )

    return {
        'tau_star': tau_star,
        'cointegration': coint_result,
        'hedge_ratio': hedge_result,
        'strategy_grade': strategy_grade,
        'is_anomaly': is_anomaly
    }
```

---

## ✅ 验收清单

### 功能验收
- [ ] 协整检验可正确识别协整/非协整关系
- [ ] 对冲比率计算结果符合理论预期
- [ ] 策略评级系统能区分高/低质量信号
- [ ] 飞书告警包含所有新增字段
- [ ] 单元测试覆盖率达标

### 性能验收
- [ ] 单币种分析时间增加 <20%
- [ ] 批量分析 150 币种 <45 分钟
- [ ] 内存占用增加 <50MB

### 文档验收
- [ ] API 文档完整
- [ ] README 更新协整检验说明
- [ ] 代码注释覆盖率 >80%

---

**文档版本**: v1.0
**创建日期**: 2025-12-26
**最后更新**: 2025-12-26
**负责人**: Development Team
**审核状态**: 待审核
