# uplift模型业界应用总结

## 何为uplift？

> uplift model 用于预测treatment的增量反馈价值，常应用在Push推送、广告投放、个性化激励等场景。比如，我们想知道对用户展现广告的价值，通常的模型只能建模用户在展示广告后的购买意愿，但事实很有可能是他们在被展示广告之前就已经很想购买了，这个时候展示广告反而会增加投放成本。因此，Push推送和广告投放等场景常采用uplift model建模增量反馈价值，在减少对用户打扰和降低成本的同时，提高业务价值，如DAU增益、广告主价值、LTV、观看时长等。

参考：

https://mp.weixin.qq.com/s/7qyJgEcdufwnSw9bApzYxQ

https://www.uplift-modeling.com/en/latest/user_guide/introduction/comparison.html

## 如何构建uplift？

### 样本构造
> uplift建模对样本的要求是比较高的，需要服从CIA ( Conditional Independence Assumption ) 条件独立假设，最简单的方式就是随机化实验A/B Test，因为通过A/B Test拆分流量得到的这两组样本在特征的分布上面是一致的，可以为Uplift Model提供无偏的样本。

> A/B Test设置两组小流量实验：一组是对照组(control组)，为原始策略；一组是实验组(treatment组)，为想要做改进的策略。两个组经过一段时间跑量，得到用于建模uplift model的样本。

### 算法求解
> uplift model求解有以下几种方法：

#### Meta-learner Algorithms

参考：Meta-learners for Estimating Heterogeneous Treatment Effects using Machine Learning，https://arxiv.org/abs/1706.03461

##### T-learner

> T-learner也叫Two-model，针对control组和treatment组分别学习一个有监督的模型，control组模型只用control组的数据，treatment组模型只用treatment组的数据，之后将两个模型的输出做差，就得到uplift。

> 这种建模方法的优点是简单容易理解，同时可以套用常见的机器学习模型，如LR，GBDT，NN等，落地成本是较低。但是该模型最大的缺点是精度有限，这一方面是因为我们独立的构建了两个模型，这两个模型在打分上面的误差容易产生累积效应，第二是我们建模的目标其实是response而不是uplift，因此对uplift的识别能力比较有限。

**缺点：双模型存在误差累加；间接计算uplift**

##### S-learner
> S-learner也称为one-model，直接把treatment作为特征放进模型，然后训练一个有监督的模型，模型在训练时直接合并control组和treatment组数据作为样本集；预测时，分别对treatment set为1/0，预测值相减则得到uplift值。

> 它和上一个模型最大差别点在于，它在模型层面做了打通，同时底层的样本也是共享的，treatment相关的变量T取值为0或1，T也可以扩展为0到N，建模multiple treatment，比如不同红包的面额，或者不同广告的素材，

> One Model版本和Two Model版本相比最大的优点是训练样本的共享可以使模型学习的更加充分，同时通过模型的学习也可以有效的避免双模型打分误差累积的问题，另外一个优点是从模型的层面可以支持multiple treatment的建模，具有比较强的实用性。同时和Two Model版本类似，它的缺点依然是其在本质上还是在对response建模，因此对uplift的建模还是比较间接，有一定提升的空间。

##### Class Transformation Method

> One Model和Two Model的缺点依然是在本质上还是在对response建模，因此对uplift的建模还是比较间接，有一定提升的空间，更为严谨的一种方式是Class Transformation Method，但该方法只适用于分类问题，具体推导如下：https://www.uplift-modeling.com/en/latest/user_guide/models/revert_label.html


其他求解方法可以参考：https://www.uplift-modeling.com/en/latest/user_guide/index.html#user-guide

#### Tree-based algorithms
待补充

#### 其他
待补充

### 评估指标

> 问题设定

定义如下：一个样本在发券后是否下单，即 $T$ 与 $Y$ 都是binary：

$$ T \in {0, 1} $$

$$ Y \in {0, 1} $$

$X$ 是validation set，用来测试模型

$u$ 是训练得到的uplift模型， $u(x)$ 是对于样本 $x$ 预测出的uplift值

$K$ 是画图时用的bins的数量，即segment个数，当我们是deciles时 $K = 10$ 

$r_k^t, r_k^c$ 是k segment里treatment组中 $Y=1$ 的样本个数和control组中 $Y=1$  的样本个数

$n_k^t, n_k^c$ 是k segment里treatment组的样本个数和control组的样本个数

$\pi$ 代表把 $u(x)$ 从大到小降序排的一个order，我们有 $u^\pi(x_{i}) > u^\pi(x_{j}), {\forall} i < j$ 

$\pi(k)$ 代表按照 $\pi$ 进行排序后的 Top k 的 样本，我们有 $u^\pi(x_{l}) < u^\pi(x_{i}), {\forall} l > k, i \leq k$ 

$R_\pi(k)$ 代表前 $k$ 个样本中 $Y=1$ 的样本个数

$R^T_\pi(k) = R_\pi(k)|T=1$ 代表前 $k$ 个样本中treatment组下 $Y=1$ 的个数

$R^C_\pi(k) = R_\pi(k)|T=0$ 代表前 $k$ 个样本中control组下 $Y=1$ 的个数

$R^C_\pi(k) + R^T_\pi(k) = R_\pi(k)$

$\overline{R}^T(k)$ 代表任意 $k$ 个样本中treatment组下 $Y=1$ 的个数

$\overline{R}^C(k)$ 代表前 $k$ 个样本中control组下 $Y=1$ 的个数

$N^T_\pi(k)$ 代表前 $k$ 个样本中treatment组的样本个数

$N^C_\pi(k)$ 代表前 $k$ 个样本中control组的样本个数

$N^C_\pi(k) + N^T_\pi(k) = k$

$N^T$ 代表样本中treatment组的样本个数

$N^C$ 代表样本中control组的样本个数

$R^T$ 代表样本中treatment组下 $Y=1$ 的个数

$R^C$ 代表样本中control组下 $Y=1$ 的个数

> uplift模型的评估指标

1. uplift by deciles graph 

a. 对 $X$ 中所有样本计算uplift $u(x)$ 并按照倒序排列

b. 对 $u(x)$ 切分成10份，找到切分边界 $b_0, b_1, ...., b_K, K=10$

c. 计算每个segment k的predicted uplift $u_{kp} = \frac{1}{n_k^t + n_k^c} \sum_{b_{k-1} < u(x) < b_{k}} u(x)$ 也就是均值

d. 计算每个segment k的actual uplift $u_{ka} = \frac{r_k^t}{n_k^t} - \frac{r_k^c}{n_k^c}$ 

e. 计算完每个segment的 $u_{kp}$ 和 $u_{ka}$ 就可以作图了


![image](https://github.com/ShaoQiBNU/uplift_model_notes/blob/main/imgs/1.jpg)


这个图就是uplift by deciles graph，我们要怎么去根据这个图和上面提到的这三个标准去评价这个模型呢？

- Monotonicity of incremental gains

这个想说的是actual和predicted的uplift在单调性上是否一致，即是不是predicted uplift越大的segment，actual uplift同样也越大。也就是说，由于predicted uplift一定是单调下降的（因为我们是按这个大小排序的），actual uplift也应该是严格单调下降的。可以看到这个图上的模型基本满足，但是在10-20%，20-30%，30-40%，90-100%这4个segment上不完全单调。

- Tight validation

每个segment里，actual uplift和predicted uplift在数值上是否足够接近，即两根柱子是不是一样长，越一样越好，表明一个预测的准确性

- Range of Predictions

最大的predicted uplift（最左那根柱子）和最小的predicted uplift（最右那根柱子），差距是否足够大。为什么要衡量这个呢？假设我们有一个模型，预测出来每个样本的uplift都是一样的，即使这个模型平均来看是比较准的，但是没有实际意义，因为我们实际上必须要求模型能够区分出uplift较大和较小的两群人，不然我们怎么发券？怎么投放？所以这里就要求模型能够在uplift上预测出足够的区分度。

此方法很难量化，毕竟是看图说话，方法之间有什么可能很难分出孰优孰劣，更多就是一个insight上的考量，去感觉这个模型靠不靠谱。不过这个方法其实已经可以比较全面的帮助我们去评判一个模型了。并且在实际应用时，Range of Predictions其实非常重要，因为我们往往并不要求全局精准，反倒是如果可以能够对最头部的segment预测准，就足以帮助我们对最大的uplift组进行投放了。

2. uplift curve 

$$ f(k) = (\frac{R^T_\pi(k)}{N^T_\pi(k)} - \frac{R^C_\pi(k)}{N^C_\pi(k)})(N^T_\pi(k) + N^C_\pi(k)) $$

- step1：按照上述提到的序 $\pi(k)$ 即把uplift降序排列
- step2：取topK个样本，统计得到 $f(k)$ ，以 $k$ 为横轴，以 $f(k)$ 为纵轴，画出Uplift Curve。如图所示：

![image](https://github.com/ShaoQiBNU/uplift_model_notes/blob/main/imgs/2.jpg)

我们希望在 $k$ 越小的地方，treatment组中 $Y=1$ 的比例与control组中 $Y=1$  的比例的差值越大，证明uplift大的样本确实是那些给treatment就更能转化的样本。但这里的差值是一个绝对值的差，这并不合理，如果本身这个测试集treatment组的数量就显著的小于control组，那就算这个uplift模型再好，这个差值可能都是负的，所以这个怎么解决后面一个方法会讲到。


3. AUUC

AUUC就是uplift curve和baseline两条线中间的面积，越大越好。明细公式如下：

$$ AUUC_\pi(k) = \sum_{i=1}^k (\frac{R^T_\pi(i)}{N^T_\pi(i)} - \frac{R^C_\pi(i)}{N^C_\pi(i)})(N^T_\pi(i) + N^C_\pi(i)) - \frac{k}{2}(\frac{\overline{R}^T(k)}{N^T_\pi(k)} - \frac{\overline{R}^C(k)}{N^C_\pi(k)})(N^T_\pi(k) + N^C_\pi(k)) $$

$$ AUUC = \int_{0}^{1} AUUC_\pi(\rho ) {\rm d}\rho $$

4. Qini curve

当 treatment 组和 control 组的 样本数量（在topK样本里）相差比较大的时候，Uplift Curve的计算可能会存在一些问题。为此 Radcliffe 提出 Qini curve ，把提升缩放treatment组的样本规模上，用一个样本比例即treatment组和control组的比例来修正，这样更加公平。

$$ g(k) = R^T_\pi(k) - R^C_\pi(k)\frac{N^T_\pi(k)}{N^C_\pi(k)} $$


不难发现， Qini Curve 与Uplift Curve的关系如下：

$$ f(k) = \frac{g(k)(N^T_\pi(k) + N^C_\pi(k))}{N^T_\pi(k)} $$

![image](https://github.com/ShaoQiBNU/uplift_model_notes/blob/main/imgs/3.jpg)


5. Qini Coefficient

Qini coefficient就是Qini curve与baseline之间的面积比上best model curve与baseline之间的面积。越大越好。具体公式如下：

$$ Q_\pi(k) = \sum_{i=1}^k (R^T_\pi(i) - R^C_\pi(i)\frac{N^T_\pi(i)}{N^C_\pi(i)}) - \frac{k}{2}( \overline{R}^T(k) - \overline{R}^C(k)\frac{N^T_\pi(k)}{N^C_\pi(k)}) $$

$$ Qini \; Coefficient = \frac{\int Q_\pi(\rho ) {\rm d}\rho}{\int Q^*_\pi(\rho ) {\rm d}\rho} $$ 

$$ {\int Q^*_\pi(\rho ) {\rm d}\rho}代表best  model算出来的面积 $$

Qini Curve在实际情况中通常会比Uplift Curve更好，主要是因为Qini Curve可以处理treatment 组和contorl 组样本数量差异大的情况，具有稳定性，所以 Qini coefficient 会比AUUC 更实用一些。


6. 图像解释

在uplift curve和Qini curve的图像里，均有best model和baseline两条线，两种算法下，两条线的形状也有差异。这里重点解释一下，假设样本集有4种情况：

Y=1|T=1 

Y=0|T=1 

Y=0|T=0

Y=1|T=0 

https://github.com/ShaoQiBNU/uplift_model_notes/blob/main/uplift%E6%A8%A1%E5%9E%8B%E5%BB%BA%E6%A8%A1%E4%B8%8E%E8%AF%84%E4%BC%B0.ipynb
里的测试集具体统计如下：

|  转化   | treatment | control  | 总计  |
|  ----   | ---- | ---- | ---- |
| Y = 1  | 1640($R^T$) | 1118($R^C$) |  2758 | 
| Y = 0  | 9113 | 9476 |  18589 | 
| 总计    | 10753($N^T$)  | 10594($N^C$) | 21347 |

- baseline

baseline直观解释就是：任意取 $k$ 个样本，计算对应的 $f(k)$ 和 $g(k)$ 。uplift curve和Qini curve最后一定会和baseline交汇，因为在全部样本下，uplift curve与Qini curve和baseline的计算结果必定相等，如下所示：

|    | 横轴值x | 纵轴值y | 斜率k | 
|  ----   | ---- | ---- | ---- |
| uplift curve  | $x \in [0, 21347]$ | $y = (\frac{1640}{10753} - \frac{1118}{10594}) * (10753 + 10594) = 1002.9705 $ | $k = \frac{1002.9705}{10753 + 10594} = 0.04698 = \frac{R^T}{N^T} - \frac{R^C}{N^C}$
| Qini curve  | $x \in [0, 21347]$ | $y = 1640 - 1118 * \frac{10753}{10594} = 505.2205$ | $k = \frac{505.2205}{10753 + 10594} = 0.023667 = \frac{R^T - R^C\frac{N^T}{N^C}}{N^T + N^C}$ 


- best model
最优模型对这4种情况赋予的uplift值有如下的排序：Y=1|T=1的样本一定排在最前面；Y=1|T=0的样本一定排在最后，因为这种人代表了一个negative effect，即“反人类”，给券和购买的行为反着来；其次是 Y=0|T=0 和 Y=0|T=1。

这个曲线首先肯定会是一条斜率为1的曲线，为什么呢？理论上最好的模型会把真正uplift最大（最有可能转化）的样本放前面（预测出来uplift也最大），所以曲线会先计算Y=1|T=1这些样本。然后把这些样本计算完之后，都是Y=0|T=1和Y=0|T=0的样本，这时候对整体的的cumulative uplift没有影响，所以会是一条直线不变。但是到了Y=0|T=1和Y=0|T=0这些样本消化完后，会轮到Y=1|T=0，这些就会对cumulative uplift产生负影响，所以会是一条斜率为负值的直线下降，直到回归baseline的终点。


|  样本 |  metric  | 横轴值x | 纵轴值y | 斜率k | 
|  ----   |  ----   | ---- | ---- | ---- |
| Y=1, T=1  | uplift curve  | $x \in [0, 1640]$ | $y = (\frac{1640}{1640} - \frac{0}{0}) * (1640 + 0) = 1640 $ | $k = \frac{1640}{1640} = 1$
| Y=1, T=1  | Qini curve  | $x \in [0, 1640]$ | $y = 1640 - 0 * \frac{1640}{0} = 1640$ | $k = \frac{1640}{1640} = 1$ | 
| Y=0, T=0  | uplift curve  | $x \in [1640, 1640+9476]$ | $y = (\frac{1640}{1640} - \frac{0}{9476}) * (1640 + 9476) = 11116$ | $k = \frac{11116-1640}{9476} = 1$ | 
| Y=0, T=0  | Qini curve  | $x \in [1640, 1640+9476]$ | $y = 1640 - 0 * \frac{1640}{9476} = 1640$ | $k = \frac{1640-1640}{9476} = 0$ | 
| Y=0, T=1  | uplift curve  | $x \in [1640+9476, 1640+9476+9113]$ | $y = (\frac{1640}{1640+9113} - \frac{0}{9476}) * (1640 + 9476 + 9113) = 3085.2376 $ | $k = \frac{3085.2376 - 11116}{9113} = -0.8812$ |
| Y=0, T=1  | Qini curve  | $x \in [1640+9476, 1640+9476+9113]$ | $y = 1640 - 0 * \frac{1640+9113}{9476} = 1640$ | $k = \frac{1640-1640}{9113} = 0$ |
| Y=1, T=0  | uplift curve  | $x \in [1640+9476+9113, 1640+9476+9113+1118]$ | $y = (\frac{1640}{1640+9113} - \frac{1118}{1118+9476}) * (1640+9476+9113+1118) = 1002.9705$ | $k = \frac{1002.9705 - 3085.2376}{1118} = -1.8624$ |
| Y=1, T=0  | Qini curve  | $x \in [1640+9476+9113, 1640+9476+9113+1118]$ | $y = 1640 - 1118 * \frac{1640+9113}{9476+1118} = 505.2205 $ | $k = \frac{505.2205 - 1640}{1118} = -1.01500 = - \frac{R^T - (R^T - R^C \frac{N^T}{N^C})}{R^C} = - \frac{N^T}{N^C} $ |

特别地，对于Qini curve，当 $T=1$ 和 $T=0$ 的样本量相同时，最后一段回归baseline终点的直线斜率为-1。

参考：

https://mp.weixin.qq.com/s/j1QXJjWTWfd7nurqLJ3lGg

https://zhuanlan.zhihu.com/p/399322196

https://blog.csdn.net/zwqjoy/article/details/124493074

https://hwcoder.top/Uplift-1#%E6%A8%A1%E5%9E%8B%E8%AF%84%E4%BC%B0

代码采用开源scikit-uplift：

https://www.uplift-modeling.com/en/latest/
https://github.com/maks-sh/scikit-uplift/tree/master


## 业界应用

> 业界应用时会先做小流量探索实验收集训练样本，用于uplift model的建模。根据业务特性和uplift model的结果上线策略调整实验。

### 求uplift增益，确定业务激励或营销程度的调整策略

#### 腾讯Push
https://zhuanlan.zhihu.com/p/451884908


#### 快手Push
用uplift建模用户DAU增益价值，control组，不发Push；treatment组，发Push；优化目标：是否DAU，二分类


#### 抖音金币增发
> 用uplift建模用户在增发金币的增益价值(LT、LTV和duration)，control组，大盘金币系数；treatment组，大盘金币系数+0.3(随机写的)；优化目标：LT、LTV和duration。

> 如果需求是金币减发，则control组，大盘金币系数；treatment组，大盘金币系数-0.3(随机写的)；优化目标：LT、LTV和duration。

> 线上实验时，根据uplift score选择k%的用户做金币增发或者减发，查看实验指标。


#### 抖音广告个性化adload(adload：推送广告占比)
> 用uplift建模用户在降低adload的LTV/duration/留存等增益价值，control组，大盘金adload系数；treatment组，大盘adload系数-0.3(随机写的)；优化目标：LTV。

> 线上实验时，根据uplift score选择k%的用户做adload调整，查看实验指标。

**由于降低adload或金币减发会导致LTV的uplift score是负值，因此uplift curve是向下弯曲的，AUUC小于0.5**


### 优惠券个性化发放

> 优惠券发放涉及到成本预算和roi的问题，因此在求解uplift model之后，还需要做运筹优化求解，在成本和roi限制下，寻找最优解(决定用户发面额多少的优惠券)。在拉新场景下，所有用户都会发放优惠券，不存在control组，全部是treatment组，此时对uplift model的评估可以采用这种方式：激励最低的组作为control组，求其余组对该组的Qini Coefficient和AUUC等。


#### 字节千人千券

**相关变量及含义**
|  字段名 | 含义  | 应用
|  ----  | ---- | ---- |
| user_id  | 用户id |  |
| coupon_id | 优惠券id |  |
| $B$ | 核销预算/发放预算 | 预算约束 |
| $C$ | 现金消耗 | 消耗约束 |
| coupon_reduce | 优惠券的赠款，即 $c_{j}$ | 预算约束 |
| coupon_threshold | 优惠券的门槛，即 $t_{j}$ | 消耗约束 |
| is_convert | 用户在优惠券 $j$ 下的转化概率，即 $v_{j}$ | 求解目标，最大化转化概率 |
| control_convert | control下的转化概率，即用户自然转化率 $v_{0}$
| is_convert_uplift | 用户在优惠券 $j$ 的转化概率增益，即 $v_{j} - v_{0}$ | 求解目标，最大化转化概率增益 |
| expected_LTV | 用户期望LTV，即 $v_{j} * LTV$ | 求解目标，最大化LTV |
| expected_LTV_uplift | 用户期望LTV增益，即 $(v_{j} - v_{0}) * LTV$ | 求解目标，最大化LTV增益 |
| expected_reduce | $v_{j} * c_{j}$ | 预算约束 |
| expected_threshold | $v_{j} * t_{j}$ |  消耗约束；求解目标，最大化现金消耗 |
| expected_threshold_uplift | $(v_{j} - v_{0}) * t_{j}$ |  消耗约束；求解目标，最大化现金消耗增益 |


**举例来说：**
> 发放预算为 $B$，消耗约束为 $C$，则 $ROI = \frac{C}{B}$,  $v_{i,j}$ 为第 $i$ 个用户使用第 $j$ 张优惠券的转化概率,  $v_{i,0}$ 为第 $i$ 个用户的自然转化概率即不发券时的转化概率,  $x_{i,j}$ 为第 $i$ 个用户是否使用第 $j$ 张优惠券,  $c_{j}$ 为第 $j$ 张优惠券的增款,  $t_{j}$ 为第 $j$ 张优惠券的门槛，${j=0}$ 表示不发优惠券, 则优惠券的分配问题可以转化为如下的**整数规划问题**：

$$ \max \sum_{i=1}^{M} \sum_{j=0}^{N} (v_{i,j} - v_{i,0}) x_{i,j} = \max \left \lbrace \sum_{i=1}^{M} \sum_{j=0}^{N} (v_{i,j} x_{i,j}  - v_{i,0} x_{i,j}) \right \rbrace $$ 

$$ = \max \left \lbrace \sum_{i=1}^{M} \sum_{j=0}^{N} v_{i,j} x_{i,j}  - \sum_{i=1}^{M} \sum_{j=0}^{N}  v_{i,0} x_{i,j} \right \rbrace = \max \left \lbrace \sum_{i=1}^{M} \sum_{j=0}^{N} v_{i,j} x_{i,j}  - \sum_{i=1}^{M} v_{i,0} \sum_{j=0}^{N} x_{i,j} \right \rbrace = \max \left \lbrace \sum_{i=1}^{M} \sum_{j=0}^{N} v_{i,j} x_{i,j} - \sum_{i=1}^{M} v_{i,0} \right \rbrace $$

$$ \iff \max \left \lbrace \sum_{i=1}^{M} \sum_{j=0}^{N} v_{i,j} x_{i,j} \right \rbrace $$

$$ s.t. \sum_{i=1}^{M}\sum_{j=1}^{N} c_{j}x_{i,j} \leq B $$

$$ \sum_{i=1}^{M}\sum_{j=1}^{N} (v_{i,j} - v_{i,0})t_{j}x_{i,j} \geq C $$

$$ \sum_{j=0}^{N} x_{i,j}=1, \forall i $$

$$ x_{i,j} \geq 0, \forall i,j $$

> 其中 $v_{i,j}$ 和 $v_{i,0}$ 表示用户在treatment组和control组下的转化率，用uplift建模，treatment组:随机发放优惠券(满60减10，满90减20，满98减30)，control组:不发优惠券；优化目标：是否转化，二分类，0代表用户领取优惠券后x天内未转化或未核销该券，1代表用户领取优惠券后x天内转化或核销该券; $M$ 和 $N$ 代表用户数和优惠券个数, $j=0$表示不发券, $j=1...N$代表 $1-N$ 张优惠券。

> **整数规划问题**的求解可以采用拉格朗日乘数法，具体如下：

$$ max L(x,\lambda) = {\max_{x}} {\min_{\lambda_B, \lambda_C}}  \sum_{i=1}^M\sum_{j=0}^Nv_{ij}x_{ij}+\lambda_B(B-\sum_{i=1}^M\sum_{j=1}^Nc_{j}x_{ij})+\lambda_C(\sum_{i=1}^M\sum_{j=1}^N(v_{ij}-v_{i0})t_{j}x_{ij}-C) $$

求解算法采用ALS(alternating least squares)进行迭代求解：

1. greedy初始化 $x_{ij}$：

$$ j_i= \arg \max_{j} v_{ij} , x_{ij_i}=1 $$

2. 对 $\lambda$ 求最小值，沿梯度方向更新：

$$ \lambda_B=\max (0, \lambda_B-\alpha{(B-\sum_{i=1}^M\sum_{j=1}^Nc_{j}x_{ij})}) $$

$$ \lambda_C=\max (0, \lambda_C-\alpha{(\sum_{i=1}^M\sum_{j=1}^N(v_{ij}-v_{i0})t_{j}x_{ij}-C)}) $$

3. 固定当前 $\lambda$，通过遍历对 $x_{ij}$ 进行更新，确定第 $j$ 张优惠券的收益最大：

$$j_i= \arg \max_{j} {v_{ij}-\lambda_Bc_j+\lambda_C(v_{ij}-v_{i0})t_j} , x_{ij_i}=1$$

4. 重复2和3，直至 $\lambda$ 收敛
其中 $\lambda_B$ 、 $\lambda_C$ 和 $\alpha$ 为超参数

线上serving的时候，已知超参数 $\lambda_B, \lambda_C$ ，对于每个请求，遍历每张优惠券，计算收益，确定符合全局收益最大化的优惠券 $x_{i,j}$

$$ \arg \max_{x_{i,j}} v_{ij}x_{ij}+\lambda_B(B-c_{j}x_{ij})+\lambda_C\{(v_{ij}-v_{i0})t_{j}x_{ij}-C\}, x_{i,j}=1$$

即(去掉了公式中的常数项)

$$\arg \max_{j} {v_{ij}-\lambda_Bc_j+\lambda_C(v_{ij}-v_{i0})t_j}$$

**NOTE**
如果没有ROI的约束，此时 $\lambda_B$ 和 $\max v_{ij}$ 呈单调负相关，可以用二分法求解。对于 $\lambda_B$ 设置上界和下界，设置初始 $\lambda_B$，求解最优 $\max v_{ij}$ ，计算发放预算与设定的发放预算 $B$ 比较，如果计算的发放预算高于 $B$ ，则调大 $\lambda_B$，如果计算的发放预算小于 $B$ ，则调小 $\lambda_B$。

业界做法DCAF，参考：https://arxiv.org/pdf/2006.09684.pdf

**激励形式有优惠券和充赠红包，优惠券有折扣券、现金券、满减券；其中冲赠红包、现金券和满减券有明确的赠款金额；而折扣券都是5折、6折或7折的打折券，没有明确的赠款金额，常基于小流量探索实验阶段收集的训练样本统计折扣券的平均赠款作为赠款金额，用于后续运筹求解**

**优惠券一般为灌发形式，直接灌发到用户的账户里，以弹窗形式展现给用户，频控限制为3天1次，因此其曝光预算等同于发放预算，发放预算高于核销预算，因为用户核销优惠券的概率不恒等于1；充赠红包需要用户充值后再发放红包，在下单页面展示，没有频控限制，一天会曝光多次，其曝光预算高于发放预算。在运筹求解中，优惠券可以用发放预算作为约束求解(如同上例，此时 $B$ 需要除以优惠券核销率得到发放预算)，也可以使用核销预算作为约束求解(如同下式)；但充赠红包用发放预算求解会出现发放与核销gap较大的情况，必须使用核销预算作为约束，因此相应的约束公式变为：**

$$ s.t. \sum_{i=1}^{M}\sum_{j=1}^{N} c_{j}v_{ij}x_{i,j} \leq B $$


**对于预算约束、消耗约束和求解目标，可以根据实际业务进行组合，求解公式如表所示：**

|  求解目标 |  预算约束   | 消耗约束  | 求解公式  |
|  ----    |  ----  | ----  | ---- |
| $v_{j}$ | $c_{j}$ | $t_{j}$ | $$\arg \max_{j} {v_{ij}-\lambda_Bc_j+\lambda_Ct_j}$$
| $v_{j}$ | $c_{j}$ | $(v_{j} - v_{0}) * t_{j}$ | $$\arg \max_{j} {v_{ij}-\lambda_Bc_j+\lambda_C(v_{ij}-v_{i0})t_j}$$
| $v_{j}$ | $v_{j} * c_{j}$ | $t_{j}$ | $$\arg \max_{j} {v_{ij}-\lambda_Bv_{ij}c_j+\lambda_Ct_j}$$
| $v_{j}$ | $v_{j} * c_{j}$ | $(v_{j} - v_{0}) * t_{j}$ | $$\arg \max_{j} {v_{ij}-\lambda_Bv_{ij}c_j+\lambda_C(v_{ij}-v_{i0})t_j}$$
| ... | ... | ... | ...


**pyspark实现代码**

1. mock一份数据

> 数据字段如下，每一行代表用户在每张优惠券下的转化概率、自然转化概率，以及该优惠券的赠款和门槛，根据业务情况自行选择 $B$ 和 $C$ 的计算逻辑。

|  字段名 | 含义  |
|  ----  | ---- |
| user_id  | 用户id |
| coupon_id | 优惠券id |
| coupon_reduce | 优惠券的赠款，用于成本约束，即 $c_{j}$
| coupon_threshold | 优惠券的门槛，用于消耗约束，即 $t_{j}$
| is_convert | 在优惠券id下的转化概率
| control_convert | control下的转化概率

- 实际业务场景中，可能不存在空白组，导致无法获得用户不发优惠券的预估概率值，但数据中必须保留每个用户不发券的概率值，将其置为0。

2. MCKP求解

根据2和3更新参数求解，得到每个用户的最优解，具体见：https://github.com/ShaoQiBNU/uplift_model_notes/blob/main/%E5%8D%83%E4%BA%BA%E5%8D%83%E5%88%B8MCKP%E6%B1%82%E8%A7%A3.ipynb

3. DCAF的二分查找求解
具体见：https://github.com/ShaoQiBNU/uplift_model_notes/blob/main/%E5%8D%83%E4%BA%BA%E5%8D%83%E5%88%B8MCKP%E6%B1%82%E8%A7%A3.ipynb


拉格朗日乘数法参考：

https://zhuanlan.zhihu.com/p/55279698

https://zhuanlan.zhihu.com/p/55532322

https://dezeming.top/wp-content/uploads/2021/09/%E6%8B%89%E6%A0%BC%E6%9C%97%E6%97%A5%E4%B9%98%E5%AD%90%E6%B3%95%E2%80%94%E2%80%94%E5%B8%A6%E4%B8%8D%E7%AD%89%E5%BC%8F%E7%BA%A6%E6%9D%9F%E9%A1%B9%E7%9A%84%E5%87%BD%E6%95%B0%E4%BC%98%E5%8C%96.pdf


3. PID控制预算约束

由于业务场景受求解人群分布与实际线上人群分布差异、其他活动流量竞争等影响，期望约束预算与实际核销预算之间存在gap，常采用PID算法控制约束进度，进而保证预算平稳消耗，可小时级调整，也可天级调整。
激励场景下的PID公式如下(以天级为例)：

$$ E_{T} = K_{P} P_{T} + K_{I} I_{T} + K_{D} D_{T} $$

$$ K_{P}， K_{I}， K_{D}是超参数，用于控制调节比例的系数 $$

$$ P是预算误差 P_{T} = B_{target} - B_{T-1}，B_{target} 是期望预算，B_{T-1} 是实际预算 $$

$$ I是预算误差的积分，I_{T} = \sum_{t=T-i}^{T}P_{T}，i是积分天数 $$

$$ D是预算误差的微分，D_{T} = P_{T} - P_{T-1} $$

预算调整公式如下：

$$ B_{T} = B_{target} + E_{T} $$

这是位置型PID，实际应用中，常采用增量型PID，方便存储和计算，具体公式如下：

$$ E_{T} = K_{P} P_{T} + K_{I} I_{T} + K_{D} D_{T} $$

$$ E_{T-1} = K_{P} P_{T-1} + K_{I} I_{T-1} + K_{D} D_{T-1} $$

$$ \Delta E_{T} = K_{P} (P_{T} - P_{T-1}) + K_{I} (I_{T} - I_{T-1}) + K_{D} (D_{T} - D_{T-1}) $$


$$ = K_{P} (P_{T} - P_{T-1}) + K_{I} P_{T} + K_{D} (P_{T} - 2 * P_{T-1} + P_{T-2}) $$

$$ E_{T} = E_{T-1} + \Delta E_{T} $$

具体代码参考：https://github.com/ShaoQiBNU/uplift_model_notes/blob/main/PID%E8%B0%83%E6%8E%A7%E7%AE%97%E6%B3%95.ipynb

