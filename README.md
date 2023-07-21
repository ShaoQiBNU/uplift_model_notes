# uplift模型业界应用总结

## 何为uplift？

> uplift model 用于预测treatment的增量反馈价值，常应用在Push推送、广告投放等场景。比如，我们想知道对用户展现广告的价值，通常的模型只能建模用户在展示广告后的购买意愿，但事实很有可能是他们在被展示广告之前就已经很想购买了，这个时候展示广告反而会增加投放成本。因此，Push推送和广告投放等场景常采用uplift model建模增量反馈价值，在减少对用户打扰和降低成本的同时，提高业务价值，如DAU增益、广告主价值、LTV、观看时长等。

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
待补充

## 业界应用

> 业界应用时会先做小流量探索实验收集训练样本，用于uplift model的建模

> 根据业务特性和uplift model的结果上线策略调整实验。其中，优惠券发放涉及到成本预算和roi的问题，因此在求解uplift model之后，还需要做运筹优化求解，在成本和roi限制下，寻找最优解(决定用户发面额多少的优惠券)。

### Push 

#### 腾讯
https://zhuanlan.zhihu.com/p/451884908

#### 快手
用uplift建模用户DAU增益价值，control组，不发Push；treatment组，发Push；优化目标：是否DAU，二分类


### 优惠券

#### 字节千人千券

> 总预算为 $B$，消耗约束为 $C$，则 $ROI = \frac{C}{B}$,  $v_{i,j}$ 为第 $i$ 个用户使用第 $j$ 张优惠券的转化概率,  $v_{i,0}$ 为第 $i$ 个用户的自然转化概率,  $x_{i,j}$ 为第 $i$ 个用户是否使用第 $j$ 张优惠券,  $c_{j}$ 为第 $j$ 张优惠券的增款,  $t_{j}$ 为第 $j$ 张优惠券的门槛，则优惠券的分配问题可以转化为如下的**整数规划问题**：

$$ \max \sum_{i=1}^{M} \sum_{j=1}^{N} v_{i,j}x_{i,j} $$

$$ s.t. \sum_{i=1}^{M}\sum_{j=1}^{N} c_{j}x_{i,j} \leq B $$

$$ \sum_{i=1}^{M}\sum_{j=1}^{N} (v_{i,j} - v_{i,0})t_{j}x_{i,j} \geq C $$

$$ \sum_{j=1}^{N} x_{i,j}=1, \forall i $$

$$ x_{i,j} \geq 0, \forall i,j $$

> 其中 $v_{i,j}$ 和 $v_{i,0}$ 表示用户在treatment组和control组下的转化率，用uplift建模，treatment组，随机发放优惠券(满60减10，满90减20，满98减30)，control组，不发优惠券；优化目标：是否转化，二分类，0代表用户领取优惠券后x天内未转化或未核销该券，1代表用户领取优惠券后x天内转化或核销该券; $M$ 和 $N$ 代表用户数和优惠券个数。

> **整数规划问题**的求解可以采用拉格朗日乘数法，具体如下：

$$ max L(x,\lambda) = {\max_{x}} {\min_{\lambda_B, \lambda_C}}  \sum_{i=1}^M\sum_{j=1}^Nv_{ij}x_{ij}+\lambda_B(B-\sum_{i=1}^M\sum_{j=1}^Nc_{j}x_{ij})+\lambda_C(\sum_{i=1}^M\sum_{j=1}^N(v_{ij}-v_{i0})t_{j}x_{ij}-C) $$

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

**激励形式有优惠券和充赠红包，优惠券有折扣券、现金券、满减券；其中冲赠红包、现金券和满减券有明确的赠款金额；而折扣券都是5折、6折或7折的打折券，没有明确的赠款金额，常基于小流量探索实验阶段收集的训练样本统计折扣券的平均赠款作为赠款金额，用于后续运筹求解**

**优惠券一般为灌发形式，直接灌发到用户的账户里，曝光预算等同于发放预算，约束为发放预算；而充赠红包，需要用户充值后再发放红包，曝光预算高于发放预算，但约束是发放预算，因此相应的约束公式变为：**

$$ s.t. \sum_{i=1}^{M}\sum_{j=1}^{N} c_{j}v_{ij}x_{i,j} \leq B $$

**对于消耗约束 $C$，有的场景限制为uplift下的消耗，有的场景限制为普通消耗，汇总两者总结下来，如表所示：**

|  字段名称   | 字段含义  | 消耗  |
|  ----  | ----  | ---- |
| expected_coupon_reduce | coupon_reduce | $B$
| expected_coupon_convert_reduce | is_convert * coupon_reduce | $B$
| expected_coupon_threshold | is_convert * coupon_threshold | $C$
| expected_coupon_threshold_uplift | (is_convert - control_convert) * coupon_threshold | $C$

**针对 $B$ 和 $C$ 的所有组合情况，求解公式如表所示：**

|  预算约束   | 消耗约束  | 求解公式  |
|  ----  | ----  | ---- |
| expected_coupon_reduce | expected_coupon_threshold | $$\arg \max_{j} {v_{ij}-\lambda_Bc_j+\lambda_Ct_j}$$
|expected_coupon_reduce | expected_coupon_threshold_uplift | $$\arg \max_{j} {v_{ij}-\lambda_Bc_j+\lambda_C(v_{ij}-v_{i0})t_j}$$
| expected_coupon_convert_reduce | expected_coupon_threshold | $$\arg \max_{j} {v_{ij}-\lambda_Bv_{ij}c_j+\lambda_Ct_j}$$
| expected_coupon_convert_reduce | expected_coupon_threshold_uplift | $$\arg \max_{j} {v_{ij}-\lambda_Bv_{ij}c_j+\lambda_C(v_{ij}-v_{i0})t_j}$$

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


- 对于促活业务场景，存在不发优惠券的情况，所以每个用户会有一行空优惠券的记录，空优惠券同样参与运筹求解；

- 对于拉新业务场景，可能存在不发优惠券的情况，为了兼容计算，也会对每个用户增加一行空优惠券的记录，相关值会置为0，虽然参与运筹求解，但始终是最小值，不会被选择。


2. MCKP求解

根据2和3更新参数求解，得到每个用户的最优解，具体见：https://github.com/ShaoQiBNU/uplift_model_notes/blob/main/%E5%8D%83%E4%BA%BA%E5%8D%83%E5%88%B8MCKP%E6%B1%82%E8%A7%A3.ipynb


拉格朗日乘数法参考：

https://zhuanlan.zhihu.com/p/55279698

https://zhuanlan.zhihu.com/p/55532322

https://dezeming.top/wp-content/uploads/2021/09/%E6%8B%89%E6%A0%BC%E6%9C%97%E6%97%A5%E4%B9%98%E5%AD%90%E6%B3%95%E2%80%94%E2%80%94%E5%B8%A6%E4%B8%8D%E7%AD%89%E5%BC%8F%E7%BA%A6%E6%9D%9F%E9%A1%B9%E7%9A%84%E5%87%BD%E6%95%B0%E4%BC%98%E5%8C%96.pdf


3. PID控制预算约束

由于业务场景受求解人群分布与实际线上人群分布差异、其他活动流量竞争等影响，期望约束预算与实际核销预算之间存在gap，常采用PID算法控制约束进度，进而保证预算平稳消耗，可小时级调整，也可天级调整。
激励场景下的PID公式如下(以天级为例)：

$$ E_{T} = K_{P} P_{T} + K_{I} I_{T} + K_{D} D_{T} $$

其中, $$ K_{P}， K_{I}， K_{D}是超参数，用于控制调节比例的系数 $$

$$ P是预算误差 P_{T} = \frac{B_{target} - B_{T-1}}{B_{target}}，B_{target} 是期望预算，B_{T-1} 是实际预算 $$

$$ I是预算误差的积分，I_{T} = \sum_{t=T-i}^{T}P_{T}，i是积分天数 $$

$$ D是预算误差的微分，D_{T} = P_{T} - P_{T-1} $$

预算调整公式如下：

$$ B_{T} = B_{T-1} + E_{T} B_{target} $$

$$ B_{0} = B_{target} $$


#### 抖音金币增发
> 用uplift建模用户在增发金币的增益价值(LT、LTV和duration)，control组，大盘金币系数；treatment组，大盘金币系数+0.3(随机写的)；优化目标：LT、LTV和duration。

> 如果需求是金币减发，则control组，大盘金币系数；treatment组，大盘金币系数-0.3(随机写的)；优化目标：LT、LTV和duration。

> 线上实验时，根据uplift score选择k%的用户做金币增发或者减发，查看实验指标。

#### 抖音广告个性化adload(adload：推送广告占比)
> 用uplift建模用户在降低adload的LTV/duration/留存等增益价值，control组，大盘金adload系数；treatment组，大盘adload系数-0.3(随机写的)；优化目标：LTV。

> 线上实验时，根据uplift score选择k%的用户做adload调整，查看实验指标。

**由于降低adload或金币减发会导致LTV的uplift score是负值，因此uplift curve是向下弯曲的，AUUC小于0.5**

待补充






