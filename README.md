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
> 用uplift建模用户在某面额优惠券的转化增益价值，control组，不发优惠券；treatment组，发优惠券，面额是c元；优化目标：是否转化，二分类？？？

运筹求解：



#### 抖音金币增发
> 用uplift建模用户在增发金币的增益价值(LT、LTV和duration)，control组，大盘金币系数；treatment组，大盘金币系数+0.3(随机写的)；优化目标：LT、LTV和duration。

> 如果需求是金币减发，则control组，大盘金币系数；treatment组，大盘金币系数-0.3(随机写的)；优化目标：LT、LTV和duration。

> 线上实验时，根据uplift score选择k%的用户做金币增发或者减发，查看实验指标。

#### 抖音广告个性化adload
> 用uplift建模用户在降低adload的LTV/duration/留存等增益价值，control组，大盘金adload系数；treatment组，大盘adload系数-0.3(随机写的)；优化目标：LTV。

> 线上实验时，根据uplift score选择k%的用户做adload调整，查看实验指标。

**由于降低adload或金币减发会导致LTV的uplift score是负值，因此uplift curve是向下弯曲的，AUUC小于0.5**

待补充






