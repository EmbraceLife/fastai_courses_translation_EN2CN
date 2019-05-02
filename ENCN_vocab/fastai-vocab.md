
<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#fastai-deep-learning-vocabulary" data-toc-modified-id="fastai-deep-learning-vocabulary-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>fastai deep learning vocabulary</a></span><ul class="toc-item"><li><span><a href="#如何快速在网上查找中文对应词-How-I-find-key-vocab-translation-online" data-toc-modified-id="如何快速在网上查找中文对应词-How-I-find-key-vocab-translation-online-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>如何快速在网上查找中文对应词 How I find key vocab translation online</a></span></li><li><span><a href="#fastai-specific-vocab-专有词汇-英中对照" data-toc-modified-id="fastai-specific-vocab-专有词汇-英中对照-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>fastai specific vocab 专有词汇 英中对照</a></span></li><li><span><a href="#第六课-Lesson-6" data-toc-modified-id="第六课-Lesson-6-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>第六课 Lesson 6</a></span><ul class="toc-item"><li><span><a href="#第五课-Lesson-5-vocab" data-toc-modified-id="第五课-Lesson-5-vocab-1.3.1"><span class="toc-item-num">1.3.1&nbsp;&nbsp;</span>第五课 Lesson 5 vocab</a></span></li></ul></li><li><span><a href="#第四课-Lesson-4" data-toc-modified-id="第四课-Lesson-4-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>第四课 Lesson 4</a></span></li><li><span><a href="#第三课--Lesson-3" data-toc-modified-id="第三课--Lesson-3-1.5"><span class="toc-item-num">1.5&nbsp;&nbsp;</span>第三课  Lesson 3</a></span></li><li><span><a href="#课程视频中的其他词汇-vocab-not-DL-specific" data-toc-modified-id="课程视频中的其他词汇-vocab-not-DL-specific-1.6"><span class="toc-item-num">1.6&nbsp;&nbsp;</span>课程视频中的其他词汇 vocab not DL specific</a></span></li></ul></li></ul></div>

# fastai deep learning vocabulary

## 如何快速在网上查找中文对应词 How I find key vocab translation online

1. 书内搜索 Search within an open source book 
[邱锡鹏深度学习教科书开源](https://nndl.github.io/nndl-book.pdf)     
2. 网站搜索 Search on google    
Google: English term 中文/中文数学
3. 在线单词表 online vocab list    
机器之心的[深度学习词汇](https://github.com/jiqizhixin/Artificial-Intelligence-Terminology)，但较长时间未更新       
谷歌开发者机器学习[词汇表](https://zhuanlan.zhihu.com/p/29884825) , [EN version](https://developers.google.com/machine-learning/glossary/)      

## fastai specific vocab 专有词汇 英中对照
:white_check_mark: = 官方认可 approved by fastai

- crappify （翻译建议采集中translation options : 1. 垃圾化；2. 残次化）
- DataBunch  数据堆 :white_check_mark:
- discriminative learning rate （翻译建议采集中translation options ：1. 判别学习率; 2. 区别学习率）

## 第六课 Lesson 6

- weight tying 权重拴连 
- generative models 生成模型
- shape 数据形状
- linear interpolations 线性插值法
- average pooling 平均汇聚
- stride 2 convolution 步长为2的卷积
- rank 3 tensor 秩为3的张量
- channel 通道
- convolution kernel 卷积核
- reflection mode 反射模式
- padding mode 填充模式
- dihedral 二面角
- weight norm 权重归一
- Covariate Shift 协变量偏移
- Batch Normalization BN 批量归一化
- instance 实例化
- module 模块
- dropout mask 随机失活的掩码 
- Beroulli trial 伯努利实验
- test time / inference time 测试期/预测期
- training time 训练期
- spiking neurons 脉冲神经元
- tabular learner 表格学习器
- long tail distributions 长尾分布
- Root Mean Squared Percentage Error 均方根百分比误差
- nomenclature 名称系统
- cardinality 集合元素数量
- preprocessors 预处理
- RMSPE (root mean squared percentage error) 均方根百分比误差
- computer vision 机器视觉
- projection 投射

### 第五课 Lesson 5 vocab 

- MAPE mean absolute percentage error 平均绝对百分比误差 感谢 @thousfeet
- super-convergence 超级收敛
- dynamic learning rate 动态学习率
- exponentially weighted moving average 指数加权移动平均值
- epoch 迭代次数
- finite differencing 有限差分法
- analytic solution 解析解
- convergence 收敛
- divergence 散度
- L2 regularization L2正则化
- learning rate annealing 学习率退火
- element-wise function 元素逐一函数, 感谢与 @Moody 的探讨
- logistic regression model 逻辑回归模型
- flatten 整平 （numpy.arrays）
- actuals 目标真实值
- constructor 构造函数
- generalization 泛化
- 2nd degree polynomial 这是2次多项式 
- Gradient Boosted Trees 梯度提升树
- Entity Embeddings 实体嵌入
- NLL (negative log likelihood)  负对数似然
- PCA (Principal Component Analysis) 主成分分析
- weight decay 权值衰减
- benchmark 基准
- cross-validation 交叉验证
- latent factors 潜在因子
- array lookup 数组查找
- one-hot encoding   one-hot编码，或者一位有效编码，（或者 独热编码 感谢 @LiuYanliang）    
- Dimensions 维度
- transpose 转置矩阵处理
- Convolutions 卷积
- Affine functions 仿射函数
- Batch Normalization 批量归一化
- multiplicatively equal 乘数分布相同 （每层都10倍递增/减 1e-5, 1e-4, 1e-3)
- diagnal edges 对角线边角
- filter 过滤器
- target 目标值
- softmax softmax函数 （转化成概率值的激活函数）
- backpropagation 反向传递
- Universal Approximation Theorem 通用近似定理
- weight tensors 参数张量
- input activations 输入激活值 

## 第四课 Lesson 4

- mask 掩码
- matrix multiplication 矩阵乘法 
- dot product vs matrix product （单个数组和单个数组的乘法 = 点积，矩阵（多数组）与矩阵（多数组）的乘法 = 矩阵乘法）**当前字幕版本对这两个词混淆使用了（全用了"点积"这个词），下个版本会做修正。**
- unfreeze 解冻模型
- freeze 封冻模型
- cross-entropy 交叉熵
- scaled sigmoid 被放缩的S函数
- layers 层
- activations 激活值/层
- parameters 参数
- Rectified Linear Unit, ReLU 线性整流函数，（或者修正线性激活函数 感谢 @LiuYanliang）    
- non-linear activation functions 非线性激活函数
- nonlinearities 非线性激活函数
- sigmoid S函数
- bias vector 偏差数组 （或者 偏置向量 感谢 @LiuYanliang）    
- embedding matrix 嵌入矩阵
- bias 偏差
- dropout  随机失活 (感谢 @Junmin )，或者 丢弃法
- root mean squared error（RMSE）均方根误差
- mean squared error（MSE）均方误差
- sum squared error  残差平方和 
- vector 数组
- spreadsheet 电子表格
- dot product 点积
- state of the art 最先进的
- time series 时间序列
- cold start problem 冷启动问题
- timestamp 时间戳
- sparse matrix 稀疏矩阵
- collaborative filtering 协同过滤
- metrics 度量函数/评估工具
- end-to-end training 端到端训练
- fully connected layer 全联接层
- meta data 元数据 
- tabular learner 表格数据学习器
- dependent variable 应变量
- data augmentation 数据增强
- processes 预处理
- transforms 变形处理/设置
- categorical variable 类别变量
- continuous variable 连续变量
- feature engineering 特征工程
- gradient boosting machines 梯度提升器
- hyperparameters 超参数
- random forest 随机森林
- discriminative learning rate 判别学习率 
- tabular data 表格数据
- momentum 动量
- decoder 解码器
- encoder 编码器
- accuracy 精度
- convergence 收敛
- overfitting 过拟合
- underfitting 欠拟合
- inputs 输入值
- weight matrix 参数数组
- matrix multiply 数组相乘
- Tokenization 分词化
- Numericalization 数值化 
- Learner 学习器 （感谢 @stas 学习器与模型的内涵[对比](https://forums.fast.ai/t/deep-learning-vocab-en-vs-cn/42297/11?u=daniel)）
- target corpus 目标文本数据集
- Supervised Learning/models 监督学习/模型
- Self-Supervised Learning 自监督学习
- pretrained model 预训练模型
- fine-tuning 微调

## 第三课  Lesson 3

- independent variable 自变量 （感谢 @Junmin 的指正）
- Image Classification 图片分类
- Image Segmentation 图片分割
- Image Regression 图片回归
- CNN Convolution Neural Network 卷积神经网络
- RNN Recurrent Neural Network 循环神经网络
- NLP Natural Language Processing  自然语言处理 
- language model 语言模型 

## 课程视频中的其他词汇 vocab not DL specific
- take it with a slight grain of salt 不可全信
- come out of left field 不常见的
- elapsed time 所经历的时间 :rescue_worker_helmet: [出现时间点](https://youtu.be/hkBa9pU-H48?t=895)
- connoisseur 鉴赏级别/专业类电影
- nomenclature 专业术语
- rule of thumb 经验法则
- asymtote 渐进
- delimiter 分隔符
- enter 回车键
- macro 宏
- unwieldy 困难
- infuriating 特别烦人
- hone in on it 精确定位目标
- hand waving 用手做比划/解释
- string 字符串
- list 序列
- 深度学习 deep learning
- 机器学习 machine learning
- 学习算法 learning algorithm
- 模型 model
- 数据集 data set 
- 示例 instance 样本 sample
- 属性 attribute 特征 feature
- 属性值 attribute value 
- 样本空间 sample space
- 特征向量 feature vector
- 维度数量 dimensionality
- 学习 learning 训练 training 
- 训练数据 training data
- 训练样本 training sample, training example
- 训练集 training set
- 假设 hypothesis
- 真相 ground-truth
- 学习器 learner = model
- 预测 prediction
- 标记 label
- 样例 example
- 标记空间 输出空间 label space 
- 分类 classification
- 回归 regression
- 二分类 binary classification
- 正类 positive class
- 反类 negative class
- 多分类 multi-class classification
- 测试 testing 
- 测试样本 testing sample
- 聚类 clustering
- 簇 cluster
- cluster = 没有标记下的分类，通过挖掘数据结构特征发现的
- class = 给定标记的分类，事先给定的
- 监督学习 supervised learning
- 无监督学习 unsupervised learning
- 泛化能力 generalization
- 分布 distribution 
- 独立同分布 independent and identically distributed i.i.d.


```python

```
