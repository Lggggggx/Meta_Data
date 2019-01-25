# Meta_Data
meta data designed for active learn

# QueryMetaData.py 我们设计主动学习策略类
将我们设计的主动学习方法，封装成为类，方便调用

# generate_metadata_main.py生成Metadata
在不同数据集上，根据已经确定的划分，生成对应数据集的Metadata，其中一条Metadata数据为[396维特征 + 4维对应模型性能变化]

# regressor_parameter.py 用于确定回归器最优的参数
在已经生成的Metadata数据集上，利用sklearn的GridSearchCV寻找最优参数

# almain.py 主动学习对比实验
选择一个数据集上测试，将剩余的数据集Metadata用于训练已经确定最优参数的回归器，将其作为主动学习选择策略的指标。
对比方法有random、uncertainty、QBC、EER(ExpectedErrorReduction)
