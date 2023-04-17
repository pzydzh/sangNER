# sangNER
基于pytorch实现的NER模型，包含了Bert+CRF和Bert两种模型结构。其中CRF使用的是pytorch_crf。

由于预训练的bert的学习率比较小，但是下游任务网络的学习率和bert的学习率不属于同一个量级，具体原因可以参考苏剑林的[这篇文章](https://kexue.fm/archives/7196)，
所以在CRF中采用了较大的学习率， 1e-3，而bert中的学习率为5e-5。

### 实验效果
在MSRA数据集上进行训练，Bert和Bert+CRF的F1都能达到0.95。

