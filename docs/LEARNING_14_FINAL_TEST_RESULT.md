# 学习 14：封存测试集最终验收

## 为什么现在才能使用测试集

模型结构、随机种子、数据划分、候选检查点和评估规则已经在测试前写入`FINAL_EVALUATION_PROTOCOL.md`。最终脚本先核对数据集和模型SHA-256，只有完全一致才会创建固定测试划分并进行推理。

本次是33,000条封存测试样本的首次正式评估。测试结果没有用于继续选择模型或修改参数。

## 最终结果

| 指标 | 验证集 | 最终测试集 | 差值 |
| --- | ---: | ---: | ---: |
| Accuracy | 56.35% | 56.38% | +0.03个百分点 |
| Macro F1 | 56.23% | 56.26% | +0.02个百分点 |

验证集和测试集的差距很小，说明候选模型在同分布封存数据上的表现较稳定。它不能证明模型在真实空口、其他设备或其他数据集上也能达到相同结果。

最终测试推理耗时约0.56秒。这个数值只统计数据已经准备完成后的GPU模型推理和指标收集，不包含数据文件加载、数据校验、模型初始化或真实在线通信延迟，因此不能当作端到端业务延迟。

## 按SNR观察

- -20 dB：Accuracy约9.58%；
- -6 dB：Accuracy约53.70%；
- 0 dB：Accuracy约79.15%；
- 2 dB：Accuracy约82.12%；
- 12 dB：Accuracy约82.67%；
- 18 dB：Accuracy约81.64%。

模型在极低SNR下仍接近11分类随机水平，从约-12 dB开始逐渐获得可辨识信息，在0 dB以上稳定在约79%至83%。

![TemporalCNN最终测试集SNR曲线](images/final_test/final_test_accuracy_by_snr.svg)

## 分类别问题

- QAM16 Recall约4.07%，大量被预测为QAM64；
- WBFM Recall约31.37%，大量被预测为AM-DSB；
- AM-SSB Recall约90.93%，但Precision约27.32%，说明模型把多种其他类别错误归入AM-SSB；
- PAM4、GFSK、BPSK等类别相对更稳定。

![TemporalCNN最终测试集混淆矩阵](images/final_test/final_test_confusion_matrix.svg)

## 最终结论

项目已经完成从数据安全检查、分层划分、PyTorch训练、基线对照、受控架构改进到封存测试集验收的完整闭环。最终候选模型为TemporalCNN，RadioML 2016.10A测试集Accuracy为56.38%，Macro F1为56.26%。

这些是完整11分类、-20 dB至18 dB全部SNR样本的总体指标。项目不会删除低SNR或困难类别来抬高最终数字。

最终测试结果JSON SHA-256：

```text
181fc5a6ba25743e9abe1928e17ca2fb6b1712f5fb9f32d36f74a4bfeb74a7bc
```
