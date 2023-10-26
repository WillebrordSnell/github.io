---

date: 2023-10-26
category:
  - 码头
tag:
  - 视频对话

---
# 视频对话方向综述性质的记录
视频对话领域在近几年顶刊上的paper寥寥无几并且在大模型的冲击下许多工作本就是换皮，本无打算特地做综述性质的报告，但由于开题需要，故留记录

## Information-Theoretic Text Hallucination Reduction for Video-grounded Dialogue
> 论文地址：

本文提出AVSD中存在：不理解问题的情况下不加区分地抄袭输入文本，其原因是由于数据集中的答案句子通常包含输入文本中的单词，因此VGD(Video-grounded Dialogue)系统过度依赖于复制输入文本中的单词，希望这些单词与地面实况文本重叠，从而学习到虚假的相关性。

## Video Dialog as Conversation about Objects Living in Space-Time
> 论文地址：


本文认为：对话的自然流程是多次转折，每次转折都建立在之前的问答基础之上。这就要求在语言上深刻理解和跟踪已经说过的话，并以视频中的视觉概念为基础，然后在新建立的语境中分析新问题。其难点主要体现在 场景的时间动态性 


## Structured Co-reference Graph Attention for Video-grounded Dialogue
> 论文地址：

本文认为视频对话难点在于：(1) 如何推断多种模态之间的共同参照(包含文本中的指代消解问题)；(2) 如何推理具有复杂时空动态的视频的丰富底层语义结构。

## BiST: Bi-directional Spatio-Temporal Reasoning for Video-Grounded Dialogues
>论文地址：

本文认为AVSD：(i) 包含空间和时间变化的视频的复杂性，以及 (ii) 在多个对话回合中查询视频中不同片段和/或不同对象的用户话语的复杂性。并且此前的方法都认为空间中所有的object都是一样重要，但其实并不是所有object对于当前问题有同样的贡献度

## Describing Unseen Videos via Multi-Modal Cooperative Dialog Agents
>论文地址：

参考意义不大

