# Grokking 现象研究项目

本项目旨在研究机器学习中的 **Grokking** 现象，特别是在模算术任务中的表现。我们首先复现了论文《Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets》中的实验，并进一步探讨了不同数据比例、不同模型架构和优化器对 Grokking 现象的影响。此外，我们还研究了更复杂的多元模加法问题，并对观察到的现象进行了解释。

## 项目结构

本项目包含以下六个主要代码文件，所有代码文件都是自包含的，可以直接运行：

1. **subtask1.py**  
   该文件实现了基础的 Transformer 模型，用于二元模加法任务。我们通过调整训练数据比例，观察 Grokking 现象的出现。

2. **subtask2.py**  
   该文件扩展了模型架构，引入了 MLP 和 LSTM 模型，并与 Transformer 模型进行对比，探讨不同模型在模加法任务中的表现。

3. **subtask3.py**  
   该文件进一步研究了不同优化器（如 AdamW、SGD、RMSprop 等）和正则化技术（如 Dropout、权重衰减）对 Grokking 现象的影响。

4. **subtask4.py**  
   该文件将问题扩展到多元模加法任务，研究了不同加数数量（K）对模型表现的影响

5. **slingshot.ipynb**  
   该文件包含了对“slingshot”现象的详细分析与实验代码。

6.**sharpness.ipynb** 
   该文件包含了对训练阶段模型参数 sharpness变化的实验代码。

## 实验内容

### 1. 复现 Grokking 现象
我们首先复现了论文中的二元模加法任务，使用 Transformer 模型进行训练，并观察了在不同训练数据比例下 Grokking 现象的出现。实验结果表明，当训练数据比例较小时，模型在长时间训练后会突然出现泛化能力的提升。

### 2. 不同模型架构的对比
为了进一步理解 Grokking 现象，我们对比了 Transformer、MLP 和 LSTM 模型在模加法任务中的表现。实验发现，Transformer 模型在 Grokking 现象上表现最为明显，而 MLP 和 LSTM 模型的泛化能力提升较为平缓。

### 3. 优化器和正则化的影响
我们研究了不同优化器（如 AdamW、SGD、RMSprop）和正则化技术（如 Dropout、权重衰减）对 Grokking 现象的影响。实验结果表明，AdamW 优化器在大多数情况下表现最佳，而过强的正则化项会导致训练不稳定。

### 4. 多元模加法任务
我们将问题扩展到多元模加法任务，研究了不同加数数量（K）对模型表现的影响。实验发现，随着加数数量的增加，模型的训练难度显著增加，且 Grokking 现象的出现时间也相应延迟。

### 5. 对模型现象的进一步探索
我们试图通过slingshot机制和对模型sharpness变化的研究来理解模型训练时出现的特殊现象

