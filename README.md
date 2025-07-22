# bert_model-fine-tuning-classify-model

基于 [BERT](https://huggingface.co/bert-base-chinese) 的中文文本分类项目，旨在通过微调（fine-tuning）`bert-base-chinese` 模型，实现一个 **情感分析模型**。
---

## 📌 项目简介

本项目聚焦于中文文本分类任务，使用预训练的 BERT 模型，结合 PyTorch 完成模型微调。训练得到的模型适用于任意中文输入句子的多分类判断场景，例如情感分析、意图识别等。

---

## 🔧 技术栈

- Python 3.x
- PyTorch
- Transformers（来自 Hugging Face）
- Anaconda 环境管理

---

## 📦 微调数据说明
- params中的two_classify_pth是微调好的二分类模型。使用的数据集为：/Users/liushanshan/my-project/bert_model-fine-tuning-classify-model/data/ChnSentiCorp
- six_classify_pth是微调好的六分类模型。使用的数据集为：/Users/liushanshan/my-project/bert_model-fine-tuning-classify-model/data/dair-ai_emotion

## 🚀 核心功能
- 加载本地 `bert-base-chinese` 模型
- 构建并训练一个多分类神经网络模型
- 文本输入，输出分类标签。如果是二分类模型，输出0，1；如果是六分类模型，输出0，1，2，3，4，5
- 模型保存与加载功能

## 🛠️ 安装与运行

### 1. 克隆项目

```bash
git clone https://github.com/jarry126/bert_model-fine-tuning-classify-model.git
cd bert_model-fine-tuning-classify-model
