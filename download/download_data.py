#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : 刘汕汕
# @Time     : 2025/7/22 
# @Version  : 1.0
# @Desc     : None
# from datasets import load_dataset
#
# ds = load_dataset("dair-ai/emotion", "split", cache_dir = "/Users/liushanshan/my-project/bert_model-fine-tuning-classify-model/data")

#
# from datasets import load_dataset
#
# ds = load_dataset("dair-ai/emotion", "unsplit", cache_dir="/Users/liushanshan/my-project/bert_model-fine-tuning-classify-model/data")
#

from datasets import load_dataset

# 加载远程数据集
dataset = load_dataset("dair-ai/emotion")

# 保存到你自己的项目目录下
dataset.save_to_disk("data/dair-ai_emotion")  # ✅注意目录名干净简单
