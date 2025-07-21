# import torch
# from transformers import BertModel
#
# #定义设备信息
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(DEVICE)
#
# #加载预训练模型
# # pretrained = BertModel.from_pretrained(r"/Users/liushanshan/my-project/ai-python/聚客ai第五期/L2/model/models--google-bert--bert-base-chinese/snapshots/8f23c25b06e129b6c986331a13d8d025a92cf0ea")
# pretrained = BertModel.from_pretrained(r"/Users/liushanshan/my-project/ai-python/聚客ai第五期/L2/model/bert-base-chinese/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")
# print(pretrained)
#
# #定义下游任务（增量模型）
# class Model(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         #设计全连接网络，实现二分类任务
#         # self.fc = torch.nn.Linear(768,2)
#         # self.fc = torch.nn.Linear(768,5)
#         self.fc = torch.nn.Linear(768,2)
#     #使用模型处理数据（执行前向计算）
#     def forward(self,input_ids,attention_mask,token_type_ids):
#         #冻结Bert模型的参数，让其不参与训练
#         with torch.no_grad():
#             out = pretrained(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
#         #增量模型参与训练
#         out = self.fc(out.last_hidden_state[:,0])
#         return out
#
# # class Model(torch.nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         self.bert = BertModel.from_pretrained(
# #             r"/Users/liushanshan/my-project/ai-python/聚客ai第五期/L2/model/models--tabularisai--multilingual-sentiment-analysis/snapshots/f0bcb3b4493d5be7da88fa86f0e0bfbd670a9e97"
# #         )
# #         self.fc = torch.nn.Linear(768, 5)  # 5 类情感分类
# #
# #     def forward(self, input_ids, attention_mask, token_type_ids):
# #         out = self.bert(input_ids=input_ids,
# #                         attention_mask=attention_mask,
# #                         token_type_ids=token_type_ids)
# #         # 取 [CLS] 的表示进行分类
# #         cls = out.last_hidden_state[:, 0]  # shape: [batch, hidden]
# #         return self.fc(cls)  # shape: [batch, 5]

import torch
from transformers import BertModel

#定义设备信息
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

#加载预训练模型
pretrained = BertModel.from_pretrained(r"D:\jukeai\demo_04\model\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f").to(DEVICE)
print(pretrained)

#定义下游任务（增量模型）
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #设计全连接网络，实现二分类任务
        self.fc = torch.nn.Linear(768,2)
    #使用模型处理数据（执行前向计算）
    def forward(self,input_ids,attention_mask,token_type_ids):
        #冻结Bert模型的参数，让其不参与训练
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        #增量模型参与训练
        out = self.fc(out.last_hidden_state[:,0])
        return out