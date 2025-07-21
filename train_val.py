#模型训练
import torch
from MyData import MyDataset
from torch.utils.data import DataLoader
from net import Model
from transformers import BertTokenizer
from torch.optim import AdamW

#定义设备信息
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#定义训练的轮次(将整个数据集训练完一次为一轮)
EPOCH = 30000
# EPOCH = 50

#加载字典和分词器
token = BertTokenizer.from_pretrained(r"/Users/liushanshan/my-project/ai-python/聚客ai第五期/L2/model/bert-base-chinese/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")

#将传入的字符串进行编码
def collate_fn(data):
    sents = [i[0]for i in data]
    label = [i[1] for i in data]
    #编码
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        # 当句子长度大于max_length(上限是model_max_length)时，截断
        truncation=True,
        max_length=512,
        # 一律补0到max_length
        padding="max_length",
        # 可取值为tf,pt,np,默认为list
        return_tensors="pt",
        # 返回序列长度
        return_length=True
    )
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    label = torch.LongTensor(label)
    return input_ids,attention_mask,token_type_ids,label

train_data = [
    # 非常负面
    ("这个产品太糟糕了，简直就是浪费钱", 0),
    ("服务态度恶劣，体验非常差", 0),
    ("简直恶心透顶", 0),
    ("再也不会买了", 0),
    # 负面
    ("等了很久才发货，不太满意", 1),
    ("不太行", 1),
    ("快递有点慢", 1),
    ("商品有点小问题", 1),
    # 中性
    ("整体还行吧，没有特别出色也没有大问题", 2),
    ("一般般，没有太多惊喜", 2),
    ("不好不坏", 2),
    ("感觉平平淡淡", 2),
    # 正面
    ("商品质量不错，快递也很快", 3),
    ("用起来挺舒服的，值得推荐", 3),
    ("物有所值", 3),
    ("体验不错", 3),
    # 非常正面
    ("超出预期，简直太棒了！", 4),
    ("完全爱上这个产品了，强烈安利！", 4),
    ("真的很不错", 4),
    ("体验非常惊艳", 4)
]


val_data = [
    # 非常负面（0）
    ("简直是灾难，不推荐购买", 0),
    ("糟糕到无以复加", 0),
    ("非常后悔购买，体验极差", 0),

    # 负面（1）
    ("使用中遇到不少小问题", 1),
    ("客服态度不太友好", 1),
    ("发票迟迟未到", 1),

    # 中性（2）
    ("感觉还可以，但没有特别惊艳", 2),
    ("中规中矩的一次购物", 2),
    ("使用了一段时间，目前没有问题", 2),

    # 正面（3）
    ("功能挺多，基本满足需求", 3),
    ("包装很细致，整体体验不错", 3),
    ("第一次购买这个品牌，感觉还行", 3),

    # 非常正面（4）
    ("爱了爱了，真的超棒！", 4),
    ("五星好评，值得推荐", 4),
    ("质量好服务也好，以后还会再来", 4)
]



#创建数据集
train_dataset = MyDataset("train")
# train_dataset = MyListDataset(train_data)
train_loader = DataLoader(
    dataset=train_dataset,
    #训练批次
    # batch_size=2,
    batch_size=50,
    #打乱数据集
    shuffle=True,
    #舍弃最后一个批次的数据，防止形状出错
    # drop_last=True,
    drop_last=False,
    #对加载的数据进行编码
    collate_fn=collate_fn
)
#创建验证数据集
val_dataset = MyDataset("validation")
# val_dataset = MyListDataset(val_data)
val_loader = DataLoader(
    dataset=val_dataset,
    #训练批次
    # batch_size=2,
    batch_size=50,
    #打乱数据集
    shuffle=True,
    #舍弃最后一个批次的数据，防止形状出错
    # drop_last=True,
    drop_last=False,
    #对加载的数据进行编码
    collate_fn=collate_fn
)
if __name__ == '__main__':
    #开始训练
    print(DEVICE)
    model = Model().to(DEVICE)
    #定义优化器
    optimizer = AdamW(model.parameters())
    #定义损失函数
    loss_func = torch.nn.CrossEntropyLoss()

    #初始化验证最佳准确率
    best_val_acc = 0.0

    for epoch in range(EPOCH):
        for i,(input_ids,attention_mask,token_type_ids,label) in enumerate(train_loader):
            #将数据放到DVEVICE上面
            input_ids, attention_mask, token_type_ids, label = input_ids.to(DEVICE),attention_mask.to(DEVICE),token_type_ids.to(DEVICE),label.to(DEVICE)
            #前向计算（将数据输入模型得到输出）
            out = model(input_ids,attention_mask,token_type_ids)
            #根据输出计算损失
            loss = loss_func(out,label)
            #根据误差优化参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #每隔5个批次输出训练信息
            if i%5 ==0:
                out = out.argmax(dim=1)
                #计算训练精度
                acc = (out==label).sum().item()/len(label)
                print(f"epoch:{epoch},i:{i},loss:{loss.item()},acc:{acc}")
        #验证模型（判断模型是否过拟合）
        #设置为评估模型
        model.eval()
        #不需要模型参与训练
        with torch.no_grad():
            val_acc = 0.0
            val_loss = 0.0
            for i, (input_ids, attention_mask, token_type_ids, label) in enumerate(val_loader):
                # 将数据放到DVEVICE上面
                input_ids, attention_mask, token_type_ids, label = input_ids.to(DEVICE), attention_mask.to(
                    DEVICE), token_type_ids.to(DEVICE), label.to(DEVICE)
                # 前向计算（将数据输入模型得到输出）
                out = model(input_ids, attention_mask, token_type_ids)
                # 根据输出计算损失
                val_loss += loss_func(out, label)
                #根据数据，计算验证精度
                out = out.argmax(dim=1)
                val_acc+=(out==label).sum().item()
            val_loss/=len(val_loader)
            val_acc/=len(val_loader)
            print(f"验证集：loss:{val_loss},acc:{val_acc}")
        # #每训练完一轮，保存一次参数
        # torch.save(model.state_dict(),f"params/{epoch}_bert.pth")
        # print(epoch,"参数保存成功！")
            #根据验证准确率保存最优参数
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(),"params/best_bert.pth")
                print(f"EPOCH:{epoch}:保存最优参数：acc{best_val_acc}")
        #保存最后一轮参数
        torch.save(model.state_dict(), "params/last_bert.pth")
        print(f"EPOCH:{epoch}:最后一轮参数保存成功！")

