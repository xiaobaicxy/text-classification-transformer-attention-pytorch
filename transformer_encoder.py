# -*- coding: utf-8 -*-
"""
文本分类 Transformer Encoder 算法
@author:
"""
import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

from data_processor import DataProcessor

# 保证每次运行生成的随机数相同
torch.manual_seed(123)
torch.cuda.manual_seed(123)

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size, pretrained_embedding=None):
        super(InputEmbedding, self).__init__()
        if pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_size)
    
    def forward(self, x):
        # x: [batch_size, seq_len]
        out = self.embedding(x) # [batch_size, seq_len, d_model]
        return out


class PositionEmbedding(nn.Module):
    def __init__(self, d_model, seq_len):
        super(PositionEmbedding, self).__init__()
        self.pe = torch.tensor(
            [
                [
                    pos / pow(10000.0, (i//2 * 2.0) / d_model) 
                    for i in range(d_model)
                ] 
            for pos in range(seq_len)]
            ) # 注意 i//2 * 2
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2]) 
        
    def forward(self):
        pe = nn.Parameter(self.pe, requires_grad=False) # [seq_len, d_model]
        return pe


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, seq_len, device, dropout):
        super(Embedding, self).__init__()
        self.input_embedding = InputEmbedding(vocab_size, d_model)
        self.position_embedding = PositionEmbedding(d_model, seq_len)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, x):
        # x: [batch_size, seq_len]
        input_embedding = self.input_embedding(x).to(self.device) # [batch_size, seq_len, d_model]
        position_embedding = self.position_embedding().to(self.device) # [seq_len, d_model]
        out = input_embedding + position_embedding # [batch_size, seq_len, d_model]
        out = self.dropout(out)
        return out


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k = None):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        
    def forward(self, Q, K, V, lengths):
        # Q, K, V: # [batch_size * num_head, seq_len, d_head]
        # lengths: [batch_size]
        batch_size = Q.size(0)
        max_seq_len = Q.size(1)
        
        for b_id, cur_len in enumerate(lengths):
            Q[b_id, cur_len:, :] = 0.0
            K[b_id, cur_len:, :] = 0.0
            V[b_id, cur_len:, :] = 0.0
        
        attention_context = torch.matmul(Q, K.permute(0, 2, 1)) # [batch_size * num_head, seq_len, seq_len]
        attention_context = attention_context.masked_fill_(attention_context==0, 1e-10)

        if self.d_k is not None:
            attention_context = attention_context / (self.d_k ** 0.5)
        
        attention_w = F.softmax(attention_context, dim=2) # [batch_size * num_head, seq_len, seq_len]
        context = torch.matmul(attention_w, V) # [batch_size * num_head, seq_len, d_head]
        return context
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_head, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.num_head = num_head
        assert d_model % num_head == 0
        self.d_head = d_model // num_head
        self.Q_fc = nn.Linear(d_model, d_model)
        self.K_fc = nn.Linear(d_model, d_model)
        self.V_fc = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(d_model)
        # self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x, lengths):
        # x: [batch_size, seq_len, d_model]
        # lengths: [batch_size]
        batch_size = x.size(0)
        Q = self.Q_fc(x).permute(0, 2, 1).contiguous() # [batch_size, d_model, seq_len]
        K = self.K_fc(x).permute(0, 2, 1).contiguous() # [batch_size, d_model, seq_len]
        V = self.V_fc(x).permute(0, 2, 1).contiguous() # [batch_size, d_model, seq_len]
        
        Q = Q.view(batch_size * self.num_head, self.d_head, -1).permute(0, 2, 1).contiguous() # [batch_size * num_head, seq_len, d_head]
        K = K.view(batch_size * self.num_head, self.d_head, -1).permute(0, 2, 1).contiguous() # [batch_size * num_head, seq_len, d_head]
        V = V.view(batch_size * self.num_head, self.d_head, -1).permute(0, 2, 1).contiguous() # [batch_size * num_head, seq_len, d_head]

        context = self.attention(Q, K, V, lengths).permute(0, 2, 1).contiguous() # [batch_size * num_head, d_head, seq_len]
        context = context.view(batch_size, self.num_head * self.d_head, -1).permute(0, 2, 1).contiguous() # [batch_size, seq_len, d_model]
        # out = self.fc(context) # [batch_size, seq_len, d_model]
        out = self.dropout(context)
        out = out + x
        out = self.layer_norm(out)
        return out


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden_size, dropout=0.0):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden_size)
        self.fc2 = nn.Linear(hidden_size, d_model)
        self.act_func = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        out = self.fc1(x) # [batch_size, seq_len, hidden_size]
        out = self.act_func(out) # [batch_size, seq_len, d_model]
        out = self.fc2(out) # [batch_size, seq_len, d_model]
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, multi_head_attention, position_wise_feed_forward):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = multi_head_attention
        self.position_wise_feed_forward = position_wise_feed_forward
    
    def forward(self, x, lengths):
        # x: [batch_size, seq_len, d_model]
        # lengths: [batch_size]
        out = self.multi_head_attention(x, lengths) # [batch_size, seq_len, d_model]
        out = self.position_wise_feed_forward(out) # [batch_size, seq_len, d_model]
        return out


class WeightSum(nn.Module):
    def __init__(self, seq_len, d_model, avg=False):
        super(WeightSum, self).__init__()
        self.avg = avg
        if not avg:
            self.weight = nn.Parameter(torch.FloatTensor(np.random.randn(d_model)), requires_grad=True)
        else:
            self.avg_pool = nn.AvgPool2d(kernel_size=(seq_len, 1))

    def forward(self, x, lengths):
        # x: [batch_size, seq_len, d_model]
        # lengths: [batch_size]
        for b_id, cur_len in enumerate(lengths):
            x[b_id, cur_len:, :] = 0.0

        if not self.avg:
            attention_context = torch.matmul(self.weight, x.permute(0, 2, 1)) # [batch_size, seq_len]
            attention_context = attention_context.masked_fill_(attention_context==0, 1e-10)
            attention_w = F.softmax(attention_context, dim=-1) # [batch_size, seq_len]
            attention_w = attention_w.unsqueeze(dim=1) # [batch_size, 1, seq_len]
            out = torch.bmm(attention_w, x)  #[batch_size, 1, d_model] 
            out = out.squeeze(dim=1)  #[batch, d_model]
        else:
            out = self.avg_pool(x).squeeze(1)
        return out


class ClassifierLayer(nn.Module):
    def __init__(self, d_model, hidden_size, num_classes, dropout=0.0):
        super(ClassifierLayer, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch_size, d_model]
        out = self.fc1(x) # [batch_size, hidden_size]
        out = self.dropout(out)
        out = self.fc2(out) # [batch_size, num_classes]
        out = F.softmax(out, dim=-1)
        return out

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.embedding = Embedding(config.vocab_size, config.d_model, config.seq_len, config.device, config.dropout)
        self.multi_head_attention = MultiHeadAttention(config.d_model, config.num_head, config.dropout)
        self.position_wise_feed_forward = PositionWiseFeedForward(config.d_model, config.hidden_size, config.dropout)
        self.encoder_layer = EncoderLayer(
            self.multi_head_attention,
            self.position_wise_feed_forward
        )
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder_layer)
            for _ in range(config.num_encoder_layer)
        ])
        self.weight_sum = WeightSum(config.seq_len, config.d_model, avg=True)
        self.classifer = ClassifierLayer(config.d_model, config.hidden_size, config.num_classes, config.dropout)
        
    
    def forward(self, x, lengths):
        # x: [batch_size, seq_len]
        # lengths: [batch_size]
        out = self.embedding(x) # [batch_size, seq_len, d_model]
        for encoder in self.encoders:
            out = encoder(out, lengths) # [batch_size, seq_len, d_model]
        out = self.weight_sum(out, lengths)
        out = self.classifer(out)
        return out


class Config:
    def __init__(self):
        # model 参数
        self.vocab_size = 50000
        self.d_model = 64
        self.num_head = 8
        self.seq_len = 100
        self.hidden_size = 128
        self.dropout = 0.2
        self.num_classes = 2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_encoder_layer = 2
        
        # 训练参数
        self.batch_size = 64
        self.lr = 0.005
        self.num_epochs = 50000
        
        # 数据路径
        self.vocab_path = "./datasets/aclImdb/imdb.vocab"
        self.train_pos_path = "./datasets/aclImdb/train/pos/" 
        self.train_neg_path = "./datasets/aclImdb/train/neg/" 
        self.test_pos_path = "./datasets/aclImdb/test/pos/" 
        self.test_neg_path = "./datasets/aclImdb/test/neg/"

        
def test(model, test_batchs, loss_func, device):
    model.eval()
    loss_val = 0.0
    corrects = 0.0
    data_size = 0
    for batch_data in test_batchs:
        # print(batch_data.shape)
        datas, labels = zip(*batch_data)
        datas = torch.LongTensor(datas).to(device)
        labels = torch.FloatTensor(labels).to(device)

        seq_len = datas.size(1)
        lengths = (datas != 0).sum(dim=-1).long()

        preds = model(datas, lengths)
        loss = loss_func(preds, labels)
        
        loss_val += loss.item() * datas.size(0)
        data_size += datas.size(0)
        
        # 获取预测的最大概率出现的位置
        preds = torch.argmax(preds, dim=1)
        labels = torch.argmax(labels, dim=1)
        corrects += torch.sum(preds == labels).item()
    test_loss = loss_val / (data_size + (1e-10))
    test_acc = corrects / (data_size + (1e-10))
    print("Test Loss: {}, Test Acc: {}".format(test_loss, test_acc))
    return test_acc

def train(model, train_iters, test_batchs, optimizer, loss_func, device, num_batches_per_epoch):
    best_val_acc = 0.0
    best_model_params = copy.deepcopy(model.state_dict())
    count = 0
    for batch_data in train_iters:
        datas, labels = zip(*batch_data)
        model.train()
        data_size = 0
        if count % num_batches_per_epoch == 0:
            loss_val = 0.0
            corrects = 0.0
        datas = torch.LongTensor(datas).to(device)
        labels = torch.FloatTensor(labels).to(device)

        seq_len = datas.size(1)
        lengths = (datas != 0).sum(dim=-1).long()

        preds = model(datas, lengths)
        loss = loss_func(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_val += loss.item() * datas.size(0)
        data_size += datas.size(0)
        
        # 获取预测的最大概率出现的位置
        preds = torch.argmax(preds, dim=1)
        labels = torch.argmax(labels, dim=1)
        corrects += torch.sum(preds == labels).item()

        train_loss = loss_val / (data_size + (1e-10))
        train_acc = corrects / (data_size + (1e-10))
        if count % (1000 * num_batches_per_epoch) == 0:
            print("Train Loss: {}, Train Acc: {}".format(train_loss, train_acc))
            test_acc = test(model, test_batchs, loss_func, device)
            if(best_val_acc < test_acc):
                best_val_acc = test_acc
                best_model_params = copy.deepcopy(model.state_dict()) # 更新最优参数
        count += 1

    model.load_state_dict(best_model_params)

    return model

if __name__ == "__main__":
    config = Config()
    processor = DataProcessor(config.vocab_size)
    train_data, train_label = processor.get_datasets(config.train_pos_path,
                                                    config.train_neg_path,
                                                    config.vocab_path,
                                                    config.seq_len)
    config.vocab_size = processor.vocab_size # 词表实际大小

    test_data, test_label = processor.get_datasets(config.test_pos_path,
                                                    config.test_neg_path,
                                                    config.vocab_path,
                                                    config.seq_len)
    
    train_set_iters = processor.batch_iter(list(zip(train_data, train_label)),
                                                 config.batch_size, 
                                                 config.num_epochs, 
                                                 shuffle=True)
    
    test_batchs = processor.test_batchs(list(zip(test_data, test_label)), config.batch_size)

    model = TransformerEncoder(config)
    model.to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_func = nn.BCELoss()
    num_batches_per_epoch = int((len(train_data) - 1) / config.batch_size) + 1
    model = train(model, train_set_iters, test_batchs, optimizer, loss_func, config.device, num_batches_per_epoch)
