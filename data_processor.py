# -*- coding: utf-8 -*-
import os
import numpy as np

np.random.seed(123) # 保证每次运行生成的随机数相同
class DataProcessor:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def load_vocab(self, vocab_path):
        word2index = dict()
        idx = 2
        with open(vocab_path, "r", encoding="utf-8") as f:
            for word in f.readlines():
                if idx > self.vocab_size + 2:
                    break
                word = word.strip()
                word2index[word] = idx
                idx += 1
        self.vocab_size = min(self.vocab_size, idx - 2) # 有效词表长度
        return word2index

    def read_text(self, pos_path, neg_path):
        #读取原始文本数据
        datas = []
        labels = []
        pos_files= os.listdir(pos_path) 
        neg_files = os.listdir(neg_path)
        
        for idx, file_name in enumerate(pos_files): 
            # if idx > 10: # 调试使用
            #     break
            file_position = pos_path + file_name
            with open(file_position, "r",encoding='utf-8') as f:  
                data = f.read()   
                datas.append(data)
                labels.append([1,0]) #正类标签维[1,0]

       
        for idx, file_name in enumerate(neg_files):
            # if idx > 10: # 调试使用
            #     break
            file_position = neg_path + file_name 
            with open(file_position, "r",encoding='utf-8') as f:
                data = f.read()
                datas.append(data)
                labels.append([0,1]) #负类标签维[0,1]
        return datas, labels
    
    def get_datasets(self, pos_path, neg_path, vocab_path, max_len):
        datas, labels = self.read_text(pos_path, neg_path)
        word2index = self.load_vocab(vocab_path)
        word2index["<pad>"] = 0 
        word2index["<unk>"] = 1
        self.vocab_size += 2 # 有效次表长度+一位pad+一位unk
        features = []
        unk_set = set()
        know_set = set()
        for data in datas:
            feature = []
            data_list = data.split()
            for word in data_list:
                word = word.lower() #词表中的单词均为小写
                if word in word2index:
                    know_set.add(word)
                    feature.append(word2index[word])
                else:
                    unk_set.add(word)
                    feature.append(word2index["<unk>"]) #词表中未出现的词用<unk>代替
                if(len(feature)==max_len): #限制句子的最大长度，超出部分直接截断
                    break
            #对未达到最大长度的句子添加padding

            feature = feature + [word2index["<pad>"]] * (max_len - len(feature))
            features.append(feature)
        print("#" * 20)
        print(f"unk size: {len(unk_set)}")
        # print(unk_set)
        print(f"know size: {len(know_set)}")
        # print(know_set)
        print("#" * 20)
        return features, labels
    
    def batch_iter(self, dataset, batch_size, num_epochs=1, shuffle=False):
        dataset = np.array(dataset)
        data_size = len(dataset)
        num_batches_per_epoch = int((len(dataset) - 1) / batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = dataset[shuffle_indices]
            else:
                shuffled_data = dataset
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]
    
    def test_batchs(self, dataset, batch_size):
        dataset = np.array(dataset)
        num_batchs = int((len(dataset) - 1) / batch_size) + 1
        batchs = []
        for batch_num in range(num_batchs):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(dataset))
            batchs.append(dataset[start_index:end_index])
        return batchs
