import json

import evaluate
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoModelForTokenClassification, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
model = AutoModelForTokenClassification.from_pretrained('./checkpoints/csl_model_1_acc(0.9664943523651244)/', num_labels=3)


class CodeSeqLabelingDataset(Dataset):
    
    def __init__(self, path) -> None:
        super(CodeSeqLabelingDataset, self).__init__()
        
        with open(path, 'r') as f:
            j = f.read()
            
        self.data = json.loads(j)
        
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

csl_dataset = CodeSeqLabelingDataset('../../../datasets/code_seq_labeling/csl_95k.json')

train_set, test_set = random_split(csl_dataset, [0.9, 0.1])



def collate_func(batch):
    code = [data['code'] for data in batch]
    labels = [data['ner_tags'] for data in batch]
    inputs = tokenizer(code, padding='max_length', max_length=320, truncation=True, return_tensors='pt')
    
    return inputs, torch.tensor(labels)

train_loader = DataLoader(train_set, shuffle=True, batch_size=64, drop_last=True, collate_fn=collate_func)
test_loader = DataLoader(test_set, shuffle=True, batch_size=32, drop_last=True, collate_fn=collate_func)




model = model.cuda()

# lr
lr = 2e-4

# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)

# loss function
criterion = nn.CrossEntropyLoss()

# max epoch
max_epoch = 6

# metrics
acc_mtc = evaluate.load('accuracy')


def validate():
    model.eval()
    with torch.inference_mode():
        for inputs, labels in test_loader:
            inputs = {k: v.cuda() for k, v in inputs.items()}
            labels = labels.cuda()
            output = model(**inputs).logits
            predict = torch.argmax(output, dim = -1)
            pred = [
                [p.item() for p, l in zip(pre, lab) if l != -100]
                for pre, lab in zip(predict, labels)
            ]            
            label = [
                [l.item() for p, l in zip(pre, lab) if l != -100]
                for pre, lab in zip(predict, labels)
            ]
            for x, y in zip(pred, label):
                acc_mtc.add_batch(predictions=x, references=y)
        return acc_mtc.compute()['accuracy']
        


def train():
    model.train()
    for epoch in range(4, max_epoch):
        step = 0
        for inputs, labels in train_loader:
            inputs = {k: v.cuda() for k, v in inputs.items()}
            labels = labels.cuda()
            
            optimizer.zero_grad()
            output = model(**inputs).logits.permute(0, 2, 1)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f'epoch: {epoch}, loss: {loss.item()}')
            step += 1

        acc = validate()
        print(f'epoch: {epoch}, acc: {acc}')
        # checkpoints
        model.save_pretrained(f'../../../checkpoints/csl_model_{epoch}_acc({acc})')



train()
