from transformers import AdamW
from transformers import AutoModelForQuestionAnswering
from lmv2 import get_ocr_words_and_boxes, subfinder, encode_dataset
from transformers import ViltProcessor, ViltModel, AutoTokenizer, \
    AutoModelForTokenClassification,  LayoutLMv3ForSequenceClassification, VisualBertForQuestionAnswering, AutoModel
import pandas as pd 
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import os
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from random import randrange
import warnings
from data import VisionDataset
import pytorch_lightning as pl
import warnings
import sklearn
warnings.filterwarnings("ignore")


device = "cuda" if torch.cuda.is_available() else "cpu"
root = '/home/daniyalaliev/Desktop/data/Receipt-AVQA-2023/images'

class BasedModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder.to(device)
        self.decoder = nn.LSTM(input_size=768, hidden_size=500, bidirectional=True, num_layers=2,batch_first=True)
        self.layer1 = nn.Linear(500, 100)
        self.layer2 = nn.Linear(100, 10)
        self.layer3 = nn.Linear(10, 4)
        self.relu = nn.LeakyReLU()
        
    def forward(self, x):
        image = []
        for file in x[0]:
            image.append(Image.open(file))
        question = list(x[1])
        # forward + back    ward + optimize
        inputs = processor(image, question, return_tensors="pt", padding=True)
        inputs = {k: v.to(device=device, non_blocking=True) for k, v in inputs.items()}
        outputs = self.encoder(**inputs)
        vectors =outputs.last_hidden_state#[attention_mask[0].bool()]
        #vectors = vectors[..., :max_length, :]
        output, (hn, cn) = self.decoder(vectors)
        x = hn[-1]
        x  = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = F.softmax(x, -1)
        #print(torch.argmax(vectors, -1))
        #print(self.min + (self.max - self.min) * torch.sigmoid(x[0]))
        return x
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), betas=(0.9, 0.999), lr=5e-5)
        self.sch = torch.optim.lr_scheduler.StepLR(self.optimizer, gamma=0.1, step_size=30)
        return  {"optimizer": self.optimizer, "lr_scheduler": self.sch, "monitor": "train_loss"}
    
    def training_step(self, batch, _):
        res = self.forward(batch)
        loss = F.cross_entropy(res, batch['answer'].long())
        self.log('train_loss', loss)
        return loss

class VisionDataset(Dataset):
    def __init__(self, df, root, mode):
        self.df = df
        self.root = root
        self.mode = mode
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        file_name = self.df.iloc[index, :].file_name
        question = self.df.iloc[index, :].question
        answer = str(self.df.iloc[index, :].answer)
        if self.mode == 'dev':
            if file_name.startswith('X'):
                path = self.root + '/dev/' + file_name + '.jpg'
            else:
                path = self.root + f'/dev/' + file_name + '.png'
        elif file_name.startswith('X'):
            path = self.root + '/train_part1/' + file_name + '.jpg'
        else:
            path = self.root + f'/train_part2/' + file_name + '.png'
        print(path)
        return (path, question, int(answer))

#config = LayoutLMv3Config(num_labels=4, max_position_embeddings=800)
model_checkpoint = "microsoft/layoutlmv3-base"
encoder =  VisualBertForQuestionAnswering.from_pretrained('uclanlp/visualbert-vqa')
df = pd.read_csv('questions_answers.csv')
df = df[(df.receipt_currency=='Indonesian rupiah') & (df.question_category == 'count') & (df.split == 'dev') & (df.answer <= 3) & (df.answer >=0)]
kick = ['X51007231331', 'X51007231336', 'X51007231338', 'X51007231341', 'X51007231344', 'X51007231346', 'X51006913024', 'X51007231274', 'X51007231276', 'X51007846400', 'X51007846387', 'X51007846379', 'X51007846321', 'X51007846290', 'X51007846400', 'X51007846397', 'X51007846400', 'X510078464033', 'X51007846632', 'X51006414708', 'X51006414715', 'X51006414728']
for ki in kick:
    df.drop(df[df['file_name'] == ki].index, inplace=True)
df.answer = df.answer.astype(int)
encoded_dataset = VisionDataset(df, root, 'dev')
dataloader = DataLoader(encoded_dataset, batch_size=4, shuffle=False)
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
encoder =  ViltModel.from_pretrained("dandelin/vilt-b32-mlm").to(device)
model = BasedModel(encoder).to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.2)
num_epochs = 500
class_weights= torch.tensor(sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(df.answer), y=df.answer.to_numpy()), dtype=torch.float).to(device)

#
checkpoint = torch.load('count_new/200_weights_for_new+weights_indonesian.pth')
model.load_state_dict(checkpoint)
model.eval()
for i in range(1):
    # path = './' + f'count/{i}' +'_weights_for_new+weights_indonesian.pth'
    # print(path)
    # if i % 100 == 0:
    #     torch.save(encoder.state_dict(), path)
    # epoch_loss = 0
    anses = torch.tensor([])
    for idx, batch in tqdm(enumerate(dataloader)):
        if not batch:
            continue
        #batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch)
        anses = torch.cat((anses, outputs.argmax(-1).cpu()))
        # batch.pop('image')
        # batch.pop('answer')
        #outputs = F.softmax(outputs.logits)
        print(outputs.argmax(-1), batch[2])

    
        # loss = F.cross_entropy(outputs.cpu(), batch['answer'].long())
        # print("Loss:", loss.item())
        # loss.backward()
        # optimizer.step()
fuck = 3
#     # forward + backward + optimize
# trainer = pl.Trainer(max_epochs=500,  log_every_n_steps=150, accelerator='cuda')
# trainer.fit(model, dataloader)