from transformers import AdamW
from transformers import AutoModelForQuestionAnswering
from lmv2 import get_ocr_words_and_boxes, subfinder, encode_dataset
from transformers import LayoutLMv3FeatureExtractor
import pandas as pd 
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import os
import torch 
from tqdm import tqdm
from random import randrange
import warnings
warnings.filterwarnings("ignore")


class VisionDataset(Dataset):
    def __init__(self, df):
        self.df = df
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        file_name = df.iloc[index, :].file_name
        question = df.iloc[index, :].question
        answer = str(df.iloc[index, :].answer * 1000)
        if file_name.startswith('X'):
            path = root + '/train_part1/' + file_name + '.jpg'
        else:
            path = root + '/train_part2/' + file_name + '.png'
        encoded = get_ocr_words_and_boxes(path, question, answer)
        try:
            return encode_dataset(encoded)
        except:
            return self.__getitem__(randrange(len(self)))

root = '/home/daniyalaliev/Desktop/data/Receipt-AVQA-2023/images'
df = pd.read_csv('questions_answers.csv')
df = df[(df.question_category =='amount') & (df.question.str[-10:-1]=='thousands') & (df.split == 'train')]

kick = ['X51007231331', 'X51007231336', 'X51007231338', 'X51007231341', 'X51007231344', 'X51007231346']
for ki in kick:
    df.drop(df[df['file_name'] == ki].index, inplace=True)


encoded_dataset = VisionDataset(df)
dataloader = DataLoader(encoded_dataset, batch_size=4, shuffle=False)
feature_extractor = LayoutLMv3FeatureExtractor()
model_checkpoint = "microsoft/layoutlmv3-base"
model = AutoModelForQuestionAnswering.from_pretrained('5_weights_for_item_price_thousands')

optimizer = AdamW(model.parameters(), lr=5e-6)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f'USING DEVICE : {device}')
model.train()
for epoch in range(6, 500):
   path = './' + str(epoch) +'_weights_for_item_price_thousands'
   model.save_pretrained(path)  
        
    # loop over the dataset multiple times
   for idx, batch in enumerate(dataloader):
        # get the inputs;
        if not batch:
            continue
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        #token_type_ids = batch["token_type_ids"].to(device)
        bbox = batch["bbox"].to(device)
        image = batch["image"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                       bbox=bbox, start_positions=start_positions, end_positions=end_positions)
        loss = outputs.loss
        print("Loss:", loss.item())
        loss.backward()
        optimizer.step()
