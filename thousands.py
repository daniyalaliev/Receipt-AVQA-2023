from transformers import AdamW
from transformers import AutoModelForQuestionAnswering
from lmv2 import get_ocr_words_and_boxes, subfinder, encode_dataset
from transformers import LayoutLMv3FeatureExtractor, LayoutLMv3Processor, AutoTokenizer
import pandas as pd 
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import os
import torch 
from tqdm import tqdm
from random import randrange
import warnings
from PIL import Image
import numpy as np
warnings.filterwarnings("ignore")


def get_ocr_words_and_boxes(image_dir, question, answer=0):
    
  # get a batch of document images
  
  # resize every image to 224x224 + apply tesseract to get words + normalized boxes
  image = [Image.open(image_dir)]
  encoded_inputs = feature_extractor(image)
  ans = dict()

  ans['image'] = encoded_inputs.pixel_values
  ans['words'] = encoded_inputs.words
  ans['boxes'] = encoded_inputs.boxes
  # ans['answers'] = [answer]
  ans['question'] = question
  
  return ans

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
       
def encode_dataset(examples, max_length=512):
  # take a batch 
  questions = examples['question']
  words = examples['words'][0]
  boxes = examples['boxes'][0]

  # encode it
  encoding = tokenizer(questions, words, boxes, max_length=max_length, padding="max_length", truncation=True)

  start_positions = []
  end_positions = []
  encoding['image'] = examples['image']
  encoding['start_positions'] = start_positions
  encoding['end_positions'] = end_positions
  for key in encoding.keys():
    if isinstance(encoding[key], list):
      encoding[key] = torch.from_numpy(np.array(encoding[key]))
  return encoding

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
        question = question[:-14] + question[-1]
        #answer = str(self.df.iloc[index, :].answer)
        if self.mode == 'test':
            if file_name.startswith('X'):
                path = self.root + '/test/' + file_name + '.jpg'
            else:
                path = self.root + f'/test/' + file_name + '.png'
        elif file_name.startswith('X'):
            path = self.root + '/train_part1/' + file_name + '.jpg'
        else:
            path = self.root + f'/train_part2/' + file_name + '.png'
        encoded = get_ocr_words_and_boxes(path, question)
        try:
            return encode_dataset(encoded) 
        except:
            return self.__getitem__(randrange(len(self)))

root = '/home/daniyalaliev/Desktop/data/Receipt-AVQA-2023/images'
df = pd.read_csv('test_questions.csv')
new_df = df[(df.question_type =='amount') & (df.question.str[-13:] == 'in thousands?')]

kick = ['X51007231275','X51007231343', 'X51007231372','X51007231331', 'X51007231336', 'X51007231338', 'X51007231341', 'X51007231344', 'X51007231346', 'X51006913024', 'X51007231274', 'X51007846268', 'X51007846283'\
        'X51007231276', 'X51007846400', 'X51007846387', 'X51007846379', 'X51007846321', 'X51007846290', 'X51007846400', 'X51007846397', 'X51007846400', 'X51007846396'\
            'X510078464033', 'X51007846632', 'X51006414708', 'X51006414715', 'X51006414728', 'X51007846303', 'X51007846304', 'X51007846310', 'X51007846303', 'X510078463355', 'X51007846358', 'X51007846371', 'X510078463392']
for ki in kick:
    new_df.drop(new_df[new_df['file_name'] == ki].index, inplace=True)



encoded_dataset = VisionDataset(new_df, root, 'test')
dataloader = DataLoader(encoded_dataset, batch_size=1, shuffle=False)
feature_extractor = LayoutLMv3FeatureExtractor()
model_checkpoint = "microsoft/layoutlmv3-base"
model = AutoModelForQuestionAnswering.from_pretrained('7_weights_for_new')
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
correct = 0
#
anses = []
for idx, batch in tqdm(enumerate(dataloader)):
    # get the inputs;
    if not batch:
        continue
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    #token_type_ids = batch["token_type_ids"].to(device)
    bbox = batch["bbox"].to(device)
    image = batch["image"].to(device)
    # start_positions = batch["start_positions"].to(device)
    # end_positions = batch["end_positions"].to(device)

    # forward + backward + optimize
    outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                    bbox=bbox) 
                    #start_positions=start_positions, end_positions=end_positions,
                    #output_hidden_states=True)
    loss = outputs.loss
            # step 3: get start_logits and end_logits
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    # step 4: get largest logit for both
    predicted_start_idx = start_logits.argmax(-1).item()
    predicted_end_idx = end_logits.argmax(-1).item()
    ans = tokenizer.decode(input_ids.squeeze()[predicted_start_idx:predicted_end_idx+1]).replace(',', '.')
    #y = tokenizer.decode(input_ids.squeeze()[start_positions:end_positions+1]).replace(',',  '.')
    if ans == '<s>':
        ans = 0
    
    #print(ans, '----------', tokenizer.decode(input_ids.squeeze()[start_positions:end_positions+1]), \
    #      '----------', batch["answer"])
    ans = tokenizer.decode(input_ids.squeeze()[predicted_start_idx:predicted_end_idx+1]).replace(',', '.')
    #y = tokenizer.decode(input_ids.squeeze()[start_positions:end_positions+1]).replace(',',  '.')
    anses.append(ans)
    print(ans)
    # try:
    #     if y == ans and ans != '<s>':
    #         correct += 1
    #         print(f'CURR RES: {correct}/{df.shape[0]}')
    # except:
    #     print(f'CURR RES: {correct}/{idx + 1}')
duck = 3
        