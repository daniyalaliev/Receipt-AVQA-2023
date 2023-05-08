import transformers
import os 
import torch
from transformers import LayoutLMv3Tokenizer, LayoutLMv3ForQuestionAnswering, LayoutLMv3Processor, AutoProcessor,LayoutLMv3FeatureExtractor,\
                        LayoutLMv3Model, LayoutLMv3Config, AutoModelForSequenceClassification, LayoutLMv3ImageProcessor, AutoTokenizer, LayoutLMv2FeatureExtractor
import requests
from PIL import Image
import pytorch_lightning as pl
from transformers import pipeline
from transformers import AdamW, AutoModelForQuestionAnswering
import re
import numpy as np

feature_extractor = LayoutLMv3FeatureExtractor()
model_checkpoint = "microsoft/layoutlmv3-base"
max_length = 200
device = "cuda" if torch.cuda.is_available() else "cpu"
#model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)


def get_ocr_words_and_boxes(image_dir, question, answer=0):
    
  # get a batch of document images
  
  # resize every image to 224x224 + apply tesseract to get words + normalized boxes
  image = [Image.open(image_dir)]
  encoded_inputs = feature_extractor(image)
  ans = dict()

  ans['image'] = encoded_inputs.pixel_values
  ans['words'] = encoded_inputs.words
  ans['boxes'] = encoded_inputs.boxes
  ans['answers'] = [answer]
  ans['question'] = question
  
  return ans

def subfinder(words_list, answer_list):  
    matches = []
    start_indices = []
    end_indices = []
    for idx, i in enumerate(range(len(words_list))):
        try:
            float(re.findall(r"(?:\d*\.*\d+)", words_list[idx].replace(',', '.'))[0])
        except:
          continue
        if float(re.findall(r"(?:\d*\.*\d+)", words_list[idx].replace(',', '.'))[0]) == float(answer_list[0]):
            matches.append(answer_list[0])
            start_indices.append(idx)
            end_indices.append(idx + len(answer_list) - 1)
        elif float(re.findall(r"(?:\d*\.*\d+)", words_list[idx].replace(',', '.').replace('.', ''))[0]) == float(answer_list[0]):
            matches.append(answer_list[0])
            start_indices.append(idx)
            end_indices.append(idx + len(answer_list) - 1)
    if matches:
      return matches[0], start_indices[0], end_indices[0]
    else:
      return None, 0, 0
    
#tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
       
def encode_dataset(examples, max_length=512):
  # take a batch 
  questions = examples['question']
  words = examples['words'][0]
  boxes = examples['boxes'][0]

  # encode it
  encoding = tokenizer(questions, words, boxes, max_length=max_length, padding="max_length", truncation=True)

  start_positions = []
  end_positions = []
  answers = examples['answers']
  answers[0] = re.findall(r"(?:\d*\.*\d+)", answers[0].replace(',', '.'))[0]

  for batch_index in range(len(answers)):
    cls_index = encoding.input_ids.index(tokenizer.cls_token_id)
    words_example = [word.lower().replace(',', '.') for word in words]
    for answer in answers:
      match, word_idx_start, word_idx_end = subfinder(words_example, answer.lower().split())
      if match:
        break
      
    if not match:
      for answer in answers:
        for i in range(len(answer)):
          answer_i = answer[:i] + answer[i+1:]
          if float(answer_i) != float(answer):
            continue
          # check if we can find this one in the context
          match, word_idx_start, word_idx_end = subfinder(words_example, answer_i.lower().split())
          if match:
            break

#     break

    if match:
      sequence_ids = encoding.sequence_ids(batch_index)
      # Start token index of the current span in the text.
      token_start_index = 0
      while sequence_ids[token_start_index] != 1:
          token_start_index += 1

      # End token index of the current span in the text.
      token_end_index = len(encoding.input_ids) - 1
      while sequence_ids[token_end_index] != 1:
          token_end_index -= 1
      
      word_ids = encoding.word_ids(batch_index)[token_start_index:token_end_index+1]
      for id in word_ids:
        if id == word_idx_start:
          start_positions.append(token_start_index)
          break
        else:
          token_start_index += 1

      for id in word_ids[::-1]:
        if id == word_idx_end:
          end_positions.append(token_end_index)
          break
        else:
          token_end_index -= 1
    
      start_position = start_positions[batch_index]
      end_position = end_positions[batch_index]
      reconstructed_answer = tokenizer.decode(encoding.input_ids[start_position:end_position+1])
    
    else:
      start_positions.append(cls_index)
      end_positions.append(cls_index)
  
  encoding['image'] = examples['image']
  encoding['start_positions'] = start_positions
  encoding['end_positions'] = end_positions
  encoding['answer'] = torch.tensor(float(answers[0]))
  for key in encoding.keys():
    if isinstance(encoding[key], list):
      encoding[key] = torch.from_numpy(np.array(encoding[key]))
  return encoding

