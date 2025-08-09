# dataset.py

import torch
from torch.utils.data import Dataset
import pandas as pd
import json
from PIL import Image
import os
import clip

class MSRVTT_Dataset(Dataset):
    """
    Custom Dataset for loading Meme-Text data as per the technical document.
    Handles loading of images, queries(captions), contexts(titles), and emotions.
    """
    def __init__(self,
                 csv_path: str,
                 json_path: str,
                 features_path: str,
                 emotion_json_path: str,
                 bert_tokenizer,
                 clip_preprocess,
                 max_words: int = 32,
                 is_train: bool = True):
        
        self.is_train = is_train
        self.data = []
        self.features_path = features_path
        self.max_words = max_words
        self.bert_tokenizer = bert_tokenizer
        self.clip_preprocess = clip_preprocess

        # --- Load Data ---
        if self.is_train:
            # For training data format
            df = pd.read_csv(csv_path)
            video_ids = df['video_id'].tolist()
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Create caption mapping
            captions = {}
            for item in json_data['sentences']:
                captions[item['video_id']] = item['caption']
            
            # Create titles mapping
            titles = json_data['titles']
            
            # Load emotions
            with open(emotion_json_path, 'r', encoding='utf-8') as f:
                emotions_data = json.load(f)
            
            # Create emotion mapping
            emotions = {}
            for item in emotions_data:
                video_id = item['video_id']
                emotion_list = []
                for emotion_dict in item['emotion']:
                    emotion_list.extend(list(emotion_dict.values()))
                emotions[video_id] = emotion_list

            for vid in video_ids:
                if vid in captions and vid in titles and vid in emotions:
                    self.data.append({
                        "video_id": vid,
                        "query": captions[vid],
                        "context": titles[vid][0] if titles[vid] else "", # Take first title
                        "emotions": emotions[vid]
                    })
        else:
            # For test data format
            with open(csv_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            with open(emotion_json_path, 'r', encoding='utf-8') as f:
                emotions_data = json.load(f)
            
            # Create emotion mapping for test data
            emotions = {}
            for item in emotions_data:
                video_id = item['video_id']
                emotion_list = []
                for emotion_dict in item['emotion']:
                    emotion_list.extend(list(emotion_dict.values()))
                emotions[video_id] = emotion_list

            for vid, content in test_data.items():
                if vid in emotions:
                    self.data.append({
                        "video_id": vid,
                        "query": content['gt'],
                        "context": content['titles'][0] if content['titles'] else "",
                        "emotions": emotions[vid]
                    })

        print(f"Loaded {len(self.data)} samples for {'training' if is_train else 'testing'}")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        video_id = item['video_id']
        
        # --- Process Image ---
        image_path = os.path.join(self.features_path, f"{video_id}.jpg")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.clip_preprocess(image)

        # --- Process Texts (Query and Context) for CLIP ---
        query_tokenized = clip.tokenize([item['query']], truncate=True)[0]
        context_tokenized = clip.tokenize([item['context']], truncate=True)[0]

        # --- Process Emotions for BERT ---
        # Flatten emotions list and convert all elements to strings
        emotion_list = []
        for emotion in item['emotions']:
            if isinstance(emotion, list):
                emotion_list.extend([str(e) for e in emotion])
            else:
                emotion_list.append(str(emotion))
        
        emotion_text = " ".join(emotion_list)
        if not emotion_text: # Handle cases with no emotions
            emotion_text = "无情感" # Neutral placeholder
            
        bert_inputs = self.bert_tokenizer(
            emotion_text,
            padding='max_length',
            truncation=True,
            max_length=16, # Emotion labels are short
            return_tensors="pt"
        )
        emotion_input_ids = bert_inputs['input_ids'].squeeze(0)
        emotion_attention_mask = bert_inputs['attention_mask'].squeeze(0)
        
        return {
            "video_id": video_id,
            "image": image_tensor,
            "query": query_tokenized,
            "context": context_tokenized,
            "emotion_input_ids": emotion_input_ids,
            "emotion_attention_mask": emotion_attention_mask
        }