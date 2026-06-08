# coding=utf-8
# Copyright 2022 Gen Luo. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import  json, re,random
import torch.utils.data as Data
from torchvision.transforms import transforms
import os
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
from util.base_prompt import *
import torch
from lavin import Tokenizer
import copy
import pickle


class VidCmsDataSet(Data.Dataset):
    # cms_type -> (训练文件, 数据键, label名, describe prompt, context prompt)
    CMS_CONFIG = {
        'eff': {
            'train_file': 'eff_training.json',
            'data_key': 'eff',
            'label': 'Effect',
            'prompt_describe': 'Question: Describe the video and predict what will be the effect. Response:',
            'prompt_context': 'What will be the effect?',
        },
        'int': {
            'train_file': 'int_training.json',
            'data_key': 'int',
            'label': 'Intention',
            'prompt_describe': 'Question: Describe the video and predict what is the intention. Response:',
            'prompt_context': 'What is the intention?',
        },
        'att': {
            'train_file': 'att_training.json',
            'data_key': 'att',
            'label': 'Attribute',
            'prompt_describe': 'Question: Describe the video and predict what is the attribute of the people. Response:',
            'prompt_context': 'What is the attribute of the people?',
        },
    }

    def __init__(self, args, split, model_path, max_words=1024, cms_type='eff'):
        super(VidCmsDataSet, self).__init__()
        self.args = args

        assert cms_type in self.CMS_CONFIG, f"cms_type must be one of {list(self.CMS_CONFIG.keys())}, got {cms_type}"
        self.cms_type = cms_type
        cms_cfg = self.CMS_CONFIG[cms_type]

        if split == 'train':
            datas = json.load(open(os.path.join(args.data_root, 'vlm_molae_json', cms_cfg['train_file'])))
        else:
            raise

        random.seed(42)
        random.shuffle(datas)
        self.datas = datas

        self.max_words = max_words
        self.tokenizer = Tokenizer(model_path=model_path + '/tokenizer.model')

        self.video_folder = args.video_folder

    def __getitem__(self, idx):
        item = self.datas[idx]
        video_name = item['video']
        caption = item['caption']
        cms_cfg = self.CMS_CONFIG[self.cms_type]
        cms_value = item[cms_cfg['data_key']]

        random_number = random.randint(0, 5)
        if random_number == 0:
            prompt_question = cms_cfg['prompt_describe']
            prompt_answer = caption + '. ' + cms_cfg['label'] + ': ' + cms_value
        else:
            prompt_question = 'Context: ' + caption + '. Question: ' + cms_cfg['prompt_context'] + ' Response:'
            prompt_answer = cms_cfg['label'] + ': ' + cms_value


        with open(f"{self.video_folder}/{video_name}", "rb") as f:
            video_fea = pickle.load(f)
        video_fea = torch.tensor(video_fea)

        example, labels, example_mask, label_mask = self.tokenize(prompt_question, prompt_answer)
        indicator = 1

        return example, labels, example_mask, video_fea, indicator

    def tokenize(self, prompt0, answer):
        example = prompt0 + answer
        # print(example)
        prompt = torch.tensor(self.tokenizer.encode(prompt0, bos=True, eos=False), dtype=torch.int64)
        example = torch.tensor(self.tokenizer.encode(example, bos=True, eos=True), dtype=torch.int64)
        # print(example.shape)
        # input("example")

        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[:self.max_words]
        labels = copy.deepcopy(example)
        labels[:len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = 0
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return example, labels, example_mask, label_mask

    def __len__(self):
        return len(self.datas)

    def shuffle_list(self, list):
        random.shuffle(list)


class InstrcutDataSet(Data.Dataset):
    def __init__(self, args,split,model_path,max_words=512,max_image_feats=1):
        super(InstrcutDataSet, self).__init__()
        self.args = args
        # --------------------------
        # ---- Raw data loading ---
        # --------------------------
        self.data = json.load(open(os.path.join(args.data_root, 'all_data.json')))[split]

        self.tokenizer = Tokenizer(model_path=model_path + '/tokenizer.model')
        self.max_words = max_words
        self.max_image_feats=max_image_feats
        self.split=split
        self.qids = [item['qid'] for item in self.data]

        print(f"number of problems in split {split}: {len(self.qids)}\n")

        self.transforms=transforms.Compose([transforms.Resize((224, 224), interpolation=Image.BICUBIC),transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])

    def tokenize(self,prompt,answer,max_words=512):
        example=prompt+answer
        # print(prompt)
        prompt=torch.tensor(self.tokenizer.encode(prompt, bos=True, eos=False), dtype=torch.int64)
        example = torch.tensor(self.tokenizer.encode(example, bos=True, eos=True), dtype=torch.int64)
        padding = max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[:self.max_words]
        labels = copy.deepcopy(example)
        labels[:len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = 0
        example_mask = example_mask.float()
        label_mask = label_mask.float()
        return example, labels, example_mask,label_mask


    def __getitem__(self, idx):

        prompt_question=self.data[idx]['instruction']
        prompt_answer=self.data[idx]['answer']

        if self.data[idx]['image'] is not None:
            # image_path='../data/images/train' if self.data[idx]['image_source']=='sqa' else '../data/images/train2014'
            if self.data[idx]['image_source'] == 'sqa':
                image = Image.open(os.path.join('../data/images/train', self.qids[idx], 'image.png')).convert('RGB')
            else:
                image = Image.open(os.path.join('../data/images/train2014',   'COCO_train2014_'+self.data[idx]['image'])).convert('RGB')
            image = self.transforms(image)
            indicator=1
        else:
            image=torch.Tensor(torch.zeros(3,224,224).float())
            indicator=0

        # print(prompt_question,prompt_answer)
        example, labels, example_mask, label_mask=self.tokenize(prompt_question,prompt_answer)

        return example, labels, example_mask, image,indicator

    def __len__(self):
        return len(self.qids)

    def shuffle_list(self, list):
        random.shuffle(list)

