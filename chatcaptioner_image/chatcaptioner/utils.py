import os
import random
from copy import deepcopy
import yaml
from PIL import Image
import matplotlib.pyplot as plt
from pycocotools.coco import COCO



class COCOHelper():
    def __init__(self, coco_path, coco_ann_path):
        # todo: make it works for test set. test set doesn't contain annotation
        self.coco_path = coco_path
        self.coco_ann = COCO(annotation_file=coco_ann_path)
        self.coco_ids = self.coco_ann.getImgIds()
        # self.split = split

    def random_img_ids(self, n):
        sample_img_ids = random.sample(self.coco_ids, n)
        return sample_img_ids
        
    def fetch_coco_img(self, image_id, split='val'):
        img_name = '%012d.jpg' % image_id
        img_path = os.path.join(self.coco_path, img_name)
        raw_image = Image.open(img_path).convert('RGB')

        ann_ids = self.coco_ann.getAnnIds(imgIds=[image_id], iscrowd=None)
        anns = self.coco_ann.loadAnns(ann_ids)
        captions = [ann['caption'] for ann in anns]
        return raw_image, captions

    
class RandomSampledDataset():
    def __init__(self, datasets_root, dataset_name):
        self.name = dataset_name
        self.dataset_path = os.path.join(datasets_root, dataset_name)
        self._ids = [file_name.split('.jpg')[0] for file_name in os.listdir(os.path.join(self.dataset_path, 'img'))]
        
        
        ann_path = os.path.join(datasets_root, dataset_name, 'annotation.yaml')
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                self.ann = yaml.safe_load(f)
                if isinstance(list(self.ann.keys())[0], int):
                    self.ann = {str(image_id): captions for image_id, captions in self.ann.items()}
        else:
            self.ann = None
        
    @property
    def ids(self):
        return deepcopy(self._ids)
    
    def random_img_ids(self, n):
        sample_img_ids = random.sample(self._ids, n)
        return sample_img_ids
    
    def fetch_img(self, image_id):
        img_path = os.path.join(self.dataset_path, 'img', '{}.jpg'.format(image_id))
        raw_image = Image.open(img_path).convert('RGB')
        
        if self.ann:
            captions = self.ann[image_id]
            
            if isinstance(captions, str):
                captions = [captions]
        else:
            captions = []
        
        return raw_image, captions
    
    
class SimPairDataset():
    def __init__(self, datasets_root, dataset_name):
        self.name = dataset_name
        self.dataset_path = os.path.join(datasets_root, dataset_name)
        
        ann_path = os.path.join(datasets_root, dataset_name, 'sim_retrieve.yaml')
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                self.ann = yaml.safe_load(f)
                if isinstance(list(self.ann.keys())[0], int):
                    self.ann = {str(image_id): captions for image_id, captions in self.ann.items()}
        else:
            self.ann = None
        self._ids = list(self.ann.keys())
        
    @property
    def ids(self):
        return deepcopy(self._ids)
    
    def fetch_img_pairs(self, pair_id):
        image_ids = list(self.ann[pair_id].keys())
        fetched = []
        for image_id in image_ids:
            img_path = os.path.join(self.dataset_path, 'img', '{}.jpg'.format(image_id))
            raw_image = Image.open(img_path).convert('RGB')
            if self.ann:
                captions = self.ann[pair_id][image_id]

                if isinstance(captions, str):
                    captions = [captions]
            else:
                captions = []
            fetched.append((image_id, raw_image, captions))
        return fetched    
    

def extractQA_chatgpt(messages):
    questions = []
    answers = []
    for message in messages:
        if 'Question: ' in message['content']:
            questions.append(message['content'].split('Question: ')[1])
        if 'Answer: ' in message['content']:
            answers.append(message['content'].split('Answer: ')[1])
    return questions, answers
    
    
def print_info(info, key='caption', variants=['BLIP2', 'BLIP2+OurPrompt', 'ChatCaptioner']):
    img_id = info['setting']['id']
    if 'GT' in info['setting']:
        gt_captions = info['setting']['GT']['caption']
        if isinstance(gt_captions, str) and len(gt_captions):
            gt_captions = [gt_captions]
            
    else:
        gt_captions = []
    
    print('Image ID {}'.format(img_id))
    for blip2_tag in info:
        if blip2_tag in ['GT', 'id', 'setting']: continue
        for variant in variants:
            if key not in info[blip2_tag][variant]:
                continue
            print('-------------------')
            print('{} {}:'.format(blip2_tag, variant))
            if key == 'chat' and isinstance(info[blip2_tag][variant][key], list):
                for message in info[blip2_tag][variant][key]:
                    print(message['content'])
            else:
                print(info[blip2_tag][variant][key])
            if key == 'chat':
                print(info[blip2_tag][variant]['caption'])
        print('===================')
    if key == 'caption' and len(gt_captions):
        print('GT:')
        [print(cap) for cap in gt_captions]


def plot_img(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
    
    
# ===================================
#           Deprecated Zone
# ===================================
    
def visualize_old(file_path):
    # out of date
    with open(file_path, 'r') as f:
        info = yaml.safe_load(f)
    print('COCO Val Image ID {}'.format(info['id']))
    print('-------------------')
    print('Ours: {}'.format(info['ours']['clip_score']))
    print(info['ours']['chat'])
    print('-------------------')
    print('BLIP2: {}'.format(info['blip2']['clip_score']))
    print(info['blip2']['caption'])
    print('-------------------')
    print('GT: {}'.format(info['gt']['clip_score']))
    [print(cap) for cap in info['gt']['caption']]
    image, _ = fetch_coco_img(info['id'])
    plot_img(image)
    
    
def print_info_old(info, key='caption', variants=['BLIP2', 'BLIP2+OurPrompt', 'ChatCaptioner']):
    if 'id' in info:
        img_id = info['id']
    else:
        img_id = info['setting']['id']
    if 'GT' in info:
        gt_captions = info['GT']['caption']
    elif 'GT' in info['setting']:
        gt_captions = info['setting']['GT']['caption']
    else:
        gt_captions = []
    
    print('Image ID {}'.format(img_id))
    for blip2_tag in info:
        if blip2_tag in ['GT', 'id', 'setting']: continue
        for variant in variants:
            if key not in info[blip2_tag][variant]:
                continue
            print('-------------------')
            print('{} {}:'.format(blip2_tag, variant))
            print(info[blip2_tag][variant][key])
        print('===================')
    if key == 'caption' and len(gt_captions):
        print('GT:')
        [print(cap) for cap in gt_captions]