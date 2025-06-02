#from numpy.lib.function_base import append
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from torchvision import transforms
import os
import numpy as np
import torch
import pandas as pd
import json

phase2label_dicts = {
    'Cholec80':{
    'Preparation':0,
    'CalotTriangleDissection':1,
    'ClippingCutting':2,
    'GallbladderDissection':3,
    'GallbladderPackaging':4,
    'CleaningCoagulation':5,
    'GallbladderRetraction':6},
    
    'M2CAI':{
    'TrocarPlacement':0,
    'Preparation':1,
    'CalotTriangleDissection':2,
    'ClippingCutting':3,
    'GallbladderDissection':4,
    'GallbladderPackaging':5,
    'CleaningCoagulation':6,
    'GallbladderRetraction':7}
}




def phase2label(phases, phase2label_dict):
    labels = [phase2label_dict[phase] if phase in phase2label_dict.keys() else len(phase2label_dict) for phase in phases]
    return labels

def label2phase(labels, phase2label_dict):
    label2phase_dict = {phase2label_dict[k]:k for k in phase2label_dict.keys()}
    phases = [label2phase_dict[label] if label in label2phase_dict.keys() else 'HardFrame' for label in labels]
    return phases



class TestVideoDataset(Dataset):
    def __init__(self, dataset, root, sample_rate, args):
        self.dataset = dataset
        self.sample_rate = sample_rate
        self.args = args
        self.videos = []
        self.labels = []
        ###      
        self.video_names = []
        if dataset =='Cholec80':
            self.hard_frame_index = 7
            self.gap = 21
        if dataset == 'M2CAI':
            self.hard_frame_index = 8
            #TODO: not sure if this will work for all datasets
            self.gap = 155

        video_feature_folder = os.path.join(root, 'video_feature')
        labels_file = os.path.join(os.getcwd(), 'DATASETS/PHASES/annotations/Original_Datasets_Splits_Annotations/json_files',
                                     f'long_term_{self.args.dataset}_{self.args.split}.json')
        
        
        with open(labels_file, "r", encoding="utf-8") as f:
            annotation_json_file = json.load(f)

        # Transformr the json file into a pandas dataframe
        annotations_dataframe = pd.DataFrame(annotation_json_file['annotations'])
       
        for v_f in os.listdir(video_feature_folder):
            v_f_abs_path = os.path.join(video_feature_folder, v_f)
            videos = np.load(v_f_abs_path)[::sample_rate,]
            #v_label_file_abs_path = os.path.join(label_folder, v_f.split('.')[0] + '.txt')
            #labels = self.read_labels(v_label_file_abs_path)
            #labels = labels[::sample_rate]

            #Correct video name and use the LED format
            video_name = f'video_{int(v_f.split('.')[0].split('video')[1]) + self.gap}'
            # Use video_name to get the labels from the annotations_dataframe
            
            labels = annotations_dataframe[annotations_dataframe['video_name'] == video_name]['phases'].values.tolist()
           
            if videos.shape[0] != len(labels):
                raise ValueError(f"Video {v_f} has {videos.shape[0]} frames but {len(labels)} labels. Please check the dataset.")
           
            self.videos.append(videos)
            self.labels.append(labels)

            # Dont undestand why this is needed, but it is in the original code
            phase = 1
            for i in range(len(labels)-1):
                if labels[i] == labels[i+1]:
                    continue
                else:
                    phase += 1

            self.video_names.append(v_f)
       
        print('VideoDataset: Load dataset {} with {} videos.'.format(self.dataset, self.__len__()))

    def __len__(self):
        return len(self.videos)
       

  
    def __getitem__(self, item):
        video, label, video_name = self.videos[item], self.labels[item], self.video_names[item]
        return video, label, video_name
    
    
    def read_labels(self, label_file):
        with open(label_file,'r') as f:
            phases = [line.strip().split('\t')[1] for line in f.readlines()]
            labels = phase2label(phases, phase2label_dicts[self.dataset])
        return labels
