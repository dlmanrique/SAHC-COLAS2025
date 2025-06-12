from torch.utils.data import Dataset
import os
import numpy as np
import torch
import pandas as pd
import json
import random

seed = 19980125

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


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


datasets_2_videos_ids = {
    'Autolaparo': {'train': [f'video_{i:02}' for i in range(1, 11)], 'valid': [f'video_{i:02}' for i in range(11, 15)], 'test': [f'video_{i:02}' for i in range(16, 21)]},
    'Cholec80': {'train': [f'video_{i:02}' for i in range(22, 62)], 'test': [f'video_{i:02}' for i in range(62, 102)]},
    'HeiChole': {'train': [f'video_{i:02}' for i in range(102, 118)], 'test': [f'video_{i:02}' for i in range(118, 126)]},
    'HeiCo': {'train': [f'video_{i:02}' for i in range(126, 133)] + [f'video_{i:02}' for i in range(136, 143)] + [f'video_{i:02}' for i in range(146, 153)],
              'test': [f'video_{i:02}' for i in range(133, 136)] + [f'video_{i:02}' for i in range(143, 146)] + [f'video_{i:02}' for i in range(153, 156)]},
    'M2CAI': {'train': [f'video_{i:02}' for i in range(156, 183)], 'test': [f'video_{i:02}' for i in range(183, 197)]},
}



def phase2label(phases, phase2label_dict):
    labels = [phase2label_dict[phase] if phase in phase2label_dict.keys() else len(phase2label_dict) for phase in phases]
    return labels

def label2phase(labels, phase2label_dict):
    label2phase_dict = {phase2label_dict[k]:k for k in phase2label_dict.keys()}
    phases = [label2phase_dict[label] if label in label2phase_dict.keys() else 'HardFrame' for label in labels]
    return phases



class VideoDataset(Dataset):
    def __init__(self, dataset, args, split):
        self.dataset = dataset
        self.sample_rate = args.sample_rate
        self.args = args
        self.split = split
        self.videos = []
        self.labels = []
        self.videos_names = []


        # Obtain all dataset videos features and filter depending on the split
        video_feature_folder = os.path.join('/media/SSD3/dlmanrique/Endovis/MICCAI2025/SOTAS/Transvnet/Trans-SVNet-COLAS-2025/Resnet_features_SAHC', dataset)
        # Filter features based on video id and split
        dataset_split_video_names = datasets_2_videos_ids[self.dataset][self.split]
        split_filtered_features_videos = [f for f in os.listdir(video_feature_folder) if any(f.startswith(v) for v in dataset_split_video_names)]

        # Load corresponding labels
        labels_file = os.path.join(os.getcwd(), 'DATASETS/PHASES/annotations/Original_Datasets_Splits_Annotations/json_files',
                                     f'long_term_{self.dataset}_{self.split}.json')
    
        with open(labels_file, "r", encoding="utf-8") as f:
            annotation_json_file = json.load(f)

        # Transform the json file into a pandas dataframe
        annotations_dataframe = pd.DataFrame(annotation_json_file['annotations'])

        frames = 0
        # Iterate through the filtered video features
        for v_f in split_filtered_features_videos:
            v_f_abs_path = os.path.join(video_feature_folder, v_f)
            video_features = np.load(v_f_abs_path)

            #Correct video name and use the LED format
            video_name = '_'.join(v_f.split('_')[:2])

            # Use video_name to get the labels from the annotations_dataframe
            labels = annotations_dataframe[annotations_dataframe['video_name'] == video_name]['phases'].values.tolist()
           
            if video_features.shape[0] != len(labels):
                raise ValueError(f"Video {v_f} has {video_features.shape[0]} frames but {len(labels)} labels. Please check the dataset.")
           

            self.videos.append(video_features)
            self.labels.append(labels)
            self.videos_names.append(v_f.split('_features')[0])
            frames += video_features.shape[0]
       
        print('VideoDataset: Load dataset {}, split {} with {} videos and {} frames.'.format(self.dataset, self.split, self.__len__(), frames))

        

    def __len__(self):
        return len(self.videos)
       

  
    def __getitem__(self, item):
        video, label, video_name = self.videos[item], self.labels[item], self.videos_names[item]
        
        return video, label, video_name
    
    
    def read_labels(self, label_file):
        with open(label_file,'r') as f:
            phases = [line.strip().split('\t')[1] for line in f.readlines()]
            labels = phase2label(phases, phase2label_dicts[self.dataset])
        return labels
