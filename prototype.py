
import os
import time
from tracemalloc import start
import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pickle
import os
import re
import argparse
import numpy as np
import random
from thop import profile
from utils import  fusion,segment_bars_with_confidence_score

from tqdm import tqdm
f_path = os.path.abspath('..')
root_path = f_path.split('surgical_code')[0]


loss_layer = nn.CrossEntropyLoss()
mse_layer = nn.MSELoss(reduction='none')


def hierarch_train(args, model, train_loader, validation_loader, device, save_dir = 'models', debug = False):
   
    model.to(device)
    num_classes = args.num_classes
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    best_epoch = 0
    best_acc = 0
    model.train()
    save_name = 'hier{}_msloss{}_trans{}'.format(args.hier,args.ms_loss,args.trans)
    save_dir = os.path.join(save_dir, args.model,save_name)
    for epoch in range(1, args.epochs + 1):
        if epoch % 30 == 0:
            args.learning_rate = args.learning_rate * 0.5
       
        correct = 0
        total = 0
        loss_item = 0
        ce_item = 0 
        ms_item = 0
        lc_item = 0
        gl_item = 0
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=1e-5)
        max_seq = 0
        mean_len = 0
        ans = 0
        max_phase = 0
        for (video, labels, video_name) in (train_loader):
        
                
             
                labels = torch.Tensor(labels).long()
              
                    
                video, labels = video.to(device), labels.to(device)

              
                predicted_list, feature_list, prototype = model(video)
               
                mean_len += predicted_list[0].size(-1)
                ans += 1
                all_out, resize_list, labels_list = fusion(predicted_list,labels, args)

                max_seq = max(max_seq, video.size(1))
              
                

                loss = 0 
                
                if args.ms_loss:
                    ms_loss = 0
                  
                    for p,l in zip(resize_list,labels_list):
                        ms_loss += loss_layer(p.transpose(2, 1).contiguous().view(-1, args.num_classes), l.view(-1))
                        ms_loss += torch.mean(torch.clamp(mse_layer(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16))
                    loss = loss + ms_loss
                    ms_item += ms_loss.item()

                optimizer.zero_grad()
                loss_item += loss.item()

             
                if args.last:
                    all_out =  resize_list[-1]
                if args.first:
                    all_out = resize_list[0]
                
                # print(all_out.size())
                loss.backward()

                optimizer.step()
               
                
                _, predicted = torch.max(all_out.data, 1)
               
                # labels = labels_list[-1]
                correct += ((predicted == labels).sum()).item()
                total += labels.shape[0]
                # total +=1

        print('Train Epoch {}: Acc {}, Loss {}, ms {}'.format(epoch, correct / total, loss_item /total,  ms_item/total))
        if debug:
            # save_dir
            test_acc, predicted, out_pro, test_video_name=hierarch_test(args, model, validation_loader, device)
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save(model.state_dict(), save_dir + '/best_{}.model'.format(epoch))
        print('Best Test: Acc {}, Epoch {}'.format(best_acc, best_epoch))

def hierarch_test(args, model, test_loader, device, random_mask=False):
   
    model.to(device)
    num_classes = args.num_classes
    
    model.eval()
   
    with torch.no_grad():
        
        
            correct = 0
            total = 0
            loss_item = 0
            all_preds = []
            center = torch.ones((1, 64, num_classes), requires_grad=False)
            center = center.to(device)
            label_correct={}
            label_total= {}
            probabilty_list = []
            video_name_list=[]
            precision=0
            recall = 0
            ce_item = 0 
            ms_item = 0
            lc_item = 0
            gl_item = 0
            max_seq = 0 
            for n_iter,(video, labels, video_name ) in enumerate(test_loader):
                
                    
                labels = torch.Tensor(labels).long()
              
                    
                video, labels = video.to(device), labels.to(device)
                max_seq = max(max_seq, video.size(1))
               
                predicted_list, feature_list, _ = model(video)
                
                all_out, resize_list,labels_list = fusion(predicted_list,labels, args)
             
                loss = 0 

              
                
                if args.ms_loss:
                    ms_loss = 0
                    for p,l in zip(resize_list,labels_list):
                        # print(p.size())
                        ms_loss += loss_layer(p.transpose(2, 1).contiguous().view(-1, args.num_classes), l.view(-1))
                        ms_loss += torch.mean(torch.clamp(mse_layer(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16))
                    loss = loss + ms_loss
                    ms_item += ms_loss.item()
               
               
                loss_item += loss.item()

                if args.last:
                    all_out =  resize_list[-1]
                if args.first:
                    all_out = resize_list[0]

                _, predicted = torch.max(all_out.data, 1)
                

                predicted = predicted.squeeze()

                # labels = labels_list[-1]
                correct += ((predicted == labels).sum()).item()
                total += labels.shape[0]

              
                video_name_list.append(video_name)

                all_preds.append(predicted)

                all_out = F.softmax(all_out,dim=1)

                probabilty_list.append(all_out.transpose(1,2))

            print('Test  Acc {}, Loss {}, ms {}'.format( correct / total, loss_item /total, ms_item/total))


            for (kc, vc), (kall, vall) in zip(label_correct.items(),label_total.items()):
                print("{} acc: {}".format(kc, vc/vall))
            return correct / total, all_preds, probabilty_list, video_name_list
        

def base_predict(model, args, device,test_loader, pki = False,split='test'):

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

    model.to(device)
    model.eval()
    save_name = '{}_hier{}_trans{}'.format(args.sample_rate,args.hier,args.trans)
   

    pic_save_dir = 'results/{}/{}/vis/'.format(args.dataset,save_name)
    results_dir = 'results/{}/{}/prediction_{}/'.format(args.dataset,save_name,args.sample_rate)

    gt_dir = root_path+'/datasets/surgical/workflow/{}/phase_annotations/'.format(args.dataset)
  
    if not os.path.exists(pic_save_dir):
        os.makedirs(pic_save_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    with torch.no_grad():
        correct= 0
        total= 0 
        for (video, labels, video_name) in tqdm(test_loader):
            labels = torch.Tensor(labels).long()

            print(video.size(),video_name,labels.size())
            video = video.to(device)
            labels = labels.to(device)
            
            # Calculate model predictions for each 4 segmente feature level
            predicted_list, feature_list, _ = model(video)
            # Part d of the diagram, join all the information and produce the final output
            all_out, resize_list,labels_list = fusion(predicted_list,labels, args)

            if args.last:
                    all_out =  resize_list[-1]

            if args.first: # For inference is True
                    all_out = resize_list[0]
            
            confidence, predicted = torch.max(F.softmax(all_out.data, 1), 1)

            # Calculate manual accuracy
            correct += ((predicted == labels).sum()).item()
            total += labels.shape[0]

            predicted = predicted.squeeze(0).tolist()
            confidence = confidence.squeeze(0).tolist()
            
            labels = [label.item() for label in labels]

            print(video_name)

        print(correct/total)

