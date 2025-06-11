import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import numpy as np
import random
from scipy.io import savemat

from utils import  fusion
from evaluation_metrics import MetricsEvaluator


# Configure device and seed everithing for reproducibility
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


loss_layer = nn.CrossEntropyLoss()
mse_layer = nn.MSELoss(reduction='none')
metrics_evaluator = MetricsEvaluator()


def hierarch_train(args, model, train_loader, validation_loader, device, save_dir = 'models', debug = False):
    
    
    # Send the model to the GPU and create the folder to save the checkpoints
    model.to(device)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Initialize variables to keep track of the best epoch and accuracy
    best_epoch = 0
    best_f1 = 0
    model.train()
    save_name = 'hier{}_msloss{}_trans{}'.format(args.hier,args.ms_loss,args.trans)
    save_dir = os.path.join(save_dir, args.model, save_name, args.datetime)

    # Training loop
    for epoch in tqdm(range(1, args.epochs + 1)):
        # Lr decay every 30 epochs
        if epoch % 30 == 0:
            args.learning_rate = args.learning_rate * 0.5
       
        correct = 0
        total = 0
        loss_item = 0
        ms_item = 0
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=1e-5)
        max_seq = 0
        mean_len = 0
        ans = 0
        
        # Iterate over the training data
        for (video, labels, video_name) in tqdm(train_loader):
                
            video_name = video_name[0]
    
            labels = torch.Tensor(labels).long()
            
            video, labels = video.to(device), labels.to(device)
            
            predicted_list, feature_list, prototype = model(video)
            
            mean_len += predicted_list[0].size(-1)
            ans += 1
            all_out, resize_list, labels_list = fusion(predicted_list, labels, args)

            max_seq = max(max_seq, video.size(1))
            
            loss = 0 
            
            if args.ms_loss: # Default True
                ms_loss = 0
                # Iterate over the 4th dimensions (F0, F1, F2, F3)
                for p,l in zip(resize_list, labels_list):
                    # Guess L_frame is this one
                    ms_loss += loss_layer(p.transpose(2, 1).contiguous().view(-1, args.num_classes), l.view(-1))
                    # Guess L_segment is this one
                    ms_loss += torch.mean(torch.clamp(mse_layer(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16))
                
                loss = loss + ms_loss
                ms_item += ms_loss.item()

            optimizer.zero_grad()
            loss_item += loss.item()

            
            if args.last:# Default is False
                all_out =  resize_list[-1]

            if args.first: # Default is True
                all_out = resize_list[0]
            

            loss.backward()

            optimizer.step()
            
            
            _, predicted = torch.max(all_out.data, 1)
            

            correct += ((predicted == labels).sum()).item()
            total += labels.shape[0]

            wandb.log({'Train Loss': loss.item()})


        print('Train Epoch {}: Acc {}, Loss {}, ms {}\n'.format(epoch, correct/total, loss_item,  ms_item))
        wandb.log({'Train Accuracy': correct/total})

        if debug:
            print('Validation at epoch {}'.format(epoch))
            # save_dir
            test_acc, predicted, out_pro, test_video_name, metrics_results = hierarch_test(args, model, validation_loader, device)

            if args.dataset == 'Cholec80' or args.dataset == 'M2CAI':
                f1_score = metrics_results[args.dataset]['mean']['f1_score']
            else:  
                f1_score = metrics_results[args.dataset]['f1_score']     

            if f1_score > best_f1:
                best_f1 = f1_score
                best_epoch = epoch
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save(model.state_dict(), save_dir + '/best_{}.model'.format(epoch))

                wandb.log({'Best metrics': metrics_results[args.dataset]})

        print('Best Test: F1 {}, Epoch {}\n'.format(best_f1, best_epoch))



def hierarch_test(args, model, test_loader, device, random_mask=False):
   
    model.to(device)    
    model.eval()
   
    with torch.no_grad():
            correct = 0
            total = 0
            loss_item = 0
            probabilty_list = []
            ms_item = 0
            max_seq = 0
            all_video_names = []
            all_predictions = []
            all_labels = []

            for n_iter,(video, labels, video_name) in enumerate(test_loader):
                
                video_name = video_name[0]

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

                all_out = F.softmax(all_out,dim=1)
                probabilty_list.append(all_out.transpose(1,2))

                # Save all labels, predictions and video id to calculate metrics
                all_predictions.append(predicted.cpu())
                all_labels.append(labels.cpu())
                all_video_names.append(video_name)
                
                # Save gt and predictions
                os.makedirs(f'Annotations_dummy/{args.dataset}/{args.datetime}', exist_ok=True)
                savemat(f'Annotations_dummy/{args.dataset}/{args.datetime}/{video_name}_annots.mat', {'Annots': labels.cpu().numpy()})

                os.makedirs(f'Predictions_dummy/{args.dataset}/{args.datetime}', exist_ok=True)
                savemat(f'Predictions_dummy/{args.dataset}/{args.datetime}/{video_name}_preds.mat', {'Preds': predicted.cpu().numpy()})


            # All metrics calculation
            metrics_results = metrics_evaluator.evaluation_per_dataset(all_predictions, all_labels, all_video_names, args.dataset)
            print('Test  Acc {}, Loss {}, ms {}'.format(correct/total, loss_item, ms_item))

            return correct / total, all_predictions, probabilty_list, all_video_names, metrics_results
        


def base_predict(model, args, device,test_loader, pki = False,split='test'):


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

