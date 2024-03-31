import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import argparse
import torch
import os
from torch.utils.data import Dataset

## copyyy


parser = argparse.ArgumentParser()

parser.add_argument('--save', type=str, default='test_results', metavar='reg',
                    help='save directory')
parser.add_argument('--test', action='store_true')

parser.add_argument('--region', type=str)
parser.add_argument('--eta', type=float,default=0.0)
args = parser.parse_args()


params_train = {'batch_size': 128,
         'shuffle': True,
         'num_workers': 0}

params_valtest = {'batch_size': 128,
         'shuffle': False,
         'num_workers': 0}

region_to_name = {0:'africa', 1:'americas', 2:'eastasia', 3:'europe', 4:'southeastasia', 5:'westasia'}


all_features = []
all_labels = []
train_features_by_class = []
train_obj = []



class GeodeModifiedDataset(Dataset):
    def __init__(self, list_of_input_tensors, labels):
        self.input_list = list_of_input_tensors
        self.labels = labels
    def __getitem__(self, index):
        return torch.unsqueeze(self.input_list[index],0), torch.tensor(self.labels[index])
    def __len__(self):
        return len(self.input_list)


classes = ["dustbin", 'hand_soap', "house", 'medicine', "religious_building",'spices']

for index, class_name in enumerate(classes):

    train_features = torch.load(f'./geode_eval/training/imagenet_pass_features/imagenet_features_{args.region}_train_{class_name}.pt')
    labels = torch.ones_like(train_features) * index
    train_features = torch.squeeze(train_features, 0)
    for j in range(train_features.shape[0]):
        all_features.append(train_features[j])
        all_labels.append(index)

    
    synth_image_sizes = [0, 50, 100, 200, 400, 800, 1600, 3200]

    synth_size = 50
    synth_feature_path = f'./geode_eval/training/synth_pass_features/geode_{class_name}_{args.region}_b{args.eta}/features_and_labels.pt'
    synth_feature_list = torch.load(synth_feature_path)
    random_indexes = np.random.permutation(len(synth_feature_list))
    ## INTRODUCE SYNTH DATA
    for j in range(min(synth_size, len(synth_feature_list))):
        i = random_indexes[j]
        all_features.append(synth_feature_list[i][0])
        all_labels.append(synth_feature_list[i][1])

print(all_features[0].shape, all_labels[0], len(all_features), len(all_labels))



train_features, val_features, train_obj, val_obj = train_test_split(all_features, all_labels, test_size=0.2, random_state = 42)

print('after split')
print(len(train_features))



    # feat_with_labels = []
    # for j in range(len(train_features)):
    #     feat_with_labels[j] = (train_features[j], index)
    # all_features_with_labels.extend(feat_with_labels)

# print(len(all_features), all_features[0].shape)
# print(all_labels)


# # for i in range(6):
# #     region_features = pickle.load(open('geode_eval/training/imagenet_pass_features/region{}.pkl'.format(i), 'rb'))
    
# #     all_obj = sorted(list(region_features.keys()))
    
# #     for o in region_features.keys():
# #         for s in ['train']:
# #             train_features.append(region_features[o][s])
# #             train_obj.append(all_obj.index(o))


# train_features, val_features, train_obj, val_obj = train_test_split(train_features, train_obj, test_size=0.2, random_state = 42)

# train_features_dict = train_features[0]
# val_features_dict = val_features[0]



# def contains_class(x):
#      classes = ['dustbin', 'hand_soap', 'house', 'medicine', 'religious_building', 'spices']
#      return any([c in x for c in classes])

# # print(train_features_dict)
# train_features = [train_features_dict[k] for k in train_features_dict if contains_class(k)]
# val_features = [val_features_dict[k] for k in val_features_dict if contains_class(k)]


# print(train_features)

# train_features = torch.Tensor(np.concatenate(train_features)).squeeze()
# train_obj = torch.Tensor(np.array(train_obj))

# val_features = torch.Tensor(np.concatenate(val_features)).squeeze()
# val_obj = torch.Tensor(np.array(val_obj))

criterion = torch.nn.CrossEntropyLoss()
#acc_function = compute_acc 
ydtype = torch.long

# classify to 6 classes
m = torch.nn.Linear(2048, len(classes))  

optimizer_setting = {
    'optimizer': torch.optim.SGD,
    'lr': 0.1,
    'momentum': 0.9
}
optimizer = optimizer_setting['optimizer']( 
                    params=m.parameters(), 
                    lr=optimizer_setting['lr'], momentum=optimizer_setting['momentum']) 
count=0
if torch.cuda.is_available(): 
    device = torch.device('cuda')
    m = m.to(device)
else:
    device = torch.device('cpu')


if not os.path.exists(args.save):
    os.makedirs(args.save)

dtype = torch.float32
best_acc = 0.0

batch_size = params_train['batch_size']
N = len(train_features)//batch_size+1

if not args.test:
    

    for e in range(500):
        m.train()
        rand_perm = np.random.permutation(len(train_features))

        print(rand_perm)


        shuffled_features = []
        shuffled_labels = []
        for j in range(len(rand_perm)):
            shuffled_features.append(train_features[j])
            shuffled_labels.append(train_obj[j])

        # train_features = train_features[rand_perm]
        # train_obj = train_obj[rand_perm]


        # train_names = [train_names[i] for i in rand_perm]
        
        for t in range(N):
            feat_batch = torch.tensor(train_features[t*batch_size:(t+1)*batch_size]).to(device=device, dtype = dtype) 
            target_batch = torch.tensor(train_obj[t*batch_size:(t+1)*batch_size]).to(device=device, dtype = ydtype) 
            
            sc = m(feat_batch)

            optimizer.zero_grad()
            
            loss = criterion(sc, target_batch)
            loss.backward()

            optimizer.step()

            if t%50==0:
               print(e, t, loss, flush=True)

        
        m.eval()

        val_scores = m(val_features.to(device))
        loss_val = criterion(val_scores, val_obj.to(device, ydtype))

        val_pred = np.argmax(val_scores.squeeze().detach().cpu().numpy(), axis=1)
        acc = np.where(val_pred==val_obj.numpy(), 1, 0).mean() 
        
        print(e, loss, acc, flush=True)

        if acc>best_acc:
            best_acc =acc
            torch.save({'model':m.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch': e, 'loss':loss, 'acc':acc}, 
                        '{}/final.pth'.format(args.save)
                        )

        torch.save({'model':m.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch': e, 'loss':loss, 'acc':best_acc}, 
                    '{}/current.pth'.format(args.save)
                    )
            

m.load_state_dict(torch.load('{}/final.pth'.format(args.save))['model'])
m = m.to(device)


## do this for only one region




# for i in range(6):

all_test_features = []
all_test_labels = []


for index, class_name in enumerate(classes):

    test_features = torch.load(f'./geode_eval/training/imagenet_pass_features/imagenet_features_{args.region}_test_{class_name}.pt')
    labels = torch.ones_like(test_features) * index
    test_features = torch.squeeze(test_features, 0)
    for j in range(test_features.shape[0]):
        all_test_features.append(test_features[j])
        all_test_labels.append(index)

    
    
    # test_feat = []
    # test_obj = []

    # for o in region.keys():
    #     for s in [ 'test']:
    #         test_region.append(region[o][s])
    #         test_obj.append(all_obj.index(o))     
all_test_labels_numpy = all_test_labels.copy()
all_test_features = torch.tensor(all_test_features)
all_test_labels = torch.tensor(all_test_labels)
softmax = torch.nn.Softmax(dim=1)


m.eval()
scores = softmax(m(all_test_features.to(device))).detach().cpu().numpy()
print(scores.shape)

pred = np.argmax(scores, axis=1).squeeze()
acc_region = np.where(scores== all_test_labels_numpy, 1, 0).mean()

with open('{}/{}_test_scores.pkl'.format(args.save, args.region), 'wb+') as handle:
        pickle.dump(scores, handle)
print('Region {} test scores: '.format(args.region), 100*acc_region)

