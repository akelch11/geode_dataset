import torch
import pickle

with open('./region0.pkl', 'rb') as pickle_file:
    feats = pickle.load(pickle_file)


classes = ['Dustbin', 'Hand soap', 'House', 'Medicine', 'Religious building', 'Spices']

# feats = dict(feats)
# # for key in feats:
# #     print(list(feats[key]['train'].values())[0])
# feats = feats['Dustbin Image']['train']


reg_indexes = {0: 'Africa', 1: 'Americas', 2: 'EastAsia', 3:'Europe', 4: 'SouthEastAsia', 5:'WestAsia'}
for ri in range(6):
    with open(f'./region{ri}.pkl', 'rb') as pickle_file:
        feats = pickle.load(pickle_file)
    region = reg_indexes[ri]
    for class_name in classes:
        full_name = f'{class_name} Image'
        for mode in ['train', 'test']:
            feat_dict = feats[full_name][mode]
            feat_list = list(feat_dict.values())
            feat_tensor = torch.tensor(feat_list)
            feat_tensor = torch.permute(feat_tensor, (1, 0, 2))
            class_str = class_name.lower().replace(" ", '_')
            print('saving', f'imagenet_features_{class_str}_{mode}.pt', feat_tensor.shape)
            torch.save(feat_tensor, f'imagenet_features_{region}_{mode}_{class_str}.pt')