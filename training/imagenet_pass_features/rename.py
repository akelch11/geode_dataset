import os

for file_name in os.listdir('.'):
    
    if 'imagenet' in file_name and '.pt' in file_name:
        parts = file_name.split('_')
        os.rename(file_name, f'geode_features_{"_".join(parts[2:])}')