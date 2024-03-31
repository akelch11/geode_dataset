import subprocess
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--region', type=str)
args = parser.parse_args()



classes = [
            'dustbin',
            'hand_soap', 
            'house',
              'medicine', 
              'religious_building',
                'spices'
                ]



for class_name in classes:
    for b in [0.0, 0.2]:
        subprocess.run(
                [
                    "sbatch",
                    f"geode_eval/training/synth_pass_features/geode_{class_name}_{args.region}_b{b}/extract_features.slurm"
                ]
            )
