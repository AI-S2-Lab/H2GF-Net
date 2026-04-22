import subprocess
import sys

for dataset in ['MIntRec']:
    for seed in range(10):
        command = [
            'python',
            'run.py',
            '--dataset', dataset,
            '--logger_name', 'mult',
            '--method', 'mult',
            '--data_mode', 'multi-class',
            '--train',
            '--save_results',
            '--seed', str(seed),
            '--gpu_id', '0',
            '--video_feats_path', 'video_feats.pkl',
            '--audio_feats_path', 'audio_feats.pkl',
            '--text_backbone', 'bert-base-uncased',
            '--config_file_name', 'mult_bert',
            '--results_file_name', 'mult_bert.csv'
        ]
        subprocess.run(command, shell=True)