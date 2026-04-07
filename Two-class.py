import subprocess

datasets = ['MIntRec']
seeds = range(10)

for dataset in datasets:
    for seed in seeds:
        command = [
            'python',
            'run.py',
            '--dataset', dataset,
            '--logger_name', 'mult',
            '--method', 'mult',
            '--train',
            '--data_mode', 'binary-class',
            '--save_results',
            '--seed', str(seed),
            '--gpu_id', '0',
            '--video_feats_path', 'video_feats.pkl',
            '--audio_feats_path', 'audio_feats.pkl',
            '--text_backbone', 'bert-base-uncased',
            '--config_file_name', 'mult_bert_bi',
            '--results_file_name', 'mult_bert_bi.csv'
        ]
        subprocess.run(command)