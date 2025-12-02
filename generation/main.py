"""
Main file for training and evaluation.
"""

from train import train, eval
import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="config.json", help='path to config.json', type=str, default="config.json")
    args = parser.parse_args()
    
    with open(args.config, 'r', encoding='utf-8') as file:
        modelConfig = json.load(file)
    if modelConfig["state"] == "train":
        print("train!")
        # Save the config file to the save_dir
        with open(f'{modelConfig["save_dir"]}/config.json', 'w', encoding='utf-8') as file:
            json.dump(modelConfig, file, ensure_ascii=False, indent=4)
        train(modelConfig)
    else:
        eval(modelConfig)
