# MetaGen
This repository is the preliminary implementation of MetaGen: LLM-Driven Generative Framework for Intelligent Metasurface Element.


#### File Directory
```
MetaGen
├── /dataset/   # code for data prepareation, include HFSS/CST simulation and data synthesis
│  ├── /cst/
│  ├── /hfss/
│  ├── /stnthetic/
│  └── README
├── /interaction/
│  ├── qwen2vl_lora_sft_ds3.yaml    # train config
│  ├── qwen2vl_lora_eval.yaml   # eval config
│  └── README
├── /generation/
│  ├── *.py
│  ├── config.json  # train/eval config
│  └── README
└── /evaluation/
   ├── *.py
   ├── config.json  # train/eval config
   └── README
```


#### Data Preperation
Please refer to the repository of [Meta-atoms-data-sharing](https://github.com/SensongAn/Meta-atoms-data-sharing) for dataset preparing. Download [Freeform.mat](https://drive.google.com/drive/folders/13qKv8_AFmJ0Ysp2CuovNBBn5KkkV53wR) for future use.


#### Code preparation
1. Get into `/interaction/`, install the [LLAMA Factory](https://github.com/hiyouga/LLaMA-Factory/blob/main)
```shell
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```
2. Download [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) for supervised fine-tuning.
3. Get into `/dataset/` to collect and process the data to the specific format.
4. Modify the `/generation/config.json` and `/evaluation/config.json`, fill in your own directory and other hyperparameters.
5. Start training! Run `python main.py` under  `/generation/` and `/evaluation/`.
6. Modify the `/interaction/qwen2vl_lora_sft_ds3.yaml` with your information.
7. Start supervised fine-tuning!
```shell
NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="0,1,2,3" llamafactory-cli /interaction/qwen2vl_lora_sft_ds3.yaml
```