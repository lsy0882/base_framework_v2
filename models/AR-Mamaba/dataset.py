import os
import torch
import random
import librosa as audio_lib
import numpy as np

from utils import util_dataset
from utils.decorators import *
from loguru import logger
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import SpeedPerturbation

@logger_wraps()
def get_dataloaders(args, dataset_config, loader_config):    
    # create dataset object for each partition
    partitions = ["test"] if args.engine_mode == "test" else ["train", "valid", "test"]
    dataloaders = {}
    for partition in partitions:
        scp_config_mix = os.path.join(dataset_config["scp_dir"], dataset_config[partition]['mixture'])
        scp_config_spk = [os.path.join(dataset_config["scp_dir"], dataset_config[partition][spk_key]) for spk_key in dataset_config[partition] if spk_key.startswith('spk')]
        scp_config_noise = os.path.join(dataset_config["scp_dir"], dataset_config[partition]['noise']) if 'noise' in dataset_config[partition] else None
        dynamic_mixing = dataset_config[partition]["dynamic_mixing"]
        dataset = MyDataset(
            partition = partition,
            wave_scp_srcs = scp_config_spk,
            wave_scp_mix = scp_config_mix,
            wave_scp_noise = scp_config_noise,
            dynamic_mixing = dynamic_mixing)
        dataloader = DataLoader(
            dataset = dataset,
            batch_size = 1 if partition == 'test' else loader_config["batch_size"],
            shuffle = True, # only train: (partition == 'train') / all: True
            pin_memory = loader_config["pin_memory"],
            num_workers = loader_config["num_workers"],
            drop_last = loader_config["drop_last"],
            collate_fn = _collate)
        dataloaders[partition] = dataloader
    return dataloaders


def _collate(egs):
    """
        Transform utterance index into a minbatch

        Arguments:
            index: a list type [{},{},{}]

        Returns:
            input_sizes: a tensor correspond to utterance length
            input_feats: packed sequence to feed networks
            source_attr/target_attr: dictionary contains spectrogram/phase needed in loss computation
    """
    def __prepare_target_rir(dict_lsit, index):
        return torch.nn.utils.rnn.pad_sequence([torch.tensor(d["src"][index], dtype=torch.float32)  for d in dict_lsit], batch_first=True)
    if type(egs) is not list: raise ValueError("Unsupported index type({})".format(type(egs)))
    num_spks = 2 # you need to set this paramater by yourself
    dict_list = sorted([eg for eg in egs], key=lambda x: x['num_sample'], reverse=True)
    mixture = torch.nn.utils.rnn.pad_sequence([torch.tensor(d['mix'], dtype=torch.float32) for d in dict_list], batch_first=True)
    src = [__prepare_target_rir(dict_list, index) for index in range(num_spks)]
    input_sizes = torch.tensor([d['num_sample'] for d in dict_list], dtype=torch.float32)
    return input_sizes, mixture, src


@logger_wraps()
class MyDataset(Dataset):
    def __init__(self, partition, wave_scp_srcs, wave_scp_mix, wave_scp_noise, dynamic_mixing):
        self.partition = partition
        for wave_scp_src in wave_scp_srcs:
            if not os.path.exists(wave_scp_src): raise FileNotFoundError(f"Could not find file {wave_scp_src}")
        self.wave_dict_srcs = [util_dataset.parse_scps(wave_scp_src) for wave_scp_src in wave_scp_srcs]
        self.wave_dict_mix = util_dataset.parse_scps(wave_scp_mix)
        self.wave_dict_noise = util_dataset.parse_scps(wave_scp_noise) if wave_scp_noise else None
        self.wave_keys = list(self.wave_dict_mix.keys())
        logger.info(f"Create MyDataset for {wave_scp_mix} with {len(self.wave_dict_mix)} utterances")
        speed_list = [0.95, 0.96, 0.97, 0.98, 0.99, 
                      1.00, 1.00, 1.00, 1.00, 1.00, 
                      1.01, 1.02, 1.03, 1.04, 1.05]
        self.speed_aug = SpeedPerturbation(16000, speed_list)
        self.dynamic_mixing = dynamic_mixing
    
    def __len__(self):
        return len(self.wave_dict_mix)
    
    def __contains__(self, key):
        return key in self.wave_dict_mix
    
    def _dynamic_mixing(self, key):
        def __match_length(wav, len_data) : 
            leftover = len(wav) - len_data
            idx = random.randint(0,leftover)
            wav = wav[idx:idx+len_data]
            return wav
        
        samps_src = []
        src_len = []
        # dyanmic source choice        
        # checking whether it is the same speaker
        while True:
            key_random = random.choice(list(self.wave_dict_srcs[0].keys()))
            tmp1 = key.split('_')[1][:3] != key_random.split('_')[3][:3]
            tmp2 = key.split('_')[3][:3] != key_random.split('_')[1][:3]
            if tmp1 and tmp2: break
        
        idx1, idx2 = (0, 1) if random.random() > 0.5 else (1, 0)
        files = [self.wave_dict_srcs[idx1][key], self.wave_dict_srcs[idx2][key_random]]
        
        # load
        for file in files:
            if not os.path.exists(file): raise FileNotFoundError("Input file {} do not exists!".format(file))
            samps_tmp, _ = audio_lib.load(file, sr=8000)
            # mixing with random gains
            gain = pow(10,-random.uniform(-2.5,2.5)/20)
            # Speed Augmentation
            samps_tmp = np.array(self.speed_aug(torch.tensor(samps_tmp))[0])
            samps_src.append(gain*samps_tmp)
            src_len.append(len(samps_tmp))
        
        # matching the audio length
        min_len = min(src_len)
        
        # add noise source dynamically if needed
        if self.wave_dict_noise:
            key_random_noise = random.choice(list(self.wave_dict_noise.keys()))
            file_noise = self.wave_dict_noise[key_random_noise]
            samps_noise, _ = audio_lib.load(file_noise, sr=8000)
            gain_noise = pow(10,-random.uniform(-2.5,2.5)/20)
            samps_noise = samps_noise*gain_noise
            if min_len > len(samps_noise):
                factor_cat = min_len//len(samps_noise) + 1
                list_pad = [samps_noise for i in range(factor_cat)]
                samps_noise = np.concatenate(list_pad, axis=0)
            
            src_len.append(len(samps_noise))    
            min_len = min(src_len)
            samps_src = [__match_length(s, min_len) for s in samps_src]
            samps_noise = __match_length(samps_noise, min_len)
            samps_mix = sum(samps_src) + samps_noise
        else:
            samps_src = [__match_length(s, min_len) for s in samps_src]
            samps_mix = sum(samps_src)
        
        # ! truncated along to the sample Length "L"
        if len(samps_mix)%8 != 0:
            remains = len(samps_mix)%8
            samps_mix = samps_mix[:-remains]
            samps_src = [s[:-remains] for s in samps_src]
        
        if self.partition != "test":
            max_len = 32000
            if len(samps_mix) > max_len:
                start = random.randint(0, len(samps_mix)-max_len)
                samps_mix = samps_mix[start:start+max_len]
                samps_src = [s[start:start+max_len] for s in samps_src]
        return samps_mix, samps_src
    
    def _direct_load(self, key):
        samps_src = []
        files = [wave_dict_src[key] for wave_dict_src in self.wave_dict_srcs]    
        for file in files:
            if not os.path.exists(file): raise FileNotFoundError(f"Input file {file} do not exists!")
            samps_tmp, _ = audio_lib.load(file, sr=8000)
            samps_src.append(samps_tmp)
        
        file = self.wave_dict_mix[key]    
        if not os.path.exists(file): raise FileNotFoundError(f"Input file {file} do not exists!")
        samps_mix, _ = audio_lib.load(file, sr=8000)
        
        # Truncate samples as needed
        if len(samps_mix) % 8 != 0:
            remains = len(samps_mix) % 8
            samps_mix = samps_mix[:-remains]
            samps_src = [s[:-remains] for s in samps_src]
        
        if self.partition != "test":
            max_len = 32000
            if len(samps_mix) > max_len:
                start = random.randint(0,len(samps_mix)-max_len)
                samps_mix = samps_mix[start:start+max_len]
                samps_src = [s[start:start+max_len] for s in samps_src]
        
        return samps_mix, samps_src
    
    def __getitem__(self, index):
        key = self.wave_keys[index]
        if any(key not in self.wave_dict_srcs[i] for i in range(len(self.wave_dict_srcs))) or key not in self.wave_dict_mix: raise KeyError(f"Could not find utterance {key}")
        samps_mix, samps_src = self._dynamic_mixing(key) if self.dynamic_mixing else self._direct_load(key)
        return {"num_sample": samps_mix.shape[0], "mix": samps_mix, "src": samps_src}