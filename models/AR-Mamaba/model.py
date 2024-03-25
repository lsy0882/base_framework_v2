import sys
sys.path.append('../')

import torch
import warnings
warnings.filterwarnings('ignore')

from utils.decorators import *
from .modules.module import *


@logger_wraps()
class Model(torch.nn.Module):
    def __init__(self, num_stages: int, num_spks: int, module_audio_enc: dict, module_feature_projector: dict, module_separator: dict, module_output_layer: dict, module_audio_dec: dict):
        super().__init__()
        self.num_stages = num_stages
        self.num_spks = num_spks
        self.audio_encoder = AudioEncoder(**module_audio_enc)
        self.feature_projector = FeatureProjector(**module_feature_projector)
        self.separator = Separator(**module_separator)
        self.out_layer = OutputLayer(**module_output_layer)
        self.audio_decoder = AudioDecoder(**module_audio_dec)
        
        # Aux_loss
        self.out_layer_bn = torch.nn.ModuleList([])
        self.decoder_bn = torch.nn.ModuleList([])
        for _ in range(self.num_stages):
            self.out_layer_bn.append(OutputLayer(**module_output_layer))
            self.decoder_bn.append(AudioDecoder(**module_audio_dec))
    
    def forward(self, x):
        encoder_output = self.audio_encoder(x)
        projected_feature = self.feature_projector(encoder_output)
        last_stage_output, each_stage_outputs = self.separator(projected_feature)
        out_layer_output = self.out_layer(last_stage_output, encoder_output)
        each_spk_output = [out_layer_output[idx] for idx in range(self.num_spks)]
        audio = [self.audio_decoder(each_spk_output[idx]) for idx in range(self.num_spks)]
        
        # Aux_loss
        audio_aux = []
        for idx, each_stage_output in enumerate(each_stage_outputs):
            each_stage_output = self.out_layer_bn[idx](torch.nn.functional.upsample(each_stage_output, encoder_output.shape[-1]), encoder_output)
            out_aux = [each_stage_output[jdx] for jdx in range(self.num_spks)]
            audio_aux.append([self.decoder_bn[idx](out_aux[jdx])[...,:x.shape[-1]] for jdx in range(self.num_spks)])
        return audio, audio_aux


if __name__ == "__main__":
    import argparse
    from torchinfo import summary as summary_
    from ptflops import get_model_complexity_info
    from thop import profile
    
    def check_parameters(net): return sum(param.numel() for param in net.parameters()) / 10**6
    
    def parse_yaml(yaml_conf):
        import os
        import yaml
        if not os.path.exists(yaml_conf): raise FileNotFoundError(f"Could not find configure files...{yaml_conf}")
        with open(yaml_conf, 'r') as f: config_dict = yaml.full_load(f)
        return config_dict
    
    parser = argparse.ArgumentParser(description="Command to start PIT training, configured by .yaml files")
    parser.add_argument(
        "--config",
        type=str,
        default="train.yaml",
        dest="config",
        help="Location of .yaml configure files for training")
    args = parser.parse_args()
    config_dict = parse_yaml(args.config)
    nnet_conf = config_dict["model"]
    nnet = Model(**nnet_conf)
    
    # ptflpos
    num_sample = 16000*5
    MACs_ptflops, params_ptflops = get_model_complexity_info(nnet, (num_sample,))
    MACs_ptflops, params_ptflops = MACs_ptflops.replace(" MMac", ""), params_ptflops.replace(" M", "")

    # thop
    input = torch.randn(1, num_sample)
    MACs_thop, params_thop = profile(nnet, inputs=(input, ), verbose=False)
    MACs_thop, params_thop = MACs_thop/1e6, params_thop/1e6    
    
    # torchinfo
    model_profile = summary_(nnet, input_size=(1, num_sample))
    MACs_torchinfo, params_torchinfo = model_profile.total_mult_adds/1e6, model_profile.total_params/1e6

    # pring detail
    print(f"ptflops: MMac: {MACs_ptflops}, Params: {params_ptflops}")
    print(f"thop: MMac: {MACs_thop}, Params: {params_thop}")
    print(f"torchinfo: MMac: {MACs_torchinfo}, Params: {params_torchinfo}")
    
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 500
    repetitions2 = 500
    timings = numpy.zeros((repetitions,1))
    
    # MEASURE PERFORMANCE
    device = torch.device("cuda")
    torch.set_num_threads(1)
    nnet = nnet.to(device)
    with torch.no_grad():
        for rep in range(repetitions+repetitions2):
            if rep > repetitions:
                dummy_input = torch.rand(1, 16000).to(device)
                starter.record()
                _ = nnet(dummy_input)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep-repetitions2] = curr_time
    print(timings.mean())