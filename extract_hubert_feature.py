# author: meiying chen
from pathlib import Path
import numpy as np
import os
import argparse

import torch
import torchaudio

def main():
    parser = argparse.ArgumentParser(description="Extract hubert code and construct directory structure")

    parser.add_argument('--root_path', default=None, type=str)
    parser.add_argument('--wavlist_path', default=None, type=str)
    parser.add_argument('--hubert_out_path', default=None, type=str)
    args = parser.parse_args()

    hubert_out_dir = Path(args.hubert_out_path)
    hubert = torch.hub.load('RF5/simple-asgan', 'hubert_base')
    if args.root_path is not None:
        rp = Path(args.root_path)
        all_files = list(args.root_path.rglob('**/*.wav'))
    else:
        with open(args.wavlist_path, "r") as f:
            all_files = []
            for l in f.readlines():
                all_files.append(Path(l.strip()))
    print(f"Processing {len(all_files)} files.")
    
    for p in all_files:
        wav, sr = torchaudio.load(p)
        
        print(wav)
        print(hubert)
        feats = hubert.get_feats_batched(wav) # (bs, seq_len, dim)
        output_path = hubert_out_dir / p.parent / f'{p.stem}.pt'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(output_path)
        print(feats.shape)
        break
        # torch.save(feats, output_path)

if __name__ == "__main__":
    main()
