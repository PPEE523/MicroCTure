"""Evaluate one saved CNN on a prepared NPZ, without training or selection."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
from train_task1 import CNN,inputs,predict,grouped,metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    spec=checkpoint['spec'];model=CNN(len(spec[1]),spec[2]);model.load_state_dict(checkpoint['state_dict'])
    with np.load(args.data) as f:data={k:f[k] for k in f.files}
    p=predict(model,inputs(data,spec));ids,y,prob=grouped(data,p)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['structure_id','true','predicted','OPCID','CHIN','CHID'])
        w.writerows([sid,int(t),int(q.argmax()),*q.tolist()] for sid,t,q in zip(ids,y,prob))
    print(json.dumps(metrics(data,p),indent=2))

if __name__=='__main__':main()
