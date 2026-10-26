#!/usr/bin/env python3
"""Independently replay saved heads against regenerated frozen encoder maps."""
import argparse,sys,json
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--suite',default='experiments');ap.add_argument('--root',type=Path,default=ROOT);args=ap.parse_args();root=args.root.resolve();directory=root/args.suite
 component=directory.parent.name=='component_ablations'
 primary=directory.name=='experiments_foreground' or component
 sys.path.insert(0,str(root/('experiments_foreground' if primary else 'experiments')))
 if primary:from run_foreground import Classifier,predict
 else:from run_experiments import Classifier,predict
 if component:
  sys.path.insert(0,str(root/'component_ablations'))
  from run_component_ablations import factory
  Classifier=factory(directory.name)
 torch.set_num_threads(4)
 summary=json.loads((directory/'summary.json').read_text())
 colors=np.load(directory/'cache/color_descriptors.npy').astype(np.float32)
 if primary:fractions=np.load(directory/'cache/foreground_fractions.npy').astype(np.float32)
 checks=[]
 for backbone,variants in summary.items():
  for variant in variants:
   x=torch.from_numpy(np.load(directory/'cache'/(backbone+('_gray' if variant=='grayscale' else '')+'.npy')).astype(np.float32))
   expected=np.load(directory/f'results/{backbone}_{variant}_oof.npy')
   for fold in range(5):
    saved=np.load(directory/f'results/{backbone}_{variant}_fold{fold}_pred.npz');idx=saved['test_index']
    checkpoint=torch.load(directory/f'results/{backbone}_{variant}_fold{fold}.pt',map_location='cpu',weights_only=False)
    model=Classifier(checkpoint['channels'],variant=='color_attention');model.load_state_dict(checkpoint['state_dict']);model.eval()
    cc=colors[idx].copy()
    if primary:
     for j in range(18):
      bad=~np.isfinite(cc[:,:,18+j]);cc[:,:,18+j][bad]=checkpoint['roi_median'][j]
    normalized=np.clip((cc-checkpoint['color_mean'])/checkpoint['color_std'],-10,10)
    if primary:normalized=np.concatenate([normalized,fractions[idx,...,None]],axis=2)
    pp=predict(model,x[idx],torch.from_numpy(normalized))
    difference=float(np.abs(pp-expected[idx]).max());assert difference<2e-6,(backbone,variant,fold,difference)
    checks.append(dict(backbone=backbone,variant=variant,fold=fold,max_probability_difference=difference,n_test=len(idx)))
 report=dict(status='passed',suite=args.suite,tolerance=2e-6,folds_replayed=len(checks),checks=checks)
 (directory/'checkpoint_replay_validation.json').write_text(json.dumps(report,indent=2))
 print('PASS',args.suite,len(checks),'saved checkpoint predictions reproduced')

if __name__=='__main__':main()
