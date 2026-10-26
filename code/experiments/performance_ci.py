#!/usr/bin/env python3
"""Day-group bootstrap intervals for fixed out-of-fold predictions."""
import argparse,csv,json
from pathlib import Path
import numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,default=Path(__file__).resolve().parent);ap.add_argument('--replicates',type=int,default=1000);ap.add_argument('--metadata',type=Path,default=Path(__file__).resolve().parent.parent/'data_audit/image_metadata.csv');args=ap.parse_args();base=args.directory.resolve()
 with open(args.metadata) as f:meta=list(csv.DictReader(f))
 y=np.array([int(r['label']) for r in meta]);names=sorted({r['day_group'] for r in meta});gmap={v:i for i,v in enumerate(names)};groups=np.array([gmap[r['day_group']] for r in meta]);ng=len(names)
 rng=np.random.default_rng(20261026);draws=rng.integers(0,ng,size=(args.replicates,ng));weights=np.array([np.bincount(d,minlength=ng) for d in draws])
 sizes=np.bincount(groups,minlength=ng);denominator=weights@sizes
 result={};boot={};point={}
 summary=json.loads((base/'summary.json').read_text())
 for backbone,variants in summary.items():
  result[backbone]={}
  for variant in variants:
   p=np.load(base/f'results/{backbone}_{variant}_oof.npy');pred=p.argmax(1)
   groupcm=np.zeros((ng,5,5),int)
   np.add.at(groupcm,(groups,y,pred),1)
   bcm=np.einsum('rg,gij->rij',weights,groupcm)
   ac=np.trace(bcm,axis1=1,axis2=2)/denominator
   tp=bcm[:,range(5),range(5)];fden=bcm.sum(1)+bcm.sum(2)
   fs=np.divide(2*tp,fden,out=np.zeros_like(tp,dtype=float),where=fden>0).mean(1)
   recall=np.divide(tp,bcm.sum(2),out=np.zeros_like(tp,dtype=float),where=bcm.sum(2)>0)
   key=(backbone,variant);boot[key]=ac;point[key]=float((pred==y).mean())
   result[backbone][variant]={'accuracy':point[key],'accuracy_95ci':np.quantile(ac,[.025,.975]).tolist(),'macro_f1':summary[backbone][variant]['pooled']['macro_f1'],'macro_f1_95ci':np.quantile(fs,[.025,.975]).tolist(),'per_class_recall_95ci':np.quantile(recall,[.025,.975],axis=0).T.tolist()}
  result[backbone]['paired_effects']={}
  for variant in [v for v in ['color_attention','grayscale'] if v in variants and 'baseline' in variants]:
   delta=boot[(backbone,variant)]-boot[(backbone,'baseline')]
   result[backbone]['paired_effects'][variant+'_minus_baseline']={'accuracy_difference':point[(backbone,variant)]-point[(backbone,'baseline')],'difference_95ci':np.quantile(delta,[.025,.975]).tolist()}
 output={'method':'paired nonparametric calendar-day/source-group bootstrap of fixed out-of-fold predictions; percentile95CI; groups sampled with replacement, all images retained within sampled groups','groups':ng,'replicates':args.replicates,'seed':20261026,'results':result}
 (base/'performance_cluster_bootstrap.json').write_text(json.dumps(output,indent=2))
 print('Saved',base/'performance_cluster_bootstrap.json')

if __name__=='__main__':main()
