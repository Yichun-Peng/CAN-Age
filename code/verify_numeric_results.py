#!/usr/bin/env python3
"""Recompute published-result counters from saved held-out probabilities."""
import csv,json,argparse
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent

def readcsv(path):
 with open(path,newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def counters(labels,prob):
 predicted=prob.argmax(1);cm=np.zeros((5,5),int);np.add.at(cm,(labels,predicted),1)
 den=cm.sum(0)+cm.sum(1);f1=np.divide(2*np.diag(cm),den,out=np.zeros(5,float),where=den>0)
 return cm,float(np.trace(cm)/cm.sum()),float(f1.mean())

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT);args=ap.parse_args();root=args.root.resolve()
 meta=readcsv(root/'data_audit/image_metadata.csv');n=len(meta);labels=np.array([int(r['label']) for r in meta])
 assert n==1143 and len({r['image_path'] for r in meta})==n
 folds=readcsv(root/'data_audit/fold_assignments.csv');testcounts=np.zeros(n,int)
 for fold in range(5):
  part={k:[r for r in folds if int(r['fold'])==fold and r['partition']==k] for k in ['train','val','test']}
  ids={k:{int(r['image_index']) for r in v} for k,v in part.items()};groups={k:{r['day_group'] for r in v} for k,v in part.items()}
  assert set.union(*ids.values())==set(range(n))
  for a,b in [('train','val'),('train','test'),('val','test')]:assert not ids[a]&ids[b] and not groups[a]&groups[b]
  for i in ids['test']:testcounts[i]+=1
 assert (testcounts==1).all()
 checked=[]
 for path in sorted(root.rglob('summary.json')):
  if 'cache' in path.parts:continue
  summary=json.loads(path.read_text())
  for backbone,variants in summary.items():
   for variant,item in variants.items():
    prob=np.load(path.parent/f'results/{backbone}_{variant}_oof.npy')
    assert prob.shape==(n,5) and np.isfinite(prob).all() and np.allclose(prob.sum(1),1,atol=1e-6)
    cm,acc,f1=counters(labels,prob);reported=item['pooled']
    assert np.array_equal(cm,np.array(reported['confusion_matrix']))
    assert abs(acc-reported['accuracy'])<1e-12 and abs(f1-reported['macro_f1'])<1e-12
    assert np.trace(cm)==reported['correct']
    checked.append(dict(suite=str(path.parent.relative_to(root)),backbone=backbone,variant=variant,n=n,accuracy=acc,macro_f1=f1))
 brightness=[]
 for path in sorted(root.rglob('brightness_robustness.json')):
  data=json.loads(path.read_text());preds=readcsv(path.parent/'brightness_robustness_predictions.csv')
  by={}
  for r in preds:by.setdefault((f'{float(r["factor"]):.1f}',r['backbone'],r['variant']),[]).append(r)
  for (factor,backbone,variant),rr in by.items():
   assert len(rr)==n and len({int(r['image_index']) for r in rr})==n
   rr.sort(key=lambda r:int(r['image_index']))
   assert [int(r['outer_fold']) for r in rr]==[int(r['outer_fold']) for r in meta]
   prob=np.array([[float(r[f'prob_age_{j+1}']) for j in range(5)] for r in rr]);cm,acc,f1=counters(labels,prob)
   reported=data['results'][factor][backbone][variant]['pooled']
   assert np.array_equal(cm,np.array(reported['confusion_matrix'])) and abs(acc-reported['accuracy'])<1e-12 and abs(f1-reported['macro_f1'])<1e-12
   brightness.append(dict(suite=str(path.parent.relative_to(root)),factor=factor,backbone=backbone,variant=variant,n=n))
 report=dict(status='passed',n_images=n,outer_folds=5,day_group_overlap=False,all_images_tested_once=True,model_conditions=checked,brightness_conditions=brightness)
 (root/'numeric_validation.json').write_text(json.dumps(report,indent=2))
 print(f'PASS: {n} images, five day-disjoint folds, {len(checked)} model conditions, {len(brightness)} brightness conditions')

if __name__=='__main__':main()
