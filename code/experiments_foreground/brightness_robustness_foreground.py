#!/usr/bin/env python3
"""Source-pixel lighting perturbations, with fixed segmentation and held-out heads."""
import csv,time,json
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from run_foreground import (HERE,META,DATA,PROCESSING,BACKBONES,Classifier,
 foreground_descriptors,rgb_to_color9,load_encoder,rows,write_json,metrics,log,DEFAULT_DEVICE,ENCODER_THREADS,register_input_signature)

FACTORS=[.6,.8,1.,1.2,1.4]
VARIANTS=['baseline','color_attention']
PROTOCOL=dict(factors=FACTORS,transform='native RGB pixels: floor(clip(factor*I,0,255)); fixed original segmentation, ROI masks, orientation, background fill[124,116,104]; anatomical descriptors recalculated before resizing; no retraining',models=BACKBONES,variants=VARIANTS,folds=5,checkpoint_selection='original inner-validation checkpoint; unchanged',factor1='original OOF predictions reused',color_normalization='unchanged original training-fold medians and normalization')

def transform_inputs(factor,meta,rawroi,masks224):
 n=len(meta);images=np.empty((n,224,224,3),np.uint8);colors=[]
 fill=np.array([124,116,104],np.uint8)
 for i,r in enumerate(meta):
  rgba=np.asarray(Image.open(PROCESSING/f'aligned_rgba/{i:04d}.png').convert('RGBA')).copy()
  rgba[:,:,:3]=np.clip(rgba[:,:,:3].astype(np.float32)*factor,0,255).astype(np.uint8)
  im=Image.fromarray(rgba).resize((224,224),Image.Resampling.BILINEAR)
  rgb=np.asarray(im.convert('RGB'));images[i]=np.where(masks224[i,...,None],rgb,fill)
  roi=np.full(18,np.nan,np.float32)
  if np.isfinite(rawroi[i]).any():
   source=np.asarray(Image.open(DATA/r['image_path']).convert('RGB'),dtype=np.float32)
   source=np.clip(source*factor,0,255).astype(np.uint8)
   if np.isfinite(rawroi[i,:9]).all():
    mask=np.asarray(Image.open(PROCESSING/f'beak_masks/{i:04d}.png').convert('L'))>=128
    roi[:9]=rgb_to_color9(source[mask].astype(np.float32)/255).mean(0)
   if np.isfinite(rawroi[i,9:]).all():
    mask=np.asarray(Image.open(PROCESSING/f'feather_masks/{i:04d}.png').convert('L'))>=128
    roi[9:]=rgb_to_color9(source[mask].astype(np.float32)/255).std(0)
  colors.append(foreground_descriptors(rgb,masks224[i],roi)[0])
  if i%200==0:log(f'FG brightness factor={factor} source descriptors {i}/{n}')
 return images,np.stack(colors)

def main():
 torch.set_num_threads(4);meta=rows(META);n=len(meta)
 signature_sha256=register_input_signature(SimpleNamespace(device=DEFAULT_DEVICE,encoder_threads=ENCODER_THREADS,threads=4,extract_batch=16))
 labels=np.array([int(r['label']) for r in meta]);outer=np.array([int(r['outer_fold']) for r in meta])
 rawroi=np.load(HERE/'cache/raw_roi_features.npy');fractions=np.load(HERE/'cache/foreground_fractions.npy');masks224=np.load(HERE/'cache/foreground_masks224.npy')
 results={};device=torch.device(DEFAULT_DEVICE)
 normmean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
 normstd=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
 predfile=HERE/'brightness_robustness_predictions.csv'
 fields=['factor','backbone','variant','image_index','image_path','outer_fold','age_label','predicted_age_label','correct']+[f'prob_age_{j+1}' for j in range(5)]
 protocol=PROTOCOL
 prespecified=HERE/'brightness_prespecified_protocol.json'
 if prespecified.exists():assert json.loads(prespecified.read_text())==protocol,'Prespecified brightness protocol changed'
 else:write_json(prespecified,protocol)
 with open(predfile,'w',newline='') as outf:
  writer=csv.DictWriter(outf,fieldnames=fields);writer.writeheader()
  for factor in FACTORS:
   start=time.time();key=f'{factor:.1f}';results[key]={}
   if factor!=1.:images,colors=transform_inputs(factor,meta,rawroi,masks224)
   for name in BACKBONES:
    if factor!=1.:
     torch.set_num_threads(ENCODER_THREADS)
     log(f'FG brightness factor={factor} loading {name}')
     enc=load_encoder(name).to(device);features=[]
     with torch.inference_mode():
      for a in range(0,n,16):
       x=torch.from_numpy(images[a:a+16].copy()).to(device).permute(0,3,1,2).float()/255
       feature=enc((x-normmean)/normstd)
       if feature.shape[-2:]!=(7,7):feature=F.adaptive_avg_pool2d(feature,(7,7))
       features.append(feature.flatten(2).transpose(1,2).cpu().half().float())
     features=torch.cat(features);del enc
     if device.type=='mps':torch.mps.empty_cache()
    results[key][name]={}
    torch.set_num_threads(4)
    for variant in VARIANTS:
     if factor==1.:probs=np.load(HERE/f'results/{name}_{variant}_oof.npy')
     else:
      probs=np.full((n,5),np.nan,np.float32)
      for fold in range(5):
       idx=np.where(outer==fold)[0]
       checkpoint=torch.load(HERE/f'results/{name}_{variant}_fold{fold}.pt',map_location='cpu',weights_only=False)
       if checkpoint.get('input_signature_sha256')!=signature_sha256:raise RuntimeError('Brightness checkpoint input signature mismatch')
       model=Classifier(checkpoint['channels'],variant=='color_attention');model.load_state_dict(checkpoint['state_dict']);model.eval()
       cc=colors[idx].copy()
       for j in range(18):
        bad=~np.isfinite(cc[:,:,18+j]);cc[:,:,18+j][bad]=checkpoint['roi_median'][j]
       normalized=np.clip((cc-checkpoint['color_mean'])/checkpoint['color_std'],-10,10)
       z=torch.from_numpy(np.concatenate([normalized,fractions[idx,...,None]],axis=2))
       with torch.no_grad():probs[idx]=model(features[idx],z).softmax(1).numpy()
     assert not np.isnan(probs).any()
     foldmetrics=[metrics(labels[outer==fold],probs[outer==fold]) for fold in range(5)]
     result=dict(pooled=metrics(labels,probs),fold_metrics=foldmetrics,fold_mean_accuracy=float(np.mean([q['accuracy'] for q in foldmetrics])),fold_std_accuracy=float(np.std([q['accuracy'] for q in foldmetrics],ddof=1)))
     results[key][name][variant]=result
     for i,r in enumerate(meta):
      pred=int(probs[i].argmax())+1
      writer.writerow(dict(factor=factor,backbone=name,variant=variant,image_index=i,image_path=r['image_path'],outer_fold=outer[i],age_label=r['age_label'],predicted_age_label=pred,correct=int(pred==int(r['age_label'])),**{f'prob_age_{j+1}':float(probs[i,j]) for j in range(5)}))
     log(f'FG brightness factor={factor} {name} {variant} OOF acc={result["pooled"]["accuracy"]:.4f}')
    if factor!=1.:del features
   outf.flush();write_json(HERE/'brightness_robustness_partial.json',results)
   log(f'FG brightness factor={factor} complete seconds={time.time()-start:.1f}')
  write_json(HERE/'brightness_robustness.json',dict(protocol=protocol,results=results))
 original=np.load(HERE/'cache/images224.npy');brightness=[]
 for i in range(n):
  gray=original[i].astype(np.float32)@np.array([.299,.587,.114],np.float32)
  brightness.append(float(gray[masks224[i]].mean()))
 brightness=np.array(brightness);cuts=np.quantile(brightness,[1/3,2/3]);tiers=np.digitize(brightness,cuts,right=True)
 natural={}
 for name in BACKBONES:
  natural[name]={}
  for variant in VARIANTS:
   p=np.load(HERE/f'results/{name}_{variant}_oof.npy');natural[name][variant]={}
   for k,label in enumerate(['low','medium','high']):
    mask=tiers==k;natural[name][variant][label]=metrics(labels[mask],p[mask])
 write_json(HERE/'natural_brightness_tertiles.json',dict(tertile_boundaries=cuts.tolist(),definition='tertiles of foreground luma on unperturbed resized inputs; original OOF predictions',results=natural))
 log('All foreground source-pixel controlled and natural brightness robustness completed')

if __name__=='__main__':main()
