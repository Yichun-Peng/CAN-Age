#!/usr/bin/env python3
"""Frozen visual encoders and color-guided attention: reproducible local eagle CV."""
import argparse, csv, hashlib, json, math, os, random, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.transforms import functional as TF
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support, confusion_matrix,
                             cohen_kappa_score, log_loss)
from scipy.stats import spearmanr, wilcoxon

HERE=Path(__file__).resolve().parent
DATA=Path(os.environ.get('EAGLE_DATA_DIR',str(Path(__file__).resolve().parents[2] / 'data')))
PROCESSING=Path(os.environ.get('EAGLE_PROCESSING_DIR',str(HERE.parent/'image_processing_cpu_semantic_v1'))).expanduser().resolve()
if os.environ.get('BWHD_ROOT'):
 allowed_root=Path(os.environ['BWHD_ROOT']).resolve()
 for managed_path in [HERE,DATA,PROCESSING]:
  if not managed_path.resolve().is_relative_to(allowed_root):raise ValueError('Server path outside BWHD_ROOT: '+str(managed_path))
DEFAULT_DEVICE=os.environ.get('EAGLE_DEVICE','mps' if torch.backends.mps.is_available() else 'cpu')
ENCODER_THREADS=int(os.environ.get('EAGLE_NUM_THREADS','4'))
META=HERE.parent/'data_audit/image_metadata.csv'
FOLDS=HERE.parent/'data_audit/fold_assignments.csv'
BACKBONES=['mobilenet_v3_small','vgg11','resnet18','densenet121']
VARIANTS=['baseline','color_attention','grayscale']
CONFIG=dict(seed=20261026,input_size=224,grid_size=7,backbones=BACKBONES,
 variants=VARIANTS,encoder_training='frozen ImageNet-1K encoders',optimizer='Adam',
 lr=0.0005,batch_size=32,max_epochs=50,patience=10,min_delta=0.0001,
 weight_decay=0.0001,dropout=0.2,visual_dim=128,color_dim=128,attention_hidden=32,
 outer_split='StratifiedGroupKFold, five folds, calendar-day groups, seed 42',
 inner_split='StratifiedGroupKFold, four folds, first fold, seed 100+outer_fold',
 loss='cross-entropy with inverse class-frequency weights computed on training only',
 early_stopping_metric='unweighted validation cross-entropy',
 color_descriptors='36 per spatial cell: local foreground mean/std RGB/HSV/CIELAB18, raw beak means9, raw feather standard deviations9',
 color_normalization='training-fold medians impute missing raw ROI descriptors; training-fold foreground-cell mean/std; clipped to [-10,10]',
 encoder_normalization='ImageNet RGB means [.485,.456,.406], std [.229,.224,.225]',
 image_preprocessing='foreground segmentation and horizontal direction alignment upstream; aligned RGBA bilinear resize224; binary mask nearest-neighbor resize224; background fixed RGB[124,116,104]; no crop',
 augmentation='none',
 color_attention_formula='a=softmax_s(w2 tanh(W1 z_s+b1)+b2+log(f_s)), with zero-foreground cells masked; v=ReLU(Wv sum_s a_s F_s+bv); c=ReLU(Wc sum_s a_s z_s+bc); g=sigmoid(wg[v;c]+bg); h=(1-g)*v+g*c; logits=Wo Dropout(h)+bo',
 baseline_formula='v=ReLU(Wv mean_s F_s+bv); logits=Wo Dropout(v)+bo',
 grayscale_formula='same baseline architecture and training on 3-channel grayscale encoder inputs',
 selection='all models evaluated on identical predefined outer folds; each fold checkpoint selected only by inner validation loss',
 primary_input='background-removed, direction-aligned images and anatomical color descriptors',
 background_fill=[124,116,104],
 raw_roi_definition='source-resolution beak RGB/HSV/CIELAB means9 plus feather RGB/HSV/CIELAB population standard deviations9; unreliable/missing anatomy is NaN then training-fold medians',
 missing_all_training_roi_policy='if a feature has no finite training observations, use 0 and record unsupported feature; no labels used to impute')
PRIMARY_ROOT=HERE
PRIMARY_PROTOCOL_CONFIG=json.loads(json.dumps(CONFIG))

def log(s):
 print(time.strftime('%Y-%m-%d %H:%M:%S'),s,flush=True)

def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

def rows(path):
 with open(path,newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def write_json(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False))

def file_sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def input_signature(args):
 """Hash every source image, relevant native ROI, fixed split, method and weight."""
 files=[]
 def add(path,logical):
  path=Path(path)
  if not path.is_file():raise FileNotFoundError('Required signed input is missing: '+str(path))
  files.append(dict(path=logical,bytes=path.stat().st_size,sha256=file_sha256(path)))
 for path in [META,FOLDS,META.parent/'splits.json',META.parent/'inventory.csv']:
  add(path,'data_audit/'+path.name)
 for name in ['processing_config.json','features.csv','raw_roi_features.npz']:
  add(PROCESSING/name,'roi/'+name)
 meta=rows(META)
 for i,r in enumerate(meta):
  add(DATA/r['image_path'],'photos/'+r['image_path'])
  for folder in ['foreground_masks','aligned_rgba','aligned_foreground_masks','beak_masks','feather_masks']:
   add(PROCESSING/f'{folder}/{i:04d}.png',f'roi/{folder}/{i:04d}.png')
 weights={'mobilenet_v3_small':models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
          'vgg11':models.VGG11_Weights.IMAGENET1K_V1,
          'resnet18':models.ResNet18_Weights.IMAGENET1K_V1,
          'densenet121':models.DenseNet121_Weights.IMAGENET1K_V1}
 weight_sources={}
 for name,weight in weights.items():
  filename=weight.url.rsplit('/',1)[-1];path=Path(torch.hub.get_dir())/'checkpoints'/filename
  add(path,'pretrained/'+filename)
  expected=filename.rsplit('-',1)[-1].split('.')[0]
  if not files[-1]['sha256'].startswith(expected):raise ValueError('Official pretrained weight hash mismatch: '+str(path))
  weight_sources[name]=dict(url=weight.url,filename=filename)
 add(Path(__file__).resolve(),'implementation/run_foreground.py')
 manifest=dict(schema='foreground-input-signature-v1',n_images=len(meta),files=files,
               method=PRIMARY_PROTOCOL_CONFIG,pretrained_weights=weight_sources,
               seed_policy='base_seed + fixed BACKBONES.index(backbone)*1000 + outer_fold*10',
               execution=dict(encoder_device=str(args.device),encoder_threads=getattr(args,'encoder_threads',ENCODER_THREADS),
                              head_threads=args.threads,extract_batch=getattr(args,'extract_batch',16),
                              python=sys.version,torch=torch.__version__,torchvision=__import__('torchvision').__version__,
                              numpy=np.__version__,pillow=__import__('PIL').__version__))
 digest=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
 return dict(sha256=digest,manifest=manifest,locations=dict(data=str(DATA.resolve()),processing=str(PROCESSING)))

def validate_signature_directory(directory,current,shared_feature_cache=False):
 """Never adopt unsigned or differently signed cached features/checkpoints."""
 directory=Path(directory);path=directory/'input_signature.json'
 if path.exists():
  previous=json.loads(path.read_text())
  if previous.get('sha256')!=current['sha256']:
   raise RuntimeError('Input signature changed; use a fresh experiment directory or archive existing cache/checkpoints: '+str(directory))
  return
 artifacts=[p for p in (directory/'results').glob('*') if p.is_file() and p.suffix in ['.pt','.npz','.npy']]
 if not shared_feature_cache:
  artifacts.extend(p for p in (directory/'cache').glob('*') if p.is_file() and (p.suffix in ['.npy','.npz','.part']))
 if artifacts:raise RuntimeError('Unsigned cache/checkpoints cannot be reused: '+str(directory))
 write_json(path,current)

def register_input_signature(args):
 log('Hashing original inputs, anatomical ROIs, fixed folds, method and official weights')
 current=input_signature(args)
 validate_signature_directory(PRIMARY_ROOT,current)
 if HERE!=PRIMARY_ROOT:validate_signature_directory(HERE,current,shared_feature_cache=True)
 log('Verified primary input signature '+current['sha256'])
 return current['sha256']

def rgb_to_color9(rgb):
 """sRGB, HSV and CIELAB D65, each represented on roughly unit scales."""
 mx=rgb.max(-1);mn=rgb.min(-1);d=mx-mn
 h=np.zeros_like(mx); mask=d>1e-8
 rm=mask&(rgb[...,0]==mx);gm=mask&(rgb[...,1]==mx)&~rm;bm=mask&~rm&~gm
 h[rm]=((rgb[...,1][rm]-rgb[...,2][rm])/d[rm])%6
 h[gm]=(rgb[...,2][gm]-rgb[...,0][gm])/d[gm]+2
 h[bm]=(rgb[...,0][bm]-rgb[...,1][bm])/d[bm]+4
 sat=np.divide(d,mx,out=np.zeros_like(d),where=mx>1e-8)
 hsv=np.stack([h/6,sat,mx],-1)
 linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
 xyz=linear@np.array([[.4124564,.2126729,.0193339],[.3575761,.7151522,.1191920],[.1804375,.0721750,.9503041]],dtype=np.float32)
 xyz/=np.array([.95047,1,1.08883],dtype=np.float32)
 delta=6/29
 q=np.where(xyz>delta**3,np.cbrt(xyz),xyz/(3*delta**2)+4/29)
 lab=np.stack([(116*q[...,1]-16)/100,500*(q[...,0]-q[...,1])/128,200*(q[...,1]-q[...,2])/128],-1)
 return np.concatenate([rgb,hsv,lab],-1).astype(np.float32)

def color_descriptor(image):
 rgb=np.asarray(image,dtype=np.float32)/255
 q=rgb_to_color9(rgb)
 cells=q.reshape(7,32,7,32,9).transpose(0,2,1,3,4).reshape(49,1024,9)
 local=np.concatenate([cells.mean(1),cells.std(1)],1)
 glob=np.r_[q.mean((0,1)),q.std((0,1))]
 z=np.concatenate([local,np.broadcast_to(glob,(49,18))],1)
 gray=rgb@np.array([.299,.587,.114],np.float32)
 stats={'overall_brightness':float(gray.mean()*255),'optical_color_descriptor':float(gray.std()*255)}
 for k,arr in [('red',rgb[...,0]),('green',rgb[...,1]),('blue',rgb[...,2]),('hue',q[...,3]),('saturation',q[...,4]),('value',q[...,5]),('lightness',q[...,6])]:
  stats[k+'_mean']=float(arr.mean())
 return z,stats

def foreground_descriptors(rgb,mask,rawroi):
 """Foreground cell statistics plus source-resolution anatomical features."""
 q=rgb_to_color9(rgb.astype(np.float32)/255)
 qc=q.reshape(7,32,7,32,9).transpose(0,2,1,3,4).reshape(49,1024,9)
 mc=mask.reshape(7,32,7,32).transpose(0,2,1,3).reshape(49,1024)
 local=np.zeros((49,18),np.float32)
 for cell in range(49):
  pix=qc[cell][mc[cell]]
  if len(pix):local[cell]=np.r_[pix.mean(0),pix.std(0)]
 z=np.concatenate([local,np.broadcast_to(rawroi,(49,18))],1)
 return z,mc.mean(1).astype(np.float32)

def load_encoder(name):
 if name=='mobilenet_v3_small':
  net=models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
  enc=net.features
 elif name=='vgg11':
  net=models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1);enc=net.features
 elif name=='resnet18':
  net=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
  enc=nn.Sequential(*list(net.children())[:-2])
 elif name=='densenet121':
  net=models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
  enc=nn.Sequential(net.features,nn.ReLU(inplace=False))
 else:raise ValueError(name)
 enc.requires_grad_(False);enc.eval()
 return enc

def extract(args):
 torch.set_num_threads(getattr(args,'encoder_threads',ENCODER_THREADS))
 register_input_signature(args)
 meta=rows(META);n=len(meta);cache=HERE/'cache';cache.mkdir(exist_ok=True)
 log(f'Extraction metadata n={n}; device={args.device}; torch={torch.__version__}')
 images=np.empty((n,224,224,3),np.uint8);grayimages=np.empty_like(images);masks224=np.empty((n,224,224),bool)
 descriptors=[];fractions=[];statrows=[]
 rawroi=np.load(PROCESSING/'raw_roi_features.npz')['features'].astype(np.float32)
 assert rawroi.shape==(n,18)
 fill=np.array(CONFIG['background_fill'],dtype=np.float32)
 for i,r in enumerate(meta):
  im=Image.open(PROCESSING/f'aligned_rgba/{i:04d}.png').convert('RGBA').resize((224,224),Image.Resampling.BILINEAR)
  mask=np.asarray(Image.open(PROCESSING/f'aligned_foreground_masks/{i:04d}.png').convert('L').resize((224,224),Image.Resampling.NEAREST))>=128
  if not mask.any():raise ValueError(f'Empty foreground mask image_index {i}')
  masks224[i]=mask
  rgb=np.asarray(im.convert('RGB'),dtype=np.float32)
  images[i]=np.where(mask[...,None],rgb,fill).astype(np.uint8)
  lum=rgb@np.array([.299,.587,.114],dtype=np.float32)
  grayimages[i]=np.where(mask[...,None],lum[...,None],fill).astype(np.uint8)
  z,frac=foreground_descriptors(rgb,mask,rawroi[i])
  descriptors.append(z);fractions.append(frac)
  statrows.append(dict(image_index=i,image_path=r['image_path'],age_label=r['age_label'],day_group=r['day_group'],foreground_fraction=float(mask.mean()),beak_roi_available=int(np.isfinite(rawroi[i,:9]).all()),feather_roi_available=int(np.isfinite(rawroi[i,9:]).all())))
 np.save(cache/'color_descriptors.npy',np.stack(descriptors))
 np.save(cache/'foreground_fractions.npy',np.stack(fractions))
 np.save(cache/'raw_roi_features.npy',rawroi)
 with open(HERE/'color_statistics.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(statrows[0]));w.writeheader();w.writerows(statrows)
 np.save(cache/'images224.npy',images)
 np.save(cache/'images_gray224.npy',grayimages)
 np.save(cache/'foreground_masks224.npy',masks224)
 device=torch.device(args.device)
 normmean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
 normstd=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
 for name in args.models:
  log('Loading pretrained '+name);enc=load_encoder(name).to(device)
  for gray in [False,True]:
   dest=cache/(name+('_gray' if gray else '')+'.npy')
   if dest.exists() and not args.overwrite:
    log('Existing cache '+dest.name);continue
   out=None;start=time.time()
   with torch.inference_mode():
    for a in range(0,n,args.extract_batch):
     source=grayimages if gray else images
     x=torch.from_numpy(source[a:a+args.extract_batch].copy()).to(device).permute(0,3,1,2).float()/255
     y=enc((x-normmean)/normstd)
     if y.shape[-2:]!=(7,7): y=F.adaptive_avg_pool2d(y,(7,7))
     value=y.flatten(2).transpose(1,2).cpu().numpy().astype(np.float16)
     if out is None:out=np.lib.format.open_memmap(str(dest)+'.part',mode='w+',dtype=np.float16,shape=(n,49,value.shape[2]))
     out[a:a+len(value)]=value
     if a==0 or (a//args.extract_batch)%8==0 or a+len(value)==n:
      log(f'extract {name} gray={gray} {a+len(value)}/{n}, elapsed={time.time()-start:.1f}s')
    out.flush();del out;os.replace(str(dest)+'.part',dest)
   log(f'Extraction finished {dest.name} seconds={time.time()-start:.1f}')
  del enc
  if device.type=='mps':torch.mps.empty_cache()
 log('All extraction complete')

class Classifier(nn.Module):
 def __init__(self,channels,attention):
  super().__init__();self.attention=attention
  self.visual=nn.Linear(channels,128)
  self.dropout=nn.Dropout(.2);self.output=nn.Linear(128,5)
  if attention:
   self.score=nn.Sequential(nn.Linear(36,32),nn.Tanh(),nn.Linear(32,1))
   self.color=nn.Linear(36,128)
   self.gate=nn.Linear(256,1)
 def forward(self,x,z,return_attention=False):
  if not self.attention:
   v=F.relu(self.visual(x.mean(1)))
   return self.output(self.dropout(v))
  frac=z[:,:,36:37]
  z=z[:,:,:36]
  scores=self.score(z)+torch.log(frac.clamp(min=1e-30))
  scores=scores.masked_fill(frac<=0,-torch.inf)
  a=scores.softmax(1)
  v=F.relu(self.visual((x*a).sum(1)))
  c=F.relu(self.color((z*a).sum(1)))
  g=self.gate(torch.cat([v,c],1)).sigmoid()
  h=(1-g)*v+g*c
  logits=self.output(self.dropout(h))
  return (logits,a,g) if return_attention else logits

def predict(model,x,z,batch=128):
 model.eval();pp=[]
 with torch.no_grad():
  for i in range(0,len(x),batch):pp.append(model(x[i:i+batch],z[i:i+batch]).softmax(1).numpy())
 return np.concatenate(pp)

def metrics(y,p):
 pred=p.argmax(1);pr,re,fs,support=precision_recall_fscore_support(y,pred,labels=np.arange(5),zero_division=0)
 cm=confusion_matrix(y,pred,labels=np.arange(5))
 errors=(y!=pred);adj=errors&(np.abs(y-pred)==1)
 return dict(n=len(y),correct=int((~errors).sum()),accuracy=float(accuracy_score(y,pred)),
  balanced_accuracy=float(balanced_accuracy_score(y,pred)),macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),
  weighted_f1=float(f1_score(y,pred,average='weighted',zero_division=0)),ordinal_mae=float(np.abs(y-pred).mean()),
  quadratic_weighted_kappa=float(cohen_kappa_score(y,pred,weights='quadratic')),
  log_loss=float(log_loss(y,p,labels=np.arange(5))),confusion_matrix=cm.tolist(),
  adjacent_error_count=int(adj.sum()),error_count=int(errors.sum()),
  adjacent_error_fraction=float(adj.sum()/errors.sum()) if errors.any() else 0.,
  per_class=[dict(age_label=i+1,precision=float(pr[i]),recall=float(re[i]),f1=float(fs[i]),n=int(support[i])) for i in range(5)])

def train(args):
 torch.set_num_threads(args.threads)
 signature_sha256=register_input_signature(args)
 meta=rows(META);ff=rows(FOLDS);n=len(meta);labels=np.array([int(r['label']) for r in meta]);cache=HERE/'cache'
 colors=np.load(cache/'color_descriptors.npy').astype(np.float32)
 fractions=np.load(cache/'foreground_fractions.npy').astype(np.float32)
 rawroi=np.load(cache/'raw_roi_features.npy').astype(np.float32)
 seed_all(CONFIG['seed'])
 config={**CONFIG,'torch':torch.__version__,'torchvision':__import__('torchvision').__version__,
         'python':sys.version,'head_device':'cpu','encoder_device':args.device,'cpu_threads':args.threads,
         'encoder_cpu_threads':getattr(args,'encoder_threads',ENCODER_THREADS),
         'processing_directory':str(PROCESSING),'input_signature_sha256':signature_sha256}
 write_json(HERE/'config.json',config)
 allsummary={};outdir=HERE/'results';outdir.mkdir(exist_ok=True)
 startall=time.time()
 for name in args.models:
  allsummary[name]={}
  for vi,variant in enumerate(args.variants):
   features=np.load(cache/(name+('_gray' if variant=='grayscale' else '')+'.npy')).astype(np.float32)
   x=torch.from_numpy(features);channels=x.shape[-1]
   probs=np.full((n,5),np.nan,np.float32);foldmetrics=[]
   for fold in range(5):
    seed=CONFIG['seed']+BACKBONES.index(name)*1000+fold*10
    resultpath=outdir/f'{name}_{variant}_fold{fold}.json'
    predpath=outdir/f'{name}_{variant}_fold{fold}_pred.npz'
    checkpoint=outdir/f'{name}_{variant}_fold{fold}.pt'
    if resultpath.exists() and predpath.exists() and checkpoint.exists() and not args.overwrite:
     result=json.loads(resultpath.read_text());assert result['seed']==seed,('Stored checkpoint seed mismatch',resultpath,result['seed'],seed)
     if result.get('input_signature_sha256')!=signature_sha256:raise RuntimeError('Stored checkpoint input signature mismatch: '+str(resultpath))
     saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
     if saved.get('seed')!=seed or saved.get('input_signature_sha256')!=signature_sha256:raise RuntimeError('Saved model input signature or seed mismatch: '+str(checkpoint))
     del saved
     npz=np.load(predpath);probs[npz['test_index']]=npz['probabilities'];foldmetrics.append(result['test_metrics']);log('Resuming completed '+resultpath.stem);continue
    seed_all(seed)
    indices={part:np.array([int(r['image_index']) for r in ff if int(r['fold'])==fold and r['partition']==part]) for part in ['train','val','test']}
    tr,va,te=indices['train'],indices['val'],indices['test']
    assert not set(tr)&set(va) and not set(tr)&set(te) and not set(va)&set(te)
    # Anatomical medians use each training image once, never test or validation.
    with np.errstate(all='ignore'):
     roi_median=np.nanmedian(rawroi[tr],axis=0)
    unsupported=np.where(~np.isfinite(roi_median))[0].tolist()
    roi_median=np.nan_to_num(roi_median,nan=0.)
    imputed=colors.copy()
    for j in range(18):
     bad=~np.isfinite(imputed[:,:,18+j]);imputed[:,:,18+j][bad]=roi_median[j]
    valid=imputed[tr][fractions[tr]>0]
    mu=valid.mean(0);sd=np.maximum(valid.std(0),1e-6)
    standardized=np.clip((imputed-mu)/sd,-10,10)
    z=torch.from_numpy(np.concatenate([standardized,fractions[...,None]],axis=2))
    y=torch.from_numpy(labels)
    model=Classifier(channels,variant=='color_attention')
    optimizer=torch.optim.Adam(model.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'])
    counts=np.bincount(labels[tr],minlength=5);weights=torch.tensor(len(tr)/(5*counts),dtype=torch.float32)
    lossfn=nn.CrossEntropyLoss(weight=weights)
    best=math.inf;best_epoch=0;bad=0;history=[];state=None;start=time.time()
    for epoch in range(1,CONFIG['max_epochs']+1):
     model.train();order=np.random.permutation(tr);total=0.;seen=0
     for a in range(0,len(order),CONFIG['batch_size']):
      idx=order[a:a+CONFIG['batch_size']];optimizer.zero_grad(set_to_none=True)
      logits=model(x[idx],z[idx]);loss=lossfn(logits,y[idx]);loss.backward();optimizer.step()
      total+=loss.item()*len(idx);seen+=len(idx)
     vp=predict(model,x[va],z[va]);vl=float(log_loss(labels[va],vp,labels=np.arange(5)))
     history.append(dict(epoch=epoch,training_weighted_cross_entropy=total/seen,validation_cross_entropy=vl,validation_accuracy=float((vp.argmax(1)==labels[va]).mean())))
     if vl<best-CONFIG['min_delta']:
      best=vl;best_epoch=epoch;bad=0;state={k:v.detach().clone() for k,v in model.state_dict().items()}
     else:bad+=1
     if bad>=CONFIG['patience']:break
    model.load_state_dict(state);tp=predict(model,x[te],z[te]);probs[te]=tp
    mm=metrics(labels[te],tp);foldmetrics.append(mm)
    torch.save(dict(state_dict=state,color_mean=mu,color_std=sd,roi_median=roi_median,unsupported_roi_features=unsupported,channels=channels,variant=variant,backbone=name,fold=fold,seed=seed,input_signature_sha256=signature_sha256),checkpoint)
    np.savez_compressed(predpath,test_index=te,probabilities=tp,labels=labels[te])
    write_json(resultpath,dict(backbone=name,variant=variant,fold=fold,seed=seed,input_signature_sha256=signature_sha256,training_n=len(tr),validation_n=len(va),test_n=len(te),class_weights=weights.tolist(),roi_median=roi_median.tolist(),unsupported_roi_features=unsupported,best_epoch=best_epoch,epochs_ran=epoch,best_validation_loss=best,test_metrics=mm,history=history,seconds=time.time()-start,trainable_parameters=sum(p.numel() for p in model.parameters())))
    log(f'train {name} {variant} fold={fold} epochs={epoch} best={best_epoch} acc={mm["accuracy"]:.4f} f1={mm["macro_f1"]:.4f} seconds={time.time()-start:.1f}')
   assert not np.isnan(probs).any()
   np.save(outdir/f'{name}_{variant}_oof.npy',probs)
   summary=dict(pooled=metrics(labels,probs),fold_metrics=foldmetrics,
      fold_mean_accuracy=float(np.mean([m['accuracy'] for m in foldmetrics])),
      fold_std_accuracy=float(np.std([m['accuracy'] for m in foldmetrics],ddof=1)),
      fold_mean_macro_f1=float(np.mean([m['macro_f1'] for m in foldmetrics])),
      fold_std_macro_f1=float(np.std([m['macro_f1'] for m in foldmetrics],ddof=1)))
   allsummary[name][variant]=summary;write_json(outdir/f'{name}_{variant}_summary.json',summary)
   write_json(HERE/'summary_partial.json',allsummary)
   log(f'OOF {name} {variant} accuracy={summary["pooled"]["accuracy"]:.4f} macroF1={summary["pooled"]["macro_f1"]:.4f}')
   del features,x
 write_json(HERE/'summary.json',allsummary)
 export_oof(meta,allsummary)
 log(f'All model training complete seconds={time.time()-startall:.1f}')

def export_oof(meta,summary):
 outdir=HERE/'results';fieldnames=['image_index','image_path','age_label','day_group','backbone','variant','outer_fold','predicted_age_label','correct']+[f'prob_age_{i+1}' for i in range(5)]
 ff=rows(FOLDS);outer={int(r['image_index']):r['fold'] for r in ff if r['partition']=='test'}
 with open(HERE/'oof_predictions.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader()
  for name in summary:
   for variant in summary[name]:
    p=np.load(outdir/f'{name}_{variant}_oof.npy')
    for i,r in enumerate(meta):
     pred=int(p[i].argmax())+1
     w.writerow(dict(image_index=i,image_path=r['image_path'],age_label=r['age_label'],day_group=r['day_group'],backbone=name,variant=variant,outer_fold=outer[i],predicted_age_label=pred,correct=int(pred==int(r['age_label'])),**{f'prob_age_{j+1}':float(p[i,j]) for j in range(5)}))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['extract','train','all']);ap.add_argument('--models',nargs='+',default=BACKBONES);ap.add_argument('--variants',nargs='+',default=VARIANTS);ap.add_argument('--device',default=DEFAULT_DEVICE);ap.add_argument('--extract-batch',type=int,default=16);ap.add_argument('--threads',type=int,default=4);ap.add_argument('--encoder-threads',type=int,default=ENCODER_THREADS);ap.add_argument('--overwrite',action='store_true');args=ap.parse_args()
 torch.set_num_threads(args.threads);seed_all(CONFIG['seed'])
 if args.mode in ['extract','all']:extract(args)
 if args.mode in ['train','all']:train(args)

if __name__=='__main__':main()
