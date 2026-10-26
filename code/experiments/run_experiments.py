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
 color_descriptors='36 per spatial cell: local and global mean/std of RGB, HSV, CIELAB',
 color_normalization='training-fold mean/std per descriptor across images and cells',
 encoder_normalization='ImageNet RGB means [.485,.456,.406], std [.229,.224,.225]',
 image_preprocessing='RGB conversion; bilinear resize directly to 224x224; no crop',
 augmentation='none',
 color_attention_formula='a=softmax_s(w2 tanh(W1 z_s+b1)+b2); v=ReLU(Wv sum_s a_s F_s+bv); c=ReLU(Wc sum_s a_s z_s+bc); g=sigmoid(wg[v;c]+bg); h=(1-g)*v+g*c; logits=Wo Dropout(h)+bo',
 baseline_formula='v=ReLU(Wv mean_s F_s+bv); logits=Wo Dropout(v)+bo',
 grayscale_formula='same baseline architecture and training on 3-channel grayscale encoder inputs',
 selection='all models evaluated on identical predefined outer folds; each fold checkpoint selected only by inner validation loss')

def log(s):
 print(time.strftime('%Y-%m-%d %H:%M:%S'),s,flush=True)

def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

def rows(path):
 with open(path,newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def write_json(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False))

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
 meta=rows(META);n=len(meta);cache=HERE/'cache';cache.mkdir(exist_ok=True)
 log(f'Extraction metadata n={n}; device={args.device}; torch={torch.__version__}')
 images=np.empty((n,224,224,3),np.uint8)
 descriptors=[];statrows=[]
 for i,r in enumerate(meta):
  im=Image.open(DATA/r['image_path']).convert('RGB').resize((224,224),Image.Resampling.BILINEAR)
  images[i]=np.asarray(im);z,stat=color_descriptor(im);descriptors.append(z)
  statrows.append(dict(image_index=i,image_path=r['image_path'],age_label=r['age_label'],day_group=r['day_group'],**stat))
 np.save(cache/'color_descriptors.npy',np.stack(descriptors))
 with open(HERE/'color_statistics.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(statrows[0]));w.writeheader();w.writerows(statrows)
 np.save(cache/'images224.npy',images)
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
     x=torch.from_numpy(images[a:a+args.extract_batch].copy()).to(device).permute(0,3,1,2).float()/255
     if gray:
      lum=x[:,0:1]*.299+x[:,1:2]*.587+x[:,2:3]*.114
      x=lum.expand(-1,3,-1,-1)
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
  a=self.score(z).softmax(1)
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
 meta=rows(META);ff=rows(FOLDS);n=len(meta);labels=np.array([int(r['label']) for r in meta]);cache=HERE/'cache'
 colors=np.load(cache/'color_descriptors.npy').astype(np.float32)
 seed_all(CONFIG['seed'])
 config={**CONFIG,'torch':torch.__version__,'torchvision':__import__('torchvision').__version__,
         'python':sys.version,'head_device':'cpu','encoder_device':args.device,'cpu_threads':args.threads,
         'encoder_cpu_threads':getattr(args,'encoder_threads',ENCODER_THREADS)}
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
    if resultpath.exists() and predpath.exists() and not args.overwrite:
     result=json.loads(resultpath.read_text());assert result['seed']==seed,('Stored checkpoint seed mismatch',resultpath,result['seed'],seed);npz=np.load(predpath);probs[npz['test_index']]=npz['probabilities'];foldmetrics.append(result['test_metrics']);log('Resuming completed '+resultpath.stem);continue
    seed_all(seed)
    indices={part:np.array([int(r['image_index']) for r in ff if int(r['fold'])==fold and r['partition']==part]) for part in ['train','val','test']}
    tr,va,te=indices['train'],indices['val'],indices['test']
    assert not set(tr)&set(va) and not set(tr)&set(te) and not set(va)&set(te)
    mu=colors[tr].mean((0,1));sd=colors[tr].std((0,1));sd=np.maximum(sd,1e-6)
    z=torch.from_numpy(np.clip((colors-mu)/sd,-10,10))
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
    checkpoint=outdir/f'{name}_{variant}_fold{fold}.pt'
    torch.save(dict(state_dict=state,color_mean=mu,color_std=sd,channels=channels,variant=variant,backbone=name,fold=fold,seed=seed),checkpoint)
    np.savez_compressed(predpath,test_index=te,probabilities=tp,labels=labels[te])
    write_json(resultpath,dict(backbone=name,variant=variant,fold=fold,seed=seed,training_n=len(tr),validation_n=len(va),test_n=len(te),class_weights=weights.tolist(),best_epoch=best_epoch,epochs_ran=epoch,best_validation_loss=best,test_metrics=mm,history=history,seconds=time.time()-start,trainable_parameters=sum(p.numel() for p in model.parameters())))
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
