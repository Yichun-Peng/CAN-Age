#!/usr/bin/env python3
"""Age-blind anatomical ROI extraction on original decoded pixels; resumable."""
import os,csv,json,time,sys,hashlib,platform,datetime,importlib.metadata
from pathlib import Path
from roi_job_control import (parse_args,assigned_indices,worker_directory,atomic_json,
 canonical_hash,record_files_hash,read_metadata,load_records,partition_lock,
 validate_worker_summaries,validate_resume_provenance,EXPECTED_IMAGES)
ARGS=parse_args()
SCRIPT_DIR=Path(__file__).resolve().parent
BASE=Path(os.environ.get('EAGLE_PROCESSING_DIR',str(SCRIPT_DIR))).resolve()
SOURCE=Path(os.environ.get('EAGLE_SOURCE_PROCESSING_DIR',str(SCRIPT_DIR))).resolve()
os.environ.setdefault('HF_HOME',str(SOURCE/'huggingface'))
os.environ.setdefault('YOLO_CONFIG_DIR',str(SOURCE/'ultralytics_config'))
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
DEVICE=os.environ.get('EAGLE_DEVICE','mps')
NUM_THREADS=int(os.environ.get('EAGLE_NUM_THREADS','4'))
DATA=Path(os.environ.get('EAGLE_DATA_DIR',str(Path(__file__).resolve().parents[2] / 'data')))
if os.environ.get('BWHD_ROOT'):
 allowed_root=Path(os.environ['BWHD_ROOT']).resolve()
 for managed_path in [BASE,SOURCE,DATA,Path(os.environ['HF_HOME']),Path(os.environ['YOLO_CONFIG_DIR'])]:
  if not managed_path.resolve().is_relative_to(allowed_root):raise ValueError('Server path outside BWHD_ROOT: '+str(managed_path))
BASE.mkdir(parents=True,exist_ok=True)
import numpy as np
CHANNELS=['red','green','blue','hue','saturation','value','lab_l','lab_a','lab_b']
META_PATH=BASE.parent/'data_audit/image_metadata.csv'
META=read_metadata(META_PATH)
CONFIG={'detector':'IDEA-Research/grounding-dino-tiny','prompt':'head. beak. feet.','fallback_head_prompt':'bird head.','detector_input':'pretrained processor original800shortedge/max1333','threshold':.18,'text_threshold':.18,'head_min_score':.18,'beak_min_score':.25,'head_area_fraction':[.001,.30],'beak_area_fraction':[.00003,.03],'beak_mask_model':'ultralytics MobileSAM official pretrained mobile_sam.pt','beak_crop':'centered at detected beak box, halfwidth/halfheight1.8timesboxdimensions','beak_mask_definition':'SAM beak+cere candidate restricted to box enlarged20percent; not intersected with coarse YOLOmask because that truncates dark beak tips','feather_mask_definition':'native YOLO bird foreground excluding accepted head box, all accepted feet boxes belowhead, beak mask, and bottom10percent of foreground verticalextent; torso feather ROI','foreground_primary_definition':'YOLO11nseg bird foreground union accepted SAM beak mask','orientation':'face right only when headscore>=.22,beakscore>=.25,headtobeak horizontal offset>=.12headwidth and |dx|>=1.25|dy|; otherwise preserve original','beak_qc':'score>=.25; candidate smallandnearhead; SAMpixelarea>=30; maskarea/boxarea0.12..2; boxoverlap>=.3; mask not within1pixel imageborder. Failure yieldsNaNs.','feather_qc':'foreground confidence>=.25,validhead,ROI>=500pixels and>=20percent foreground','raw_measurement':'all channel statistics from original decoded native-resolution RGB, no resize,enhancement,adjustment or reencoding. RGBHSVLabexactrun_experiments.rgb_to_color9','mottling':'native gray populationSD; highpassMADofgray-Gaussian(sigma3pixels) over3pixel-erodedROI; Laplacian variance; grayhist entropy256bins','target_use':'age labels copied asmetadataonly; never used detection,ROI,orientation,QC or missingvaluecompletion','seed':20261026}
CONFIG.update({'image_formats':['JPEG','PNG'],'data_directory':str(DATA),'processing_directory':str(BASE),'source_processing_directory':str(SOURCE),'model_device':DEVICE,'cpu_threads':NUM_THREADS,'sample_detection_cache_policy':'Never import old sample predictions; detections and checkpoint records are read only from this processing directory.','legacy_texture_fields':'Original fields retained; mask-normalized texture correction is written separately.'})
CONFIG.update({'metadata_sha256':hashlib.sha256(META_PATH.read_bytes()).hexdigest(),
 'foreground_manifest_sha256':hashlib.sha256((SOURCE/'foreground_manifest.csv').read_bytes()).hexdigest(),
 'job_control_sha256':hashlib.sha256((SCRIPT_DIR/'roi_job_control.py').read_bytes()).hexdigest()})
CONFIG_HASH=canonical_hash(CONFIG)
SCRIPT_HASH=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def clipbox(b,w,h):
 return np.array([max(0,float(b[0])),max(0,float(b[1])),min(w,float(b[2])),min(h,float(b[3]))])
def rectangle_mask(box,w,h):
 m=np.zeros((h,w),bool);x0,y0,x1,y1=np.floor(box).astype(int);x0=max(0,x0);y0=max(0,y0);x1=min(w,max(x0+1,x1));y1=min(h,max(y0+1,y1));m[y0:y1,x0:x1]=True;return m
def detections(im,processor,model,prompt='head. beak. feet.'):
 inp=processor(images=im,text=prompt,return_tensors='pt').to(DEVICE)
 with torch.inference_mode():out=model(**inp)
 result=processor.post_process_grounded_object_detection(out,inp.input_ids,threshold=.18,text_threshold=.18,target_sizes=[im.size[::-1]])[0]
 labels=result.get('text_labels',result.get('labels'))
 return {'labels':list(labels),'scores':result['scores'].cpu().numpy().tolist(),'boxes':result['boxes'].cpu().numpy().tolist()}
def select_regions(d,fg,w,h):
 allc=[]
 for label,score,b in zip(d['labels'],d['scores'],d['boxes']):
  box=clipbox(b,w,h);bw=box[2]-box[0];bh=box[3]-box[1]
  if bw<=1 or bh<=1:continue
  m=rectangle_mask(box,w,h);area=bw*bh;cover=float(fg[m].mean()) if m.any() else 0
  allc.append(dict(label=label,score=float(score),box=box,area=area,coverage=cover))
 heads=[a for a in allc if a['label'] in ['head','bird head'] and .001*w*h<=a['area']<=.3*w*h and a['score']>=.18 and a['coverage']>.20]
 head=max(heads,key=lambda a:a['score']) if heads else None
 beaks=[]
 for a in allc:
  if a['label']!='beak' or not(.00003*w*h<=a['area']<=.03*w*h) or a['score']<.25:continue
  b=a['box'];center=(b[:2]+b[2:])/2
  if head is not None:
   hb=head['box'];expand=(hb[2:]-hb[:2])*.2
   if not np.all(center>=hb[:2]-expand) or not np.all(center<=hb[2:]+expand):continue
   if a['area']>=head['area']*.85:continue
  # A beak cannot be a distant yellow foot even if tokenprediction errs: near upper/side forebody.
  if head is None and center[1]>.85*h and a['score']<.5:continue
  beaks.append(a)
 beak=max(beaks,key=lambda a:a['score']) if beaks else None
 feet=[]
 for a in allc:
  if a['label']!='feet' or a['score']<.25 or a['area']>.25*w*h:continue
  center=(a['box'][:2]+a['box'][2:])/2
  if center[1]>(head['box'][1]+head['box'][3])/2+.10*h if head is not None else center[1]>.65*h:feet.append(a)
 return head,beak,feet

def extract_beak(im,box,sam):
 w,h=im.size;b=box;cx=(b[0]+b[2])/2;cy=(b[1]+b[3])/2;bw=b[2]-b[0];bh=b[3]-b[1]
 cb=[max(0,int(cx-bw*1.8)),max(0,int(cy-bh*1.8)),min(w,int(cx+bw*1.8)),min(h,int(cy+bh*1.8))]
 crop=im.crop(cb);pb=(b-np.array(cb[:2]*2)).tolist()
 result=sam.predict(crop,bboxes=[pb],device=DEVICE,verbose=False,project=str(BASE/'inference_runs'),save=False,save_txt=False)[0]
 mask=np.zeros((h,w),bool);samconf=0.
 if result.masks is not None and len(result.masks.xy):
  k=int(result.boxes.conf.argmax());sm=Image.new('L',crop.size);ImageDraw.Draw(sm).polygon([tuple(t) for t in result.masks.xy[k]],fill=255)
  mask[cb[1]:cb[3],cb[0]:cb[2]]=np.asarray(sm)>0;samconf=float(result.boxes.conf[k])
 # Restrict SAM awayfromneighboring feather/iriswhileretaining fullbeakoutline.
 pad=np.array([bw*.2,bh*.2,bw*.2,bh*.2]);limit=b+pad*np.array([-1,-1,1,1]);mask &=rectangle_mask(limit,w,h)
 return mask,samconf

def stats(rgb,gray,mask,prefix,okay):
 # Native RGBconvertedonlyatROI locations; no resize.
 values=rgb_to_color9(rgb[mask]) if mask.any() else np.empty((0,9),np.float32)
 out={prefix+'_pixels':int(mask.sum())}
 for j,c in enumerate(CHANNELS):
  out[prefix+'_'+c+'_mean']=float(values[:,j].mean()) if len(values) and okay else float('nan')
  out[prefix+'_'+c+'_std']=float(values[:,j].std()) if len(values) and okay else float('nan')
 vals=gray[mask]
 out[prefix+'_gray_mean']=float(vals.mean()) if len(vals) and okay else float('nan');out[prefix+'_gray_std']=float(vals.std()) if len(vals) and okay else float('nan')
 if len(vals) and okay:
  core=binary_erosion(mask,iterations=3);core=core if core.sum()>=100 else mask
  hp=gray-gaussian_filter(gray,sigma=3);v=hp[core];out[prefix+'_highpass_mad']=float(np.median(np.abs(v-np.median(v))))
  out[prefix+'_laplacian_variance']=float(laplace(gray)[core].var())
  hist=np.bincount(np.clip(vals.astype(int),0,255),minlength=256);prob=hist[hist>0]/hist.sum();out[prefix+'_gray_entropy']=float(-(prob*np.log2(prob)).sum())
 else:
  for c in ['highpass_mad','laplacian_variance','gray_entropy']:out[prefix+'_'+c]=float('nan')
 return out

def save_all(records):
 records=sorted(records,key=lambda x:x['image_index'])
 for r in records:
  r['original_head_direction']=('left' if r['head_to_beak_dx']<0 else 'right') if r['orientation_confidence']>0 else 'uncertain'
  r['visible_anatomical_side']='unknown'
  r['BCI_rgb_mean']=float(np.mean([r['beak_red_mean'],r['beak_green_mean'],r['beak_blue_mean']])*255) if r['beak_ok'] else float('nan')
  r['OCD_foreground_gray_std']=r['foreground_gray_std']
  r['PFD_feather_gray_std']=r['feather_gray_std']
  # Backward-compatible legacy alias; main OCD is the foreground descriptor above.
  r['OCD_feather_gray_std']=r['feather_gray_std']
  r['OB_foreground_gray_mean']=r['foreground_gray_mean']
  r['foreground_method']='MobileSAM_manual_point_QC' if r['image_index']==229 else 'YOLO11n_seg_bird_class14_imgsz640'
 with (BASE/'features.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
 features=np.full((len(META),18),np.nan,np.float32)
 for r in records:
  i=r['image_index'];features[i,:9]=[r['beak_'+c+'_mean'] for c in CHANNELS];features[i,9:]=[r['feather_'+c+'_std'] for c in CHANNELS]
 np.savez(BASE/'raw_roi_features.npz',features=features,channels=CHANNELS,image_index=np.arange(len(META)),beak_ok=np.array([next((int(r['beak_ok']) for r in records if r['image_index']==i),0) for i in range(len(META))]),feather_ok=np.array([next((int(r['feather_ok']) for r in records if r['image_index']==i),0) for i in range(len(META))]))

def write_main_config():
 config_path=BASE/'processing_config.json'
 if config_path.exists() and not (BASE/'processing_config_initial.json').exists():
  (BASE/'processing_config_initial.json').write_bytes(config_path.read_bytes())
 atomic_json(config_path,CONFIG)

def aggregate():
 start=time.time()
 with partition_lock(BASE,range(len(META))):
  records=load_records(BASE,META,require_complete=True,require_artifacts=True)
  workers=validate_worker_summaries(BASE,META,ARGS.shards,CONFIG_HASH,SCRIPT_HASH)
  # No Torch, Transformers, SAM, PIL, or experiment module is imported in this mode.
  save_all(records)
  write_main_config()
  summary={'n_processed':len(records),'foreground_ok':sum(r['foreground_ok'] for r in records),
   'beak_ok':sum(r['beak_ok'] for r in records),'feather_ok':sum(r['feather_ok'] for r in records),
   'orientation_flipped':sum(r['orientation_flip'] for r in records),
   'orientation_reliable':sum(r['orientation_confidence']>0 for r in records),
   'elapsed_seconds':time.time()-start,'timing_scope':'Aggregation only; inference worker runtimes reported separately',
   'mode':'aggregate_only','shards':ARGS.shards,'device':DEVICE,'cpu_threads_per_worker':NUM_THREADS,
   'worker_elapsed_seconds':[w['elapsed_seconds'] for w in workers],
   'worker_resume_ids':[w['resume_id'] for w in workers],
   'config_sha256':CONFIG_HASH,'source_script_sha256':SCRIPT_HASH,
   'records_sha256':record_files_hash(BASE,range(len(META))),
   'worker_summaries':[str(worker_directory(BASE,ARGS.shards,k)/'processing_summary.json') for k in range(ARGS.shards)]}
  atomic_json(BASE/'processing_summary.json',summary)
  print('AGGREGATE_COMPLETE '+json.dumps(summary),flush=True)

def initialize_models():
 global torch,Image,ImageDraw,JpegImagePlugin,PngImagePlugin,gaussian_filter,binary_erosion,laplace,rgb_to_color9
 import torch
 from PIL import Image,ImageDraw,JpegImagePlugin,PngImagePlugin
 from scipy.ndimage import gaussian_filter,binary_erosion,laplace
 from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
 from ultralytics import SAM
 sys.path.insert(0,str(BASE.parent/'experiments'))
 from run_experiments import rgb_to_color9
 torch.set_num_threads(NUM_THREADS)
 processor=AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-tiny',local_files_only=True)
 model=AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-tiny',local_files_only=True).to(DEVICE).eval()
 sam=SAM(str(SOURCE/'models/mobile_sam.pt'))
 return processor,model,sam

def process_images():
 indices=assigned_indices(len(META),ARGS.shards,ARGS.shard_id) if ARGS.records_only else list(range(len(META)))
 job_dir=worker_directory(BASE,ARGS.shards,ARGS.shard_id) if ARGS.records_only else BASE
 with partition_lock(BASE,indices):
  job_dir.mkdir(parents=True,exist_ok=True)
  for directory in ['detections','beak_masks','feather_masks','aligned_beak_masks','aligned_feather_masks','aligned_rgba','aligned_foreground_masks','feature_records','roi_overlays']:
   (BASE/directory).mkdir(exist_ok=True)
  records=load_records(BASE,META,indices=indices,require_artifacts=True)
  done={r['image_index'] for r in records};start=time.time();initial_count=len(records)
  pending=[i for i in indices if i>=ARGS.start and (ARGS.limit is None or i<ARGS.limit) and i not in done]
  if ARGS.records_only:
   if records:validate_resume_provenance(job_dir,CONFIG_HASH,SCRIPT_HASH,indices)
   atomic_json(job_dir/'processing_config.json',CONFIG)
  else:
   write_main_config()
  resume_id=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+str(os.getpid())
  resume_provenance={'resume_id':resume_id,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
   'host':platform.node(),'platform':platform.platform(),'device':DEVICE,
   'torch_version':importlib.metadata.version('torch'),'cpu_threads':NUM_THREADS,
   'data_directory':str(DATA),'processing_directory':str(BASE),'source_processing_directory':str(SOURCE),
   'mode':'records_only' if ARGS.records_only else 'single_process','shards':ARGS.shards,'shard_id':ARGS.shard_id,
   'assigned_indices':indices,'checkpoint_records':initial_count,'checkpoint_indices':sorted(done),
   'checkpoint_records_sha256':record_files_hash(BASE,done),'config_sha256':CONFIG_HASH,
   'source_script_sha256':SCRIPT_HASH,
   'checkpoint_policy':'Existing records inside this processing version are preserved; no records or detections are imported from another version.',
   'device_policy':'All anatomical ROI predictions in this processing version use '+DEVICE}
  with (job_dir/'resume_provenance.jsonl').open('a') as provenance_file:provenance_file.write(json.dumps(resume_provenance)+'\n')
  print('RESUME '+json.dumps({k:v for k,v in resume_provenance.items() if k not in ['checkpoint_indices','assigned_indices']}),flush=True)
  if pending:
   processor,model,sam=initialize_models()
  FG=list(csv.DictReader((SOURCE/'foreground_manifest.csv').open()))
  for i in pending:
   m=META[i]
   t0=time.time();im=Image.open(DATA/m['image_path'],formats=['JPEG','PNG']).convert('RGB');w,h=im.size;fg=np.asarray(Image.open(SOURCE/'foreground_masks'/f'{i:04d}.png',formats=['PNG']))>0
   tload=time.time();detpath=BASE/'detections'/f'{i:04d}.json'
   if detpath.exists():det=json.loads(detpath.read_text())
   else:det=detections(im,processor,model);detpath.write_text(json.dumps(det))
   head,beak,feet=select_regions(det,fg,w,h)
   if head is None:
    secondary=detections(im,processor,model,'bird head.');det['fallback_head']=secondary
    combined={'labels':det['labels']+secondary['labels'],'scores':det['scores']+secondary['scores'],'boxes':det['boxes']+secondary['boxes']};head,beak,feet=select_regions(combined,fg,w,h);detpath.write_text(json.dumps(det))
   tdet=time.time();beakmask=np.zeros((h,w),bool);samconf=0.;beak_ok=False;beakratio=np.nan;overlap=np.nan;clipped=False
   if beak is not None:
    beakmask,samconf=extract_beak(im,beak['box'],sam);beakratio=beakmask.sum()/beak['area'];boxmask=rectangle_mask(beak['box'],w,h);overlap=(beakmask&boxmask).sum()/max(1,beakmask.sum());clipped=bool(beakmask[:2].any() or beakmask[-2:].any() or beakmask[:,:2].any() or beakmask[:,-2:].any())
    beak_ok=beakmask.sum()>=30 and .12<=beakratio<=2 and overlap>=.3 and not clipped
   tsam=time.time();feather=fg.copy()
   if head is not None:feather &=~rectangle_mask(head['box'],w,h)
   for foot in feet:feather &=~rectangle_mask(foot['box'],w,h)
   feather &=~beakmask
   ys,xs=np.where(fg)
   if len(ys):feather[int(ys.min()+.90*(ys.max()-ys.min())):]=False
   foreground_ok=bool(int(FG[i]['foreground_ok']));head_ok=head is not None;feather_ok=foreground_ok and head_ok and feather.sum()>=max(500,int(fg.sum()*.2))
   flip=False;oconf=0.;dx=dy=np.nan
   if head is not None and beak is not None and beak_ok:
    hc=(head['box'][:2]+head['box'][2:])/2;bc=(beak['box'][:2]+beak['box'][2:])/2;dx,dy=(bc-hc).tolist();oconf=min(head['score'],beak['score'])
    reliable=head['score']>=.22 and abs(dx)>=(head['box'][2]-head['box'][0])*.12 and abs(dx)>=1.25*abs(dy)
    if reliable:flip=dx<0
    else:oconf=0.
   primaryfg=fg|(beakmask if beak_ok else False)
   for directory,mask in [('beak_masks',beakmask),('feather_masks',feather)]:Image.fromarray((mask*255).astype(np.uint8)).save(BASE/directory/f'{i:04d}.png',compress_level=1)
   for directory,mask in [('aligned_foreground_masks',primaryfg),('aligned_beak_masks',beakmask),('aligned_feather_masks',feather)]:
    a=np.fliplr(mask) if flip else mask;Image.fromarray((a*255).astype(np.uint8)).save(BASE/directory/f'{i:04d}.png',compress_level=1)
   rgba=im.copy();rgba.putalpha(Image.fromarray((primaryfg*255).astype(np.uint8)));rgba=rgba.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if flip else rgba;rgba.save(BASE/'aligned_rgba'/f'{i:04d}.png',compress_level=1)
   tsave=time.time();rgb=np.asarray(im,dtype=np.float32)/255;gray=(rgb@np.array([.299,.587,.114],np.float32))*255
   qc=[]
   if not foreground_ok:qc.append('foreground_uncertain')
   if not head_ok:qc.append('head_not_found')
   if not beak_ok:qc.append('beak_uncertain')
   if clipped:qc.append('beak_clipped')
   if not feather_ok:qc.append('feather_uncertain')
   if oconf==0:qc.append('orientation_preserved_uncertain')
   record={'image_index':i,'image_path':m['image_path'],'age_label':int(m['age_label']),'day_group':m['day_group'],'width':w,'height':h,'foreground_ok':int(foreground_ok),'foreground_confidence':float(FG[i]['foreground_confidence']),'foreground_pixels':int(fg.sum()),'foreground_fraction':float(fg.mean()),'head_ok':int(head_ok),'head_confidence':head['score'] if head else 0.,'head_box_json':json.dumps(head['box'].tolist()) if head else '', 'beak_ok':int(beak_ok),'beak_confidence':beak['score'] if beak else 0.,'beak_box_json':json.dumps(beak['box'].tolist()) if beak else '', 'sam_confidence':samconf,'beak_mask_box_area_ratio':float(beakratio),'beak_mask_box_overlap':float(overlap),'beak_clipped':int(clipped),'feather_ok':int(feather_ok),'orientation_flip':int(flip),'orientation_confidence':oconf,'head_to_beak_dx':dx,'head_to_beak_dy':dy,'roi_qc':'|'.join(qc) or 'pass'}
   for name,mask,okay in [('foreground',fg,foreground_ok),('beak',beakmask,beak_ok),('feather',feather,feather_ok)]:record.update(stats(rgb,gray,mask,name,okay))
   bg=~primaryfg
   record['background_gray_mean']=float(gray[bg].mean()) if bg.any() else np.nan;record['background_gray_std']=float(gray[bg].std()) if bg.any() else np.nan
   tstats=time.time();checkpoint=BASE/'feature_records'/f'{i:04d}.json'
   atomic_json(checkpoint,record);records.append(record)
   with (job_dir/'image_runtime_provenance.jsonl').open('a') as provenance_file:
    provenance_file.write(json.dumps({'image_index':i,'resume_id':resume_id,'device':DEVICE,'host':platform.node(),
     'shards':ARGS.shards,'shard_id':ARGS.shard_id,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})+'\n')
   if len(records)-initial_count<=5:print(f'TIMING image{i} load={tload-t0:.2f} DINO={tdet-tload:.2f} SAM={tsam-tdet:.2f} masksPNG={tsave-tsam:.2f} stats={tstats-tsave:.2f}',flush=True)
   # Deterministic every25th+knownchallengingcases, sourcepixelsoverlaidwithoutvisualenhancement.
   if i%25==0 or i in [47,49,186,229,424,470,627,787,829,924,925]:
    over=np.asarray(im).copy();over[beakmask]=(over[beakmask]*.35+np.array([255,40,40])*.65).astype(np.uint8);over[feather]=(over[feather]*.75+np.array([40,180,210])*.25).astype(np.uint8);pim=Image.fromarray(over);pim.thumbnail((640,640));pim.save(BASE/'roi_overlays'/f'{i:04d}.jpg',quality=92)
   if len(records)%25==0 or i==pending[-1]:
    if not ARGS.records_only:save_all(records)
    print(f'ROI {len(records)}/{len(indices)} shard={ARGS.shard_id} latest={i} beakOK={sum(r["beak_ok"] for r in records)} featherOK={sum(r["feather_ok"] for r in records)} flip={sum(r["orientation_flip"] for r in records)} elapsed={time.time()-start:.1f}s',flush=True)
   if not ARGS.records_only and len(records)<=3:save_all(records)
  if not ARGS.records_only and records:save_all(records)
  complete=len(records)==len(indices) and sorted(r['image_index'] for r in records)==indices
  summary={'n_processed':len(records),'n_expected':len(indices),'complete':complete,
   'foreground_ok':sum(r['foreground_ok'] for r in records),'beak_ok':sum(r['beak_ok'] for r in records),
   'feather_ok':sum(r['feather_ok'] for r in records),'orientation_flipped':sum(r['orientation_flip'] for r in records),
   'orientation_reliable':sum(r['orientation_confidence']>0 for r in records),'elapsed_seconds':time.time()-start,
   'timing_scope':'This resumed worker process only' if ARGS.records_only else 'This resumed process only',
   'resumed_checkpoint_records':initial_count,'new_records':len(records)-initial_count,'device':DEVICE,
   'cpu_threads':NUM_THREADS,'resume_id':resume_id,'shards':ARGS.shards,'shard_id':ARGS.shard_id,
   'mode':'records_only' if ARGS.records_only else 'single_process','completed_indices':sorted(r['image_index'] for r in records),
   'config_sha256':CONFIG_HASH,'source_script_sha256':SCRIPT_HASH,
   'records_sha256':record_files_hash(BASE,[r['image_index'] for r in records])}
  atomic_json(job_dir/'processing_summary.json',summary)
  print(('WORKER_COMPLETE ' if complete else 'WORKER_PARTIAL ')+json.dumps(summary),flush=True)

if __name__=='__main__':
 if ARGS.aggregate_only:aggregate()
 else:process_images()
