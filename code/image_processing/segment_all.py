import os,csv,json,time
from pathlib import Path
BASE=Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR']=str(BASE/'ultralytics_config')
import numpy as np
from PIL import Image,ImageDraw
from ultralytics import YOLO
DATA=Path(os.environ.get("EAGLE_DATA_DIR", str(Path(__file__).resolve().parents[2] / 'data')))
rows=list(csv.DictReader((BASE.parent/'data_audit/image_metadata.csv').open()))
for d in ['foreground_masks','unaligned_rgba','aligned_rgba','aligned_foreground_masks']:(BASE/d).mkdir(exist_ok=True)
model=YOLO(str(BASE/'models/yolo11n-seg.pt'))
reports=[];start=time.time()
for i,r in enumerate(rows):
 im=Image.open(DATA/r['image_path']).convert('RGB')
 res=model.predict(im,classes=[14],conf=.10,imgsz=640,device='mps',verbose=False)[0]
 candidates=[]
 if res.masks is not None:
  for j,xy in enumerate(res.masks.xy):
   mask=Image.new('L',im.size);ImageDraw.Draw(mask).polygon([tuple(x) for x in xy],fill=255)
   area=int(np.asarray(mask).astype(bool).sum());candidates.append((area,float(res.boxes.conf[j]),mask))
 if candidates:area,conf,mask=max(candidates,key=lambda x:x[0]);okay=conf>=.25 and area>im.width*im.height*.08
 else:area,conf,mask,okay=0,0.,Image.new('L',im.size),False
 name=f'{i:04d}.png';mask.save(BASE/'foreground_masks'/name)
 rgba=im.copy();rgba.putalpha(mask);rgba.save(BASE/'unaligned_rgba'/name)
 # Initial direction preserved. ROI pipeline updates aligned copies after reliable orientation.
 rgba.save(BASE/'aligned_rgba'/name);mask.save(BASE/'aligned_foreground_masks'/name)
 reports.append(dict(image_index=i,image_path=r['image_path'],foreground_ok=int(okay),foreground_confidence=conf,foreground_pixels=area,foreground_fraction=area/(im.width*im.height),foreground_mask=str(BASE/'foreground_masks'/name)))
 if i%25==0 or i==len(rows)-1:
  with (BASE/'foreground_manifest.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=reports[0]);w.writeheader();w.writerows(reports)
  print(f'FG {i+1}/{len(rows)} okay={sum(r["foreground_ok"] for r in reports)} elapsed={time.time()-start:.1f}s',flush=True)
print('Foreground complete',flush=True)
