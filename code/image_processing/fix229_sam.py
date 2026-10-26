import os,csv
from pathlib import Path
BASE=Path(__file__).resolve().parent;os.environ['YOLO_CONFIG_DIR']=str(BASE/'ultralytics_config')
import numpy as np
from PIL import Image,ImageDraw
from ultralytics import SAM
r=list(csv.DictReader((BASE.parent/'data_audit/image_metadata.csv').open()))[229]
p=Path(os.environ.get("EAGLE_DATA_DIR", str(Path(__file__).resolve().parents[2] / 'data')))/r['image_path'];im=Image.open(p).convert('RGB')
sam=SAM(str(BASE/'models/mobile_sam.pt'));res=sam.predict(im,bboxes=[[5,5,672,649]],points=[[[220,300],[400,420],[100,70],[260,595],[600,345],[500,80]]],labels=[[1,1,1,1,0,0]],device='mps',verbose=False)[0]
mask=Image.new('L',im.size)
for xy in res.masks.xy:ImageDraw.Draw(mask).polygon([tuple(t) for t in xy],fill=255)
a=int((np.asarray(mask)>0).sum());confidence=float(res.boxes.conf.max());print('SAMmask',a,confidence,flush=True)
mask.save(BASE/'foreground_masks/0229.png');rgba=im.copy();rgba.putalpha(mask);rgba.save(BASE/'unaligned_rgba/0229.png');rgba.save(BASE/'aligned_rgba/0229.png');mask.save(BASE/'aligned_foreground_masks/0229.png')
bg=Image.new('RGB',im.size,(200,200,200));bg.paste(im,mask=mask);bg.save(BASE/'fix229_sam_review.jpg')
reports=list(csv.DictReader((BASE/'foreground_manifest.csv').open()));reports[229]['foreground_pixels']=a;reports[229]['foreground_fraction']=a/(im.width*im.height);reports[229]['foreground_confidence']=confidence;reports[229]['foreground_ok']=0
with (BASE/'foreground_manifest.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=reports[0]);w.writeheader();w.writerows(reports)
