from pathlib import Path
import csv,json,os
import numpy as np
from PIL import Image,ImageOps,ImageDraw
import cv2
B=Path(__file__).resolve().parent.parent
D=Path(os.environ.get('EAGLE_DATA_DIR',str(Path(__file__).resolve().parents[1] / 'data')))
rows=list(csv.DictReader((B/'code/data_audit/phash_candidates.csv').open()))
meta={r['image_path']:r for r in csv.DictReader((B/'code/data_audit/image_metadata.csv').open())}
cache={}
def img(p):
 if p not in cache:cache[p]=np.asarray(Image.open(D/p).convert('L').resize((128,128),Image.Resampling.BILINEAR),np.float32)
 return cache[p]
def ssim(a,b):
 mu1=cv2.GaussianBlur(a,(11,11),1.5);mu2=cv2.GaussianBlur(b,(11,11),1.5)
 v1=cv2.GaussianBlur(a*a,(11,11),1.5)-mu1*mu1;v2=cv2.GaussianBlur(b*b,(11,11),1.5)-mu2*mu2
 v12=cv2.GaussianBlur(a*b,(11,11),1.5)-mu1*mu2
 score=(2*mu1*mu2+6.5025)*(2*v12+58.5225)/((mu1*mu1+mu2*mu2+6.5025)*(v1+v2+58.5225))
 return float(score[5:-5,5:-5].mean())
for r in rows:
 r['cross_outer_fold']=int(meta[r['image_a']]['outer_fold']!=meta[r['image_b']]['outer_fold'])
 r['ssim_128']=ssim(img(r['image_a']),img(r['image_b']))
rows.sort(key=lambda r:r['ssim_128'],reverse=True)
p=B/'code/data_audit/near_duplicate_qc.csv'
with p.open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
cross=[r for r in rows if r['cross_outer_fold']]
summary={'candidate_pairs':len(rows),'cross_fold_candidate_pairs':len(cross),'ssim_ge_0.98_crossfold':sum(r['ssim_128']>=.98 for r in cross),'ssim_ge_0.95_crossfold':sum(r['ssim_128']>=.95 for r in cross),'max_ssim_crossfold':max(r['ssim_128'] for r in cross),'status':'Candidates for visual comparison; SSIM and pHash do not establish shared individual identity.'}
(B/'code/data_audit/near_duplicate_qc.json').write_text(json.dumps(summary,indent=2))
# Six highest scoring cross-fold pairs; no image content changes beyond display scaling.
sel=cross[:12];sheet=Image.new('RGB',(1200,6*230),'white');draw=ImageDraw.Draw(sheet)
for j,r in enumerate(sel):
 x=(j%2)*600;y=(j//2)*230
 for k,key in enumerate(['image_a','image_b']):
  im=Image.open(D/r[key]).convert('RGB');im.thumbnail((290,195));sheet.paste(im,(x+k*300,y))
 draw.text((x+4,y+199),f"SSIM {r['ssim_128']:.3f}   ages {meta[r['image_a']]['age_label']}/{meta[r['image_b']]['age_label']}   folds {meta[r['image_a']]['outer_fold']}/{meta[r['image_b']]['outer_fold']}",fill='black')
sheet.save(B/'code/data_audit/near_duplicate_top_crossfold.jpg')
print(summary)
