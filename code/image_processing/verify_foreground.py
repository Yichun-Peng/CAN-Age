from pathlib import Path
import csv,json,hashlib,os
import numpy as np
from PIL import Image,JpegImagePlugin,PngImagePlugin
SCRIPT_DIR=Path(__file__).resolve().parent
BASE=Path(os.environ.get('EAGLE_PROCESSING_DIR',str(SCRIPT_DIR))).resolve()
SOURCE=Path(os.environ.get('EAGLE_SOURCE_PROCESSING_DIR',str(SCRIPT_DIR))).resolve()
DATA=Path(os.environ.get('EAGLE_DATA_DIR',str(Path(__file__).resolve().parents[2] / 'data')))
rows=list(csv.DictReader((BASE.parent/'data_audit/image_metadata.csv').open()));fg=list(csv.DictReader((SOURCE/'foreground_manifest.csv').open()))
issues=[];checks=[]
for i,r in enumerate(rows):
 a=np.asarray(Image.open(DATA/r['image_path'],formats=['JPEG','PNG']).convert('RGB'));b=np.asarray(Image.open(SOURCE/'unaligned_rgba'/f'{i:04d}.png',formats=['PNG']));mask=np.asarray(Image.open(SOURCE/'foreground_masks'/f'{i:04d}.png',formats=['PNG']))
 if a.shape!=b[:,:,:3].shape or not np.array_equal(a,b[:,:,:3]):issues.append({'image_index':i,'issue':'RGBA_RGB_pixels_changed'})
 if not np.array_equal(mask,b[:,:,3]):issues.append({'image_index':i,'issue':'alpha_mask_mismatch'})
 if mask.shape!=a.shape[:2]:issues.append({'image_index':i,'issue':'mask_shape_mismatch'})
 if not mask.any():issues.append({'image_index':i,'issue':'empty_mask'})
 checks.append({'image_index':i,'mask_sha256':hashlib.sha256((SOURCE/'foreground_masks'/f'{i:04d}.png').read_bytes()).hexdigest(),'rgba_sha256':hashlib.sha256((SOURCE/'unaligned_rgba'/f'{i:04d}.png').read_bytes()).hexdigest(),'foreground_method':'MobileSAM_manual_point_QC' if i==229 else 'YOLO11n_seg_bird_class14_imgsz640'})
with (BASE/'foreground_checksums.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=checks[0]);w.writeheader();w.writerows(checks)
report={'n_images':len(rows),'all_native_RGB_pixels_preserved':not any(x['issue']=='RGBA_RGB_pixels_changed' for x in issues),'all_alpha_equal_raw_mask':not any(x['issue']=='alpha_mask_mismatch' for x in issues),'issues':issues,'segmentation_model':'Ultralytics YOLO11n-seg pretrained COCO bird class14, imgsz640, conf0.10, largest area subject','confident_count':sum(x['foreground_ok']=='1' for x in fg),'uncertain_count':sum(x['foreground_ok']=='0' for x in fg),'special_case_229':'YOLO no valid foreground at0.10; lowerconfidence1280mask visiblytruncated; replacedwithMobileSAMfullbirdbox plus4positive2negativepoints inspected against original; excludesbackgroundcrow; keptforeground_ok0','originals_modified':False}
(BASE/'foreground_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
