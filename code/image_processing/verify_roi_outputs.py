from pathlib import Path
import csv,json,collections,hashlib,sys,os,platform
import numpy as np
from PIL import Image,JpegImagePlugin,PngImagePlugin
SCRIPT_DIR=Path(__file__).resolve().parent
BASE=Path(os.environ.get('EAGLE_PROCESSING_DIR',str(SCRIPT_DIR))).resolve()
SOURCE=Path(os.environ.get('EAGLE_SOURCE_PROCESSING_DIR',str(SCRIPT_DIR))).resolve()
DATA=Path(os.environ.get('EAGLE_DATA_DIR',str(Path(__file__).resolve().parents[2] / 'data')))
META=list(csv.DictReader((BASE.parent/'data_audit/image_metadata.csv').open()));features=list(csv.DictReader((BASE/'features.csv').open()))
assert len(features)==len(META)==1143
assert [int(r['image_index']) for r in features]==list(range(1143))
ROIS=np.load(BASE/'raw_roi_features.npz');assert ROIS['features'].shape==(1143,18)
ORIGINAL_HASHES={r['image_path']:r['sha256'] for r in csv.DictReader((BASE.parent/'data_audit/inventory.csv').open())}
issues=[];samplechecks=[];channels=['red','green','blue','hue','saturation','value','lab_l','lab_a','lab_b']
sys.path.insert(0,str(BASE.parent/'experiments'))
from run_experiments import rgb_to_color9
for i,(m,f) in enumerate(zip(META,features)):
 if hashlib.sha256((DATA/m['image_path']).read_bytes()).hexdigest()!=ORIGINAL_HASHES[m['image_path']]:issues.append({'image_index':i,'issue':'original_file_bytes_changed'})
 im=np.asarray(Image.open(DATA/m['image_path'],formats=['JPEG','PNG']).convert('RGB'));flip=bool(int(f['orientation_flip']))
 mask={name:np.asarray(Image.open((SOURCE if name=='foreground' else BASE)/d/f'{i:04d}.png',formats=['PNG']))>0 for name,d in [('foreground','foreground_masks'),('beak','beak_masks'),('feather','feather_masks')]}
 for name,a in mask.items():
  if a.shape!=im.shape[:2]:issues.append({'image_index':i,'issue':'shape_'+name})
 if (mask['beak']&mask['feather']).any():issues.append({'image_index':i,'issue':'beak_feather_overlap'})
 if (mask['feather']&~mask['foreground']).any():issues.append({'image_index':i,'issue':'feather_not_subset_fg'})
 desiredfg=mask['foreground']|(mask['beak'] if int(f['beak_ok']) else False)
 am=np.asarray(Image.open(BASE/'aligned_foreground_masks'/f'{i:04d}.png',formats=['PNG']))>0
 expected=np.fliplr(desiredfg) if flip else desiredfg
 if not np.array_equal(am,expected):issues.append({'image_index':i,'issue':'aligned_fg_mismatch'})
 rgba=np.asarray(Image.open(BASE/'aligned_rgba'/f'{i:04d}.png',formats=['PNG']))
 if not np.array_equal(rgba[:,:,:3],np.fliplr(im) if flip else im):issues.append({'image_index':i,'issue':'aligned_RGB_changed'})
 if not np.array_equal(rgba[:,:,3]>0,am):issues.append({'image_index':i,'issue':'alpha_mismatch'})
 for name,d in [('beak','aligned_beak_masks'),('feather','aligned_feather_masks')]:
  a=np.asarray(Image.open(BASE/d/f'{i:04d}.png',formats=['PNG']))>0
  if not np.array_equal(a,np.fliplr(mask[name]) if flip else mask[name]):issues.append({'image_index':i,'issue':'aligned_'+name+'_mismatch'})
 validbeak=bool(int(f['beak_ok']));validfeather=bool(int(f['feather_ok']))
 if validbeak!=bool(np.isfinite(ROIS['features'][i,:9]).all()):issues.append({'image_index':i,'issue':'beak_valid_NaN_mismatch'})
 if validfeather!=bool(np.isfinite(ROIS['features'][i,9:]).all()):issues.append({'image_index':i,'issue':'feather_valid_NaN_mismatch'})
 if i%75==0 or i in [229,627,787,925,1142]:
  for name,op,sl,valid in [('beak','mean',slice(None,9),validbeak),('feather','std',slice(9,None),validfeather)]:
   if valid:
    vals=rgb_to_color9(im[mask[name]].astype(np.float32)/255);raw=vals.mean(0) if op=='mean' else vals.std(0);err=float(np.abs(raw-ROIS['features'][i,sl]).max())
    if err>1e-6:issues.append({'image_index':i,'issue':name+'_raw_feature_error','max_abs':err})
    samplechecks.append({'image_index':i,'region':name,'raw_recalculation_max_abs':err})
 if i%200==0:print('Verified',i,flush=True)
report={'n_images':len(META),'n_features_rows':len(features),'n_beak_ok':sum(int(f['beak_ok']) for f in features),'n_feather_ok':sum(int(f['feather_ok']) for f in features),'n_foreground_ok':sum(int(f['foreground_ok']) for f in features),'n_flipped':sum(int(f['orientation_flip']) for f in features),'original_head_directions':dict(collections.Counter(f['original_head_direction'] for f in features)),'beak_valid_by_label':{str(c):sum(int(f['beak_ok']) for f in features if f['age_label']==str(c)) for c in range(1,6)},'feather_valid_by_label':{str(c):sum(int(f['feather_ok']) for f in features if f['age_label']==str(c)) for c in range(1,6)},'issues':issues,'n_raw_feature_recalculation_checks':len(samplechecks),'raw_feature_recalculation_checks':samplechecks,'rgb_preserved_and_coordinate_alignment_verified':not issues}
report.update({'validation_host':platform.node(),'data_directory':str(DATA),'processing_directory':str(BASE),'source_processing_directory':str(SOURCE),'image_formats':['JPEG','PNG'],'validation_device':os.environ.get('EAGLE_DEVICE','cpu'),'cpu_threads':int(os.environ.get('EAGLE_NUM_THREADS','4'))})
(BASE/'roi_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
assert not issues,issues[:10]
