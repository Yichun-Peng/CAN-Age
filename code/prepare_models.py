#!/usr/bin/env python3
"""Prepare pinned official models after installing the platform-specific lock."""
import os,json,hashlib,urllib.request,sys
from importlib import metadata
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=ROOT/'image_processing'
os.environ.setdefault('HF_HOME',str(BASE/'huggingface'))
os.environ.setdefault('TORCH_HOME',str(ROOT/'model_cache/torch'))
if os.environ.get('BWHD_ROOT'):
 allowed=Path(os.environ['BWHD_ROOT']).resolve()
 for path in [ROOT,BASE,Path(os.environ['HF_HOME']),Path(os.environ['TORCH_HOME'])]:
  if not path.resolve().is_relative_to(allowed):raise ValueError('Model cache outside BWHD_ROOT: '+str(path))
if sys.platform.startswith('linux'):
 try:headless=metadata.version('opencv-python-headless')
 except metadata.PackageNotFoundError:headless=None
 if headless!='5.0.0.93':raise RuntimeError('Install the platform lock, then explicitly reinstall opencv-python-headless==5.0.0.93 before model preparation; see server/REPRODUCE_SERVER.md')
try:import cv2
except ImportError as error:raise RuntimeError('cv2 import failed. On Linux, reinstall the same-version headless build after all locked dependencies; see server/REPRODUCE_SERVER.md') from error
models=BASE/'models';models.mkdir(parents=True,exist_ok=True)
sources=json.loads((ROOT/'model_sources.json').read_text())
urls={'yolo11n-seg.pt':'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt','mobile_sam.pt':'https://github.com/ultralytics/assets/releases/download/v8.3.0/mobile_sam.pt'}
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
for name,url in urls.items():
 path=models/name
 if not path.exists():
  tmp=path.with_suffix('.download');urllib.request.urlretrieve(url,tmp);tmp.replace(path)
 expected=sources['files'][name]['sha256']
 if digest(path)!=expected:raise ValueError('Official weight checksum differs: '+name)
 print('Verified',name,flush=True)
from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
identifier=sources['grounding_dino']['repository'];revision=sources['grounding_dino']['revision']
AutoProcessor.from_pretrained(identifier,revision=revision)
model=AutoModelForZeroShotObjectDetection.from_pretrained(identifier,revision=revision)
del model
from torchvision import models as tv
for constructor,weights in [(tv.mobilenet_v3_small,tv.MobileNet_V3_Small_Weights.IMAGENET1K_V1),(tv.vgg11,tv.VGG11_Weights.IMAGENET1K_V1),(tv.resnet18,tv.ResNet18_Weights.IMAGENET1K_V1),(tv.densenet121,tv.DenseNet121_Weights.IMAGENET1K_V1)]:
 model=constructor(weights=weights);del model
print('All official pretrained models are ready for local processing.')
