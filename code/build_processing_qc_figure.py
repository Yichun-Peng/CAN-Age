from pathlib import Path
import os
import numpy as np,pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parent.parent;P=Path(os.environ.get('EAGLE_PROCESSING_DIR',B/'code/image_processing'));O=B/'outputs/新增实验_20261026';D=Path(os.environ.get('EAGLE_DATA_DIR',str(Path(__file__).resolve().parents[1] / 'data')))
f=pd.read_csv(P/'features.csv').sort_values('image_index');assert len(f)==1143 and (P/'ROI_VALIDATED.flag').exists()
plt.rcParams.update({'font.family':os.environ.get('BWHD_FONT_FAMILY','Arial'),'font.size':7.2,'axes.titlesize':7.4,'axes.labelsize':7.2,'xtick.labelsize':7,'ytick.labelsize':7,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
fig=plt.figure(figsize=(7.087,8.9));outer=fig.add_gridspec(3,4,height_ratios=[1.18,1.18,1],hspace=.27,wspace=.18)
examples=[0,400,707,1142,229,627,787,925];example_rows=[]
for j,i in enumerate(examples):
 r=f[f.image_index==i].iloc[0];a=np.asarray(Image.open(D/r.image_path).convert('RGB'));v=a.copy()
 bm=np.asarray(Image.open(P/'beak_masks'/f'{i:04d}.png'))>0;fm=np.asarray(Image.open(P/'feather_masks'/f'{i:04d}.png'))>0
 v[fm]=(.72*v[fm]+.28*np.array([42,167,184])).astype(np.uint8);v[bm]=(.3*v[bm]+.7*np.array([211,69,41])).astype(np.uint8)
 sub=outer[j//4,j%4].subgridspec(2,1,height_ratios=[.25,1],hspace=.03)
 title=fig.add_subplot(sub[0]);title.axis('off');title.text(0,1,chr(97+j),fontweight='bold',fontsize=10,va='top');age='5+' if r.age_label==5 else str(r.age_label)
 title.text(.15,1,f'Image {i} · class {age}\nBeak {"valid" if r.beak_ok else "uncertain"}\nTorso {"valid" if r.feather_ok else "uncertain"}',va='top',fontsize=7,linespacing=1.1)
 ax=fig.add_subplot(sub[1]);ax.imshow(v);ax.axis('off');example_rows.append(r.to_dict())
bottom_grid=outer[2,:].subgridspec(1,2,wspace=.36)
ax=fig.add_subplot(bottom_grid[0,0]);groups=[('beak_ok','Beak','#b95a42'),('feather_ok','Torso','#3b8da0'),('foreground_ok','Foreground','#6d8b63')]
for j,(key,name,color) in enumerate(groups):
 vals=[f[f.age_label==a][key].mean()*100 for a in range(1,6)];ax.bar(np.arange(5)+(j-1)*.24,vals,width=.22,label=name,color=color)
ax.set(xticks=np.arange(5),xticklabels=['1','2','3','4','5+'],ylim=(0,105),ylabel='Measurements passing QC (%)',xlabel='Annotated winter-age class');ax.legend(frameon=False,fontsize=7,ncol=3,loc='upper center',bbox_to_anchor=(.5,1.17));ax.text(-.15,1.06,'i',transform=ax.transAxes,fontweight='bold',fontsize=10)
ax=fig.add_subplot(bottom_grid[0,1]);directions=['left','right','uncertain'];bottom=np.zeros(5)
for key,color in zip(directions,['#517c9c','#bda264','#a9afb3']):
 vals=np.array([(f[f.age_label==a].original_head_direction==key).mean()*100 for a in range(1,6)])
 ax.bar(np.arange(5),vals,bottom=bottom,color=color,label=key,width=.68);bottom+=vals
ax.set(xticks=np.arange(5),xticklabels=['1','2','3','4','5+'],ylim=(0,100),ylabel='Original image direction (%)',xlabel='Annotated winter-age class');ax.legend(frameon=False,fontsize=7,ncol=3,loc='upper center',bbox_to_anchor=(.5,1.17));ax.text(-.15,1.06,'j',transform=ax.transAxes,fontweight='bold',fontsize=10)
fig.subplots_adjust(left=.08,right=.98,bottom=.06,top=.97)
for ext in ['pdf','svg','png']:fig.savefig(O/'figures'/f'Figure_E0_processing_quality.{ext}',dpi=600 if ext=='png' else 150,bbox_inches='tight')
plt.close(fig)
pd.DataFrame(example_rows).to_csv(O/'tables/E0_processing_examples.csv',index=False)
coverage=f.groupby('age_label')[['beak_ok','feather_ok','foreground_ok']].agg(['sum','count','mean']);coverage.to_csv(O/'tables/E0_qc_coverage.csv')
(O/'Figure_E0_中文图注.txt').write_text('图E0｜图像处理与原生部位测量质量。a–d为预先指定的常规样例，e–h为预先指定的漏检修复、低头及遮挡样例；红色显示喙及蜡膜候选，青色显示躯干羽毛掩膜。valid/uncertain结合统一检测、几何规则及全体照片的年龄盲化语义复核给出。i为五类照片的喙、羽毛及前景有效比例（每类n=400、227、161、138、217）。j为原始图像头部朝向分布；仅可靠朝左图像进行水平镜像，uncertain保留原方向。颜色数值从原分辨率RGB像素计算，叠色仅用于展示。\n')
print('E0 processing QC figure and exact tables created')
