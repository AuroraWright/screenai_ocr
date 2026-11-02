import sys, time, numpy as np
from PIL import Image, ImageDraw, ImageFont
from ai_edge_litert.interpreter import Interpreter
it=Interpreter(model_path=sys.argv[1], num_threads=8); it.allocate_tensors()
ins={d['name']:d for d in it.get_input_details()}; outs={d['name']:d for d in it.get_output_details()}
img=Image.new('L',(1280,1280),255); d=ImageDraw.Draw(img)
f=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 28)
for k in range(5): d.text((100,200+k*60), "The quick brown fox jumps over the lazy dog %d"%k, font=f, fill=0)
for n,dd in ins.items():
    x=np.full(dd['shape'],255,np.uint8)
    if n=='input_features_1': x[0,:,:,0]=np.array(img)
    it.set_tensor(dd['index'],x)
t=time.time(); it.invoke(); print('invoke %.2fs'%(time.time()-t))
sig=lambda z:1/(1+np.exp(-z))
for n in ['Identity_%d'%i for i in range(1,11)]+['Identity']:
    o=it.get_tensor(outs[n]['index'])[0]; p=sig(o[...,0])
    ys,xs=np.where(p>0.5)
    print(n, o.shape, 'max p %.3f'%p.max(), 'cells>0.5:',len(ys), ('rows %d-%d cols %d-%d'%(ys.min(),ys.max(),xs.min(),xs.max())) if len(ys) else '')
