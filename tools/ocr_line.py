# SPDX-License-Identifier: MPL-2.0
import sys, numpy as np
sys.path.insert(0, sys.argv[1])
from pool import load_pool
from google.protobuf import message_factory
from PIL import Image, ImageDraw, ImageFont
from ai_edge_litert.interpreter import Interpreter
pool,_,_=load_pool(sys.argv[2]); L=sys.argv[3]; model=sys.argv[4]; lm=sys.argv[5]; text=sys.argv[6]
LM=message_factory.GetMessageClass(pool.FindMessageTypeByName('google_ocr.LabelClassIDMap'))()
LM.ParseFromString(open(f'{L}/{lm}','rb').read())
labels={l.class_id:l.name for l in LM.label}
print('num_classes', LM.num_classes, 'labels', len(labels), 'max id', max(labels))
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 22)
w=int(font.getlength(text))+8
img=Image.new('L',(w,32),255); ImageDraw.Draw(img).text((4,3),text,font=font,fill=0)
it=Interpreter(model_path=f'{L}/{model}'); runner=it.get_signature_runner('serving_default')
# chunk: 136 px payload + 16 px padding each side = 168 px input, frame width 4 px
arr=np.array(img); W=arr.shape[1]; frames=[]
for x0 in range(0, W, 136):
    chunk=np.full((32,168),255,np.uint8)
    lo=max(0,x0-16); hi=min(W,x0+136+16)
    chunk[:, lo-(x0-16):hi-(x0-16)]=arr[:,lo:hi]
    out=runner(image_tensor=chunk[None,:,:,None])
    lg=out['decode_logits'][0]  # [42, C] quantized
    frames.append(lg[4:4+34])   # drop 16px/4 = 4 padding frames each side
lg=np.concatenate(frames)[: (W+3)//4]
ids=lg.argmax(-1)
blank=LM.num_classes-1 if (LM.num_classes-1) not in labels else None
prev=-1; s=[]
for i in ids:
    if i!=prev and i in labels and i!=blank: s.append(labels[i])
    prev=i
print('blank guess', blank, '| uniq ids', sorted(set(ids.tolist()))[:20])
print('INPUT :', text); print('OUTPUT:', ''.join(s))
