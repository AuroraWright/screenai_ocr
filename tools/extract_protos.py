# SPDX-License-Identifier: MPL-2.0
import re, sys, os
from google.protobuf import descriptor_pb2
data = open(sys.argv[1],'rb').read(); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
def varint(b,i):
    r=s=0
    while True:
        c=b[i]; i+=1; r|=(c&0x7f)<<s; s+=7
        if c<0x80: return r,i
VALID={1:2,2:2,3:2,4:2,5:2,6:2,7:2,8:2,9:2,10:0,11:0,12:2,13:2,14:0,15:2,16:0}
def scan(i):
    start=i; seen=0
    while i<len(data):
        try:
            tag,j=varint(data,i)
        except IndexError: break
        f,wt=tag>>3,tag&7
        if f not in VALID or VALID[f]!=wt:
            if not (wt==2 and f<=20): break
        if wt==2:
            l,j=varint(data,j)
            if l>len(data)-j: break
            j+=l
        else:
            _,j=varint(data,j)
        i=j; seen+=1
    return data[start:i]
seen=set(); n=0
for m in re.finditer(rb'\x0a([\x01-\x7f])([A-Za-z0-9_./\-]+\.proto)', data):
    if len(m.group(2))!=m.group(1)[0]: continue
    blob=scan(m.start())
    fd=descriptor_pb2.FileDescriptorProto()
    try: fd.ParseFromString(blob)
    except Exception: continue
    if fd.name in seen or not (fd.message_type or fd.enum_type): continue
    seen.add(fd.name); n+=1
    p=os.path.join(out, fd.name.replace('/','__')+'.pbtxt')
    open(p,'w').write(str(fd))
print('extracted', n)
for s in sorted(seen): print(s)
