import sys
sys.path.insert(0, sys.argv[1])
from pool import load_pool
from google.protobuf import message_factory, text_format, descriptor_pb2
pool,_,_=load_pool(sys.argv[2])
if sys.argv[3]=='msgs':
    fd=pool.FindFileByName(sys.argv[4])
    def walk(m,ind=''):
        print(ind+'message',m.full_name)
        for f in m.fields: print(ind+'  ',f.number,f.name,f.message_type.full_name if f.message_type else (f.enum_type.full_name if f.enum_type else f.type))
        for n in m.nested_types: walk(n,ind+'  ')
    for m in fd.message_types_by_name.values(): walk(m)
else:
    m=message_factory.GetMessageClass(pool.FindMessageTypeByName(sys.argv[3]))()
    m.ParseFromString(open(sys.argv[4],'rb').read())
    print(text_format.MessageToString(m, descriptor_pool=pool, expand_any=True) if 'expand_any' in text_format.MessageToString.__code__.co_varnames else text_format.MessageToString(m, descriptor_pool=pool))
