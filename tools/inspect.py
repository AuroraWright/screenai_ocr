import sys, glob, os
from ai_edge_litert.interpreter import Interpreter
from ai_edge_litert import schema_py_generated as S
def ops(path):
    buf=open(path,'rb').read()
    m=S.Model.GetRootAsModel(buf,0)
    names=set()
    bmap={v:k for k,v in vars(S.BuiltinOperator).items() if not k.startswith('_')}
    for i in range(m.OperatorCodesLength()):
        oc=m.OperatorCodes(i)
        code=max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())
        if code==S.BuiltinOperator.CUSTOM: names.add('CUSTOM:'+oc.CustomCode().decode())
        else: names.add(bmap.get(code,code))
    md=[m.Metadata(i).Name().decode() for i in range(m.MetadataLength())]
    return sorted(map(str,names)), md
for p in sys.argv[1:]:
    print('=====', os.path.relpath(p, '/Users/aurora/.config/screen_ai/resources'))
    o,md=ops(p)
    print('ops:', ', '.join(o)); print('metadata:', md)
    try:
        it=Interpreter(model_path=p)
        sigs=it.get_signature_list(); print('signatures:', sigs)
        for d in it.get_input_details(): print('  IN ', d['name'], d['shape'].tolist(), d['shape_signature'].tolist(), d['dtype'].__name__, d['quantization'])
        for d in it.get_output_details(): print('  OUT', d['name'], d['shape'].tolist(), d['shape_signature'].tolist(), d['dtype'].__name__)
        try:
            it.allocate_tensors(); print('  allocate: OK')
        except Exception as e: print('  allocate FAIL:', str(e)[:200])
    except Exception as e: print('  LOAD FAIL:', str(e)[:300])
