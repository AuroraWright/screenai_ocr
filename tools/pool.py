import glob, sys
from google.protobuf import descriptor_pb2, descriptor_pool, text_format, message_factory
from google.protobuf.internal import decoder
def load_pool(d):
    fds={}
    for p in glob.glob(d+'/*.pbtxt'):
        fd=descriptor_pb2.FileDescriptorProto(); text_format.Parse(open(p).read(), fd, allow_unknown_field=True); fds[fd.name]=fd
    pool=descriptor_pool.DescriptorPool(); done=set(); failed={}; global stubs; stubs=set()
    def add(n, stack=()):
        if n in done or n in failed: return n in done
        fd=fds.get(n)
        if fd is None:
            fd=descriptor_pb2.FileDescriptorProto(name=n, package='stub'); fds[n]=fd; stubs.add(n)
        for dep in fd.dependency:
            if dep not in stack: add(dep, stack+(n,))
        try: pool.Add(fd); done.add(n); return True
        except Exception as e: failed[n]=str(e)[:120]; return False
    for n in list(fds): add(n)
    return pool, done, failed
if __name__=='__main__':
    pool,done,failed=load_pool(sys.argv[1])
    print('loaded',len(done),'failed',len(failed),'stubs',sorted(stubs))
    for k,v in list(failed.items())[:30]: print(' ',k,v)
