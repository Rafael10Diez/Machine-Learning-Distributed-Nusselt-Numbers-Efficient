
# ------------------------------------------------------------------------
#                  Generic Libraries
# ------------------------------------------------------------------------

from  os.path      import  dirname
from  collections  import  namedtuple
from  math         import  prod
from  datetime     import  datetime

# ------------------------------------------------------------------------
#                  Basic Functions
# ------------------------------------------------------------------------

lmap         =  lambda   f,x:  list(map   (f,x))
lfilter      =  lambda   f,x:  list(filter(f,x))
import_json  =  lambda fname: eval(' '.join(reader(fname)))
deepdirname  =  lambda   x,n:  x if n<1 else deepdirname(dirname(x),n-1)
tstamp       =  lambda      :  datetime.now().strftime("%Y%m%d_%H%M%S")
format_dt    =  lambda   x  :  "%02d:%02d:%02d [hh:mm:ss]" % (x//3600, (x%3600)//60, x%60)

def reader(fname, clean = False):
    with open(fname,'r') as f:
        A = [x.rstrip('\n') for x in f]
    if clean: A = lfilter(None,A)
    return A

def pop1(A):
    x, = A
    return x

def as_namedtuple(D, name):
    keys,values = [],[]
    for k,v in D.items():
        keys  .append(k)
        values.append(v)
    return namedtuple(name,keys)(*values)

def as_list(x, dtype):
    if hasattr(x,'__iter__') and ((prod(x.shape)>1) if hasattr(x,'shape') else True):
        return [ as_list(c,dtype) for c in x ]
    else:
        return dtype(x)

# ------------------------------------------------------------------------
#                  FPrint
# ------------------------------------------------------------------------

class FPrint:
    def __init__(self, fname, track_all = False, check_path = None):
        self.fname        =  fname
        self.fwrite       =  open(fname, 'w')
        self.__track_all  =  track_all
        self.__A          =  []
        
    def __call__(self,*line):
        
        if line and line[-1] == '__only_to_file__':
            line = ' '.join(map(str,line[:-1]))
        else:
            line = ' '.join(map(str,line))
            print(line)
        
        self.fwrite.write( line + '\n' )
        self.fwrite.flush()
        
        if self.__track_all: self.__A.append(line)
    
    def get_all(self):
        assert self.__track_all
        return self.__A
    
    close   = __del__ = lambda self: self.fwrite.close()
    
