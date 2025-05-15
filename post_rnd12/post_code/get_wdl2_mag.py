# ------------------------------------------------------------------------
#                  Generic Libraries
# ------------------------------------------------------------------------

import  zipfile
from    os.path      import  dirname, abspath, basename, isfile
from    os.path      import  join                                as  pjoin
from    sys          import  path                                as  sys_path
from    collections  import  OrderedDict
import  torch

# ------------------------------------------------------------------------
#                  Basic Functions
# ------------------------------------------------------------------------

lfilter               =  lambda f,x: list(filter(f,x))

def pop1(A):
    x, = A
    return x

def reader(fname):
    with open(fname,'r') as f:
        return [x.rstrip('\n') for x in f]

def deepdirname(x,n):
    for _ in range(n):
        x = dirname(x)
    return x

def linux_path(x):
    cL  = r' / '.replace(' ','')
    cW  = r' \ '.replace(' ','')
    for c in [cW,cL*2]:
        while c in x:
            x = x.replace(c,cL)
    return x

def fix_path(x, folder):
    p0    =  linux_path(folder)
    _,p1  =  linux_path(x).split(basename(folder))
    assert (p0[-1] != '/') and (p1[0] == '/')
    return linux_path(p0+p1)

CLOUD_FOLDER = deepdirname(abspath(__file__),7)
assert basename(CLOUD_FOLDER) == 'Dropbox'

# ------------------------------------------------------------------------
#                  Custom Libraries
# ------------------------------------------------------------------------

def get_net(tag, args_net, seed):
    sys_path.append(pjoin(deepdirname(abspath(__file__),3), 'source' , 'models'))
    if tag == 'alt2_orig':  from  cnn_alt2_orig  import  Net_Circular_Alt2_Orig  as  Model_net
    sys_path.pop()
    return Model_net(args_net, seed)


def import_json(fname, cloud_folder = CLOUD_FOLDER):
    fname       =  fix_path(fname, cloud_folder)
    saved_zip   =  fname + '.zip'
    local_name  =  basename(fname)
    assert isfile(saved_zip),saved_zip
    A = zipfile.Path(saved_zip, at = local_name).read_text().split('\n')
    return eval(' '.join(A))

# ------------------------------------------------------------------------
#                  Scan Log File
# ------------------------------------------------------------------------

class Get_wdl2_mag:
    def __init__(self, saver, log_fname):
        self.saver      =  saver
        self.log_fname  =  log_fname

        self.info_log   =  self.scan_logfile(self.log_fname)

        net             =  get_net(self.info_log['tag_net'], self.info_log['args_net'], lambda _: None)

        net.load_state_dict(import_json(saver['Net_Dict']))
        
        p_change        =  [p for p in net.parameters() if p.requires_grad]
        self.wdl2_mag   =  sum(float((p**2).sum().item()) for p in p_change if p.requires_grad)
    
    @staticmethod
    def scan_logfile(log_fname):
        
        A  =  reader(log_fname)
        def get_rhs(tag, sep=':'):
            line = pop1(lfilter(lambda x: x[:len(tag)] == tag, A))
            return sep.join(line.split(sep)[1:])
        
        return {'args_net': eval(get_rhs('    args_net        :'))               ,
                'tag_net' :      get_rhs('    net_type        :').replace(' ','')}
