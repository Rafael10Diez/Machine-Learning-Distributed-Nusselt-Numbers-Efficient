# ------------------------------------------------------------------------
#                  Generic Libraries
# ------------------------------------------------------------------------

# import  numpy     as      np
from    os.path   import  dirname, abspath, basename, isfile, isdir
from    os.path   import  join                                       as  pjoin
from    sys       import  path                                       as  sys_path
from    os        import  listdir                                    as  os_listdir

# ------------------------------------------------------------------------
#                  Custom Libraries
# ------------------------------------------------------------------------

sys_path.append(dirname(abspath(__file__)))
from get_wdl2_mag import Get_wdl2_mag
sys_path.pop()

# ------------------------------------------------------------------------
#                  Basic Functions
# ------------------------------------------------------------------------

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

lmap                  =  lambda f,x: list(map   (f,x))
lfilter               =  lambda f,x: list(filter(f,x))

listdir_full          =  lambda x  : sorted(pjoin(x,y) for y in os_listdir(x))
listdir_full_files    =  lambda x  : lfilter(isfile,listdir_full(x))
listdir_full_folders  =  lambda x  : lfilter(isdir ,listdir_full(x))

# ------------------------------------------------------------------------
#                  Scan Log File
# ------------------------------------------------------------------------

class ScoreSavers_Log_File:
    def __init__(self, folder, fprint = None):
        if fprint is None:
            fprint = lambda _: None
        
        self.fprint     =  fprint 
        self.folder     =  folder
        self.log_fname  =  pop1(lfilter(lambda x: '.log' in basename(x),
                                        listdir_full_files(folder)     ))
        
        self.A_log       =  reader(self.log_fname)
        
        self.all_epochs  =  []
        self.all_savers  =  []
        
        for arr, mk_entry, tag in [[self.all_epochs, self.mk_epoch_info, 'Epoch : '                                                       ],
                                   [self.all_savers, self.mk_savers    , '------------------------- Begin Saver -------------------------']]:
            
            for i in lfilter(lambda i: self.A_log[i][:len(tag)] == tag, range(len(self.A_log))):
                arr.append(mk_entry(self.A_log,i))
        
        self.score_savers()
        
    @staticmethod
    def mk_epoch_info(A_log, i):
        D           =  {'info_dsets': ScoreSavers_Log_File.get_dsets_info(A_log, i)}
        D['epoch']  =  int  (A_log[i].split('Epoch :' )[1].split('(')[0])
        D['loss']   =  float(A_log[i].split('(loss = ')[1].split(')')[0])
        return D
    
    @staticmethod
    def mk_savers(A_log, i):
        D  =  {'info_dsets': ScoreSavers_Log_File.get_dsets_info(A_log, i)}
        for j,tag in [[i+4,'---->  Net_Dict       :'],
                      [i+5,'---->  Optim_Dict     :'],
                      [i+6,'---->  Field Vars     :']]:      
            assert A_log[j][:len(tag)] == tag
            D[tag.replace('Field Vars','Field_Vars').replace('---->','').replace(':','').replace(' ','')] = (':'.join(A_log[j].split(':')[1:])).lstrip(' ').rstrip(' ')
        D['epoch'] = pop1(set(    D[key].split('_epoch_')[1].split('_')[0] for key in ['Net_Dict', 'Optim_Dict', 'Field_Vars']    ))
        if D['epoch'] != 'MiTrAbDiEp':
            D['epoch'] = int(D['epoch'])
        return D
    
    @staticmethod
    def get_dsets_info(A_log, i):
        D  =  {}
        for j in range(3):
            j        =  i+1+j
            dset     =  (A_log[j].split('(dset = ')[1].split(')')[0]).replace(' ','')
            D[dset]  =  {}
            for metric in [ 'diff/total'   ,
                           '|diff|/total'  ,
                           '|diff|/|total|']:
                D[dset][metric]  =  float(A_log[j].split(f'({metric} = ')[1].split('%)')[0])/100.
        return D
    
    def score_savers(self, all_span = [500], weight = {'|diff|/|total|': 0, 'diff/total': 3.}):
        min_saver = {'score': float('inf')}
        for saver in self.all_savers:
            epoch  =  saver['epoch']
            if type(epoch) == int:
                score = []
                for span in all_span:
                    for d_epoch in lfilter(lambda D: (epoch-span) <= D['epoch'] <= epoch, self.all_epochs):
                        s = 0.
                        for metric in ['|diff|/|total|', 'diff/total']:
                            s += d_epoch['info_dsets']['train'][metric] * weight[metric]
                        score.append(s)
                saver['score']  =  max(score)#/len(score)
                if saver['score'] < min_saver['score']:
                    min_saver = saver
        
        min_saver['wdl2_mag'] = Get_wdl2_mag(min_saver, self.log_fname).wdl2_mag

        self.best_saver = min_saver
        
        line = '-'*40
        self.fprint(f"\n\n{line} Best Saver {line}")
        self.fprint(f"    Folder       :  {basename(self.folder)}")
        self.fprint(f"    Log Filename :  {basename(self.log_fname)}")
        self.fprint(f"    Epoch        :  {min_saver['epoch']}")
        self.fprint(f"    wdl2_mag     :  {min_saver['wdl2_mag']:.3f}")
        self.print_info_dsets(min_saver['info_dsets'], self.fprint, pad=8)
    
    @staticmethod
    def print_info_dsets(info_dsets, fprint, pad=0):
        all_dsets   =  ['train'     ,  'valid'      ,  'test'         ]
        all_metric  =  ['diff/total', '|diff|/total', '|diff|/|total|']
        
        assert set(info_dsets.keys()) == set(info_dsets)
        
        for dset in all_dsets:
            A = [f"(dset = {dset:5s})"]
            metrics = info_dsets[dset]
            assert set(metrics.keys()) == set(all_metric)
            for key in all_metric:
                val = metrics[key]
                A.append(f"({key} = {(val*100):.3f} %)")
            fprint(' '*pad + ' '.join(A))

# ------------------------------------------------------------------------
#                  Direct Runner
# ------------------------------------------------------------------------

if __name__ == '__main__':
    line = '-'*60
    for qmode in ['Cf_classic','Nu_classic']:
        print(f'\n\n\n\n{line} {qmode} {line}')
        all_folders   =  listdir_full_folders(pjoin(deepdirname(abspath(__file__),3),'data',f'output_{qmode}'))
        all_logfiles  =  lmap(lambda x: ScoreSavers_Log_File(x,fprint=print), all_folders)
    
    