
# ------------------------------------------------------------------------
#                  Generic Libraries
# ------------------------------------------------------------------------

import  zipfile
import  numpy     as      np
from    os.path   import  dirname, abspath, basename, isfile, isdir
from    os.path   import  join                                       as  pjoin
from    sys       import  path                                       as  sys_path
from    sys       import  argv
from    os        import  mkdir                                      as  os_mkdir
from    os        import  listdir                                    as  os_listdir
from    pprint    import  pformat
from    time      import  time
from    datetime  import  datetime
from    copy      import  deepcopy
import  random
random.seed(0)

# ------------------------------------------------------------------------
#                  Global Variables
# ------------------------------------------------------------------------

APPLY_AVG    =  True
SMOOTH_MODE  =  'COST_FUNCTION'
RELAX_GRAD   =  1

# ------------------------------------------------------------------------
#                  Custom Libraries
# ------------------------------------------------------------------------

sys_path.append(dirname(abspath(__file__)))
from relax_grad_engine  import  Relax_Grad_Engine
from score_savers       import  ScoreSavers_Log_File
if   SMOOTH_MODE == 'COST_FUNCTION':  from filtering.smooth_predictions  import  Smooth_Predictions  as Joint_Predictions
else                               :  raise Exception(f'Unrecognized (SMOOTH_MODE = {SMOOTH_MODE})')
sys_path.pop()

# ------------------------------------------------------------------------
#                  Basic Functions
# ------------------------------------------------------------------------

def pick2(A):
    A = deepcopy(A)
    assert set(map(type,A)) == {int,}
    if len(A) > 2:
        random.shuffle(A)
        A = sorted(A[:2])
    else:
        assert len(A) == 2
    return sorted(A)

lfilter               =  lambda f,x: list(filter(f,x))
lmap                  =  lambda f,x: list(map(f,x))

deepdirname           =  lambda x,n:  x if n<1 else deepdirname(dirname(x),n-1)

listdir_full          =  lambda x: sorted([pjoin(x,y) for y in os_listdir(x)])
listdir_full_files    =  lambda x: lfilter(isfile,listdir_full(x))
listdir_full_folders  =  lambda x: lfilter(isdir ,listdir_full(x))
fmt_p                 =  lambda x: f'{(x*100.):.4f} %'

apply_pformat         =  lambda x: pformat(x, sort_dicts=False, indent = 2)

almost                =  lambda x,y, tol = 1e-10: np.fabs(x-y).max() < tol
format_dt             =  lambda x               :  "%02d:%02d:%02d [hh:mm:ss]" % (x//3600, (x%3600)//60, x%60)
tstamp                =  lambda                 :  datetime.now().strftime("%Y%m%d_%H%M%S")

def mkdir_p(x):
    assert not isfile(x)
    if not isdir(x):
        mkdir_p(dirname(x))
        os_mkdir(x)
    assert isdir(x)

def reader(fname):
    with open(fname,'r') as f:
        return [x.rstrip('\n') for x in f]

def pop1(A):
    x, = A
    return x

def linux_path(x):
    cL  = r' / '.replace(' ','')
    cW  = r' \ '.replace(' ','')
    for c in [cW,cL*2]:
        while c in x:
            x = x.replace(c,cL)
    return x

def first_int(x):
    result = []
    i      = 0
    while i<len(x) and x[i].isdigit():
        result.append(x[i])
        i  +=  1
    return int(''.join(result))

def lset_brute(A):
    unique = []
    for x in A:
        if not (x in unique):
            unique.append(x)
    # This check is only useful for the current program.
    #     It's ok if "unique" has multiple elements. 
    assert len(unique) == 1
    return     unique

def get_rhs(tag, A, use_set = True, remove_spaces = True):
    A = lfilter(lambda x: tag in x, A)
    for i,x in enumerate(A):
        assert x.count(':') == 1
        A[i] = x.split(':')[1]
        if remove_spaces:
            A[i] = A[i].replace(' ','')
    if use_set:
        A = lset_brute(A)
    return pop1(A)

locate_file  =  lambda tag, folder: pop1(lfilter(lambda x: tag in basename(x),
                                                 listdir_full_files(folder)  ))

# import_json  =  lambda x: eval(' '.join(reader(x)))

def import_json(fname):
    saved_zip   =  fname + '.zip'
    local_name  =  basename(fname)
    assert isfile(saved_zip)
    A = zipfile.Path(saved_zip, at = local_name).read_text().split('\n')
    return eval(' '.join(A))

def first_dot_split(x, sep=':'):
    x = x.split(sep)
    return x[0],sep.join(x[1:])

def fix_path(x, folder):
    p0    =  linux_path(folder)
    _,p1  =  linux_path(x).split(basename(folder))
    assert (p0[-1] != '/') and (p1[0] == '/')
    return linux_path(p0+p1)

def zip_str_write(fname, x):
    
    if type(x) != str:
        assert not hasattr(x,'shape')
        x = str(x)
    
    assert not isfile(fname)
    
    target_zip   = fname + '.zip'
    local_fname  = basename(fname)
    
    assert not isfile(target_zip)
    
    with zipfile.ZipFile(target_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(local_fname, x)

# ------------------------------------------------------------------------
#                  Utilities
# ------------------------------------------------------------------------

def fix_keys(x):
    if type(x) in [list,dict]:
        for i in {list: range(len(x)), dict: x}[type(x)]:
            x[i] = fix_keys(x[i])
    return x

def reduce_longest_common(A):
    assert A
    A = sorted(A,key=len)
    # assert len(set(map(len,A))) == 1
    result = [(i if all(x[i]==c for x in A) else False) for i,c in enumerate(A[0])]
    # [ 0, 1, 2, False, 4, 5, 6, False, ... ]
    result  = [lmap(int,lfilter(None,x.split(' '))) for x in ' '.join(map(str,result)).split('False')]
    maxL    = max(map(len,result))
    result, = [x for x in result if len(x)==maxL]
    return ''.join(A[0][i] for i in result)

# ------------------------------------------------------------------------
#                  Import Previous Functions
# ------------------------------------------------------------------------

post_output   =  pjoin(dirname(dirname(abspath(__file__))), 'post_output')

# ------------------------------------------------------------------------
#                  Scan Folder
# ------------------------------------------------------------------------
class Scan_Folder:
    def __init__(self, folder, n_savers = 'best_score'):
        line = '---------------------------------------------------------------'
        print(line)
        print(f'        Scanning Folder : {folder}' )
        print(f'        Number of Savers: {n_savers}')
        t0                     =  time()
        self.folder            =  folder
        self.n_savers          =  n_savers
        self.m_log             =  ScanLogFile(folder, n_savers)
        
        if type(n_savers) == int:
            self.last_savers       =  self.m_log.savers[-n_savers:]
        else:
            assert len(self.m_log.savers) == 1
            self.last_savers   =  self.m_log.savers
            
        self.all_preds         =  self.fetch_preds(self.last_savers)
        self.ref_vars          =  import_json(self.m_log.ref_vars)
        print(f'        Elapsed Time: {format_dt(time()-t0)}')
        print(line)
    @staticmethod
    def fetch_preds(last_savers):
        all_pred   = lmap(lambda d: import_json(d['Field_Vars']),last_savers) # [Field_Vars, ...] = [{'train': [...], ...}, ...]
        
        keys       = list(all_pred[0].keys())
        assert (set(keys) == {'train','valid','test'}) and all(list(p.keys())==keys for p in all_pred)
        
        # result has shape:
        #     {dset: [ ML_preds for e in epochs_used] for dset in ('train','valid','test')}
        #     result[dset][epoch][indpred]
        return {key: [list(np.array(p[key])) for p in all_pred] for key in keys} # apply list to separate surfs
    
    def get_data(self, dset, ind):
        assert (type(dset) == str) and (type(ind)==int)
        pred_data  =  self.all_preds[dset]
        ref_data   =  self.ref_vars [dset]
        # ref_data[dtype][indpred] has data with predictions
        mask       =  lfilter(lambda i: ref_data['full_info'][i]['ind']==ind, range(len(ref_data['full_info'])))
        assert len(mask) == 2
        
        def fmask(A, is_info=False, check_prev_flip=False, check_post_flip=False):
            assert (type(A)==list) and (type(is_info)==bool)
            A = [A[i] for i in mask]
            assert len(A) == 2
            if is_info:
                return pop1(lset_brute(fix_keys(A)))
            else:
                a,b = lmap(np.array,A)
                if check_prev_flip: assert almost(a,b)
                assert (a.shape == b.shape) and (len(a.shape)==len(b.shape)==4) and (a.shape[:2] == b.shape[:2]==(1,1))
                b   = np.flip(np.copy(b),-1)
                if check_post_flip: assert almost(a,b)
                def fget(arr):
                    assert (len(arr.shape)==4) and (arr.shape[:2] == (1,1))
                    return arr[0,0,:,:]
                if APPLY_AVG:
                    mid   =  0.5*(a+b)
                    a, b  =  mid+0., mid+0.
                return fget(a), fget(b)
            
        k_full_info = 'full_info'
        k_X_map     = 'X_map'
        assert all((k in list(ref_data.keys())) for k in [k_full_info,k_X_map])
        # pred_data has the form [ ML_preds for e in epochs_used]
        # therefore, we must map fmask
        return {'pred_data_sides': lmap(fmask, pred_data)                                                  ,
                'ref_data'       : {k: fmask(ref_data[k]                        ,
                                             is_info         = (k==k_full_info ),
                                             check_prev_flip = (k==k_X_map     ),
                                             check_post_flip = True             ) for k in ref_data.keys()}}

# ------------------------------------------------------------------------
#                  Main Runner
# ------------------------------------------------------------------------

class Process_IndPred:
    def __init__(self, args):
        self.args           =  args
        qmode               =  args['qmode']
        dset                =  args['dset' ]
        ind                 =  args['ind'  ]
        
        assert  dset in ['valid','test']
        mkdir_p(dirname(args['tec_fname']))
        assert  isdir(dirname(args['tec_fname'])) and (not isfile(args['tec_fname']))
        
        d         =  self.reduce_data()
        
        info      =  d['ref_data']['full_info'] # this is a bit strange, but full_info is now a dictionary
        
        assert type(info) == dict
        
        # for key,val in [['N_xyz',[280,278,140]],['dpdx',1.],['source_q',1.]]:
        #     if not key in info['props']:
        #         info['props'][key] = val
        #         print(f"WARNING: {key} not found in info['props'] (assumed key = {info['props'][key]})")
        
        tec_order, data, qmod_alt, extra  =  self.mk_data(d, qmode, info['props'])
        Nx, _, Nz                         =  info['props']['N_xyz']
        
        self.sample  =  {'nx'                   : Nx                                         ,
                         'nz'                   : Nz                                         ,
                         'qmode'                : qmode                                      ,
                         'pred_mode'            : dset                                       ,
                         'ind_pred'             : ind                                        ,
                         'K_trials'             : args['K_trials']                           ,
                         'tec_fname'            : args['tec_fname']                          ,
                         'tec_order'            : tec_order                                  ,
                         'info'                 : info                                       ,
                         'metrics'              : self.mk_metrics(data, qmode, qmod_alt)     ,
                         'text_prev_epochs'     : d['text_prev_epochs']                      ,
                         'data'                 : data                                       ,
                         'extra'                : extra                                      ,
                         'qmod_alt'             : qmod_alt                                   ,
                         'data_folders'         : d['use_data_folders']                      ,
                         'log_files'            : d['use_log_files']                         }
        self.write_tecfile()
    
    def reduce_data(self):
        args           =  self.args         
        use_m_folders  =  args['use_m_folders']
        dset           =  args['dset']
        ind            =  args['ind'  ]
        qmode          =  args['qmode']
        f_relax        =  Relax_Grad_Engine(args['relax_grad'])
        
        def reduce_ref(key):
            all_found = [m.get_data(dset,ind)['ref_data'][key] for m in use_m_folders]
            if key == 'full_info':
                result, = lset_brute(fix_keys(all_found))
            else:
                result = all_found[0][0]
                assert hasattr(result,'shape')
                assert all(almost(result,arr) for ab in all_found for arr in ab)
            return result
        
        def mk_avg():
            all_pred_data_sides = [m.get_data(dset,ind)['pred_data_sides'] for m in use_m_folders]
            all_flat            = [arr.tolist() for mp in all_pred_data_sides for e_ab in mp for arr in e_ab]
            return f_relax(Joint_Predictions(all_flat).result)
            
        def mk_text_prev_epochs():
            return [lmap(lambda d: d['text'], m.last_savers) for m in use_m_folders]
        
        return {'ref_data'        :  {key: reduce_ref(key) for key in ( 'X_map'         ,
                                                                        'H_map'         ,
                                                                        'Z_map'         ,
                                                                       f'Labels_{qmode}',
                                                                        'full_info'     )},
                'avg_data'        :  mk_avg()                                             ,
                'text_prev_epochs':  mk_text_prev_epochs()                                ,
                'use_data_folders':  [m.folder           for m in  use_m_folders]         ,
                'use_log_files'   :  [m.m_log.fname_log  for m in  use_m_folders]         }
    
    def write_tecfile(self):
        sample     =  self.sample
        pad        =  ' '
        line       =  '----------------------------------'
        
        self.tec_fname  =  tec_fname  =  sample['tec_fname']
        mkdir_p(dirname(tec_fname))
        assert isdir(dirname(tec_fname)) and (not isfile(tec_fname))
        
        header = { 'Original_Work_Folders':  sample['data_folders']     ,
                   'Log_Files'            :  sample['log_files']        ,
                   'K_trials'             :  sample['K_trials']         ,
                   'Ind_Pred'             :  sample['ind_pred']         ,
                   'Prediction_Mode'      :  sample['pred_mode']        ,
                   'qmode'                :  sample['qmode']            ,
                   'qmod_alt'             :  sample['qmod_alt']         ,
                   'Bulk_Flow_Properties' :  sample['info']             ,
                   'Tecplot_fname'        :  tec_fname                  ,
                   'Accuracy_Avg_File'    :  sample['metrics']          ,
                   'SMOOTH_ARGS'          : {'APPLY_AVG'  :  APPLY_AVG              ,
                                             'SMOOTH_MODE':  SMOOTH_MODE            ,
                                             'relax_grad' :  self.args['relax_grad']},
                   'Tecplot_order'        :  sample['tec_order']        ,
                   'extra_predictions'    :  sample['extra']            , 
                   'Parent_Epochs'        :  sample['text_prev_epochs'] ,
                   'Raw_Log_Files'        :  'PYTHON_HOLDER_RAW_LOG_FILE'}
        header = apply_pformat(header).replace( "'PYTHON_HOLDER_RAW_LOG_FILE'"        ,
                                                f'{lmap(reader, sample["log_files"])}').split('\n')
        print(f"    -> writing tecfile: {basename(tec_fname)}")
        
        A_write = []
        fwrite  = lambda x: A_write.append(str(x).rstrip('\n')+'\n')
        cwrite  = lambda x: fwrite(f'# {x}')
        
        cwrite(f'{line} Begin_About_Tecplotfile {line}')
        lmap(cwrite, header)
        cwrite(f'{line} End___About_Tecplotfile {line}')
        cwrite(f'{line} Begin_Main__Tecplotfile {line}')
        fwrite(f'VARIABLES ="'+'", "'.join(sample['tec_order'])+'"')
        fwrite(f"ZONE I={sample['nz']} K={sample['nx']} F=POINT ")
        for      i in range(sample['nx']):
            for  j in range(sample['nz']):
                fwrite(' '.join(map(str,[sample['data'][key][i,j] for key in sample['tec_order']])))
        
        zip_str_write(tec_fname,'\n'.join(map(lambda x: str(x).rstrip('\n'),
                                              A_write                      ))+'\n')
        
    @staticmethod
    def mk_data(d, qmode, props):
        extra      =  {}
        qmod_alt   =  {'Cf_classic'  : 'Fx'  ,
                       'Nu_classic'  : 'Qdot',
                       'St_classic'  : 'Qdot'}[qmode]
        
        tec_order  =  [ "X", "Y", "Z", "H",
                       f"qmode_{qmode}_dns"        , f"qmode_{qmode}_pred"        , f"qmode_{qmode}_error"  ,
                       f"qmod_alt_{qmod_alt}_dns", f"qmod_alt_{qmod_alt}_pred", f"qmod_alt_{qmod_alt}_error"]
        
        data = {'X'                           : d['ref_data']['X_map']          ,
                'Y'                           : d['ref_data']['H_map']          ,
                'Z'                           : d['ref_data']['Z_map']          ,
                'H'                           : d['ref_data']['H_map']          ,
                f"qmode_{qmode}_dns"          : d['ref_data'][f'Labels_{qmode}'],
                f"qmode_{qmode}_pred"         : d['avg_data']                   }
        
        prefix_alt_dns   =  f"qmod_alt_{qmod_alt}_dns"
        prefix_alt_pred  =  f"qmod_alt_{qmod_alt}_pred"
        
        rho              =  props['rho']
        
        def get_delta_A_wall_vol():
            Lx,Ly__,Lz       =  props['L_xyz']
            delta            =  0.5 * Ly__
            A_wall           =  Lx*Lz
            vol              =  A_wall*delta
            return delta, A_wall, vol
        delta, A_wall, vol  =  get_delta_A_wall_vol()
        
        if   qmode == 'Cf_classic':
            assert qmod_alt == 'Fx'
            
            mk_Fx    =  lambda Cf,Ub: A_wall*Cf*(rho*(Ub**2))/2.
            
            Cf_dns   =  data[f"qmode_{qmode}_dns"]   #  Cf for qmode == 'Cf_classic'
            Cf_pred  =  data[f"qmode_{qmode}_pred"]  #  Cf for qmode == 'Cf_classic'
            
            data[prefix_alt_dns ]         =  mk_Fx(Cf_dns, props['Ub']) # Fx = prefix_alt_dns for qmode == 'Cf_classic'
            
            extra['Fx_pred']  =  Fx_pred  =  vol * props['dpdx']
            extra['Ub_pred']  =  Ub_pred  =  np.sqrt(2*Fx_pred/A_wall/(rho*np.mean(Cf_pred)))
            data[prefix_alt_pred]         =  mk_Fx(Cf_pred, Ub_pred)
            extra['Cf_global_pred']       =  np.mean(Cf_pred)
            
        elif qmode   == 'Nu_classic':
            assert qmod_alt == 'Qdot'
            cp           =  props['cp']
            nu           =  props['nu']
            Pr           =  props['Pr']
            Tw           =  props['Tw']
            Ret          =  props['Ret']
            conductivity = 1/(Ret*Pr)
            assert abs(Tw) < 1e-12
            
            mk_Qdot =  lambda Nu,Tb: A_wall*Nu*(conductivity*(Tb-Tw)/delta)
            
            assert qmode == 'Nu_classic'
            Nu_dns   =  data[f"qmode_{qmode}_dns"]
            Nu_pred  =  data[f"qmode_{qmode}_pred"]
            
            data[prefix_alt_dns ]             =  mk_Qdot(Nu_dns, props['Tb']) # Fx = prefix_alt_dns for qmode == 'Cf_classic'
            
            extra['Qdot_pred']  =  Qdot_pred  =  vol * props['source_q']
            extra['Tb_pred'  ]  =  Tb_pred    =  Qdot_pred/A_wall/np.mean(Nu_pred)/conductivity*delta + Tw
            data[prefix_alt_pred]             =  mk_Qdot(Nu_pred, Tb_pred)
            extra['Nu_global_pred']           =  np.mean(Nu_pred)
            
        else:
            raise Exception(f'ERROR: Conversion required between qmode and qmod_alt not implemented. (qmode = {qmode}) (qmod_alt = {qmod_alt})')
        
        for prefix in [f"qmode_{qmode}",f"qmod_alt_{qmod_alt}"]:
            data[f'{prefix}_error']  =  data[f'{prefix}_pred'] - data[f'{prefix}_dns']
        
        assert 13.84845 < float(np.mean(data[prefix_alt_pred])) < 17.84845
        
        assert set(tec_order) == set(data.keys())
        return tec_order, data, qmod_alt, extra
    
    @staticmethod
    def mk_metrics(data, qmode, qmod_alt):
        # (diff/total = 146.8845 %) (|diff|/total = 318.4859 %) (|diff|/|total| = 263.6910 %)
        result = {}
        for prefix in [f"qmode_{qmode}",f"qmod_alt_{qmod_alt}"]:
            error           =  data[f"{prefix}_error"]
            dns             =  data[f"{prefix}_dns"]
            
            diff            =          error .sum()
            diff_abs        =  np.fabs(error).sum()
            total           =          dns   .sum()
            total_abs       =  np.fabs(dns  ).sum()
            result[prefix]  =  { 'diff/total'   : fmt_p(diff    /total    ),
                                '|diff|/total'  : fmt_p(diff_abs/total    ),
                                '|diff|/|total|': fmt_p(diff_abs/total_abs)}
        return result

# ------------------------------------------------------------------------
#                  Scan Log File
# ------------------------------------------------------------------------
class ScanLogFile:
    def __init__(self, folder, n_savers):
        self.folder     =  folder
        self.fname_log  =  locate_file('.log', folder)
        self.A_log      =  reader(self.fname_log)
        
        if type(n_savers) == int:
            self.best_saver = None
        else:
            assert n_savers == 'best_score'
            self.best_saver  =  ScoreSavers_Log_File(dirname(self.fname_log)).best_saver
        
        self.__begin_saver  =  '--- Begin Saver ---'
        self.__end_saver    =  '--- End   Saver ---'
        self.__arrow        =  '---->'
        
        self.qmode          =  get_rhs('qmode           :', self.A_log)
        
        # assert '_K10_' in self.fname_log
        self.Ktrial         =  first_int(self.fname_log.split('runid_')[1])
        
        self.inds_dset      =  {key: eval(get_rhs(   f'inds_{key:5s}      :', self.A_log)) for key in ('train','test','valid')}
        
        self.scan_savers()
        
        self.ref_vars       = pjoin(folder,'saved', '00_reference_variables.dat') # locate_file('00_reference_variables.dat', pjoin(folder,'saved'))
    
    def scan_savers(self):
        
        find_inds   = lambda tag: [i for i,x in enumerate(self.A_log) if (tag in x)]
        self.savers = []
        
        assert (len(find_inds(self.__begin_saver)) ==
                len(find_inds(self.__end_saver  ))  )
        
        for i_start,i_end in zip(find_inds(self.__begin_saver),
                                 find_inds(self.__end_saver  )):
            assert (i_end-i_start) == 7
            D  =  {'text': self.A_log[i_start+1:i_end]}
            for x in D['text']:
                if 'Prediction' in x:
                    k       =  'acc_' + x.split('dset = ')[1].split(')')[0].replace(' ','') 
                    E       = {}
                    for y in first_dot_split(x)[1].replace(')','').split('(')[1:]:
                        p0,p1 = y.replace(' ','').split('=')
                        p0    = p0.replace('elapsedtime', 'elapsed_time')
                        E[p0] = p1
                    assert set(E.keys()) == {'diff/total','|diff|/total','|diff|/|total|','elapsed_time'}
                    D[k] = E
                elif self.__arrow in x:
                    p0,p1 = first_dot_split(x)
                    k     = p0.replace(self.__arrow,'').replace(' ','') 
                    k     = k .replace('FieldVars', 'Field_Vars')
                    D[k]  = fix_path(p1.replace(' ',''), self.folder)
            
            D['ref_epoch'] = pop1(set(    D[key].split('_epoch_')[1].split('_')[0] for key in ['Net_Dict', 'Optim_Dict', 'Field_Vars']    ))
            if D['ref_epoch'] != 'MiTrAbDiEp': D['ref_epoch'] = int(D['ref_epoch'])
            
            assert set(D.keys()) == {'acc_train','acc_valid','acc_test','Net_Dict','Optim_Dict','Field_Vars','text','ref_epoch'}
            
            if self.best_saver:
                if D['ref_epoch'] == self.best_saver['epoch']:
                    self.savers.append(D)
                # else: pass
            else:
                self.savers.append(D)

# ------------------------------------------------------------------------
#                  Main Routine
# ------------------------------------------------------------------------

def main_runner(tag_run, qmode, relax_grad=RELAX_GRAD):
    assert tag_run  in  ['_sel40_', '_rnd40_'   , '_rnd16_', '_rnd12_', '_rnd08_']
    assert qmode    in  ['Cf_classic', 'Nu_classic']
    fname_archive         =  pjoin(post_output, f'archive_new_tecfiles_tag_run_{tag_run}_qmode_{qmode}.dat'.replace('__','_'))
    
    # easy check
    if isfile(fname_archive): raise Exception('Archive of files already built!')
    with open(fname_archive, 'w') as f:  f.write('\n')
    
    archive_new_tecfiles  =  []
    
    # folders belonging to [tag_run,qmode]
    all_folders           = lfilter(lambda x: all((k in basename(x)) for k in [tag_run, f'_{qmode}_']),
                                    listdir_full_folders(pjoin(deepdirname(abspath(__file__),3),'data',f'output_{qmode}')))
    
    print('all_folders',all_folders)
    # assert len(all_folders) == 10
    
    # m_folder_objects belonging to [tag_run, qmode]
    all_m_folders  =  lmap(Scan_Folder, all_folders)
    
    # write tecfiles
    for      dset  in  ['test','valid']:
        
        # indexes (at all) in [qmode, tag_run, dset]
        get_inds  =  lambda dset:  sorted(set(i for m in all_m_folders for i in m.m_log.inds_dset[dset]))
        
        for  ind  in  get_inds(dset): # all indexes in [qmode, tag_run, dset]
            
            get_Ktrial         =  lambda m:  m.m_log.Ktrial
            
            # all_use_m_folders: filtered by [qmode,tag_run,dset,ind]
            #     -> all folders from [qmode,tag_run] that have [ind] in their [dset]
            all_use_m_folders  =  lambda: sorted(lfilter(lambda m: ind in m.m_log.inds_dset[dset], all_m_folders),
                                                 key = get_Ktrial                                                )
            # K-trials to consider
            all_K_trials       =  lmap(get_Ktrial, all_use_m_folders())
            
            # assert len(all_K_trials) == 2
            
            # iterate over K-trial combinations
            for K_trials in (lmap(lambda x: [x], all_K_trials) + [pick2(all_K_trials)]):# for K_trials in  [all_K_trials]:
                
                print(f"(tag_run = {tag_run}) (qmode = {qmode}) (dset = {dset}) (ind = {ind}) (K_trials = {K_trials})")
                args = dict( qmode          =  qmode                                                             ,
                             dset           =  dset                                                              ,
                             ind            =  ind                                                               ,
                             K_trials       =  K_trials                                                          ,
                             relax_grad     =  relax_grad                                                        ,
                             # all folders in [qmode,tag_run,dset,ind,K_trial]
                             use_m_folders  =  lfilter(lambda m: m.m_log.Ktrial in K_trials, all_use_m_folders()))
                
                def mk_tec_fname():
                    # get base tag (qmode,tag_run) (we add [dset,ind,K_trials] later)
                    tag  =  reduce_longest_common(lmap(lambda m: basename(m.m_log.fname_log), all_use_m_folders()))
                    while tag and tag[-1].isdigit(): tag = tag[ :-1]
                    while tag and tag[ 0].isdigit(): tag = tag[1:  ]
                    assert ((tag[0]=='_') and (tag[-7:]=='_runid_'))
                    tag  =  tag.replace('_runid_','')
                    assert tag_run in tag
                    assert qmode   in tag
                    # add missing information to tag
                    return pjoin(post_output, 'tecplot_files', qmode, f'{tstamp()}{tag}_dset_{dset}_ind_{ind}_Ktrials_{"_".join(map(str,K_trials))}.tec')
                
                args['tec_fname'] = mk_tec_fname()
                
                m_process = Process_IndPred(args) 
                archive_new_tecfiles.append(dict(tec_fname     =  m_process.tec_fname                                    ,
                                                 tag_run       =  tag_run                                                ,
                                                 qmode         =  qmode                                                  ,
                                                 dset          =  dset                                                   ,
                                                 ind           =  ind                                                    ,
                                                 K_trials      =  K_trials                                               ,
                                                 relax_grad    =  relax_grad                                             ,
                                                 used_logs     = lmap(lambda m: m.m_log.fname_log, args['use_m_folders']),
                                                ))
    
    with open(fname_archive, 'w') as f:
        f.write(apply_pformat(archive_new_tecfiles)+'\n')

# ------------------------------------------------------------------------
#                  Direct Runner
# ------------------------------------------------------------------------

if __name__ == '__main__':
    tag_run = '_rnd12_'
    qmode   = 'Nu_classic'
    main_runner(tag_run, qmode)
