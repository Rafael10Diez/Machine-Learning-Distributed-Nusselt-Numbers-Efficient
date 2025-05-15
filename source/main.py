
# ------------------------------------------------------------------------
#                  Generic Libraries
# ------------------------------------------------------------------------

from    os.path   import  abspath, isdir, dirname, basename, isfile
from    os.path   import  join                                       as  pjoin
from    os        import  listdir                                    as  os_listdir
from    os        import  system                                     as  os_system
from    sys       import  path                                       as  sys_path
from    os        import  mkdir
from    sys       import  argv
from    datetime  import  datetime
from    copy      import  deepcopy

# ------------------------------------------------------------------------
#                  Custom Libraries
# ------------------------------------------------------------------------

_folder_  =  dirname(abspath(__file__))

sys_path.append(_folder_)
from core.external      import  tstamp, lmap, lfilter
from core.io_access_ml  import  FPrint_ML, output_dir_ml, Copy_Zipped
from core.gather_data   import  All_Datasets
from core.ml_runner     import  ML_Runner
sys_path.pop()

# ------------------------------------------------------------------------
#                  Basic Functions
# ------------------------------------------------------------------------

listdir_full          =  lambda x: sorted(pjoin(x,y) for y in os_listdir(x))
listdir_full_files    =  lambda x: lfilter(isfile, listdir_full(x))
listdir_full_folders  =  lambda x: lfilter(isdir , listdir_full(x))

def splitpath(x):
    A = []
    iters, Nmax  =  0, max(len(x)*10,int(1e4))
    while basename(x):
        A.append(basename(x))
        x      = dirname(x)
        iters += 1
        assert iters < Nmax
    return A[::-1]

def pop1(A):
    x, = A
    return x

# ------------------------------------------------------------------------
#                  Data Folders
# ------------------------------------------------------------------------

folders  =  [f'data_Ret_180_Pr_1_r{i:02d}' for i in range(40)]

# ------------------------------------------------------------------------
#                  Seen Argv
# ------------------------------------------------------------------------

def get_log_argv(folder):
    fname, = lfilter(lambda x: '.log' in basename(x),
                     listdir_full_files(folder)     )
    result = []
    with open(fname,'r') as f:
        for x in f:
            x = x.rstrip('\n')
            if '    given_argv' == x[:14]:
                result.append(eval(':'.join(x.split(':')[1:])))
    return pop1(result)

get_seen_argv = lambda qmode: lmap(get_log_argv, listdir_full_folders(pjoin(  dirname(dirname(__file__)),
                                                                             'data'                     ,
                                                                            f'output_{qmode}'           )))

def parse_argv(A):
    A     = deepcopy(A)
    icuda = pop1(lfilter(lambda i: 'cuda' in A[i], [-2,-3]))
    assert 'cuda' in A[icuda]
    assert 'valid_' == A[3][:6]
    assert 'test_'  == A[4][:5]
    A[3]  = 'jointvt_'+'_'.join(map(str,sorted(map(int, A[3].lstrip('valid_').lstrip('test_').split('_') +
                                                        A[4].lstrip('valid_').lstrip('test_').split('_') ))))
    A[4]     = None
    A[icuda] = None
    return A

# ------------------------------------------------------------------------
#                  Run Function
# ------------------------------------------------------------------------

def main_runner(run_tag, qmode, inds_train, inds_valid, inds_test, Cseq, dseq, run_id, device, register_run, weight_decay_w, given_argv):
    
    if parse_argv(given_argv) in lmap(parse_argv, get_seen_argv(qmode)): return
    
    register_run  =  bool(register_run)
    SEED          =  int(run_id)
    
    assert not (set(inds_train) & set(inds_valid))
    assert not (set(inds_train) & set(inds_test ))
    assert not (set(inds_valid) & set(inds_test ))
    
    # -------------------------------------------
    #                  Arguments
    # -------------------------------------------
    
    # arguments (build datasets) 
    args_dsets  =  { 'side'       :  'bottom'                     ,
                     'qmode'      :  qmode                        ,
                     'folders'    :  folders                      , 
                     'inds_train' :  inds_train                   ,
                     'inds_test'  :  inds_test                    ,
                     'inds_valid' :  inds_valid                   ,
                     'as_float'   :  True                         ,
                   }
    
    # arguments (cnn) 
    args_net   =  {'Cseq' : Cseq,
                   'dseq' : dseq}
    
    # arguments (ML runner) 
    args_ML     =  { 'qmode'                :  qmode                                           ,
                     'weight_decay_w'       :  weight_decay_w                                  ,
                     'optim_w_type'         : 'adam'                                           ,
                     'lr_w'                 :  0.001                                           ,
                     'lr_min_w'             :  1e-05                                           ,
                     'Epochs'               :  10001                                           ,
                     'args_net'             :  args_net                                        ,
                     'betas_w'              :  (0.9, 0.999)                                    ,
                     'eps'                  :  1e-7                                            ,
                     'loss_function'        :  'lambda delta: torch.abs(delta).sum()'          , #+ 3*torch.abs(delta.sum())'          , # torch.mean(torch.abs(delta))',# 'lambda delta: torch.mean((delta)**2)' 'lambda delta: torch.mean((delta)**2)'
                     'loss_type'            :  'l1'                                            ,
                     'net_type'             :  'alt2_orig'                                     , # alt2_bigwindow alt2_super2
                     'batch_size'           :  None                                            ,
                     'min_saver_dt'         :  60.                                             ,
                     'breaker'              :  None                                            ,
                     'epoch_save_freq'      :  500                                             ,
                     'epoch_n_saves'        :  None                                            ,
                     'epoch_predict'        :  10                                              ,
                     'epoch_report'         :  10                                              ,
                     'SEED'                 :  SEED                                            ,
                     'register_run'         :  register_run                                    ,
                     'given_argv'           :  given_argv                                      ,
                   }
    
    args_ML['seeder'] = { 'epochs_per_seed':  2000                     ,
                          'n_seeds_try'    :  6 if register_run else 10,
                          'epochs_avg'     :  100                      ,
                          'metric_name'    :  '|diff|/|total|'         ,
                          'use_dsets'      :  ['train','valid'] if register_run else ['train'],
                        }
    
    args_ML['Epochs'] += args_ML['seeder']['epochs_per_seed']*(args_ML['seeder']['n_seeds_try']-1)
    
    assert (args_ML['seeder']['epochs_per_seed'] % args_ML['epoch_save_freq']) == 0
    
    # -------------------------------------------
    #                  Outout Folder
    # -------------------------------------------
    assert type(run_id) == str
    if args_ML['weight_decay_w']:
        tag_wdl2 = "_wdl2_" + f"{(args_ML['weight_decay_w']):.2e}".replace('+','')
    else:
        tag_wdl2 = ''
    subfolder      =  tstamp() + f"_{run_tag}_{args_ML['loss_type']}_{args_ML['qmode']}_net_{args_ML['net_type']}{tag_wdl2}_runid_{run_id}"
    
    output_folder  =  pjoin(output_dir_ml(qmode), subfolder)
    
    if not isdir(output_dir_ml(qmode)):
        mkdir(   output_dir_ml(qmode))
    
    assert not isdir(output_folder)
    mkdir(      output_folder         )
    mkdir(pjoin(output_folder,'saved'))
    
    # -------------------------------------------
    #                  Copy Code
    # -------------------------------------------
    
    Copy_Zipped(  dirname(abspath(__file__))  ,
                  output_folder               )
    
    # -------------------------------------------
    #                  Logger
    # -------------------------------------------
    
    # logger
    fprint     =  FPrint_ML( pjoin( output_folder, subfolder + '.log') )
    
    # -------------------------------------------------------------
    #                  Datasets (train/valid/test)
    # -------------------------------------------------------------
    
    # build datasets
    get_all_data   =  lambda: All_Datasets(args_dsets, fprint)
    
    # -------------------------------------------------------------
    #                 ML Runner
    # -------------------------------------------------------------
    
    # ml runner
    ml_runner  =  ML_Runner(get_all_data, args_ML, fprint, output_folder, abspath(__file__), device)
    
    # make predictions
    ml_runner.global_predict()
    
# ------------------------------------------------------------------------
#                  Fetch Inds from Argv
# ------------------------------------------------------------------------

def fetch_inds(A, tag):
    A = A.replace(' ','')
    assert A[:(len(tag)+1)]  ==  (tag + '_')
    A = A.split('_')[1:]
    return [int(x.lstrip('0') or '0') for x in A]

def parse_str(A,s):
    A = A.split('_')
    assert A[0] == s
    A = A[1:]
    for i,x in enumerate(A):
        if 'x' in x:
            A[i] = lmap(int,x.split('x'))
        else:
            A[i] = int(x)
    return A

# ------------------------------------------------------------------------
#                  Direct Runner
# ------------------------------------------------------------------------

if __name__ == '__main__':
    def add_empty_subfolders():
        for subfolder in [pjoin(dirname(_folder_),x) for x in ['data/archive_Cf_classic',
                                                               'data/archive_Nu_classic',
                                                               'data/output_Cf_classic',
                                                               'data/output_Nu_classic',
                                                               'data/started_Cf_classic',
                                                               'data/started_Nu_classic',
                                                               'data/wdl2_output_runs',
                                                               'post_rnd12/post_output/pics_s2d_2d/Nu_classic',
                                                               'post_rnd12/post_output/pics_s2d_2d/Cf_classic',
                                                               'post_rnd12/post_output/tecplot_files/Nu_classic',
                                                               'post_rnd12/post_output/tecplot_files/Cf_classic',
                                                               'post_rnd12/post_output/presentations',
                                                               'post_rnd12/post_output/his_plots',
                                                               'post_rnd12/post_output/corr_plots']]:
            if not isdir(subfolder):
                print(f'Creating folder: {subfolder}')
                os_system(f'mkdir -p "{subfolder}"')
    add_empty_subfolders()

    run_tag, qmode                     =  lmap( lambda s: s.replace(' ','') ,
                                                argv[1:3]                   )
    
    inds_train, inds_valid, inds_test  =  lmap(  lambda x: fetch_inds(x[0],x[1])          , 
                                                 zip(argv[3:6],['train','valid','test'])  )
    
    Cseq           =  parse_str(argv[6],'Cseq')
    dseq           =  parse_str(argv[7],'dseq')
    run_id         =       argv[ 8]
    device         =       argv[ 9]
    register_run   =  eval(argv[10])
    weight_decay_w =  eval(argv[11])
    given_argv     =       argv[1:]
    main_runner(run_tag, qmode, inds_train, inds_valid, inds_test, Cseq, dseq, run_id, device, register_run, weight_decay_w, given_argv)