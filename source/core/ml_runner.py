# ------------------------------------------------------------------------
#                  Generic Libraries
# ------------------------------------------------------------------------
import  socket
import  zipfile
from    os.path      import  basename, abspath, dirname, isfile, isdir
from    os.path      import  join                                       as  pjoin
from    time         import  time, sleep
from    math         import  prod
from    collections  import  OrderedDict
from    pprint       import  pformat

import  torch

# ------------------------------------------------------------------------
#                  Random Seed
# ------------------------------------------------------------------------

class Apply_Seed_Torch:
    def __init__(self, SEED):
        self.SEED = SEED
    
    def __call__(self, torch):
        torch.     manual_seed(self.SEED)
        torch.cuda.manual_seed(self.SEED)
        # torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

# ------------------------------------------------------------------------
#                  Custom Libraries
# ------------------------------------------------------------------------

from core  .external            import  format_dt, reader, tstamp, lmap
from core  .loaders             import  All_Loaders
from models.cnn_alt2_orig       import  Net_Circular_Alt2_Orig

# ------------------------------------------------------------------------
#                  Basic Definitions
# ------------------------------------------------------------------------

# number of trainable parameters:
calc_net_params  =  lambda net: sum( p.numel() for p in net.parameters() if p.requires_grad )
# calc_L1_params   =  lambda net: sum( torch_abs(p).sum() for p in net.parameters() if p.requires_grad )

apply_pformat    =  lambda D: pformat(D, sort_dicts=False, indent = 2)
fmt_D            =  lambda D: str(D).replace(' ','')

fmt_p            =  lambda x: f'{(x*100):.5f} %'

own_socket       =  socket.gethostname()

class fmtTensor:
    def __init__(self, A, tensor_keyword):
        assert hasattr(A,'shape')
        self.dtype = str(A.dtype)
        self.A     = A.cpu().detach().numpy().tolist() # convert tensor to list
        assert type(self.A) in [list,int,float]
        assert type(tensor_keyword) == bool
        self.tensor_keyword = tensor_keyword
    def __print(self):
        s = fmt_D(self.A)
        assert type(s) == str
        return f'torch.tensor({s},dtype={self.dtype})' if self.tensor_keyword else s
    __repr__ = __str__ = __print

def To_Str_State_Dict(state_dict):
    def dfs(A):
        if type(A) in [dict,OrderedDict]:
            return type(A)([[k,dfs(v)] for k,v in A.items()])
        
        elif type(A) in [list,tuple]:
            return type(A)([dfs(v) for v in A])
        
        elif type(A) in [float,int,str,bool]:
            return type(A)(A)
        
        elif A is None:
            return None
        
        elif hasattr(A,'shape'):
            return fmtTensor(A,tensor_keyword=True)
        
        else:
            raise Exception(f'Unrecognized (type = {type(A)}) (state_dict={state_dict})')
    
    return fmt_D(dfs(state_dict))

# def write_torch_state_dict(state_dict, fname): # nothing wrong with this code, it's just unused...
#     with open(fname,'w') as f:
#         f.write(To_Str_State_Dict(state_dict)+'\n')

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
#                  ML Runner
# ------------------------------------------------------------------------

class ML_Runner:
    def __init__(self, get_all_data, args, fprint, output_folder, main_fname, device):
        assert 'cuda' in device
        # Register Arguments:
        self.device         =  device
        self.args           =  args
        self.fprint         =  fprint
        self.output_folder  =  output_folder
        self.__main_fname   =  main_fname
        self.pred_stats     =  {}
        
        seed_torch          =  Apply_Seed_Torch(args['SEED'])
        seed_torch(torch)
        
        self.__record_results('ini')
        
        # Direct writer to file
        self.fwrite  =  fwrite  =  lambda s: fprint( s , '__only_to_file__' )
        
        
        # Import Network
        f_build_net = {  'alt2_orig':  Net_Circular_Alt2_Orig,
                      }[args['net_type']]
        
        self.__n_calls_mk_new = 0
        
        def mk_new_net_opt(state_dicts=None, seed_id = None):
            assert ( ((state_dicts is     None) and (seed_id is     None)) or
                     ((state_dicts is not None) and (seed_id is not None))   ) # when we restore a state_dict, we should also retrieve the original seed
            net                    = f_build_net(args['args_net'], Apply_Seed_Torch(args['SEED']+self.__n_calls_mk_new))
            self.__seed_id         =  self.__n_calls_mk_new if (seed_id is None) else int(seed_id)
            self.__n_calls_mk_new += 1
            if state_dicts:
                net.load_state_dict(eval(state_dicts['net'    ]))
            net  =  net.to(device)
            
            p_change  =  [p for p in net.parameters() if p.requires_grad]
            
            if args['optim_w_type'] == 'adam':
                optim_w  =  torch.optim.Adam( p_change                                ,
                                              lr            =  args['lr_w']           ,
                                              betas         =  args['betas_w']        ,
                                              weight_decay  =  args['weight_decay_w'] ,
                                              eps           =  args['eps']            )
                if state_dicts:
                    optim_w.load_state_dict(eval(state_dicts['optim_w']))
            else:
                raise Exception(f"Error: Unrecognized Optimizer (optim_w_type = {args['optim_w_type']})")
            self.net  =  net
            self.p_change  =  p_change
            self.optim_w   =  optim_w
        
        self.mk_new_net_opt = mk_new_net_opt
        
        self.mk_new_net_opt()
        
        # Define criterion
        self.criterion  =  eval(args['loss_function'])

        # make data loaders
        
        self.total_params  =  calc_net_params(self.net)
        
        # Print Header:
        fprint( '\n--------------------------------------------------\n    ML Runner\n--------------------------------------------------\n')
        fprint(f'\nLog Filename (this file)   :  {fprint.fname}')
        fprint(f'\nSeed                       :  {args["SEED"]}')
        fprint(f'\nHost                       :  {own_socket}')
        fprint(f'\nDevice                     :  {device}')
        fprint(   'Input Arguments (ML_Runner):')
        for k,v in args.items():
            fprint(f'    {k:15s} :  {v}')
        
        fprint(f'\nNumber of Parameters: {self.total_params}')
        
        self.all_data  =  get_all_data()
        self.loaders    =  All_Loaders( self.all_data      ,
                                        args['batch_size'] )
        fprint('')
        fprint('\n--------------------------------------------------\n    Scaling Parameters\n--------------------------------------------------\n')
        fprint(f'    input_scale: {self.loaders.input_scale}')
        fprint(f'    label_scale: {self.loaders.label_scale}')
        fprint('')
        
        fwrite('')
        fwrite('\n--------------------------------------------------\n    Code Neural Network\n--------------------------------------------------\n')
        fwrite(f'File Path: {self.net.get_file()}')
        for line in reader(self.net.get_file()):
            fwrite(f'    ->  {line}')
        fwrite('')
        
        fwrite('')
        fwrite('\n--------------------------------------------------\n    Code Main Runner\n--------------------------------------------------\n')
        fwrite(f'File Path: {main_fname}')
        for line in reader(main_fname):
            fwrite(f'    ->  {line}')
        fwrite('')
        
        fwrite('')
        fwrite('\n--------------------------------------------------\n    Details Neural Network\n--------------------------------------------------\n')
        fwrite( self.net )
        fwrite('')
        
        self.write_ref_vars = True
        
        self.runner()
        
    def runner(self):
        line              =  '--------------------'
        args              =  self.args
        criterion         =  self.criterion
        trainloader       =  self.loaders.trainloader
        fprint            =  self.fprint
        
        min_saver_dt      =  self.args['min_saver_dt']
        epoch_predict     =  self.args['epoch_predict']
        epoch_report      =  self.args['epoch_report']
        fwrite            =  self.fwrite
        
        bestSeed                      = {}
        bestSeed['track_epochs']      =  set(lmap(lambda i: i*self.args['seeder']['epochs_per_seed'],
                                                  range(1,self.args['seeder']['n_seeds_try']+1)     ))
        bestSeed['epoch_final_load']  =  max(bestSeed['track_epochs']) + 1
        bestSeed['record']            =  float('inf')
        bestSeed['state_dicts']       =  None
        bestSeed['seed_id']           =  None
        
        def seeder_current_metric():
            n_use       = self.args['seeder']['epochs_avg']//epoch_predict
            metric_name = self.args['seeder']['metric_name']
            use_dsets   = self.args['seeder']['use_dsets']
            assert list(self.args['seeder']['use_dsets']) == ['train']
            all_val     = []
            for dset in use_dsets:
                all_val  += [abs(val) for val in lmap(lambda d: d[metric_name],
                                                      self.pred_stats[dset][-n_use:])]
            return sum(all_val)/len(all_val)
        
        bestSeed['seeder_current_metric'] = seeder_current_metric
        
        saved_epochs      = list(range(0,args['Epochs']+1,self.args['epoch_save_freq']))[1:]
        while saved_epochs[-1] > args['Epochs']:
            saved_epochs.pop()
        
        if type(args['epoch_n_saves']) in [str,int]:
            saved_epochs = saved_epochs[-int(args['epoch_n_saves']):]
        print(f'INFORMATION: saved_epochs = {saved_epochs}')
        
        breaker = float('inf') if (args['breaker'] is None) else args['breaker']
        
        min_saved_epochs     =  min(saved_epochs)
        
        self.__save_special  =  {'started'  : False        ,
                                 'best'     : float('inf') ,
                                 'dict'     : None         }
        
        t0 = t1 = t2 = t3 = time()
        prev_epoch   =  0
        assert not args['batch_size']
        
        for epoch in range(1,args['Epochs']+1):  # loop over the dataset multiple times
            
            if (time()-t3) >= 60.:
                t3       = time()
                while any(map(lambda x: own_socket in x                                            ,
                              reader(pjoin( dirname(dirname(abspath(__file__))), 'host_pause.dat')))):
                    sleep(60.)
                    print('Sleeping...')
            
            if epoch > saved_epochs[-1]: break
            
            if epoch == bestSeed['epoch_final_load']:
                assert epoch == (max(bestSeed['track_epochs'])+1)
                assert bestSeed['state_dicts']
                self.mk_new_net_opt(state_dicts = bestSeed['state_dicts'],
                                    seed_id     = bestSeed['seed_id'])
                fprint(f'{line} Reloading Best (Seed = {self.__seed_id}) {line}')
            
            self.net.train()
            
            self.__current_epoch  =  epoch
            
            # main step training weights
            self.set_grads(self.p_change, True)
            
            self.optim_w.zero_grad()
            running_loss = self.get_loss('trainloader', backward=True)
            self.optim_w.step()
            
            if (epoch % epoch_report)==0:
                fprint(f'Epoch : {epoch:8d} (loss = {running_loss:.4e}) (wdl2_mag = {self.get_wdl2_mag():.4e}) (total time = {format_dt(time()-t0)}) (loop = {format_dt(time()-t1)}) (avg. time/epoch = {((time()-t1)/(epoch-prev_epoch)):10.6f})')
                prev_epoch  =  float(epoch)
                t1 = time()
            
            if (epoch % epoch_predict)==0:
                self.__save_special['started']  =  (epoch >= min_saved_epochs)
                self.__save_stats    =  True
                self.global_predict()
                self.__save_stats    =  False
            
            if epoch in saved_epochs:
                assert (time()-t2) >= min_saver_dt
                assert (epoch % epoch_report )==0
                assert (epoch % epoch_predict)==0
                self.saver(self.__mk_save_dict(epoch))
                t2 = time()
            
            if epoch in bestSeed['track_epochs']:
                fprint(f'{line} Evaluating Results Seed {line}')
                current_metric  =  seeder_current_metric()
                if current_metric < bestSeed['record']:
                    fprint(f'    New Record Seed: (new best = {fmt_p(current_metric)}) (old = {fmt_p(bestSeed["record"])})')
                    bestSeed['record']       =  current_metric
                    bestSeed['state_dicts']  =  {'net'    :  To_Str_State_Dict(self.net    .state_dict()),
                                                 'optim_w':  To_Str_State_Dict(self.optim_w.state_dict())}
                    bestSeed['seed_id']      =  int(self.__seed_id)
                else:
                    fprint(f'    Seed Discarded: (metric = {fmt_p(current_metric)}) (best = {fmt_p(bestSeed["record"])})')
                fprint(f'{line} Starting New Seed {line}')
                self.mk_new_net_opt()
                
            if (time()-t0)>breaker:
                break
        
        fprint(f'Finished Training (elapsed time = {format_dt(time()-t0)})')
        self.__record_results('end')
        self.saver(self.__save_special['dict'])
    
    def get_loss(self, tag, backward=False):
        running_loss = 0.
        data         = getattr(self.loaders,tag).data
        K            = 1./len(data)
        for inputs, labels in data:
            inputs, labels  =  inputs.to(self.device), labels.to(self.device)
            loss            =  self.criterion(self.net(inputs) - labels) * K
            if backward:
                loss.backward()
            running_loss  +=  float(loss.item())
        return running_loss
    
    def set_grads(self, A, val):
        assert type(val) == bool
        for p in A:
            p.requires_grad = val
    
    def get_wdl2_mag(self):
        return sum(float((p**2).sum().item()) for p in self.p_change if p.requires_grad)
    
    def global_predict(self, fp = None):
        if fp is None:
            fp = self.fprint
        self.predict('train', fp)
        self.predict('valid', fp)
        self.predict('test' , fp)
    
    def predict(self, dset, fp):
        
        t0      =  time()
        loader  =  { 'train': self.loaders.trainloader ,
                     'valid': self.loaders.full_valid  ,
                     'test' : self.loaders.full_test   }[dset]
        
        fabs    =  torch.abs
        
        total, diff, abstotal, absdiff  =  0., 0., 0., 0.
        
        self.net.eval()
        
        for inputs, labels in loader.data:
            
            inputs, labels  =  inputs.to(self.device), labels.to(self.device)
            
            delta     =  self.net(inputs) - labels
            
            abstotal +=  fabs(labels).sum().item()
            absdiff  +=  fabs(delta ).sum().item()
            total    +=  fabs(labels .sum()).item()
            diff     +=  fabs(delta  .sum()).item()
        
        self.net.train() 
        
        new_stats = { 'diff/total'   : diff    / total     ,
                     '|diff|/total'  : absdiff / total     ,
                     '|diff|/|total|': absdiff / abstotal  ,
                     'epoch'         : self.__current_epoch}
        
        if self.__save_stats:
            if not dset in self.pred_stats: self.pred_stats[dset] = []
            self.pred_stats[dset].append(new_stats)
        
        mk_str = lambda : ' '.join(f'({k} = {(v*100):10.5f} %)' for k,v in new_stats.items() if k!='epoch')
        fp(f"Prediction (dset = {dset:5s}): {mk_str()} (elapsed time = {(time()-t0):10.6f})")
        
        if self.__save_special['started'] and (dset == 'train') and (new_stats['|diff|/|total|'] < self.__save_special['best']):
            self.__save_special['best']  =  new_stats['|diff|/|total|']
            self.__save_special['dict']  =  self.__mk_save_dict(f'MiTrAbDiEp_{self.__current_epoch}')
    
    def __record_results(self, mode):
        if self.args['register_run']:
            qmode     =  self.args['qmode']
            subfolder =  {'ini':  f'started_{qmode}', 'end': f'archive_{qmode}'}[mode]
            
            mk_fname = lambda suffix:  pjoin(   pjoin(dirname(dirname(self.output_folder)),subfolder), # folder_archive
                                            f"{basename(self.fprint.fname).replace('.log',f'{suffix}.dat')}" )
            fname = mk_fname('')
            ii    = 0
            while isfile(fname):
                fname = mk_fname(f'_{ii}')
                ii   += 1
            assert     isdir (dirname(fname))
            assert not isfile(fname)
            
            D_run_info  =  {'last_stats'    : self.pred_stats                        ,
                            'ml_runner_info': {'args'         : self.args            ,
                                            'fname_log'    : self.fprint.fname    ,
                                            'output_folder': self.output_folder   ,
                                            'main_fname'   : self.__main_fname    ,
                                            'log_run_data' : self.fprint.get_all()}}
            
            zip_str_write(fname                         ,
                          apply_pformat(D_run_info)+'\n')
            # with open(fname,'w') as f:
            #     f.write(apply_pformat(D_run_info)+'\n')
            self.fprint(f'--------------- Recorded Results at (mode = {mode}) (fname = {fname}) ---------------')
    
    def __mk_save_dict(self,epoch):
        outpath       =  lambda fname:  pjoin(self.output_folder, 'saved', fname)
        all_loaders   =  [  self.loaders.trainloader  ,
                            self.loaders.full_valid   ,
                            self.loaders.full_test    ]
        
        Awrite        =  []
        fwrite        =  lambda x: Awrite.append(x)
        
        
        fname_net     =  outpath( f'seed_{self.__seed_id}_epoch_{epoch}_statedict_net.dat'                  )
        fname_optim   =  outpath( f'seed_{self.__seed_id}_epoch_{epoch}_statedict_optim_w.dat'              )
        fname_fields  =  outpath( f"seed_{self.__seed_id}_epoch_{epoch}_predicted_{self.args['qmode']}.dat" )
        
        fwrite('\n------------------------- Begin Saver -------------------------')
        self.global_predict(fp = fwrite)
        fwrite( f'---->  Net_Dict       :  {fname_net}'   )
        fwrite( f'---->  Optim_Dict     :  {fname_optim}' )
        fwrite( f'---->  Field Vars     :  {fname_fields}' )
        fwrite('------------------------- End   Saver -------------------------')
        fwrite('')
        
        sd_net      =  To_Str_State_Dict( self.net    .state_dict())
        sd_optim_w  =  To_Str_State_Dict( self.optim_w.state_dict())
        
        net  =  self.net
        net.eval()
        
        D = {}
        for loader in all_loaders:
            
            predicted = []
            for inputs, _ in loader.data:
                field  =  net(inputs.to(self.device)) * self.loaders.label_scale
                predicted.append( fmtTensor(field,tensor_keyword=False) )
            
            D[loader.tag] =  predicted
        
        D = D_pred = fmt_D(D)
        net.train()
        
        return {'fname_net'   :  fname_net   ,
                'fname_optim' :  fname_optim ,
                'fname_fields':  fname_fields,
                'Awrite'      :  Awrite      ,
                'D_pred'      :  D_pred      ,
                'sd_net'      :  sd_net      ,
                'sd_optim_w'  :  sd_optim_w  }
    
    def saver(self, info):
        outpath       =  lambda fname:  pjoin(self.output_folder, 'saved', fname)
        all_loaders   =  [  self.loaders.trainloader  ,
                            self.loaders.full_valid   ,
                            self.loaders.full_test    ]
        
        for line in info['Awrite']:
            self.fwrite(line)
        
        def quick_write(sd,fname):
            zip_str_write(fname       ,
                          str(sd)+'\n')
            # with open(fname,'w') as f:
            #     f.write(str(sd)+'\n')
        
        quick_write( info['sd_net']     , info['fname_net'   ] )
        quick_write( info['sd_optim_w'] , info['fname_optim' ] )
        quick_write( info['D_pred']     , info['fname_fields'] )
        
        if self.write_ref_vars:
            
            self.write_ref_vars = False
            
            fname_base = outpath( f'00_reference_variables.dat' )
            
            D = {}
            
            for loader in all_loaders:
                
                D[loader.tag] =  { 'X_map'                        :  loader.X_coords  ,
                                   'Z_map'                        :  loader.Z_coords  ,
                                   'H_map'                        :  loader.H_coords  ,
                                   f"Labels_{self.args['qmode']}" :  loader.Field_var ,
                                   'full_info'                    :  loader.full_info }
                
                loader.X_coords  = None
                loader.Z_coords  = None
                loader.H_coords  = None
                loader.Field_var = None
            
            zip_str_write(fname_base   ,
                          fmt_D(D)+'\n')
            # with  open(fname_base,'w')  as  f:
            #   f.write(fmt_D(D)+'\n')