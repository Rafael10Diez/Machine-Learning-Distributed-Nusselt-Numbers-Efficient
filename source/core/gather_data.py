# ------------------------------------------------------------------------
#                  Generic Libraries
# ------------------------------------------------------------------------

from    os.path  import  join                                       as  pjoin
from    sys      import  path                                       as  sys_path
from    os.path  import  basename, abspath, dirname, isfile, isdir
from    os       import  listdir
from    time     import  time
from    math     import  prod
import  torch 

# ------------------------------------------------------------------------
#                  Root Directory (interp + ml)
# ------------------------------------------------------------------------

_folder_           =  dirname(abspath(__file__))
root_dir           =  dirname(dirname(_folder_))
interp_output_dir  =  pjoin(root_dir, 'orig_data')

# ------------------------------------------------------------------------
#                  Custom Libraries
# ------------------------------------------------------------------------

sys_path.append(_folder_)
from external import lmap, lfilter, reader, format_dt, pop1
sys_path.pop()

# ------------------------------------------------------------------------
#                  Basic Definitions
# ------------------------------------------------------------------------

# utilities
almost         =  lambda a, b   :  torch.abs(a-b).max().item() < 1e-10
linspace       =  lambda a, b, n:  torch.linspace(a, b, n, dtype = torch.double)

listdir_full   =  lambda x  : sorted([pjoin(x,y) for y in listdir(x)])

def find_once(tag , A):
    # find element in A[i] that contains "tag"
    n = 0  # number of repetitions
    for x in A:
        if tag in x:
            result  = x            # record line found
            n      += x.count(tag) # count number of matches
    assert n == 1                  # assert only one match was found
    return result

# ------------------------------------------------------------------------
#                  Load Single Tecplot File
# ------------------------------------------------------------------------
#     (a staggered DNS grid is described by multiple files)

class Load_Single_Tec:
    def __init__(self, fname):
        
        with open(fname,'r') as f:
            # read header, zone
            getline   =  lambda: next(iter(f)).rstrip('\n')
            header    =  getline()
            zone      =  getline()
            
            # read matrix
            A         =  []
            for line in f:
                A.append(lmap(float,
                              filter(None,line.rstrip('\n').split(' '))
                             ))
        
        # define order
        order  =  header.split('=')[1].replace('"',' ').replace(',',' ')
        order  =  lfilter(None,order.split(' '))
        
        # convert to torch.tensor
        A      =  torch.tensor(A, dtype = torch.double)
        
        # register all variables
        self.fname   =  fname
        self.header  =  header
        self.zone    =  zone
        self.order   =  order
        self.data    =  { key:A[:,i] for i,key in enumerate(order) }
        
        self.make_all_2d()
    
    def make_all_2d(self):
        
        inds_2d = self.find_inds_2d(self.data['X'],
                                    self.data['Z'])
        
        for k,v in self.data.items():
            self.data[k] = v[inds_2d]
    
    @staticmethod
    def find_inds_2d(x, z):
        
        # mk_inds:
        def mk_inds(a, a1):
            to_ind  =  {val.item():i for i,val in enumerate(a1)}
            return  [to_ind[val.item()]  for val in  a]
        
        # 1d distributions
        x1   =  torch.unique(x)
        z1   =  torch.unique(z)
        
        # shape 2d
        shape2 = len(x1), len(z1)
        
        # assert shapes are consistent
        #     (floating point errors in "unique" would be detected here)
        assert prod(shape2) == len(x) == len(z)
        
        # build index maps
        #     (convert xz positions into indexes)
        #     e.g., [0.5 ,1.5, 0.5, ... ] -> [0, 1, 0,...]
        ix  =  mk_inds(x, x1)
        iz  =  mk_inds(z, z1)
        
        # define final array
        #     inds = [[0,1,5,6],[9,3,2]]
        #       (this indicates how elements should be regrouped to create 2d array)
        
        inds = torch.zeros( shape2 , dtype = torch.long )
        
        for k in range(len(x)):
            # inds[a,b] goes at position (ix[k],iz[k])
            inds[ix[k],iz[k]] = k
        
        return inds
        

# ------------------------------------------------------------------------
#                  Load All Tecplot Data (for one DNS case)
# ------------------------------------------------------------------------
#     (this class is intended to be a data-holder attribute for
#      ML_Data_Single_Case)

class Load_All_Tec:
    def __init__(self, subfolder, side, qmode, fprint, apply_check = True):
        
        self.fprint           =  fprint
        self.subfolder        =  subfolder
        self.full_data_folder =  pjoin( interp_output_dir, self.subfolder )
        
        self.side             =  side
        
        self.load_data ()
        if apply_check:
            self.check_data()
        self.__mk_internal_vars()
        
        if   qmode == 'Fx':
            self.Fx  =  self.__Fx
        
        elif qmode == 'Qdot':
            self.Qdot  =  self.__Qdot
        
        elif qmode == 'Cf_classic':
            self.Cf_classic = self.__Cf_classic
        
        elif qmode == 'St_classic':
            self.St_classic = self.__St_classic
        
        elif qmode == 'Nu_classic':
            self.Nu_classic = self.__Nu_classic
        
        else:
            raise Exception(f'Unrecognized (qmode = {qmode}) (expected qmode in [Fx, Qdot, Cf_classic, St_classic, Nu_classic])')
    
    def load_data(self):
        
        for var in 'UVWPT':
            fname = pjoin( self.full_data_folder, self.subfolder + f'_quadint_interp2center_{self.side}_{var}.tec' )
            setattr(self, f'tec_{var}', Load_Single_Tec(fname))
    
    def check_data(self):
        x  =  self.x_2d  =  self.tec_P.data['X']
        y                =  self.tec_P.data['Y']
        z  =  self.z_2d  =  self.tec_P.data['Z']
        h =   self.h_2d  =  self.tec_P.data['H']
        
        for key in 'UVWPT':
            data  =  getattr(self, f'tec_{key}').data
            # print(key,"torch.abs(y-data['Y']).max().item()",torch.abs(y-data['Y']).max().item())
            assert all( almost(arr,data[key]) for arr,key in ((x,'X'),(z,'Z'),(h,'H'),(y,'Y')) )
    
    def __load_bulk_data(self):
        fname = pop1(lfilter(lambda x: basename(x)[-11:]=='_DUplus.dat',
                             listdir_full(self.full_data_folder)       ))
        A     = reader(fname)
        
        def get_val(tag,f=float):
            n = 0
            for x in A:
                if tag in x:
                    assert ':' in x
                    result  = f(x.split(':')[1])
                    n      += x.count(tag)
            assert n == 1
            return result
        
        self.props                              =  {}
        self.props['Ret']   = Ret               =  get_val(' Ret ')
        self.props['L_xyz'] = __Lx, Ly, __Lz    =  get_val(' L_xyz ',f=lambda x: lmap(float,x.replace('[','').replace(']','').split(',')))
        self.props['N_xyz'] = __Nx,  _, __Nz    =  get_val(' N_xyz ',f=lambda x: lmap(int  ,x.replace('[','').replace(']','').split(',')))
        self.props['Ub']                        =  get_val(' Bulk Velocity')
        self.props['Tb']                        =  get_val(' Bulk Temperature')
        self.props['Pr']                        =  get_val(' Pr ')
        self.props['dpdx']                      =  1.
        self.props['source_q']                  =  1.
        self.props['rho']                       =  1.
        self.props['cp']                        =  1.
        self.props['Tw']                        =  0.
        
        self.props['__A_elem']                  =  (__Lx * __Lz)/(__Nx * __Nz)
        
        self.props['nu']   =  nu                =  1./Ret
        self.props['Reb']                       =  self.props['Ub'] * Ly / nu
        
        self.fprint('    -------------------------------------------------')
        self.fprint('                    Reading Data File')
        self.fprint('    -------------------------------------------------')
        
        self.fprint(f'        fname  = {fname}')
        for k,v in self.props.items():
            self.fprint(f'        {k:5s}  = {v}')
        self.props['WARNING'] = "        " + f"WARNING: the following constant properties have been assumed (rho = {self.props['rho']}) (cp = {self.props['cp']}) (Tw = {self.props['Tw']}) (dpdx = {self.props['dpdx']}) (source_q = {self.props['source_q']})".replace(' ','_')
        self.fprint(self.props['WARNING'])
        
    def __mk_internal_vars(self):
        self.__mk_Fx()
        self.__mk_Qdot()
        self.__load_bulk_data()
        self.__mk_Cf_St_Nu_classic()
    
    def __mk_Fx(self):
        
        data_U  =  self.tec_U.data
        data_V  =  self.tec_V.data
        data_W  =  self.tec_W.data
        data_P  =  self.tec_P.data
        
        self.__Fx    =  ( data_U["inv_Ret_2.0_dUdx_nx"] + data_U["inv_Ret_1.0_dUdy_ny"] + data_U["inv_Ret_1.0_dUdz_nz"] + 
                                                          data_V["inv_Ret_1.0_dVdx_ny"] + 
                                                                                          data_W["inv_Ret_1.0_dWdx_nz"] + 
                          data_P["minus_P_nx"]          )
    
    def __mk_Qdot(self):
        
        data_T       =  self.tec_T.data
        self.__Qdot  =  ( data_T["inv_Pet_dTdx_nx"] +
                          data_T["inv_Pet_dTdy_ny"] +
                          data_T["inv_Pet_dTdz_nz"] )
    
    def __mk_Cf_St_Nu_classic(self):
        
        rho     =  self.props['rho']
        Ub      =  self.props['Ub']
        cp      =  self.props['cp']
        Tw      =  self.props['Tw']
        Tb      =  self.props['Tb']
        Pr      =  self.props['Pr']
        Reb     =  self.props['Reb']
        A_elem  =  self.props['__A_elem']
        
        assert (hasattr(self.__Fx  ,'shape') and
                hasattr(self.__Qdot,'shape') and
                (type(rho) ==
                 type(Ub ) == 
                 type(cp ) == 
                 type(Tw ) == 
                 type(Tb ) == 
                 type(Pr ) == 
                 type(Reb) ==  float)
               )
        
        self.__Cf_classic  =  2*self.__Fx   / A_elem / (rho*(Ub**2))
        self.__St_classic  =    self.__Qdot / A_elem / (rho*Ub*cp*(Tb-Tw))
        self.__Nu_classic  =  self.__St_classic * (0.5*Reb) * Pr           # remember that Reb is defined with Ly above
        
        def avg(tag, all_x, a, b):
            x  =  float(torch.mean(all_x))
            print(tag, f'{x:.8f}')
            assert (0.5*min(a,b)) < x < (1.5*max(a,b)), ('tag,x,a,b: ',tag,x,a,b)
        
        avg('__Cf_classic', self.__Cf_classic,  0.0150 ,  0.0175 )
        avg('__Nu_classic', self.__Nu_classic, 14.01   , 14.83   )
        avg('__St_classic', self.__St_classic,  0.0068 ,  0.0077 )
    
    def _get_full_export(self):
        result = {}
        for var_ in 'UVWPT':
            data = getattr(self, f"tec_{var_}").data
            for key in data:
                if not key in result:
                    result[key] = data[key]
        result.update(dict(Cf_classic  =  self.__Cf_classic,
                           Fx          =  self.__Fx        ,
                           Qdot        =  self.__Qdot      ,
                           Nu_classic  =  self.__Nu_classic))
        return {key: val.tolist()  for key,val in  result.items()}, self.full_data_folder

# ------------------------------------------------------------------------
#                  Build ML Data (one case)
# ------------------------------------------------------------------------

class ML_Data_Single_Case:
    def __init__(self, subfolder, ind, side, qmode, fprint):
        
        self.all_tec   =  Load_All_Tec(subfolder, side, qmode, fprint)
        self.info      = {'ind'         : ind                 ,
                          'props'       : self.all_tec.props  }
        
        # fetch images
        self.maps = {}
        self.maps['X']    =  self.fmt_bc( self.all_tec.x_2d )
        self.maps['Z']    =  self.fmt_bc( self.all_tec.z_2d )
        self.maps['H']    =  self.fmt_bc( self.all_tec.h_2d )
        self.maps[qmode]  =  self.fmt_bc(getattr(self.all_tec,qmode))
        
        assert ( self.maps['X']  .shape == 
                 self.maps['Z']  .shape == 
                 self.maps['H']  .shape == 
                 self.maps[qmode].shape )
    
    def fmt_bc(self, x):
        
        # add channel to 2d shape:
        #   (x,z) -> (1,1,x,z)
        
        # get shape
        shape  =  x.shape
        
        # assert it's 2d
        assert len(shape) == 2 
        
        new = x.view(1,1,*shape)
        
        assert almost(new[0,0,:,:], x)
        
        return new

# ------------------------------------------------------------------------
#                  Dataset Partition (Unified)
# ------------------------------------------------------------------------

# DataSplit:
#    hold a unified dataset (train/valid/test)

class DataSplit:
    def __init__(self, all_ml_cases, tag, qmode, as_float):
        
        # register tag, as_float
        self.tag          =  tag
        self.qmode        =  qmode
        self.as_float     =  as_float
        
        self.all_info     =  lmap(lambda x: x.info,
                                  all_ml_cases    )
        
        # mk_tensors
        self.X_input       =  self.gather_data('X'  , all_ml_cases)
        self.Z_input       =  self.gather_data('Z'  , all_ml_cases)
        self.H_input       =  self.gather_data('H'  , all_ml_cases)
        self.labels_qmode  =  self.gather_data(qmode, all_ml_cases)
        
        # assert tensors have the same shape
        self.check_shapes()
        
        assert (len(self.X_input     )  == 
                len(self.Z_input     )  == 
                len(self.H_input     )  ==  
                len(self.labels_qmode)    )
    
    def gather_data(self, name, all_ml_cases):
        
        # collect tensors (with property = name)
        A  =  [ml.maps[name] for ml in all_ml_cases]
        
        # check whether to convert results into single precision
        if self.as_float:
            A  =  [t.float() for t in A]
        
        return A
    
    def check_shapes(self):
        
        # pick reference shape
        shape = self.H_input[0].shape
        
        # assert shape is 1+1+2d
        assert (len(shape) == 4) and (shape[0]==shape[1]==1)
        
        # assert all shapes are identical (to ref. shape)
        for A in [self.X_input, self.Z_input, self.H_input, self.labels_qmode]:
            for t in A:
                assert t.shape == shape

# ------------------------------------------------------------------------
#                  Global ML Data
# ------------------------------------------------------------------------
 
class All_Datasets:
    def __init__(self, args, fprint):
        
        # Print Header:
        fprint( '\n--------------------------------------------------\n    Building Datasets\n--------------------------------------------------\n')
        fprint( '\nInput Arguments (All_Datasets):')
        for k,v in args.items():
            if k == 'folders':
                fprint(f'    {k:15s} :', '__only_to_file__')
                for vv in v:
                    fprint(' '*23 + f'{vv}', '__only_to_file__')
            else:
                fprint(f'    {k:15s} :  {v}')
        
        # track initial time
        t0 = time()
        
        # register arguments
        self.args  =  args
        
        # load all data
        inds_used     =  set(args['inds_train'] + args['inds_valid'] + args['inds_test'])
        all_ML_data   =  { i:ML_Data_Single_Case(args['folders'][i], i, args['side'], args['qmode'], fprint)  for i in  inds_used }
        
        # make one dataset (train/valid/test)
        mk_dataset    =  lambda inds, name: DataSplit( [all_ML_data[i] for i in inds] , name, args['qmode'], args['as_float'])
        
        # split into datasets
        self.test     =  mk_dataset( args['inds_test']  , 'test'  )
        self.valid    =  mk_dataset( args['inds_valid'] , 'valid' )
        self.train    =  mk_dataset( args['inds_train'] , 'train' )
        
        fprint(f'Elapsed Time: {format_dt(time()-t0)} (build train/valid/test datasets)')
