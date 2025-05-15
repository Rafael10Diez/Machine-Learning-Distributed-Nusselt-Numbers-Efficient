
# ------------------------------------------------------------------------
#                  Generic Libraries
# ------------------------------------------------------------------------
import  zipfile
from    PIL                import  Image                                      as  PIL_Image
from    sys                import  path                                       as  sys_path
from    os.path            import  join                                       as  pjoin
from    os                 import  listdir                                    as  os_listdir
from    os                 import  mkdir                                      as  os_mkdir
from    os.path            import  basename, abspath, dirname, isfile, isdir
import  numpy              as      np
import  matplotlib.pyplot  as      plt
from    matplotlib         import  colors, cm
from    pprint             import  pformat

# ------------------------------------------------------------------------
#                  Custom Libraries
# ------------------------------------------------------------------------

_folder_  =  dirname(abspath(__file__))

sys_path.append(_folder_)
from  tec_processing  import  lmap, lfilter
sys_path.pop()

# ------------------------------------------------------------------------
#                  Basic Definitions
# ------------------------------------------------------------------------

post_output         =  pjoin(dirname(_folder_),'post_output')

apply_pformat       =  lambda x: pformat(x, sort_dicts=False, indent = 2)

cempty              =  255
array               =  np.array
full_listdir        =  lambda x:  sorted([pjoin(x,y) for y in os_listdir(x)])
full_listdir_files  =  lambda x: lfilter(isfile, full_listdir(x))
full_listdir_folders=  lambda x: lfilter(isdir , full_listdir(x))

fmt1                =  lambda x  : str(round(x,1))
fmt_p               =  lambda ref: lambda x,_: fmt1(x/ref*100) + '%'
fmt_x               =  lambda ref: lambda x,_: fmt1(x/ref    )

def updated_folder_path(fname):
    fname  =  fname.split('/')
    fname  =  pjoin(dirname(_folder_),'post_output', *fname[fname.index('tecplot_files'):])
    assert isfile(fname + '.zip')
    return fname

def get_avg(x):
    x = list(x)
    return sum(x)/len(x)

def pop1(A):
    assert (type(A) == list) and (len(A)==1)
    return A[0]

def read_arr(fname):
    with open(fname,'r') as f:
        return [lmap(float,lfilter(None,x.rstrip('\n').split())) for x in f]

my_colormaps = {key: read_arr(pjoin(_folder_,'colorscales',f'{key}.dat')) for key in ['extended_coolwarm', 'standard_coolwarm']}

def get_cmap(cmap_name,MODE_PLOT='s2d'):
    
    scale = my_colormaps[cmap_name]
    
    if    MODE_PLOT in ['pyplt', 's2d']:
        return colors.ListedColormap(scale, name = cmap_name)
    elif  MODE_PLOT == 'mlab':
        return np.array([ list(np.array(row)*cempty)+[cempty] for row in scale])
    else:
        raise Exception(f'ERROR: Unrecognized (MODE_PLOT = {MODE_PLOT})')

def reader(fname):
    with open(fname,'r') as f:
        return [x.rstrip('\n') for x in f]

def writer(x, fname):
    assert type(x) == str
    with open(fname,'w') as f:
        f.write(str(x).rstrip('\n')+'\n')

def mk_dir_p(x):
    assert not isfile(x)
    if not isdir(x):
         mk_dir_p(dirname(x)) # ensure parent folder is created
         os_mkdir(x)          # create current folder

import_json = lambda x: eval(' '.join(reader(x)))

def reader_zip(fname):
    saved_zip   =  fname.removesuffix('.zip') + '.zip'
    local_name  =  basename(fname)
    assert not isfile(fname)
    assert     isfile(saved_zip)
    return zipfile.Path(saved_zip, at = local_name).read_text().split('\n')

# ------------------------------------------------------------------------
#                  Import TecPlot File
# ------------------------------------------------------------------------

class ImportTec:
    def __init__(self, fname):
        
        #  extract numbers from line
        as_floats   =  lambda line: list(map(float,  filter(None,line.split(' '))  ))
        
        #  read filename
        self.fname   =  fname
        self.header  =  []
        A            =  []
        
        for x in reader_zip(fname):
            if not x: continue
            if x.lstrip(' ')[0]=='#':
                self.header.append(x)
            else:
                A.append(x)
        
        self.order  =  A[0].split('=')[1].replace('"','').replace(' ','').split(',')
        
        assert  self.order[:4]  ==  ["X", "Y", "Z", "H"]
        assert  'ZONE' in A[1]
        
        A          =  array( list(map(as_floats,A[2:])) )
        self.data  =  { label: A[:,i] for i,label in enumerate(self.order) }
        
        inds       =  self.mk_inds_2d(self.data['X'], self.data['Z'])
        
        items      =  list(self.data.items())
        for key,values in items:
            self.data[key] = values[inds]

            if key.endswith('_error'):
                error_key                =  key
                abserror_key             =  key.removesuffix('_error') + '_abserror'
        
                self.data[abserror_key]  =  np.fabs(self.data[error_key])
                self.order.append(abserror_key)

        self.__mk_json_about()
        
    def mk_inds_2d(self, X, Z):
        
        ix, Nx  =  self.__inds_1d( X )
        iz, Nz  =  self.__inds_1d( Z )
        
        shape   =  Nx, Nz
        inds    =  np.zeros(shape, dtype = int)
        
        assert np.prod(shape) == len(X) == len(Z)
        
        for k in range(len(X)):
            inds[ix[k],iz[k]]  =  k
        
        return inds
    
    def __inds_1d(self, X):
        
        x1  =  sorted(np.unique(X))
        D   =  dict(zip(  x1              ,
                          range(len(x1))  ,
                       ))
        
        return [D[xx]  for xx in  X] ,  len(x1)
    
    def __mk_json_about(self):
        begin = '--- Begin_About_Tecplotfile ---'
        end   = '--- End___About_Tecplotfile ---'
        iget  = lambda tag: pop1([i for i,x in enumerate(self.header) if tag in x])
        A     = lmap(lambda x: x.lstrip('# ')              ,
                     self.header[(iget(begin)+1):iget(end)])
        self.json_about = eval(' '.join(A))
        
# ------------------------------------------------------------------------
#                  PNG Cropper
# ------------------------------------------------------------------------

class Crop_Png:
    def __init__(self, fname):
        assert fname[-4:] == '.png'
        data = Crop_Png.__crop_array(np.asarray(PIL_Image.open(fname)))
        PIL_Image.fromarray(data).save(fname)
    
    @staticmethod
    def __crop_array(data):
        
        mask   =  np.any(data != cempty, axis = -1)
        
        assert mask.shape == data.shape[:2]
        
        def get_bounds(axis):
            i, = np.where(np.any(mask, axis=axis))
            return i.min(),i.max()+1
        
        i0,L0 = get_bounds(1)
        i1,L1 = get_bounds(0)
        return data[i0:L0,
                    i1:L1]

# ------------------------------------------------------------------------
#                  Plot 2-D Surfaces (s2d)
# ------------------------------------------------------------------------

class Plot_Surf_2D:
    def __init__(self, tecdata, args, width_inches = 6.69):
        
        self.tecdata    =  tecdata
        self.args       =  args
        self.fname_png  =  fname_png = args['fname_png']
        
        X, Z, V         =  [tecdata.data[k].transpose() for k in ('X','Z',args['key_scalar'])]
        
        fig, ax         =  plt.subplots()
        
        plt.rcParams["font.size"]     =  "15"
        plt.rcParams["font.family"]   =  "Times New Roman"
        plt.rcParams['axes.xmargin']  =  0
        
        pcm  =  ax.pcolormesh(  self.toedges(X)                          ,
                                self.toedges(Z)                          ,
                                V                                        ,
                                cmap     =  get_cmap(args['cmap_name'])  ,
                                vmin     =  args['vmin']                 ,
                                vmax     =  args['vmax']                 ,
                                shading  =  'flat'                       )
        
        ax.set_axis_off()
        ax.margins (x=0)
        plt.margins(0,0)

        if 'cbar_location' in args:
            assert args['cbar_location'] in ['top','bottom']
            shrink = 1.2 if '%' in args['fmt_cbar'](0,None) else 0.93
            fig.colorbar(  pcm                                                     ,
                        ax           =  [ax]                                       ,
                        location     =  args['cbar_location']                      ,
                        shrink       =  shrink                                     ,
                        pad          =  0.085                                      ,
                        ticks        =  np.linspace(args['vmin'],args['vmax'], 5)  ,
                        format       =  args['fmt_cbar']                           ,
                        aspect       =  30                                         )
        
        plt.gca().set_aspect('equal', adjustable='box')
        
        fig.savefig(  fname_png                 ,
                      bbox_inches  =  'tight'   ,
                      dpi          =  8*fig.dpi ,
                      pad_inches   =  0         )
        plt.close(fig)
    
    def toedges(self, X):
        
        assert len(X.shape) == 2
        
        def mk_edges(x):
            extrapolate  =  lambda a,b: [2*a-b]
            x            =  list( x[:-1] + 0.5*np.diff(x) )
            return extrapolate(x[0],x[1]) + x + extrapolate(x[-1],x[-2])
        
        for _ in range(2):
            X  =  array(list(map(mk_edges, X))) # this will interpolate to "edge_positions" along every direction
            X  =  X.transpose()                 # first transpose switches to work in remaining axis, second transpose reverts to normal order
        
        return X

# ------------------------------------------------------------------------
#                  Direct Runner
# ------------------------------------------------------------------------

def list_all_info_tec(archive_fname):
    result                =  []
    for info_tec in import_json(archive_fname):
        info_tec['tec_fname'] = updated_folder_path(info_tec['tec_fname'])
        result.append(info_tec)
    return result

def get_allowed_tecfiles(archive_fname, n_ktrials):
    result = {}
    for d in list_all_info_tec(archive_fname):
        if len(d['K_trials']) == n_ktrials:
            if not (d['ind'] in result):
                result[d['ind']] = d['tec_fname']
    return [v for _,v in sorted(result.items())]

def main_runner(tag_run, qmode, n_ktrials):
    
    archive_fname = pjoin(post_output, f'archive_new_tecfiles_tag_run_{tag_run}_qmode_{qmode}.dat')
    assert isfile(archive_fname)

    allowed_tecfiles  =  get_allowed_tecfiles(archive_fname, n_ktrials)
    
    print(f'\nMain Run: (tag_run: {tag_run}) (qmode: {qmode}) (n_ktrials: {n_ktrials})')
    _ = lmap(lambda x: print(f'    {x}'), allowed_tecfiles)


    all_info_tec = lfilter(lambda info_tec: info_tec['tec_fname'] in allowed_tecfiles,
                           list_all_info_tec(archive_fname)                          )
    
    assert len(all_info_tec) == len(allowed_tecfiles)

    memo_td           =  {info_tec['tec_fname']: ImportTec(info_tec['tec_fname']) for info_tec in all_info_tec}
    all_td            =  list(memo_td.values())
    archive_pics      =  []

    for info_tec in all_info_tec:
        tecname  =  info_tec['tec_fname']
        
        assert tecname in allowed_tecfiles
        
        td  =  memo_td[info_tec['tec_fname']]
        
        qmode_keys    =  lfilter(lambda x: ('qmode_' in x) and not ('qmod_alt' in x) , td.order)
        qmod_alt_keys =  lfilter(lambda x:                          'qmod_alt' in x , td.order)
        
        assert     len(qmode_keys) == len(qmod_alt_keys)  == 4
        assert len(set(qmode_keys) &  set(qmod_alt_keys)) == 0
        print(basename(tecname), 'qmode_keys', qmode_keys, 'qmod_alt_keys', qmod_alt_keys)

        for all_key in [qmode_keys   ,
                        qmod_alt_keys,
                        ['H']        ]:

            get_vm      =  lambda f, use_key=all_key, use_td = all_td: f(f(iter(td.data[key].flatten())) for key in use_key for td in use_td)

            def get_rel(f, dns_key):
                result    =  None
                for td in all_td:
                    val     =  get_vm(f, all_key, [td]) / get_vm(get_avg, [dns_key], [td])
                    result  =  val if result is None else f(result, val)
                return result

            if 'H' in all_key:
                scale_cbar  = 1.
                fmt_cbar    = fmt_p(scale_cbar)
                vmin, vmax  =  get_vm(min), get_vm(max)
                # good as ever
                
            else:
                dns_key,   =  lfilter(lambda x: '_dns' in x, all_key)
                dns_avg    =  get_vm(get_avg, [dns_key], [td])
                
                scale_cbar =  dns_avg
                fmt_cbar   =  fmt_x(scale_cbar)

                vmin      =  get_rel(min, dns_key) * dns_avg
                vmax      =  get_rel(max, dns_key) * dns_avg

            for key in all_key:
                if ('Nu_classic' in key) or ('Qdot' in key):
                    vmin = 0.
                elif ('Cf_classic' in key) or ('Fx' in key) or (key=='H'):
                    vmax = max(abs(vmin), abs(vmax))
                    vmin = -vmax
                else:
                    raise Exception(f'Unrecognized (key = {key})')
                fname_png = td.fname.replace('tecplot_files','pics_s2d_2d').replace('.tec',f'_{key}.png')
                mk_dir_p(dirname(fname_png))
                args  =  {'key_scalar'   :  key                ,
                          'cmap_name'    :  'extended_coolwarm',
                          'vmin'         :  vmin               ,
                          'vmax'         :  vmax               ,
                          'cbar_location':  'bottom'           ,
                          'fmt_cbar'     :  fmt_cbar           ,
                          'fname_png'    :  fname_png          }
                Plot_Surf_2D(td, args)
                args['info_tec']   = info_tec
                args['td_fname']   = td.fname 
                args['fmt_cbar']   = str('fmt_cbar')
                args['scale_cbar'] = scale_cbar 
                archive_pics.append(args)
        
    writer(apply_pformat(archive_pics)                                     ,
           pjoin(dirname(archive_fname), f"pics_{basename(archive_fname).replace('new_tecfiles_','')}"))
    return archive_pics

# ------------------------------------------------------------------------
#                  Direct Runner
# ------------------------------------------------------------------------

if __name__ == '__main__':
    all_archive_pics                 =  {}
    tag_run                          =  'rnd12'
    qmode                            =  'Nu_classic'
    n_ktrials                        =  2
    all_archive_pics[tag_run,qmode]  =  main_runner(tag_run, qmode, n_ktrials)
    