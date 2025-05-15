# ------------------------------------------------------------------------
#                  Generic Libraries
# ------------------------------------------------------------------------

from  shutil   import  make_archive
from  os.path  import  abspath
from  os.path  import  join     as  pjoin

# ------------------------------------------------------------------------
#                  Custom Libraries
# ------------------------------------------------------------------------

from  core.external  import  ( FPrint,               #  io_access
                               tstamp, deepdirname,  #  utils
                             )

# ------------------------------------------------------------------------
#                  Output Directory (ML)
# ------------------------------------------------------------------------

output_dir_ml  =  lambda qmode: pjoin(  deepdirname(abspath(__file__),3),
                                       'data'                           ,
                                      f'output_{qmode}'                 )

# ------------------------------------------------------------------------
#                  Modified FPrint Class
# ------------------------------------------------------------------------

class FPrint_ML(FPrint):
    def __init__(self, fname):
        
        # initialize parent class
        super().__init__( fname , check_path = False, track_all = True)

# ------------------------------------------------------------------------
#                  Copy Zipped Folder
# ------------------------------------------------------------------------

def Copy_Zipped(source, target):
    
    make_archive(  pjoin(target, 'python_code')  ,
                   'zip'                         ,
                   source                        )