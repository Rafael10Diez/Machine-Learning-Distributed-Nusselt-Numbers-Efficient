# ------------------------------------------------------------------------
#                  Generic Libraries
# ------------------------------------------------------------------------

import  torch
from    torchvision.transforms.functional  import hflip as  H_Flip

# ------------------------------------------------------------------------
#                  Custom Libraries
# ------------------------------------------------------------------------

from core.external  import  lmap, as_list, pop1

# ------------------------------------------------------------------------
#                  Random Seed
# ------------------------------------------------------------------------

from  os.path  import  join              as  pjoin
from  os.path  import  dirname, abspath

# ------------------------------------------------------------------------
#                  Basic Definitions
# ------------------------------------------------------------------------

clone           =  torch.clone
searchsorted    =  torch.searchsorted

# # unused
# # searchsorted(a, v, right = ?)
# #   left:             v =< a[i]
# #   right:  a[i-1] <= v
#
# fbegin  =  lambda a,v: searchsorted(a, v, right=False).item()
#
# # fend: range a[:x] lower than "v"
# fend    =  lambda a,v: searchsorted(a, v, right=True ).item()

def add_hflip(arr0, to_list = False):
    assert type(arr0) == list

    arr  =  arr0  +  lmap(H_Flip, arr0)

    arr  =  lmap(torch.clone, arr)

    if to_list:
        return [ as_list(t,float) for t in arr]
    else:
        return arr

# ------------------------------------------------------------------------
#                  Base Initializer (Loaders)
# ------------------------------------------------------------------------

class Base_Loader:
    def __init__(self, dset, tag):

        # register tag
        self.tag    =  tag

        # load inputs (H_map)
        inputs      =  dset.H_input

        # find labels (force or heat transfer)
        labels  =  dset.labels_qmode

        # assert inputs and labels are a list (of 1+3d tensors)
        assert type(inputs) == type(labels) == list

        # add flipped images
        self.data  =  list(zip(  add_hflip(inputs)  ,
                                 add_hflip(labels)  ,
                              ))

        # save field variables
        self.X_coords   =  add_hflip(dset.X_input , to_list = True)
        self.Z_coords   =  add_hflip(dset.Z_input , to_list = True)
        self.H_coords   =  add_hflip(dset.H_input , to_list = True)
        self.Field_var  =  add_hflip(labels       , to_list = True) # (labels have not been edited yet)

        self.full_info  =  list(dset.all_info) + list(dset.all_info)

# ------------------------------------------------------------------------
#                  Train Loader
# ------------------------------------------------------------------------

class Train_Loader(Base_Loader):
    def __init__  (self, dset, batch_size, tag):
        assert batch_size is None
        super().__init__(dset, tag)

        self.batch_size   =  batch_size
        self.total_numel  =  sum( labels.numel() for _,labels in self.data )

# ------------------------------------------------------------------------
#                  Full Loader (for prediction)
# ------------------------------------------------------------------------

class Full_Loader(Base_Loader):
    def __init__  (self, dset, tag):
        super().__init__(dset, tag)

# ------------------------------------------------------------------------
#                  Make All Loaders
# ------------------------------------------------------------------------

class All_Loaders:
    def __init__(self, all_data, batch_size):

        # register type of analysis (force/heat_transfer/Cf/St/Nu)
        self.qmode        =  pop1(list(set([all_data.train.qmode,
                                            all_data.valid.qmode,
                                            all_data.test .qmode])))

        # make train loader
        self.trainloader  =  Train_Loader(all_data.train, batch_size, 'train')

        self.full_valid   =  Full_Loader( all_data.valid, 'valid')
        self.full_test    =  Full_Loader( all_data.test , 'test')

        self.input_scale  =  max( torch.abs(inputs).max() for inputs,_      in self.trainloader.data )
        self.label_scale  =  max( torch.abs(labels).max() for _     ,labels in self.trainloader.data )

        self.normalize(self.trainloader)
        self.normalize(self.full_valid )
        self.normalize(self.full_test  )

    def normalize(self, loader):

        data  =  loader.data

        for i in range(len(data)):

            inputs, labels  =  data[i]
            data[i]         =  ( inputs / self.input_scale ,
                                 labels / self.label_scale )
