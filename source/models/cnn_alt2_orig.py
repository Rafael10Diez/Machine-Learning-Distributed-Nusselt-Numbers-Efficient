# ------------------------------------------------------------------------
#                  Generic Libraries
# ------------------------------------------------------------------------

from    os.path  import  abspath
import  torch

# ------------------------------------------------------------------------
#                  Basic Definitions
# ------------------------------------------------------------------------

nn  =  torch.nn

# ------------------------------------------------------------------------
#                  Auxiliary Functions
# ------------------------------------------------------------------------

# as_tuple2: ensure integer variables are formatted in tuple-format (?,?)
def as_tuple2(x):
    if    type(x) in [tuple,list]:
        return tuple(x)
    elif  type(x) == int:
        return (x,x)
    else:
        raise Exception(f'Unrecognized variable type (x = {x}) (type(x) = {type(x)})')


# get_padding_odd_simple: calculate padding for convolution with odd-kernels
def get_padding_odd_simple(K_, dilation):

    # assert data types
    assert  type(K_) == type(dilation) == tuple

    # ensure all kernels are odd
    assert  all( (k&1)  for k in  K_ )

    # return padding
    return  tuple( (k//2)*d  for k,d in  zip(K_,dilation) )

# ------------------------------------------------------------------------
#                  Circular Convolution
# ------------------------------------------------------------------------

class Conv2d_Circular(nn.Module):
    def __init__(self, C_in, C_out, K, dilation, groups = 1, bias = True):
        super().__init__()

        K         =  as_tuple2(K)
        dilation  =  as_tuple2(dilation)

        assert all( k_&1 for k_ in K)

        self.A  =  nn.Conv2d( C_in                                                ,
                              C_out                                               ,
                              K                                                   ,
                              dilation      =  dilation                           ,
                              padding       =  get_padding_odd_simple(K,dilation) ,
                              padding_mode  =  'circular'                         ,
                              bias          =  bias                               ,
                              groups        =  groups                             )

    def forward(self, x):
        return self.A(x)

def SepConv2d(C_in, C_out, K, dilation, groups = 1, bias = True):
    assert bias
    groups = min(C_in, C_out)
    return nn.Sequential(  Conv2d_Circular(C_in , C_out, K, dilation, groups = groups, bias = bias) , nn.BatchNorm2d(C_out), nn.PReLU(C_out),
                           Conv2d_Circular(C_out, C_out, 1, 1       , groups = 1     , bias = bias) , nn.BatchNorm2d(C_out), nn.PReLU(C_out), nn.BatchNorm2d(C_out), nn.PReLU(C_out),
                        )

# ------------------------------------------------------------------------
#                  Main CNN
# ------------------------------------------------------------------------

class Net_Circular_Alt2_Orig(nn.Module):
    def __init__(self, args, seed_torch):
        seed_torch(torch)
        super().__init__()

        assert type(args) == dict

        # register arguments
        self.args  =  args
        dseq       =  args['dseq']
        Cseq       =  args['Cseq']

        arr  = lambda d: [d,d] if type(d)==int else list(d)

        def does_grow(d1,d2):
            geq  = lambda v1,v2: v1 in [v2, 2*v2]
            d1   = arr(d1)
            d2   = arr(d2)
            assert all(geq(x,y) for x,y in    [d1,d2]) # check          dilation in streamwise direction is larger (or equal)
            return all(geq(b,a) for a,b in zip(d1,d2)) # check upcoming dilation                         is larger (or equal)

        assert (len(Cseq)-1) == len(dseq)                                      # this is important for the main-loop
        assert all(does_grow(dseq[i],dseq[i+1]) for i in range(len(dseq)-1))
        assert (Cseq[0] == Cseq[-1] == 1) and (arr(dseq[0]) in [arr(1),arr(2),arr((2,1)),arr((1,2))])

        # def arbitrary_check():
        #     def equal_seq(A,B):
        #         assert all((arr(a)==arr(b)) for a,b in zip(A,B))
        #         return True
        #     assert equal_seq(dseq , [  1   ,  #        SepConv2d( 1, C, 3,   1    )
        #                              ( 2,1),  #        SepConv2d( C, C, 3, ( 2,1) )
        #                              ( 4,2),  #        SepConv2d( C, C, 3, ( 4,2) )
        #                              ( 4,2),  #        SepConv2d( C, C, 3, ( 4,2) )
        #                              ( 8,4),  #        SepConv2d( C, C, 3, ( 8,4) )
        #                              ( 8,4),  #        SepConv2d( C, C, 3, ( 8,4) )
        #                              (16,8),  #        SepConv2d( C, C, 3, (16,8) )
        #                              (16,8)])  #  Conv2d_Circular( C, 1, 3, (16,8)   , bias = False)
        #     C = 20
        #     assert equal_seq(Cseq , [1,       #        SepConv2d( 1, C, 3,   1    )
        #                              C,       #        SepConv2d( C, C, 3, ( 2,1) )
        #                              C,       #        SepConv2d( C, C, 3, ( 4,2) )
        #                              C,       #        SepConv2d( C, C, 3, ( 4,2) )
        #                              C,       #        SepConv2d( C, C, 3, ( 8,4) )
        #                              C,       #        SepConv2d( C, C, 3, ( 8,4) )
        #                              C,       #        SepConv2d( C, C, 3, (16,8) )
        #                              C,       #  Conv2d_Circular( C, 1, 3, (16,8)   , bias = False)
        #                              1])
        # arbitrary_check()

        L_loop = len(dseq)
        last   = L_loop-1

        A      = []
        for i in range(L_loop):
            assert i <= last
            c1,c2 = Cseq[i], Cseq[i+1]
            d     = dseq[i]
            f,K   = [SepConv2d,3] if i<last else [Conv2d_Circular,1]
            # assert K == 3
            A.append(f(c1, c2, K, d if K>1 else 1))

        self.A         =  nn.Sequential(*A)

        # # factory for convolutions:
        # # based on  C:\Users\rafae\surfdrive\workspace\code\202109_stag_interp_v3\ml_adjasb4_masks_use_ExistingGrid\v5\runner.py
        # self.A  =  nn.Sequential(  SepConv2d( 1, C, 3,   1    ) , # Conv2d(1,C,3,stride=(2,1)), PReLU+BN2d(C),
        #                            SepConv2d( C, C, 3, ( 2,1) ) , # Conv2d(C,C,3,stride=2    ), PReLU+BN2d(C),
        #                            SepConv2d( C, C, 3, ( 4,2) ) , # Conv2d(C,C,3             ), PReLU+BN2d(C),
        #                            SepConv2d( C, C, 3, ( 4,2) ) , # Conv2d(C,C,3,stride=2    ), PReLU+BN2d(C),
        #                            SepConv2d( C, C, 3, ( 8,4) ) , # Conv2d(C,C,3             ), PReLU+BN2d(C),
        #                            SepConv2d( C, C, 3, ( 8,4) ) , # Conv2d(C,C,3,stride=2    ), PReLU+BN2d(C),
        #                            SepConv2d( C, C, 3, (16,8) ) , # Conv2d(C,C,3             ), PReLU+BN2d(C),
        #                      Conv2d_Circular( C, 1, 3, (16,8)   , bias = False),                    # Conv2d(C,1,3,bias=False  ),
        #                         )

    def forward(self, x):
        return self.A(x)

    def get_file(self):
        return abspath(__file__)