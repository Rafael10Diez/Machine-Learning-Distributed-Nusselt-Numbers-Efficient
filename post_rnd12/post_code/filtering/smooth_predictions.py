# ------------------------------------------------------------------------
#                  Generic Libraries
# ------------------------------------------------------------------------
import  torch
from    time    import  time
from    math    import  pi  as  _PI
from    pprint  import  pformat

# ------------------------------------------------------------------------
#                  Basic Definitions
# ------------------------------------------------------------------------

nn             =  torch.nn
format_dt      =  lambda x:  "%02d:%02d:%02d [hh:mm:ss]" % (x//3600, (x%3600)//60, x%60)
apply_pformat  =  lambda x: pformat(x, sort_dicts=False, indent = 2)

# ------------------------------------------------------------------------
#                  Combine Predictions
# ------------------------------------------------------------------------

class _CombinedPreds(nn.Module):
    def __init__(self, all_x, device, K = 10):
        super().__init__()
        assert all(type(x) == list for x in all_x)
        self.all_x = all_x = [torch.tensor(x                            ,
                                           dtype         = torch.float64,
                                           device        = device       ,
                                           requires_grad = False        ) for x in all_x]
        shape2,            = set(x.shape for x in all_x)
        self.all_a         = nn.Parameter(torch.zeros((len(all_x),*shape2)  ,
                                                      dtype  = torch.float64,
                                                      device = device       ))

        # adjustable saturation constants (commented, since they didn't improve results)
        #self.all_K        = nn.Parameter(torch.zeros((len(all_x),*shape2)  ,
        #                                             dtype  = torch.float64,
        #                                             device = device       ))
        self.K = K
        assert type(self.K) == int

    def forward(self):
        sigmoid = lambda a: torch.exp(a)#1 + a/(self.K+torch.abs(a)) # 1 + a/(10+torch.abs(K)+torch.abs(a))
        all_s   = [sigmoid(a) for a in self.all_a]  # ,K in zip(self.all_a,self.all_K)]
        denom   =  sum(all_s)
        return sum(x*(s/denom) for x,s in zip(self.all_x,all_s))

class Smooth_Predictions:
    def __init__(self, all_x, args = None):

        if args is None:
            args = { 'loss_type': 'lapla' ,
                     'device'   : 'cuda'  ,
                     'lr_ini'   : 1e-3    ,
                     'lr_tol'   : 1e-12   ,
                     'k_less'   : 0.5     ,
                     'k_more'   : 1.8     ,
                     'max_iters': 10000   }

        self.args    =   args
        sq           =   lambda x: (x**2).sum()
        self.Jcost   =  {'mag'  :  sq                                                       ,
                         'grad' :  lambda x: sum(sq(torch.roll(x, 1,i)-x) for i in range(2)),
                         'lapla':  lambda x: sum(sq(torch.roll(x, 1,i)+
                                                    torch.roll(x,-1,i)-
                                                    2*x                 ) for i in range(2)),
                        }[args['loss_type']]
        self.comb_preds  =  _CombinedPreds(all_x, args['device'])
        self.runner()

    def runner(self):
        line = '----------------------------------------'
        print(line)
        print('        Smoothing Predictions')
        print(line)
        print('')
        print('    args = ')
        for line in apply_pformat(self.args).split('\n'):
            print('            '+line)
        print('')

        lr          =  self.args['lr_ini']
        k_less      =  self.args['k_less']
        k_more      =  self.args['k_more']
        lr_tol      =  self.args['lr_tol']
        max_iters   =  self.args['max_iters']
        comb_preds  =  self.comb_preds

        assert lr        <= 1
        assert k_less    <= 0.5 < 1.5 <= k_more
        assert lr_tol    <= 1e-12
        assert max_iters >= 10000

        p_change  =  [p for p in comb_preds.parameters() if p.requires_grad]
        mk_p_old  =  lambda: [torch.clone(p.data) for p in p_change]
        f_Jfloat  =  lambda: float(self.Jcost(comb_preds()))

        J_old     =  f_Jfloat()
        p_old     =  mk_p_old()

        t0  =  t1  =  time()
        iii        =  0
        mk_print   =  lambda tag, iii, J_old, lr: print(f'{tag:18s}: {iii:6d} (Jcost = {J_old:13.6e}) (lr = {lr:13.6e}) (Total Elapsed Time: {format_dt(time()-t0)}) (iters/sec = {(iii/max(1e-12,time()-t0)):7.2f})')
        mk_print('Initial Iteration', iii, J_old, lr)
        while lr > lr_tol:
            iii+=1
            if iii>max_iters: break

            # zero_grad
            for p in p_change:
                if not p.grad is None:
                    p.grad *= 0.

            # backward
            self.Jcost(comb_preds()).backward()

            # update
            for p in p_change:
                if not p.grad is None:
                    p.data -= lr * p.grad

            J_now  =  f_Jfloat()
            if J_now < J_old:
                lr    *= k_more
                # define old (next iteration)
                J_old  =  J_now
                p_old  =  mk_p_old()
            else:
                lr    *= k_less
                # reverse p_change
                for p,po in zip(p_change,p_old):
                    p.data  *=  0.
                    p.data  +=  po
            if (time()-t1)>10.:
                t1 = time()
                mk_print('Iteration'        , iii, J_old, lr)
        mk_print(        'Final   Iteration', iii, J_old, lr)
        print('')
        self.result = comb_preds().cpu().detach().numpy()

# ------------------------------------------------------------------------
#                  Direct Runner (testing)
# ------------------------------------------------------------------------

if __name__ == '__main__':

    def mk_all_x(a=1, b=100, n0=20, n1=20,rdiv=5):
        # x      = torch.linspace(-2, 2,n0).view(-1, 1)
        # y      = torch.linspace(-1, 3,n1).view( 1,-1)
        # base   = (a-x)**2 + b*((y-x**2)**2)
        x      = torch.linspace(0, _PI,n0)[:-1].view(-1, 1)
        y      = torch.linspace(0, _PI,n1)[:-1].view( 1,-1)
        base   = (torch.sin(x)**2) * (torch.sin(y)**2)
        spikes = (torch.rand(base.shape)*(base.max()-base.min())-0.5)*0.8
        spikes[:       ,(n1//rdiv):] = 0.
        spikes[(n0//rdiv):,:       ] = 0.
        result = [base + torch.roll(spikes,shifts=(rdiv  , rdiv  ),dims=(0,1)) + torch.rand(base.shape)*1e-6,
                 base + torch.roll(spikes,shifts=(rdiv*2, rdiv*2),dims=(0,1)) + torch.rand(base.shape)*1e-6]
        return list(map(lambda x: x.cpu().detach().numpy().tolist(), result))
    from  os.path  import  abspath, dirname
    from  os.path  import  join              as  pjoin
    def print_comps(all_x, comb_preds,loss_type, print_all_x):
        import matplotlib.pyplot as plt
        D = {'mag'  : 'loss  = y^2'            ,
             'grad' : 'loss  = grads(y)^2 '    ,
             'lapla': 'loss = laplacians(y)^2 '}
        if print_all_x:
            vmin = torch.tensor(all_x,dtype=torch.double).min()
            vmax = torch.tensor(all_x,dtype=torch.double).max()
            for x,tag in zip(all_x                                      + [comb_preds().cpu().detach().numpy().tolist()],
                             [f'Input_{i}' for i in range(len(all_x))] + [f'Combined_Predictions_{loss_type}']       ):
                fig,ax = plt.subplots()
                ax.imshow(x, vmin = vmin, vmax = vmax)
                # ax.set_title(tag)
                plt.axis('off')
                fig.savefig( pjoin(dirname(abspath(__file__)), tag+'.png') ,
                             dpi          =  8*fig.dpi                     ,
                             bbox_inches  =  'tight'                       ,
                             pad_inches   =  0.0                           )
                plt.close(fig)

    device      =  'cuda'
    all_x       =  mk_all_x()
    for loss_type in ['lapla']:# 'mag','grad',
        args = {'loss_type': loss_type ,
                'device'   : 'cuda' ,
                'lr_ini'   : 1e-3   ,
                'lr_tol'   : 1e-12  ,
                'k_less'   : 0.5    ,
                'k_more'   : 1.8    ,
                'max_iters': 10000  }
        comb_preds = Smooth_Predictions(all_x, args).comb_preds
        print_comps(all_x                             ,
                    comb_preds                        ,
                    loss_type                         ,
                    print_all_x = (loss_type=='lapla'))