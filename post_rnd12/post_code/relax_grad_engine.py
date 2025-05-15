# python /home/rafael/Dropbox/workspace/code/202109_stag_interp_v5/classic_ML_use_Cf_Nu_search/post/post_code/relax_grad_engine.py
# ------------------------------------------------------------------------
#                  Generic Libraries
# ------------------------------------------------------------------------
from    math     import  prod
from    sys      import  path              as  sys_path
from    os.path  import  abspath, dirname
from    time     import  time
import  torch

# ------------------------------------------------------------------------
#                  Basic Functions
# ------------------------------------------------------------------------
format_dt      =  lambda x:  "%02d:%02d:%02d [hh:mm:ss]" % (x//3600, (x%3600)//60, x%60)

# ------------------------------------------------------------------------
#                  Relaxation Engine
# ------------------------------------------------------------------------

class Relax_Grad_Engine:
    def __init__(self, alpha, device='cuda'):
        assert 0<= alpha <=1  # alpha must be a ratio (0.9, etc.)
        print(f'Relax_Grad_Engine (alpha = {alpha})')
        self.alpha   =  alpha
        self.device  =  device
        
    def __call__(self, x, gammas = {'low': 0, 'high': 0.35}, tol_gamma=1e-12):
        
        # trivial exits
        if abs(self.alpha-1)<1e-12: return x
        if abs(self.alpha  )<1e-12: return x*0.
        
        assert len(x.shape) == 2
        get_avg  =  lambda x: float(x.sum())/prod(x.shape)
        
        # basic functions
        fnorm       =  lambda x: float(torch.sqrt(torch.mean(x**2)).item())
        
        # initialize data
        x           =  torch.tensor(x, device=self.device)
        x_ini       =  torch.clone(x)
        norm_ini    =  fnorm  (x_ini)
        avg_ini     =  get_avg(x_ini)
        
        assert len(x.shape) == 2
        
        gamma_high  =  gammas['high']
        gamma_low   =  gammas['low' ]
        
        t0          =  time()
        
        while True:
            x           =  self.__solve(x, x_ini, gamma_high)
            alpha_calc  =  fnorm(x)/norm_ini
            if alpha_calc > self.alpha: # we need to erase more
                gamma_low, gamma_high  =  gamma_high, 2*gamma_high # notice gamma hadn't done enough, os it's also the lower threshold
            else:
                break
        
        while abs(gamma_high - gamma_low)>tol_gamma:
            gamma       =  0.5*(gamma_high + gamma_low)
            x           =  self.__solve(x, x_ini, gamma)
            alpha_calc  =  fnorm(x)/norm_ini
            
            if alpha_calc >= self.alpha: # we didn't erase enough -> increase gamma
                gamma_low  = gamma
            if alpha_calc <= self.alpha: # we erased too much -> lower gamma
                gamma_high = gamma
            # print('alphas calc,ref',alpha_calc,self.alpha,'gammas',gamma_low,gamma_high)
        result  = x.cpu().detach().numpy()
        result += avg_ini - get_avg(result) # average correction
        print(f'(final_gamma: {gamma:.6f}) (Total Elapsed Time: {format_dt(time()-t0)})')
        return result
    
    # @staticmethod
    def __solve(self, x, x_ini, gamma, lr_ini = 1e-3, k_more = 1.9, k_less = 0.5, lr_tol=1e-12):
        
        sq         =  lambda x: float((x**2).sum().item())
        Jcost      =  lambda x: sq(x-x_ini) + gamma*(sq(x-torch.roll(x,1,0)) + sq(x-torch.roll(x,1,1)))
        get_grad   =  lambda x: (1 + 4*gamma)*x - x_ini - gamma*sum(torch.roll(x,shift,dim) for shift in [-1,1] for dim in range(2)) # doesn't matter what roll goes first
        lr         =  lr_ini
        
        x_old      =  x
        Jcost_old  =  Jcost(x)
        
        while lr>lr_tol:
            x          =  x - lr*get_grad(x)
            Jcost_now  =  Jcost(x)
            if Jcost_now<Jcost_old:
                lr        *=  k_more
                x_old      =  x
                Jcost_old  =  Jcost_now
            else:
                x    =  x_old
                lr  *=  k_less
        return x

# ------------------------------------------------------------------------
#                  Direct Runner
# ------------------------------------------------------------------------

if __name__ == '__main__':
    # this is only a manual quick test
    #    -> please update filepath (fname) to run it again
    fname = '/home/rafael/Dropbox/workspace/code/202109_stag_interp_v5/classic_ML_use_Cf_Nu_search/post/post_output/tecplot_files/20220724_122746_rnd33_K11_l1_Cf_classic_net_alt2_orig_runid_9/valid/20220724_122746_rnd33_K11_l1_Cf_classic_net_alt2_orig_runid_9_indpred_18.tec'
    
    sys_path.append(dirname(abspath(__file__)))
    from gen_pictures_updated  import ImportTec
    sys_path.pop()
    
    td = ImportTec(fname)
    
    x = td.data['qmode_Cf_classic_pred']
    print('ratio',x.max()/np.average(x))
    relax = Relax_Grad_Engine(0.97)
    x_new = relax(x+0.)
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.imshow(np.concatenate((x,x_new),axis=1))
    plt.show()
    
    plt.imshow(x_new)
    plt.show()