


# python /media/rafael/DATA/Dropbox/workspace/code/202109_stag_interp_v5/classic_ML_use_Cf_Nu_Github/source/job_scheduler.py

# ------------------------------------------------------------------------
#                  Basic Libraries
# ------------------------------------------------------------------------

from  os       import  system                      as  os_system
from  os.path  import  join                        as  pjoin
from  os.path  import  dirname, basename, abspath

import random
random.seed(0)

# ------------------------------------------------------------------------
#                  Basic Functions
# ------------------------------------------------------------------------

_folder_    =  dirname(abspath(__file__))
group       =  lambda A: '_'.join(map(str,A))
lfilter     =  lambda f,x: list(filter(f,x))
lmap        =  lambda f,x: list(map   (f,x))

# ------------------------------------------------------------------------
#                  Utility Functions
# ------------------------------------------------------------------------

def deepdirname(x, n, tag=None):
    for _ in range(n):
        x = dirname(x)
    if not (tag is None):
     assert tag in basename(x)
    return x

def shuffled(n):
    A = list(range(n))
    random.shuffle(A)
    return A

# ------------------------------------------------------------------------
#                  Utility Functions
# ------------------------------------------------------------------------

def mk_rnd12_K03():
    from    math    import  ceil

    N          =  40
    T          =  12
    places     =  2
    V          =  N - T
    K          =  ceil(places * N / V)
    assert type(K) == int
    result     =  None

    while True:

        for _ in range(100_000):
            failure = False

            # training sets
            all_K =  [[] for _ in range(K)]

            for i in shuffled(N):
                # loop over indexes

                success = 0

                # try to place "i" in a Ktrial
                for kk in shuffled(K):

                    # break if "i" was already placed enough times
                    if success == places:
                        break

                    # see if k-fold is possible
                    if (not i in all_K[kk]) and (len(all_K[kk])<V):
                        # place available for ind
                        all_K[kk].append(i)
                        success += 1

                if success != places: # i could not be placed
                    # report failure of attempt, break this loop
                    failure = True
                    break

            all_K = sorted(map(sorted,all_K))

            # if all inds were placed, check if K-folds are different and break loop
            if not failure:
                if len(set( tuple(sorted(arr)) for arr in all_K )) == K:
                    result = all_K
                    break

        # if result -> break, otherwise continue
        if result:
            break
        else:
            K += 1

    # pad result with missing indexes:
    for arr in result:
        arr += list(filter(lambda i: not i in arr, shuffled(N)))[:(V-len(arr))]
        arr.sort()

    # convert result to training, validation, testing
    result = [[sorted(set(range(N))-set(arr)),
               arr[:(len(arr)//2) ]          ,
               arr[ (len(arr)//2):]          ] for arr in result]
    return result

def mk_rnd12_K03_shuffle():
    N  =  40
    A  =  shuffled(N)
    T  =  12
    K  =   3
    all_trains = [A[i:i+T] for i in range(0,N,T)][:K]
    assert len(set(x for A in all_trains for x in A)) == (T*K)
    result = []
    for ii in range(K):
        train  = sorted(all_trains[ii])
        arr    = sorted(set(range(N)) - set(train))
        imid   = len(arr)//2
        valid  = arr[:imid ]
        test   = arr[ imid:]
        assert not len(set(train) & set(valid))
        assert not len(set(train) & set(test))
        assert not len(set(valid) & set(test))
        result.append([train, valid, test])
    assert not len(set(result[0][0]) & 
                   set(result[1][0]) & 
                   set(result[2][0]))
    return result

# ------------------------------------------------------------------------
#                  Main Runner
# ------------------------------------------------------------------------

if __name__ == '__main__':

    python_path    =  'python'
    main_runner    =  pjoin(        _folder_ , 'main.py')
    cache_abspath  =  pjoin(dirname(_folder_), 'clean_cache.py')
    qmode          =  'Nu_classic'
    run_tag        =  'rnd12_K03'
    device         =  'cuda'
    str_Cseq       =  'Cseq_1_20_20_20_20_20_20_20_1'
    str_dseq       =  'dseq_1x1_2x1_4x2_8x4_16x8_32x16_64x32_64x32'

    register_run   =  False
    tuples_inds    =  list(mk_rnd12_K03_shuffle())

    weight_decay_w = 0
    run_id         = 0
    for inds_train, inds_valid, inds_test in tuples_inds:
        os_system(f'{python_path} {cache_abspath}')
        os_system(f'{python_path} {main_runner} {run_tag} {qmode} train_{group(inds_train)} valid_{group(inds_valid)} test_{group(inds_test)} {str_Cseq} {str_dseq} {run_id} {device} {register_run} {weight_decay_w}')
        os_system(f'{python_path} {cache_abspath}')
        run_id += 1