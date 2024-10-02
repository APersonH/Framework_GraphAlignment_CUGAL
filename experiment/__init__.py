import torch
from sacred import Experiment
from sacred.observers import FileStorageObserver
import logging
#from algorithms import gwl, conealign, grasp as grasp, regal, eigenalign, NSD, isorank2 as isorank, netalign, klaus, sgwl,Grampa,GraspB,GrampaS,Fugal,Fugal2,QAP
from algorithms import Fugal, Cugal
#GraspBafter Grampa
from cugal.config import SinkhornMethod, HungarianMethod

ex = Experiment("ex")

ex.observers.append(FileStorageObserver('runs'))

# create logger
logger = logging.getLogger('e')
logger.setLevel(logging.INFO)
logger.propagate = False

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

ex.logger = logger


_GW_args = {
    'opt_dict': {
        'epochs': 1,
        'batch_size': 1000000,
        'use_cuda': False,
        'strategy': 'soft',
        # 'strategy': 'hard',
        # 'beta': 0.1,
        'beta': 1e-1,
        'outer_iteration': 400,  # M
        'inner_iteration': 1,  # N
        'sgd_iteration': 300,
        'prior': False,
        'prefix': 'results',
        'display': False
    },
    'hyperpara_dict': {
        'dimension': 90,
        # 'loss_type': 'MSE',
        'loss_type': 'L2',
        'cost_type': 'cosine',
        # 'cost_type': 'RBF',
        'ot_method': 'proximal'
    },
    # 'lr': 0.001,
    'lr': 1e-3,
    # 'gamma': 0.01,
    # 'gamma': None,
    'gamma': 0.8,
    # 'max_cpu': 20,
    # 'max_cpu': 4
}

_SGW_args = {
    'ot_dict': {
        'loss_type': 'L2',  # the key hyperparameters of GW distance
        'ot_method': 'proximal',
         #'beta': 0.025,#euroroad
        'beta': 0.025,#netscience,eurorad,arenas
        #'beta': 0.1,#dense ex fb, socfb datasets
        #'beta': 0.025,# 0.025-0.1 depends on degree
        # outer, inner iteration, error bound of optimal transport
        'outer_iteration': 2000,  # num od nodes
        'iter_bound': 1e-10,
        'inner_iteration': 2,
        'sk_bound': 1e-30,
        'node_prior': 1000,
        'max_iter': 4,  # iteration and error bound for calcuating barycenter
        'cost_bound': 1e-26,
        'update_p': False,  # optional updates of source distribution
        'lr': 0,
        #'alpha': 0
        'alpha': 1
    },
    "mn": 1,  # gwl
    # "mn": 1,  # s-gwl-3
    # "mn": 2,  # s-gwl-2
    # "mn": 3,  # s-gwl-1
    'max_cpu': 40,
}

_CONE_args = {
    'dim': 512,  # clipped by Src[0] - 1
    'window': 10,
    'negative': 1.0,
    'niter_init': 10,
    'reg_init': 1.0,
    'nepoch': 5,
    'niter_align': 10,
    'reg_align': 0.05,
    'bsz': 10,
    'lr': 1.0,
    'embsim': 'euclidean',
    'alignmethod': 'greedy',
    'numtop': 10
}

_GRASP_args = {
    'laa': 2,
    'icp': False,
    'icp_its': 3,
    'q': 100,
    'k': 20,
    #'n_eig': Src.shape[0] - 1
    'n_eig': 100,
    'lower_t': 1.0,
    'upper_t': 50.0,
    'linsteps': True,
    'base_align': True
}

_REGAL_args = {
    'attributes': None,
    'attrvals': 2,
    'dimensions': 128,  # useless
    'k': 10,            # d = klogn
    'untillayer': 2,    # k
    'alpha': 0.01,      # delta
    'gammastruc': 1.0,
    'gammaattr': 1.0,
    'numtop': 10,
    'buckets': 2
}

_LREA_args = {
    'iters': 40,
    'method': "lowrank_svd_union",
    'bmatch': 3,
    'default_params': True
}

_NSD_args = {
    'alpha': 0.8,
    'iters': 20
}

_ISO_args = {
    'alpha': 0.9,
    'tol': 1e-12,
    'maxiter': 100,
    'lalpha': 10000,
    'weighted': True
}

_NET_args = {
    'a': 1,
    'b': 2,
    'gamma': 0.95,
    'dtype': 2,
    'maxiter': 100,
    'verbose': True
}

_KLAU_args = {
    'a': 1,
    'b': 1,
    'gamma': 0.4,
    'stepm': 25,
    'rtype': 2,
    'maxiter': 100,
    'verbose': True
}
_Grampa_args = {
   'eta': 0.2,
   'lalpha':10000,
   'initSim':0,
   'Eigtype':0
}
_GrampaS_args = {
    'eta': 0.2,
    'lalpha':10000
}
_GRASPB_args = {
    'laa': 3,
    'icp': True,
    'icp_its': 3,
    'q': 20,
    'k': 20,
    #'n_eig': Src.shape[0] - 1
    #n_eig': 100,
    'lower_t': 0.1,
    'upper_t': 50.0,
    'linsteps': True,
    'ba_': True,
    'corr_func': 3,
    'k_span':40

}
_Fugal_args={
    'iter': 15,
    #'iter': 15, for xx dataset.
    'simple': True,
    'mu': 2,#1 MM,are,net --0.1 ce--2 eu
}
_Fugal2_args={
    'iter': 15,
    #'iter': 15, for xx dataset.
    'simple': True,
    'mu': 0.3,#1 MM,are,net --0.1 ce--2 eu
}

_Cugal_args={
    'iter': 15,
    'simple': True,
    'mu': 2,#1 MM,are,net --0.1 ce--2 eu
    'path': "cugal",
    'sparse': False,
    'cache': 1,
    'sinkhorn_method': SinkhornMethod.MIX,
}

_Cugal_sparse_args={
    'iter': 15,
    'simple': True,
    'mu': 0.5,#1 MM,are,net --0.1 ce--2 eu
    'path': "cugal",
    'sparse': True,
    'cache': 0,
    'sinkhorn_method': SinkhornMethod.MIX,
    'sinkhorn_iterations': 1000,
    'sinkhorn_threshold': 1e-3,

}

_Cugal_sparse_cache_args={
    'iter': 15,
    'simple': True,
    'mu': 2,#1 MM,are,net --0.1 ce--2 eu
    'path': "cugal",
    'sparse': True,
    'cache': 1,
    'sinkhorn_method': SinkhornMethod.MIX,
    'sinkhorn_iterations': 1000,
    'sinkhorn_threshold': 1e-3,
}

_Cugal_sparse_cache_log_args = dict( _Cugal_sparse_cache_args, sinkhorn_method=SinkhornMethod.LOG)
_Cugal_sparse_cache_mix_args = dict( _Cugal_sparse_cache_args, sinkhorn_method=SinkhornMethod.MIX)

_Cugal_mix_lambda_pol_args = dict( _Cugal_sparse_cache_mix_args, lambda_func = lambda x: x**2)
_Cugal_mix_lambda_exp_args = dict( _Cugal_sparse_cache_mix_args, lambda_func = lambda x: 2**x)

_Cugal_mix_fw_001_args = dict( _Cugal_sparse_cache_mix_args, frank_wolfe_threshold=0.01)
_Cugal_mix_fw_005_args = dict( _Cugal_sparse_cache_mix_args, frank_wolfe_threshold=0.05)
_Cugal_mix_fw_010_args = dict( _Cugal_sparse_cache_mix_args, frank_wolfe_threshold=0.1)
_Cugal_mix_fw_020_args = dict( _Cugal_sparse_cache_mix_args, frank_wolfe_threshold=0.2)
_Cugal_mix_fw_050_args = dict( _Cugal_sparse_cache_mix_args, frank_wolfe_threshold=0.5)
_Cugal_mix_fw_100_args = dict( _Cugal_sparse_cache_mix_args, frank_wolfe_threshold=1.0)

_Cugal_log_fw_010_args = dict( _Cugal_sparse_cache_log_args, frank_wolfe_threshold=0.1)
_Cugal_log_fw_020_args = dict( _Cugal_sparse_cache_log_args, frank_wolfe_threshold=0.2)


_Cugal_cache_test_02_args = dict( _Cugal_sparse_cache_mix_args, cache=2)
_Cugal_cache_test_03_args = dict( _Cugal_sparse_cache_mix_args, cache=3)
_Cugal_cache_test_05_args = dict( _Cugal_sparse_cache_mix_args, cache=5)
_Cugal_cache_test_10_args = dict( _Cugal_sparse_cache_mix_args, cache=10)


_Cugal_hungarian_test = _Cugal_sparse_cache_mix_args.copy()
_Cugal_hungarian_CuLAP = dict( _Cugal_hungarian_test)#, hungarian=HungarianMethod.CULAP)
_Cugal_hungarian_rand_test = dict( _Cugal_hungarian_test, hungarian=HungarianMethod.RAND)
_Cugal_mix_hungarian_more_greed_test = dict( _Cugal_hungarian_test, hungarian=HungarianMethod.DOUBLE_GREEDY)
_Cugal_log_hungarian_more_greed_test = dict( _Cugal_sparse_cache_log_args, hungarian=HungarianMethod.DOUBLE_GREEDY)
_Cugal_hungarian_entro_greedy = dict( _Cugal_hungarian_test)#, hungarian=HungarianMethod.ENTRO_GREEDY)
_Cugal_hungarian_parallel_greedy = dict( _Cugal_hungarian_test, hungarian=HungarianMethod.PARALLEL_GREEDY)
_Cugal_JV_args = dict(_Cugal_sparse_cache_mix_args)#, hungarian=HungarianMethod.JV)
_Cugal_log_sparse_hungarian = dict( _Cugal_sparse_cache_log_args, hungarian=HungarianMethod.SPARSE)

_Cugal_hungarian_test_FW_20 = dict( _Cugal_hungarian_test, frank_wolfe_threshold=0.2)
_Cugal_hungarian_CuLAP_FW_20 = dict( _Cugal_hungarian_CuLAP, frank_wolfe_threshold=0.2)
_Cugal_hungarian_rand_FW_20 = dict( _Cugal_hungarian_rand_test, frank_wolfe_threshold=0.2)
_Cugal_mix_hungarian_more_greed_FW_10 = dict( _Cugal_mix_hungarian_more_greed_test, frank_wolfe_threshold=0.1)
_Cugal_mix_hungarian_more_greed_FW_20 = dict( _Cugal_mix_hungarian_more_greed_test, frank_wolfe_threshold=0.2)
_Cugal_hungarian_entro_greedy_FW_20 = dict( _Cugal_hungarian_entro_greedy, frank_wolfe_threshold=0.2)
_Cugal_hungarian_parallel_greedy_FW_20 = dict( _Cugal_hungarian_parallel_greedy, frank_wolfe_threshold=0.2)
_Cugal_JV_FW20 = dict(_Cugal_JV_args, frank_wolfe_threshold=0.2)
_Cugal_log_hungarian_more_greed_FW_10 = dict( _Cugal_log_hungarian_more_greed_test, frank_wolfe_threshold=0.1)
_Cugal_log_hungarian_more_greed_FW_20 = dict( _Cugal_log_hungarian_more_greed_test, frank_wolfe_threshold=0.2)

_Cugal_mix_sink_thres_1e_3 = dict( _Cugal_sparse_cache_mix_args, sinkhorn_threshold=1e-3)
_Cugal_mix_sink_thres_1e_4 = dict( _Cugal_sparse_cache_mix_args, sinkhorn_threshold=1e-4)
_Cugal_mix_sink_thres_1e_2 = dict( _Cugal_sparse_cache_mix_args, sinkhorn_threshold=1e-2)
_Cugal_mix_sink_thres_1e_6 = dict( _Cugal_sparse_cache_mix_args, sinkhorn_threshold=1e-6)
_Cugal_log_sink_thres_1e_3 = dict( _Cugal_sparse_cache_log_args, sinkhorn_threshold=1e-3)
_Cugal_log_sink_thres_1e_4 = dict( _Cugal_sparse_cache_log_args, sinkhorn_threshold=1e-4)
_Cugal_log_sink_thres_1e_2 = dict( _Cugal_sparse_cache_log_args, sinkhorn_threshold=1e-2)
_Cugal_log_sink_thres_1e_6 = dict( _Cugal_sparse_cache_log_args, sinkhorn_threshold=1e-6)
_Cugal_log_sink_thres_1e_1 = dict( _Cugal_sparse_cache_log_args, sinkhorn_threshold=0.1)
_Cugal_mix_sink_thres_1e_1 = dict( _Cugal_sparse_cache_mix_args, sinkhorn_threshold=0.1)
_Cugal_log_sink_thres_03 = dict( _Cugal_sparse_cache_log_args, sinkhorn_threshold=0.3)
_Cugal_log_sink_thres_05 = dict( _Cugal_sparse_cache_log_args, sinkhorn_threshold=0.5)
_Cugal_log_sink_thres_10 = dict( _Cugal_sparse_cache_log_args, sinkhorn_threshold=1.0)
_Cugal_mix_sink_thres_03 = dict( _Cugal_sparse_cache_mix_args, sinkhorn_threshold=0.3)
_Cugal_mix_sink_thres_05 = dict( _Cugal_sparse_cache_mix_args, sinkhorn_threshold=0.5)
_Cugal_mix_sink_thres_10 = dict( _Cugal_sparse_cache_mix_args, sinkhorn_threshold=1.0)

_Cugal_OT_args={
    'iter': 15,
    'simple': True,
    'mu': 1,#1 MM,are,net --0.1 ce--2 eu
    'path': "cugal",
    'device': 'cpu',
    #'sinkhorn_method': SinkhornMethod.OT_CPU
}

_Cugal_OT_GPU_args={
    'iter': 15,
    'simple': True,
    'mu': 0.5,#1 MM,are,net --0.1 ce--2 eu
    'path': "cugal",
    #'sinkhorn_method': SinkhornMethod.OT_GPU
}

_Cugal_CPU_args={
    'iter': 15,
    'simple': True,
    'mu': 2,#1 MM,are,net --0.1 ce--2 eu
    'path': "cugal",
    'device': 'cpu',
    'sinkhorn_method': SinkhornMethod.STANDARD,
    'dtype': torch.float64,
    'use_fugal': True,
}


_algs = [ # (alg, args, ?, name)
    #(gwl, _GW_args, [3], "GW"),
    #(conealign, _CONE_args, [-3], "CONE"),
    #(grasp, _GRASP_args, [-3], "GRASP"),
    #(regal, _REGAL_args, [-3], "REGAL"),
    #(eigenalign, _LREA_args, [3], "LREA"),
    #(NSD, _NSD_args, [30], "NSD"),

    #(isorank, _ISO_args, [3], "ISO"),
    #(netalign, _NET_args, [3], "NET"),
    #(klaus, _KLAU_args, [3], "KLAU"),
    #(sgwl, _SGW_args, [3], "SGW"),
    #(Grampa, _Grampa_args, [3], "GRAMPA"),
    #(GraspB, _GRASPB_args, [-96], "GRASPB"),
    #(Fugal2, _Fugal_args, [3], "FUGALB"),
    #(GrampaS, _GrampaS_args, [4], "GRAMPAS"),
    (Fugal, _Fugal_args, [3], "FUGAL"), # 0
    (Cugal, _Cugal_args, [3], "CUGAL"), # 1
    (Cugal, _Cugal_sparse_args, [3], "CUGAL_Sparse"), # 2 
    (Cugal, _Cugal_sparse_cache_args, [3], "CUGAL_Sparse"), # 3
    (Cugal, _Cugal_sparse_cache_log_args, [3], "CUGAL_Sparse_LOG"), # 4
    (Cugal, _Cugal_sparse_cache_mix_args, [3], "CUGAL_Sparse_MIX"), # 5
    (Cugal, _Cugal_OT_args, [3], "CUGAL_OT"), # 6
    (Cugal, _Cugal_OT_GPU_args, [3], "CUGAL_OT_GPU"), # 7
    (Cugal, _Cugal_mix_fw_001_args, [3], "CUGAL_FW_001"), # 8
    (Cugal, _Cugal_mix_fw_005_args, [3], "CUGAL_FW_005"), # 9
    (Cugal, _Cugal_mix_fw_010_args, [3], "CUGAL_FW_010"),
    (Cugal, _Cugal_mix_fw_020_args, [3], "CUGAL_FW_020"), # 11
    (Cugal, _Cugal_mix_fw_050_args, [3], "CUGAL_FW_050"),
    (Cugal, _Cugal_mix_fw_100_args, [3], "CUGAL_FW_100"),
    (Cugal, _Cugal_cache_test_02_args, [3], "CUGAL_CACHE_02"), # 14
    (Cugal, _Cugal_cache_test_03_args, [3], "CUGAL_CACHE_03"),
    (Cugal, _Cugal_cache_test_05_args, [3], "CUGAL_CACHE_05"),
    (Cugal, _Cugal_cache_test_10_args, [3], "CUGAL_CACHE_10"),
    (Cugal, _Cugal_hungarian_test_FW_20, [3], "CUGAL_HUNGARIAN_FW_20"), # 18
    (Cugal, _Cugal_hungarian_rand_test, [3], "CUGAL_HUNGARIAN_RAND"),
    (Cugal, _Cugal_mix_hungarian_more_greed_test, [3], "CUGAL_HUNGARIAN_MORE_GREEDy"), # 20
    (Cugal, _Cugal_hungarian_rand_FW_20, [3], "CUGAL_HUNGARIAN_RAND_FW_20"),
    (Cugal, _Cugal_mix_hungarian_more_greed_FW_20, [3], "CUGAL_HUNGARIAN_MORE_GREEDY_FW_20"), # 22
    (Cugal, _Cugal_hungarian_entro_greedy, [3], "CUGAL_HUNGARIAN_ENTRO_GREEDY"),
    (Cugal, _Cugal_hungarian_entro_greedy_FW_20, [3], "CUGAL_HUNGARIAN_ENTRO_GREEDY_FW_20"), 
    (Cugal, _Cugal_CPU_args, [3], "CUGAL_CPU"), # 25
    (Cugal, _Cugal_JV_args, [3], "CUGAL_JV"), # 26
    (Cugal, _Cugal_JV_FW20, [3], "CUGAL_JV_FW20"), # 27
    (Cugal, _Cugal_hungarian_parallel_greedy, [3], "CUGAL_HUNGARIAN_PARALLEL_GREEDY"), # 28
    (Cugal, _Cugal_hungarian_parallel_greedy_FW_20, [3], "CUGAL_HUNGARIAN_PARALLEL_GREEDY_FW_20"), # 29
    (Cugal, _Cugal_hungarian_CuLAP, [3], "CUGAL_HUNGARIAN_CULAP"), # 30
    (Cugal, _Cugal_hungarian_CuLAP_FW_20, [3], "CUGAL_HUNGARIAN_CULAP_FW_20"), # 31
    (Cugal, _Cugal_mix_lambda_pol_args, [3], "CUGAL_MIX_LAMBDA_POL"), # 32
    (Cugal, _Cugal_mix_lambda_exp_args, [3], "CUGAL_MIX_LAMBDA_EXP"),  # 33
    (Cugal, _Cugal_mix_sink_thres_1e_3, [3], "CUGAL_MIX_SINK_1e_3"), # 34
    (Cugal, _Cugal_mix_sink_thres_1e_4, [3], "CUGAL_MIX_SINK_1e_4"), # 35
    (Cugal, _Cugal_mix_sink_thres_1e_2, [3], "CUGAL_MIX_SINK_1e_2"), # 36
    (Cugal, _Cugal_mix_sink_thres_1e_6, [3], "CUGAL_MIX_SINK_1e_6"), # 37
    (Cugal, _Cugal_log_sink_thres_1e_3, [3], "CUGAL_LOG_SINK_1e_3"), # 38
    (Cugal, _Cugal_log_sink_thres_1e_4, [3], "CUGAL_LOG_SINK_1e_4"), # 39
    (Cugal, _Cugal_log_sink_thres_1e_2, [3], "CUGAL_LOG_SINK_1e_2"), # 40
    (Cugal, _Cugal_log_sink_thres_1e_6, [3], "CUGAL_LOG_SINK_1e_6"), # 41
    (Cugal, _Cugal_log_sink_thres_1e_1, [3], "CUGAL_LOG_SINK_1e_1"), # 42
    (Cugal, _Cugal_mix_sink_thres_1e_1, [3], "CUGAL_MIX_SINK_1e_1"), # 43
    (Cugal, _Cugal_log_sink_thres_03, [3], "CUGAL_LOG_SINK_03"), # 44
    (Cugal, _Cugal_log_sink_thres_05, [3], "CUGAL_LOG_SINK_05"), # 45
    (Cugal, _Cugal_log_sink_thres_10, [3], "CUGAL_LOG_SINK_10"), # 46
    (Cugal, _Cugal_mix_sink_thres_03, [3], "CUGAL_MIX_SINK_03"), # 47
    (Cugal, _Cugal_mix_sink_thres_05, [3], "CUGAL_MIX_SINK_05"), # 48
    (Cugal, _Cugal_mix_sink_thres_10, [3], "CUGAL_MIX_SINK_10"), # 49
    (Cugal, _Cugal_log_fw_010_args, [3], "CUGAL_LOG_FW_010"), # 50
    (Cugal, _Cugal_log_fw_020_args, [3], "CUGAL_LOG_FW_020"), # 51
    (Cugal, _Cugal_log_hungarian_more_greed_test, [3], "CUGAL_LOG_HUNGARIAN_MORE_GREEDY"), # 52
    (Cugal, _Cugal_log_hungarian_more_greed_FW_10, [3], "CUGAL_LOG_HUNGARIAN_MORE_GREEDY_FW_10"), # 53
    (Cugal, _Cugal_log_hungarian_more_greed_FW_20, [3], "CUGAL_LOG_HUNGARIAN_MORE_GREEDY_FW_20"), # 54
    (Cugal, _Cugal_mix_hungarian_more_greed_FW_10, [3], "CUGAL_HUNGARIAN_MORE_GREEDY_FW_10"), # 55
    (Cugal, _Cugal_log_sparse_hungarian, [3], "CUGAL_LOG_SPARSE_HUNGARIAN"), # 56
    
    #(Fugal2, _Fugal2_args, [3], "FUGALB"),
    #(QAP, _Fugal_args, [3], "QAP"),

]   

_acc_names = [
    "acc",
    "EC",
    "ICS",
    "S3",
    "jacc",
    "mnc",
]
