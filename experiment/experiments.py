from . import ex, _algs, _CONE_args, _GRASP_args, _GW_args, _ISO_args, _KLAU_args, _LREA_args, _NET_args, _NSD_args, _REGAL_args,_Grampa_args,_GrampaS_args
from generation import generate as gen
#from algorithms import regal, eigenalign, conealign, netalign, NSD, klaus, gwl, Fugal, isorank2 as isorank,Grampa,Fugal,GrampaS,Fugal
from algorithms import Fugal, Cugal
import networkx as nx
import numpy as np
from enum import Enum

# mprof run workexp.py with playground run=[1,2,3,4,5] iters=2 win

class Algs(Enum):
    FUGAL = 0
    CUGAL = 1
    CUGAL_SPARSE = 2
    CUGAL_CHACHE_SPARSE = 3
    CUGAL_CHACHE_SPARSE_LOG = 4
    CUGAL_CHACHE_SPARSE_MIX = 5
    CUGAL_OT = 6
    CUGAL_OT_GPU = 7
    CUGAL_MIX_FW_001 = 8
    CUGAL_MIX_FW_005 = 9
    CUGAL_MIX_FW_010 = 10
    CUGAL_MIX_FW_020 = 11
    CUGAL_MIX_FW_050 = 12
    CUGAL_MIX_FW_100 = 13
    CUGAL_CHACHE_TEST_02 = 14
    CUGAL_CHACHE_TEST_03 = 15
    CUGAL_CHACHE_TEST_05 = 16
    CUGAL_CHACHE_TEST_10 = 17
    CUGAL_HUNGARIAN_FW_20 = 18
    CUGAL_HUNGARIAN_RAND = 19
    CUGAL_HUNGARIAN_MORE_GREED = 20
    CUGAL_HUNGARIAN_RAND_FW_20 = 21
    CUGAL_HUNGARIAN_MORE_GREED_FW_20 = 22
    CUGAL_HUNGARIAN_ENTRO_GREEDY = 23
    CUGAL_HUNGARIAN_ENTRO_GREEDY_FW_20 = 24
    CUGAL_CPU = 25
    CUGAL_JV = 26
    CUGAL_JV_FW_20 = 27
    CUGAL_HUNGARIAN_PARALLEL_GREEDY = 28
    CUGAL_HUNGARIAN_PARALLEL_GREEDY_FW_20 = 29
    CUGAL_HUNGARIAN_CULAP = 30
    CUGAL_HUNGARIAN_CULAP_FW_20 = 31
    CUGAL_MIX_LAMBDA_POL = 32
    CUGAL_MIX_LAMBDA_EXP = 33
    CUGAL_MIX_SINK_THRESH_1E_3 = 34
    CUGAL_MIX_SINK_THRESH_1E_4 = 35
    CUGAL_MIX_SINK_THRESH_1E_2 = 36
    CUGAL_MIX_SINK_THRESH_1E_6 = 37
    CUGAL_LOG_SINK_THRESH_1E_3 = 38
    CUGAL_LOG_SINK_THRESH_1E_4 = 39
    CUGAL_LOG_SINK_THRESH_1E_2 = 40
    CUGAL_LOG_SINK_THRESH_1E_6 = 41
    CUGAL_LOG_SINK_THRESH_1E_1 = 42
    CUGAL_MIX_SINK_THRESH_1E_1 = 43
    CUGAL_LOG_SINK_THRESH_03 = 44
    CUGAL_LOG_SINK_THRESH_05 = 45
    CUGAL_LOG_SINK_THRESH_10 = 46
    CUGAL_MIX_SINK_TRHESH_03 = 47
    CUGAL_MIX_SINK_TRHESH_05 = 48
    CUGAL_MIX_SINK_TRHESH_10 = 49
    CUGAL_LOG_FW_10 = 50
    CUGAL_LOG_FW_20 = 51
    CUGAL_LOG_HUN_MORE_GREED= 52
    CUGAL_LOG_HUN_MORE_GREED_FW_10 = 53
    CUGAL_LOG_HUN_MORE_GREED_FW_20 = 54
    CUGAL_MIX_HUN_MORE_GREED_FW_10 = 55
    CUGAL_LOG_HUN_SPARSE = 56

class Data(Enum):
    CA_NETSCIENCE = "ca-netscience"            # 379   / 914   / connected
    VOLES = "voles"
    HIGH_SCHOOL = "high-school"
    YEAST = [f"real world/MultiMagna/yeast{n}_Y2H1" for n in range(0, 26, 5)]
    MULTIMAGNA = "MultiMagna"
    BIO_CELEGANS = "bio-celegans"             # 453   / 2k    / connected
    IN_ARENAS = "in-arenas"                   # 1.1k  / 5.4k  / connected
    ARENAD = "arenad"
    INF_EUROROAD = "inf-euroroad"             # 1.2K  / 1.4K  / disc - 200
    INF_POWER = "inf-power"                   # 4.9K  / 6.6K  / connected
    CA_GRQC = "ca-GrQc"                       # 4.2k  / 13.4K / connected - (5.2k  / 14.5K)?
    BIO_DMELA = "bio-dmela"                   # 7.4k  / 25.6k / connected
    CA_ASTROPH = "CA-AstroPh"                 # 18k   / 195k  / connected
    SOC_HAMSTERSTER = "soc-hamsterster"       # 2.4K  / 16.6K / disc - 400
    SOCFB_BOWDOIN47 = "socfb-Bowdoin47"       # 2.3K  / 84.4K / disc - only 2
    SOCFB_HAMILTON46 = "socfb-Hamilton46"     # 2.3K  / 96.4K / disc - only 2
    SOCFB_HAVERFORD76 = "socfb-Haverford76"   # 1.4K  / 59.6K / connected
    SOCFB_SWARTHMORE42 = "socfb-Swarthmore42" # 1.7K  / 61.1K / disc - only 2
    SOC_FACEBOOK = "soc-facebook"
    SCC_ENRONONLY = "scc_enron-only"
    SCC_FB_FORUM = "scc_fb-forum"
    SCC_FB_MESSAGES = "scc_fb-messages"
    SCC_INFECT_HYPER = "scc_infect-h"         # 4k    / 87k   / connected
    CA_ERDOS = "ca-Erdos992"                  # 6.1K  / 7.5K  / disc - 100 + 1k disc nodes
    EMAIL_ENRON = "email-Enron"               # 36K   / 183K  / connected
    CA_HEP = "ca-HepPh-remapped"                # 12K   / 118K  / connected

def aaa(vals, dist_type=0):
    g = []
    for val in vals:
        if dist_type == 0:
            dist = np.random.randint(15, 21, val)
        if dist_type == 1:
            dist = nx.utils.powerlaw_sequence(val, 2.5)
            dist = np.array(dist)
            dist = dist.round()
            dist += 1
            dist = dist.tolist()
        if dist_type == 2:
            dist = np.random.normal(10, 1, val)
            # dist = np.random.normal(val, 1, 2**14)
        if dist_type == 3:
            dist = np.random.poisson(lam=1, size=val)
            dist = np.array(dist)
            dist += 1
            dist = dist.tolist()

        dist = [round(num) for num in dist]
        usum = sum(dist)
        if usum % 2 == 1:
            max_value = max(dist)
            max_index = dist.index(max_value)
            dist[max_index] = dist[max_index]-1
        G2 = nx.configuration_model(dist, nx.Graph)
        G2.remove_edges_from(nx.selfloop_edges(G2))
        g.append((lambda x: x, (G2,)))
    return g
    # normald = np.random.normal(10, 2, 1000) make it 1 for standard

def aa1(vals):
    g = []
    for val in vals:
        G2=nx.newman_watts_strogatz_graph, (val, 7, 0.1)
        g.append((lambda x: x, (G2,)))
    return g
def ggg(vals):
    return [str(x) for x in vals]


@ex.named_config
def scaling():

    # Greedied down
   # _algs[0][2][0] = 2
   # _algs[1][2][0] = -2
   # _algs[2][2][0] = -2
   # _algs[3][2][0] = -2
   # _algs[4][2][0] = 2
   # _algs[5][2][0] = 2
   # _algs[6][2][0] = 2

  #  _GW_args["max_cpu"] = 40
    # _CONE_args["dim"] = 1000
   # _CONE_args["dim"] = 256
   # _GRASP_args["n_eig"] = 256
   # _ISO_args["alpha"] = 0.9
   # _ISO_args["lalpha"] = 100000  # full dim

   # run = [1, 2, 3, 4, 5, 6]
    #run= [1, 6,9,10,11,14,15]
    run= [1]
    iters = 3

    tmp = [
        #2**i for i in range(3, 4)
        # 2**i for i in range(10, 14)
        # 2 ** 15,
        # 2 ** 16,
        # 2 ** 17,
        10, 100, 1000, 10000
    ]

    # graphs = aaa(tmp, dist_type=0)
    # xlabel = "kdist"
    # graphs = aaa(tmp, dist_type=1)
    # xlabel = "powerlaw"
    graphs = aaa(tmp, dist_type=2)
    #graphs = aa1(tmp)
    xlabel = "normal"
    
    # graphs = aaa(tmp, dist_type=3)
    # xlabel = "poisson"
    graphs = []

    graph_names = ggg(tmp)

    noises = [
        # 0.00,
        0.01,
        # 0.02,
        # 0.03
        # 0.04,
    ]

    #s_trans = (2, 1, 0, 3)
    s_trans = (0, 2, 1, 3)
    #xlabel = list(tmp[1][0].keys())[0]

def alggs(tmp):
    alg, args, mtype, algname = _algs[tmp[0]]
    return [
        # (alg, {**args, **update}, mtype, f"{algname}{list(update.values())[0]}") for update in tmp[1]
        (alg, {**args, **update}, mtype, str(list(update.values())[0])) for update in tmp[1]
    ]


@ex.named_config
def tuning():

    # tmp = [
    #     1,  # CONE
    #     [
    #         {'dim': 128 * i} for i in range(1, 17)
    #     ]
    # ]

    # tmp = [
    #     2,  # grasp
    #     [
    #         {'n_eig': x} for x in [128, 512, 1024]
    #     ]
    # ]
    # _algs[2][2][0] = -2

    # tmp = [
    #     3,  # REGAL
    #     [
    #         {'untillayer': x} for x in range(1, 6)
    #     ]
    # ]

    # tmp = [
    #     4,  # LREA
    #     [
    #         {'iters': 8 * i} for i in range(1, 9)
    #     ]
    # ]

    # tmp = [
    #     5,  # NSD
    #     [
    #         {'iters': x} for x in [15, 20, 25, 30, 35, 40]
    #     ]
    # ]

   # tmp = [
    #    6,  # ISO
    #    [
            # {'lalpha': x} for x in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 99999]
    #        {'alpha': x} for x in [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.999, 0.9999]
    #    ]
   # ]
    #tmp = [
    #    10,  # Grampa
     #   [
    #        {'Eigtype': x} for x in [0, 1, 2, 3, 4]
    #    ]
    #]
    tmp = [
        13, # Grampa
        [
            {'mu': x} for x in [0.5, 1, 1.5, 2]
        ]
    ]

    # _ISO_args["alpha"] = 0.8
    #_ISO_args["lalpha"] = 40
    # _ISO_args["weighted"] = False

    _algs[:] = alggs(tmp)

    run = list(range(len(tmp[1])))

    iters = 5

    graph_names = [
        "in-arenas",
        "inf-euroroad",
        "ca-netscience",
        "bio-celegans",
       # "MultiMagna"
        #"facebook",
        # "astro",
        # "gnp"
        #"soc-hamsterster",
        #"socfb-Bowdoin47",
        #"socfb-Hamilton46",
        #"socfb-Haverford76",
        #"socfb-Swarthmore42"
    ]

    #graphs = [
    #    (gen.loadnx, ("in-arenas",)),
    #    (gen.loadnx, ("inf-euroroad",)),
     #   (gen.loadnx, ("voles",)),
        #(gen.loadnx, ('data/facebook.txt',)),
        # (gen.loadnx, ('data/CA-AstroPh.txt',)),
        # (nx.gnp_random_graph, (2**15, 0.0003)),
    #]
    graphs = rgraphs(graph_names)
    noises = [
        0.00,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
    ]

    s_trans = (0, 2, 1, 3)
    xlabel = list(tmp[1][0].keys())[0]


def namess(tmp):
    return [name[-15:] for name in tmp[1]]
def namessgt(tmp):
    return [name[-15:] for name in tmp[2]]

def graphss(tmp):
    return [
        (lambda x:x, [[
            tmp[0],
            target,
            None
        ]]) for target in tmp[1]
    ]

def graphss1(tmp):
    x=len(tmp[2])
    return [
        (lambda x:x, [[
            tmp[0],
            tmp[1][i],
            tmp[2][i]
        ]]) for i in range(x)
            
    ]

@ex.named_config
def real_noisetest():

   
    tmp = [
        "data/real world/arenas/arenas_orig.txt",
        [
            f"data/real world/arenas/noise_level_10/edges_{i}.txt" for i in [
                 1]
        ],
        [
            f"data/real world/arenas/noise_level_10/gt_{i}.txt" for i in [
               1]
        ]

    ]
    #xlabel = "CA-AstroPh"
    xlabel = "arenas"
    graph_names = namess(tmp)
    graphs = graphss1(tmp)
    print(graphs)
    run=[11]
    iters = 1

    noises = [
        1.0
    ]

    s_trans = (2, 1, 0, 3)

    # (g,alg,acc,n,i)
    # s_trans = (3, 1, 2, 0, 4)




@ex.named_config
def real_noise():

    #tmp = [
    #   "data/real world/contacts-prox-high-school-2013/contacts-prox-high-school-2013_100.txt",
    #   [
    #       f"data/real world/contacts-prox-high-school-2013/contacts-prox-high-school-2013_{i}.txt" for i in [
    #           99, 95, 90, 80]
    #   ]
    #]
    #xlabel = "high-school-2013"

   # tmp = [
   #     "data/real world/mamalia-voles-plj-trapping/mammalia-voles-plj-trapping_100.txt",
   #     [
   #         f"data/real world/mamalia-voles-plj-trapping/mammalia-voles-plj-trapping_{i}.txt" for i in [
   #             99, 95, 90, 80]
   #     ]
   #  ]
   # xlabel = "mammalia-voles"

    tmp = [
        "data/real world/MultiMagna/yeast0_Y2H1.txt",
        [
             f"data/real world/MultiMagna/yeast{i}_Y2H1.txt" for i in [
                 5]#, 10, 15, 20, 25]
        ]
    ]
    xlabel = "yeast_Y2H1"
    #tmp = [
    #    "data/real world/arenas/arenas_orig.txt",
    #    [
    #        f"data/real world/arenas/noise_level_0/edges_{i}.txt" for i in [
    #            1, 2, 3, 4,5]
    #    ],
    #    [
    #        f"data/real world/arenas/noise_level_0/gt_{i}.txt" for i in [
    #            1, 2, 3, 4,5]
    #    ]

    #]
    #xlabel = "yeast_Y2H1"

    graph_names = namess(tmp)
    #graphs = graphss1(tmp)
    graphs = graphss(tmp)
    print(graphs)
    #run=[9,13,14,15]
    run=[
        Algs.CUGAL_CHACHE_SPARSE_LOG.value,
        Algs.FUGAL.value,
    ]
    iters = 3
    #accs=[0,1,2,3,4,5]
    noises = [
        1.0
    ]


    #s_trans = (2, 1, 0, 3)

    # (g,alg,acc,n,i)
    # s_trans = (3, 1, 2, 0, 4)


def rgraphs(gnames):
    return [
        (gen.loadnx, [(f"data/{name}.txt",)]) for name in gnames
    ]

@ex.named_config
def OT_test():
    run = [
        #Algs.CUGAL_CHACHE_SPARSE.value,
        #Algs.CUGAL_CHACHE_SPARSE_LOG.value,
        #Algs.CUGAL_CHACHE_SPARSE_MIX.value,
        Algs.FUGAL.value,
        Algs.CUGAL_OT.value,
        #Algs.CUGAL_OT_GPU.value
    ]
    iters = 1
    graph_names = [
        Data.CA_NETSCIENCE.value[0]
        ]
    
    graphs = rgraphs(graph_names)

    noises = [ 
        0.00,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        ]        

    noise_type = 1

@ ex.named_config
def hungarian_test():
    run = [
        Algs.CUGAL_HUNGARIAN_CULAP.value,
    ]
    iters = 1
    graph_names = [
        Data.BIO_DMELA.value,
    ]
    graphs = rgraphs(graph_names)
    noises = [ 0.10 ]

@ ex.named_config
def cugal_test():
    run = [
        Algs.CUGAL_CHACHE_SPARSE_LOG.value,
        Algs.CUGAL_LOG_HUN_MORE_GREED.value,
        Algs.CUGAL_LOG_HUN_SPARSE.value,
    ]
    iters = 1
    graph_names = [
        Data.INF_EUROROAD.value,
    ]
    graphs = rgraphs(graph_names) 
    #accs=[0,1,2,3,4,5]
    noises = [
        0, 0.05, 0.10,
    ]
    noise_type = 1

@ ex.named_config
def fugal_test():
    run = [
        Algs.FUGAL.value,
        ]
    iters = 1
    graph_names = [
        Data.CA_GRQC.value
    ]
    graphs = rgraphs(graph_names)
    noises = [ 0.00, 0.05, 0.10, 0.15, 0.20, 0.25 ]

@ ex.named_config
def real():

    #run = [1, 2, 3, 4, 5, 6]
    run = [
        Algs.FUGAL.value,
        #Algs.CUGAL_CHACHE_SPARSE_LOG.value,
        #Algs.CUGAL_CHACHE_SPARSE_MIX.value,
    ]
    #run=[13,14,15]
    iters = 1
    #print("start")
    graph_names = [             # n     / e
        #"ca-netscience",       # 379   / 914   / connected
        #"voles",
        #"high-school",
        #"yeast",
        #"MultiMagna",
        #"bio-celegans",         # 453   / 2k    / connected
        #"in-arenas",            # 1.1k  / 5.4k  / connected
        #"arenad",
        #"inf-euroroad",         # 1.2K  / 1.4K  / disc - 200
        #"inf-power",  
                  # 4.9K  / 6.6K  / connected
        #"ca-GrQc",              # 4.2k  / 13.4K / connected - (5.2k  / 14.5K)?
        #"bio-dmela",            # 7.4k  / 25.6k / connected
        #"CA-AstroPh",  
                 # 18k   / 195k  / connected
        #"soc-hamsterster",      # 2.4K  / 16.6K / disc - 400
        #"socfb-Bowdoin47",      # 2.3K  / 84.4K / disc - only 2
        #"socfb-Hamilton46",     # 2.3K  / 96.4K / disc - only 2
        #"socfb-Haverford76",    # 1.4K  / 59.6K / connected
        #"socfb-Swarthmore42",   # 1.7K  / 61.1K / disc - only 2
        #"soc-facebook",

        #"scc_enron-only",
        #"scc_fb-forum",
        #"scc_fb-messages",
        #"scc_infect-hyper"
                 # 4k    / 87k   / connected
        #"ca-Erdos992",          # 6.1K  / 7.5K  / disc - 100 + 1k disc nodes
    ]
    print("done")
    graphs = rgraphs(graph_names)

    accs=[0, 1, 2, 3, 4, 5]

    noises = [
        0.00,
        #0.01,
        0.02,
        # 0.03,
        0.04,
        0.06,
    ]
    noise_type = 1


@ ex.named_config
def synthetic():

    # use with 'mall'

    iters = 1
    #run = [1,6,9,10,11,14,15]
    #run = [1,6,14,15] #9,10,11
    run=[
        Algs.FUGAL.value,
        Algs.CUGAL_CHACHE_SPARSE_MIX.value,
        #Algs.CUGAL_MIX_FW_010.value,
        #Algs.CUGAL_MIX_FW_020.value,
        Algs.CUGAL_HUNGARIAN_MORE_GREED.value,
        #Algs.CUGAL_MIX_HUN_MORE_GREED_FW_10.value,
        #Algs.CUGAL_HUNGARIAN_MORE_GREED_FW_20.value,
        Algs.CUGAL_CHACHE_SPARSE_LOG.value,
        #Algs.CUGAL_LOG_FW_10.value,
        #Algs.CUGAL_LOG_FW_20.value,
        Algs.CUGAL_LOG_HUN_MORE_GREED.value,
        #Algs.CUGAL_LOG_HUN_MORE_GREED_FW_10.value,
        #Algs.CUGAL_LOG_HUN_MORE_GREED_FW_20.value,
    ]
    graph_names = [
        #"arenas",
        #"powerlaw",
        #"nw_str",
        #"watts_str",
        #"gnp",
        #"barabasi",

        #"nw_str512",
        #"nw_str1024",
        #"nw_str2048",
        #"nw_str4096",
        #"nw_str8192",
    ]

    graphs = [
        #newman_watts:
        [nx.newman_watts_strogatz_graph, (1000, 7, 0.03)] for p in np.linspace(0.05, 0.25, num=20)
        #[nx.newman_watts_strogatz_graph, (1000, k, 0.5)] for k in [7, 10, 15, 25, 50, 100]
        #Barabasi:
        #[nx.barabasi_albert_graph, (1000, m)] for m in [3, 5, 7, 10, 15, 25, 50, 100]
        #Erdos-Renyi:
        #[nx.gnp_random_graph, (1000, p)] for p in np.linspace(0.001, 0.005, num=5)
        #Powerlaw:
        #[nx.powerlaw_cluster_graph, (1000, 2, p)] for p in np.linspace(0.2, 1, num=5)
        #Lobster:
        #[nx.random_lobster, (1000, 0.5, 0.5)] for p in np.linspace(0.2, 0.8, num=1)
    ]

    noises = [
        0.05
    ]
    accs = [0, 1, 2, 3, 4, 5]


@ ex.named_config
def tuned():
    _CONE_args["dim"] = 512
    _LREA_args["iters"] = 40
    _ISO_args["alpha"] = 0.9
    _ISO_args["lalpha"] = 100000  # full dim
    # _ISO_args["lalpha"] = 25


@ ex.named_config
def test():

    graph_names = [
        "test1",
        "test2",
    ]

    graphs = [
        # (gen.loadnx, ('data/arenas.txt',)),
        (nx.gnp_random_graph, (50, 0.5)),
        (nx.barabasi_albert_graph, (50, 3)),
    ]

    run = [1, 3, 5]

    iters = 4

    noises = [
        0.00,
        0.01,
        0.02,
        0.03,
        0.04,
    ]
