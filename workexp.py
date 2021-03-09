from algorithms import regal, eigenalign, conealign, netalign, NSD, klaus, gwl
from data import ReadFile, similarities_preprocess
from evaluation import evaluation, evaluation_design
from sacred import Experiment
import numpy as np
import scipy.sparse as sps

ex = Experiment("experiment")


# def print(*args):
#     pass


@ex.config
def global_config():
    noise_level = 1
    edges = 1
    _lim = 0

    data1 = "data/arenas_orig.txt"
    data2 = f"data/noise_level_{noise_level}/edges_{edges}.txt"
    gt = f"data/noise_level_{noise_level}/gt_{edges}.txt"

    gma, gmb = ReadFile.gt1(gt)
    G1 = ReadFile.edgelist_to_adjmatrix1(data1)
    G2 = ReadFile.edgelist_to_adjmatrix1(data2)
    adj = ReadFile.edgelist_to_adjmatrixR(data1, data2)

    Ae = np.loadtxt(data1, int)
    Be = np.loadtxt(data2, int)


@ex.named_config
def demo():
    _lim = 10
    # _data = "data/arenas_orig.txt"

    gma = np.arange(_lim)
    gmb = np.arange(_lim)
    np.random.shuffle(gmb)

    Ae = np.loadtxt("data/arenas_orig.txt", int)
    Ae = Ae[np.where(Ae < _lim, True, False).all(axis=1)]
    Be = gmb[Ae]

    # print(Ae.T)
    # print(Be.T)
    print(np.array([gma, gmb]).T)


@ex.capture
def eval_regal(_log, gma, gmb, adj):
    alignmatrix = regal.main(adj)
    mar, mbr = evaluation.transformRAtoNormalALign(alignmatrix)
    acc = evaluation.accuracy(gma, gmb, mbr, mar)
    _log.info(f"acc1: {acc}")

    return acc


@ex.capture
def eval_eigenalign(_log, gma, gmb, G1, G2):
    ma1, mb1, _, _ = eigenalign.main(G1, G2, 8, "lowrank_svd_union", 3)
    acc = evaluation.accuracy(gma+1, gmb+1, mb1, ma1)
    _log.info(f"acc: {acc}")

    return acc


@ex.capture
def eval_conealign(_log, gma, gmb, G1, G2):
    alignmatrix = conealign.main(G1, G2)
    mar, mbr = evaluation.transformRAtoNormalALign(alignmatrix)
    acc = evaluation.accuracy(gma, gmb, mbr, mar)
    _log.info(f"acc1: {acc}")

    return acc


@ex.capture
def eval_netalign(_log, Ae, Be, gma, gmb):
    Ai, Aj = Ae.T
    n = max(max(Ai), max(Aj)) + 1
    nedges = len(Ai)
    Aw = np.ones(nedges)
    A = sps.csr_matrix((Aw, (Ai, Aj)), shape=(n, n), dtype=int)
    A = A + A.T

    Bi, Bj = Be.T
    m = max(max(Bi), max(Bj)) + 1
    medges = len(Bi)
    Bw = np.ones(medges)
    B = sps.csr_matrix((Bw, (Bi, Bj)), shape=(m, m), dtype=int)
    B = B + B.T

    L = similarities_preprocess.create_L(A, B, alpha=2)
    S = similarities_preprocess.create_S(A, B, L)
    li, lj, w = sps.find(L)

    matching = netalign.main(S, w, li, lj, a=0)
    print(matching.T)

    ma, mb = matching
    mab = gmb[ma]
    # print(np.array([mab, mb]).T)
    matches = np.sum(mab == mb)
    print("acc:", (matches/len(gma), matches/len(ma)))


@ex.capture
def eval_NSD(_log, gma, gmb, G1, G2):
    ma, mb = NSD.run(G1, G2)
    print(ma)
    print(mb)
    print(gmb)
    acc = evaluation.accuracy(gma, gmb, mb, ma)
    print(acc)


@ex.capture
def eval_klaus(_log, Ae, Be, gma, gmb):
    Ai, Aj = Ae.T
    n = max(max(Ai), max(Aj)) + 1
    nedges = len(Ai)
    Aw = np.ones(nedges)
    A = sps.csr_matrix((Aw, (Ai, Aj)), shape=(n, n), dtype=int)
    A = A + A.T

    Bi, Bj = Be.T
    m = max(max(Bi), max(Bj)) + 1
    medges = len(Bi)
    Bw = np.ones(medges)
    B = sps.csr_matrix((Bw, (Bi, Bj)), shape=(m, m), dtype=int)
    B = B + B.T

    L = similarities_preprocess.create_L(A, B, alpha=4)
    S = similarities_preprocess.create_S(A, B, L)
    li, lj, w = sps.find(L)

    # S = "data/karol/S.txt"
    # li = "data/karol/li.txt"
    # lj = "data/karol/lj.txt"

    # S = ReadFile.edgelist_to_adjmatrix1(S)
    # li = np.loadtxt(li, int)
    # lj = np.loadtxt(lj, int)
    # li -= 1
    # lj -= 1
    # w = np.ones(len(li))

    matching = klaus.main(S, w, li, lj, a=0, maxiter=10)
    print(matching.T)

    ma, mb = matching
    mab = gmb[ma]
    # print(np.array([mab, mb]).T)
    matches = np.sum(mab == mb)
    print("acc:", (matches/len(gma), matches/len(ma)))


@ex.capture
def eval_gwl(_log, data1, data2):
    A = np.loadtxt(data1, int)
    n = np.amax(A) + 1
    B = np.loadtxt(data2, int)
    m = np.amax(B) + 1

    data = {
        'src_index': {float(i): i for i in range(n)},
        'src_interactions': A.tolist(),
        'tar_index': {float(i): i for i in range(m)},
        'tar_interactions': B.tolist(),
        'mutual_interactions': None
    }

    gwl.main(data)


@ex.automain
def main(data1):
    # eval_regal()
    # eval_eigenalign()
    # eval_conealign()
    # eval_netalign()
    # eval_NSD()
    eval_klaus()
    # eval_gwl()
