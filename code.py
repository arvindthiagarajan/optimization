from itertools import permutations
import numpy as np

def check_weight(A,B,weightfrac=0.25):
    return (1-weightfrac) <= B/A <= 1

def check_vol(A,B,volfrac=0.25):
    return (1-volfrac) <= np.prod(B)/np.prod(A) <= volfrac

def check_dim(A,B,dimfrac=0.25):
    for pA in permutations(A):
        v = B/pA
        if ((1-dimfrac) < v).all() and (v < dimfrac).all():
            return True
    return False

def full_check(A,B):
    return check_weight(A[-1],B[-1]) and (check_dim(A[:-1],B[:-1]) or check_vol(A[:-1],B[:-1]))


def create_graph(filepath):
    with open(filepath, 'r') as f:
        boxes = [np.array(tuple(map(float, a.split(' ')))) for a in f.read().split('\n') if a != '']

    check = lambda i,j: i == j or full_check(boxes[i], boxes[j])
    return np.fromfunction(check, (len(boxes), len(boxes)), dtype=int)

def find_cover(graph, N = 40, cstr_wt = (1,1024),
               dir_bal = 0.5, beta = (1,7)):

    params = {'db' : dir_bal, 'beta' : beta, 'lr' : cstr_wt}
    for k,v in list(params.items()):
        if not isinstance(v, tuple): v = (v, v)
        params[k] = np.geomspace(*v, N)

    choices = []
    satisfied = set()
    uncovered = np.logical_not(np.any(graph[choices], axis=0))

    for loss_ratio, beta, directional_balance in zip(params['lr'], params['beta'], params['db']):
        loss = len(choices) + loss_ratio * uncovered.sum()
        for j in range(5000):
            add = np.nonzero(np.dot(graph, uncovered))[0]
            sub = choices[:]

            if len(add) > 0 and (len(sub) == 0 or np.random.rand() > directional_balance):
                new_choices = choices + [np.random.choice(add)]
            else:
                new_choices = np.random.choice(sub, size=len(choices)-1).tolist()

            new_uncovered = np.logical_not(np.any(graph[new_choices], axis=0))
            new_coverage_loss = new_uncovered.sum()
            new_loss = len(new_choices) + loss_ratio * new_coverage_loss
            if np.random.exponential() > beta*(new_loss - loss):
                uncovered = new_uncovered
                choices = new_choices
                loss = new_loss
                if new_coverage_loss == 0:
                    satisfied.add(tuple(sorted(choices)))

    return min(list(satisfied), key=len)
