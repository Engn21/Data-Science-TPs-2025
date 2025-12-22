import math

# joint pmf as dictionary {(u,v,w): p}
pmf = {
    (0,0,0): 1/4,
    (0,0,1): 0,
    (0,1,0): 1/4,
    (0,1,1): 1/8,
    (1,0,0): 0,
    (1,0,1): 1/8,
    (1,1,0): 0,
    (1,1,1): 1/4
}

def entropy(prob_dict):
    return -sum(p*math.log(p,2) for p in prob_dict.values() if p>0)

# marginals
def marginal(var):
    res={}
    for key,p in pmf.items():
        v=key[var]
        res[v]=res.get(v,0)+p
    return res

HU = entropy(marginal(0))
HV = entropy(marginal(1))
HW = entropy(marginal(2))

def conditional_entropy(X,Y):
    marginal_Y = marginal(Y)
    cond_H=0
    for y,py in marginal_Y.items():
        cond={}
        for (u,v,w),p in pmf.items():
            vals=(u,v,w)
            if vals[Y]==y:
                x=vals[X]
                cond[x]=cond.get(x,0)+p
        for x in cond:
            cond[x] /= py
        cond_H += py * entropy(cond)
    return cond_H

HU_V = conditional_entropy(0,1)
HV_U = conditional_entropy(1,0)
HW_U = conditional_entropy(2,0)

def mutual_info(X,Y):
    return entropy(marginal(X)) - conditional_entropy(X,Y)

IUV = mutual_info(0,1)
IUW = mutual_info(0,2)
IVW = mutual_info(1,2)

HUVW = entropy({k:p for k,p in pmf.items()})

# Results
print("H(U) =", HU)
print("H(V) =", HV)
print("H(W) =", HW)

print("H(U|V) =", HU_V)
print("H(V|U) =", HV_U)
print("H(W|U) =", HW_U)

print("I(U;V) =", IUV)
print("I(U;W) =", IUW)
print("I(V;W) =", IVW)

print("H(U,V,W) =", HUVW)
