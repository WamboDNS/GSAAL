import random

def featureSubspaceSelection(dimension, k):
    dims = random.choices(range(1,dimension), k=k)
    subspaces = [[0,1]]
    for i in range(k-1):
        newSpace = random.sample(range(dimension), dims[i])
        if i != 0:
            loop = True
            while loop:
                for j in range(i-1,-1,-1):
                    if set(newSpace) == set(subspaces[j]):
                        newSpace = random.sample(range(dimension), dims[i])
                        loop = True
                        break
                    loop = False
        subspaces.append(newSpace)
    with open('subspaces.txt', 'w') as f:
        for i in range(len(subspaces)):
            f.write("{}\n".format(subspaces[i]))
    return subspaces