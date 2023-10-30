import numpy as np

def Assort():
    graph = np.genfromtxt('signed_net.csv',delimiter = ',')
    posMat = np.multiply(graph,graph>0)
    negMat = abs(np.multiply(graph,graph<0))
    l = []
    print(negMat)
    
    N = graph.shape[0]
    print(N)

    M = -1
    jkMult = 0
    jkSum = 0
    jkSumSq = 0

    

    print("r^+(+,+)")
    M = posMat.sum()/2
    print("M",M)
    for j in range(N-1):
        for k in range(j+1,N):
            if (posMat[j,k] == 0):
                continue
            kDeg = posMat[k,:].sum()
            jDeg = posMat[j,:].sum()
            jkMult += kDeg * jDeg
            jkSum += kDeg + jDeg
            jkSumSq += (kDeg ** 2) + (jDeg ** 2)
    arg1Num = 1/M * jkMult
    arg2Num = (1/M * 1/2 * jkSum)**2
    arg1Den = 1/M * 1/2 * jkSumSq
    arg2Den = (1/M * 1/2 * jkSum)**2
    l.append(np.nan_to_num((arg1Num-arg2Num)/(arg1Den-arg2Den),nan=0))
    print(l)
    

    
    print("\n^-(+,+)")
    M = negMat.sum()/2
    print("M",M)
    jkMult = 0
    jkSum = 0
    jkSumSq = 0

    for j in range(N-1):
        for k in range(j+1,N):
            if (negMat[j,k] == 0):
                continue
            kDeg = posMat[k,:].sum()
            jDeg = posMat[j,:].sum()
            jkMult += kDeg * jDeg
            jkSum += kDeg + jDeg
            jkSumSq += (kDeg ** 2) + (jDeg ** 2)
    arg1Num = 1/M * jkMult
    arg2Num = (1/M * 1/2 * jkSum)**2
    arg1Den = 1/M * 1/2 * jkSumSq
    arg2Den = (1/M * 1/2 * jkSum)**2
    l.append(np.nan_to_num((arg1Num-arg2Num)/(arg1Den-arg2Den),nan=0))
    print(l[-1])

    print("\n^+(-,-)")
    M = posMat.sum()/2
    #print(M)
    jkMult = 0
    jkSum = 0
    jkSumSq = 0

    for j in range(N-1):
        for k in range(j+1,N):
            if (posMat[j,k] == 0):
                continue
            jDeg = negMat[j,:].sum()
            kDeg = negMat[k,:].sum()
            jkMult += kDeg * jDeg
            jkSum += kDeg + jDeg
            jkSumSq += (kDeg ** 2) + (jDeg ** 2)
    arg1Num = 1/M * jkMult
    arg2Num = (1/M * 1/2 * jkSum)**2
    arg1Den = 1/M * 1/2 * jkSumSq
    arg2Den = (1/M * 1/2 * jkSum)**2
    l.append(np.nan_to_num((arg1Num-arg2Num)/(arg1Den-arg2Den),nan=0))
    print(l[-1])


    print("\n^-(-,-)")
    M = negMat.sum()/2
    #print(M)
    jkMult = 0
    jkSum = 0
    jkSumSq = 0

    for j in range(N-1):
        for k in range(j+1,N):
            if (negMat[j,k] == 0):
                continue
            jDeg = negMat[j,:].sum()
            kDeg = negMat[k,:].sum()
            jkMult += kDeg * jDeg
            jkSum += kDeg + jDeg
            jkSumSq += (kDeg ** 2) + (jDeg ** 2)
    arg1Num = 1/M * jkMult
    arg2Num = (1/M * 1/2 * jkSum)**2
    arg1Den = 1/M * 1/2 * jkSumSq
    arg2Den = (1/M * 1/2 * jkSum)**2
    l.append(np.nan_to_num((arg1Num-arg2Num)/(arg1Den-arg2Den),nan=0))
    print(l[-1])


    print("\n^+(+,-)")
    M = posMat.sum()/2
    #print(M)
    jkMult = 0
    jkSum = 0
    jkSumSq = 0

    for j in range(N-1):
        for k in range(j+1,N):
            if (posMat[j,k] == 0):
                continue
            jDeg = posMat[j,:].sum()
            kDeg = negMat[k,:].sum()
            jkMult += kDeg * jDeg
            jkSum += kDeg + jDeg
            jkSumSq += (kDeg ** 2) + (jDeg ** 2)
    arg1Num = 1/M * jkMult
    arg2Num = (1/M * 1/2 * jkSum)**2
    arg1Den = 1/M * 1/2 * jkSumSq
    arg2Den = (1/M * 1/2 * jkSum)**2
    l.append(np.nan_to_num((arg1Num-arg2Num)/(arg1Den-arg2Den),nan=0))
    print(l[-1])

    print("\n^-(+,-)")
    M = negMat.sum()/2
    #print(M)
    jkMult = 0
    jkSum = 0
    jkSumSq = 0

    for j in range(N-1):
        for k in range(j+1,N):
            if (negMat[j,k] == 0):
                continue
            jDeg = posMat[j,:].sum()
            kDeg = negMat[k,:].sum()
            jkMult += kDeg * jDeg
            jkSum += kDeg + jDeg
            jkSumSq += (kDeg ** 2) + (jDeg ** 2)
    arg1Num = 1/M * jkMult
    arg2Num = (1/M * 1/2 * jkSum)**2
    arg1Den = 1/M * 1/2 * jkSumSq
    arg2Den = (1/M * 1/2 * jkSum)**2
    l.append(np.nan_to_num((arg1Num-arg2Num)/(arg1Den-arg2Den),nan=0))
    print(l[-1])

    
    return