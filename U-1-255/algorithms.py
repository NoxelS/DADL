from utils import swap, timeit

def PrintPerm(A, n):
    if(n==1):
        print(" ".join([str(x) for x in A]))
    else:
        for i in range(n-1):
            PrintPerm(A, n-1)
            if(n%2==0):
                swap(A, i, n-1)
            else:
                swap(A, 0, n-1)
        PrintPerm(A, n-1)

def PermIter(A, n):
    if(n==1):
        yield A
    else:
        for i in range(n-1):
            yield from PermIter(A, n-1)
            if(n%2==0):
                swap(A, i, n-1)
            else:
                swap(A, 0, n-1)
        yield from PermIter(A, n-1)