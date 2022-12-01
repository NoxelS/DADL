import time

timeit_output = []
ts = time.time()

# Wrapper to track time spent in a function
def timeit(func):
    def wrapper(*args, **kw):
        ts = time.time()
        result = func(*args, **kw)
        te = time.time()
        timeit_output.append([func.__name__, args, kw, te-ts])
        return result
    return wrapper

# Wrapper to output function name and arguments and return value
def trace(func):
    def wrapper(*args, **kw):
        print("Calling %s with args: %s, %s" % (func.__name__, args, kw))
        result = func(*args, **kw)
        print("Returning %s" % result)
        return result
    return wrapper

# Output all timeit results
def output_timeit():
    elapsed = time.time() - ts
    for line in timeit_output:
        name, args, kw, time_e = line
        rel_time = time_e / elapsed
        kw = ", ".join(["%s=%s" % (k,v) for k,v in kw.items()])
        kw = ", " + kw if kw else kw
        args = str(args).replace("(", "").replace(")", "") + kw
        print("%s(%s) took %.4fs (%.2f %% of total time))" % (name, args, time_e, rel_time * 100))


def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)


def catalan(n):
    return (factorial(2*n)//(factorial(n+1)*factorial(n)))

def swap(A, i, j):
    A[i], A[j] = A[j], A[i]