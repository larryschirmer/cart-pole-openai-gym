import multiprocessing as mp
import numpy as np


def get_slice(list_name, step, i): return list_name[step*i:step*i+step]


def square(x):
    return np.square(x)


thread_count = mp.cpu_count()
pool = mp.Pool(thread_count)

x = np.arange(thread_count**2)
print("x: {}".format(x))

number_to_process = [get_slice(x, thread_count, i) for i in range(thread_count)]
squared = pool.map(square, number_to_process)
print(squared)
