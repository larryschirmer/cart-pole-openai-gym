import multiprocessing as mp
import numpy as np


def get_slice(list_name, step, i): return list_name[step*i:step*i+step]


def square(i, x, queue):
    print("In process {}".format(i,))
    queue.put(np.square(x))


thread_count = mp.cpu_count()
pool = mp.Pool(thread_count)
processes = []
queue = mp.Queue()
x = np.arange(thread_count**2)

print("x: {}".format(x))

for thread_index in range(thread_count):
    proc = mp.Process(target=square, args=(
        thread_index, get_slice(x, thread_count, thread_index), queue))

    proc.start()
    processes.append(proc)

for proc in processes:
    proc.join()

for proc in processes:
    proc.terminate()

squared_results = []
while not queue.empty():
    squared_results.append(queue.get())

print(squared_results)
