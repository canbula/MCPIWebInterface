import os
import time
import random
from multiprocessing import Process, Array, Condition
from threading import Thread
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import numpy as np
import io
import base64


g_pi = 0
g_i = 0
g_n = 0
g_t = 0
g_l = 0
g_p = np.array([])
g_x = np.array([])
g_y = np.array([])
g_done = 'n'


app = Flask(__name__)


def append_new_line(file_name, text_to_append):
    with open(file_name, "a+") as file_object:
        file_object.seek(0)
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        file_object.write(text_to_append)


class TicToc:
    def __init__(self):
        self.t1 = 0
        self.t2 = 0

    def tic(self):
        self.t1 = time.time()

    def toc(self):
        self.t2 = time.time()
        return self.t2 - self.t1


class FindPi:
    def __init__(self):
        self.n = 0
        self.i = 0

    def throw_points(self, nn, p, all_i, all_n, all_s, file_lock):
        for _ in range(nn):
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            if file_lock.acquire():
                append_new_line("points.txt", f"{p:2d} {x:7.4f} {y:7.4f}")
                file_lock.notify()
                file_lock.release()
            self.n += 1
            if (x**2 + y**2) <= 1:
                self.i += 1
            all_i[p] = self.i
            all_n[p] = self.n
        all_s[p] = 0


def calculate_pi(n, p):
    tt = TicToc()
    tt.tic()
    find_pis = []
    processes = []
    shared_i = Array('i', [0]*p)
    shared_n = Array('i', [0]*p)
    shared_s = Array('i', [1]*p)
    file_lock = Condition()
    for i in range(p):
        find_pis.append(FindPi())
        processes.append(Process(target=find_pis[i].throw_points,
                                 args=(int(n/p), i, shared_i, shared_n, shared_s, file_lock)))
    for process in processes:
        process.start()
    while sum(shared_n) == 0:
        time.sleep(0.1)
    while sum(shared_s) > 0:
        globals()['g_pi'] = 4 * sum(shared_i) / sum(shared_n)
        globals()['g_i'] = sum(shared_i)
        globals()['g_n'] = sum(shared_n)
        globals()['g_t'] = tt.toc()
    globals()['g_done'] = 'y'


@app.route('/plot.html')
def build_plot():
    file_exists = False
    while not file_exists:
        if os.path.exists("points.txt") and globals()['g_l'] < globals()['g_n']:
            file_exists = True
        time.sleep(0.1)
    p, x, y = np.loadtxt('points.txt', skiprows=g_l, usecols=(0, 1, 2), unpack=True)
    globals()['g_l'] += p.size
    globals()['g_p'] = np.append(g_p, p, axis=0)
    globals()['g_x'] = np.append(g_x, x, axis=0)
    globals()['g_y'] = np.append(g_y, y, axis=0)
    img = io.BytesIO()
    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])
    plt.plot(g_x, g_y, 'o', markersize=0.8, color='blue')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().add_patch(plt.Circle((0, 0), 1, color='r', fill=False))
    plt.axis('off')
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return '<img src="data:image/png;base64,{}">'.format(plot_url)


@app.route('/results.html')
def results():
    outputs = {'pi': f"{g_pi:.8f}", 'inner': f"{g_i}", 'total': f"{g_n}", 'time': f"{g_t:.1f}", 'done': f"{g_done}"}
    return render_template("results.html", outputs=outputs)


@app.route('/', methods=['GET', 'POST'])
def index():
    globals()['g_pi'] = 0
    globals()['g_i'] = 0
    globals()['g_n'] = 0
    globals()['g_t'] = 0
    globals()['g_l'] = 0
    globals()['g_p'] = np.array([])
    globals()['g_x'] = np.array([])
    globals()['g_y'] = np.array([])
    globals()['g_done'] = 'n'
    n = request.form.get('number_of_points')
    p = request.form.get('number_of_processors')
    show_output = False
    if n is not None:
        if os.path.exists("points.txt"):
            os.remove("points.txt")
        show_output = True
        Thread(target=calculate_pi, args=(int(n), int(p))).start()
        number_of_points = int(int(n)/int(p))*int(p)
        number_of_processors = p
    else:
        number_of_points = 100
        number_of_processors = 1
    return render_template("index.html",
                           title="Monte Carlo PI Web Interface",
                           number_of_points=number_of_points,
                           number_of_processors=number_of_processors,
                           max_number_of_processors=os.cpu_count(),
                           show_output=show_output)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
