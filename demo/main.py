import warnings
warnings.filterwarnings("ignore")

import cv2

import panel as pn
import matplotlib.pyplot as plt

from pyscript import document, display
from my_utils import define_setup, load_models, sample, get_features, get_labels, eval_models, init_features

pn.extension(sizing_mode="stretch_width")

#document.querySelector("img").setAttribute("src", img)

slider = pn.widgets.FloatSlider(start=-10, end=10, value=0, name='Shift')

def callback(new):
    results = refresh_models()
    refresh_plot(results)

def run(event):
    slider.value = 0

def generate_test_samples(n, adj, mechanism, noise, shift_interventions):
    return sample(n, adj, mechanism, noise, shift_interventions)

def refresh_models():
    # anti-causal feature
    intervention = {2: slider.value}

    test_data = generate_test_samples(n, adj, mechanism, noise, intervention)
    X_test = get_features(test_data)
    y_test = get_labels(test_data)
    results = eval_models(X_test, y_test, models, features)

    return results

def add_bar_labels(x, y):
    labels = []
    for i in range(len(x)):
        labels.append(ax.text(i, y[i]+0.01, f"{y[i]:.3f}", ha='center'))
    return labels

def remove_bar_labels(labels):
    for l in labels:
        l.remove()

def refresh_plot(results):
    global graph, bl

    remove_bar_labels(bl)
    graph.remove()
    graph = ax.bar(names, [results['all'], results['causal']], color=['red', 'green'])
    bl = add_bar_labels(names, [results['all'], results['causal']])
    plt_pane.param.trigger("object")


case = 'TC-conf'
conf = 'observed'
n_noise = 'normal'
scm = 'linear'
n = 1000

fig_g, ax_g = plt.subplots()
graph_img = cv2.imread("graph.svg")
ax_g.imshow(graph_img)
plt.close(fig_g)
graph_pane = pn.pane.Matplotlib(fig_g)
pn.Row(graph_pane).servable(target="graph_img")
#document.querySelector("img").setAttribute("src", graph_pane)

adj, noise, mechanism, _  = define_setup(case, n_noise, scm)
models = load_models(case, conf, n_noise, scm, 'lr', n)
features = init_features(case, conf)

results = refresh_models()

names = ['all-features', 'causal']
fig, ax = plt.subplots()

graph = ax.bar(names, [results['all'], results['causal']], color=['red', 'green'])
bl = add_bar_labels(names, [results['all'], results['causal']])

ax.set_ylabel("OOD accuracy")
ax.set_title("intervention: anti-causal")

plt.close(fig)
plt_pane = pn.pane.Matplotlib(fig)

pn.Row(slider, pn.bind(callback, slider)).servable(target='simple_app')
pn.Row(plt_pane).servable(target='plot')