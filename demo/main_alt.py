import warnings
warnings.filterwarnings("ignore")

import panel as pn
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from my_utils import define_setup, load_models, sample, get_features, get_labels, eval_models, init_features

pn.extension(template='fast', sizing_mode="stretch_width")

sliders = []
sliders.append(pn.widgets.FloatSlider(start=-10, end=10, value=0, name='T shift').servable(target='simple_app'))
sliders.append(pn.widgets.FloatSlider(start=-10, end=10, value=0, name='C shift').servable(target='simple_app'))
sliders.append(pn.widgets.FloatSlider(start=-10, end=10, value=0, name='AC shift').servable(target='simple_app'))
sliders.append(pn.widgets.FloatSlider(start=-10, end=10, value=0, name='S shift').servable(target='simple_app'))
sliders.append(pn.widgets.FloatSlider(start=-10, end=10, value=0, name='TCC shift').servable(target='simple_app'))
sliders.append(pn.widgets.FloatSlider(start=-10, end=10, value=0, name='TACC shift').servable(target='simple_app'))
sliders.append(pn.widgets.FloatSlider(start=-10, end=10, value=0, name='TSC shift').servable(target='simple_app'))


case = '3-conf'
conf = 'observed'
n_noise = 'normal'
scm = 'linear'
n = 1000

adj, noise, mechanism, _  = define_setup(case, n_noise, scm)
models = load_models(case, conf, n_noise, scm, 'lr', n)
features = init_features(case, conf)

names = ['all-features', 'causal']

def show_img():
    img = np.asarray(Image.open('graph.png'))
    fig, ax = plt.subplots()
    ax.imshow(img)
    plt.axis('off')
    plt.close(fig)
    return fig

def plot(*shifts):
    interventions = {}
    for i, s in enumerate(shifts):
        interventions[i] = s

    test_data = sample(n, adj, mechanism, noise, interventions)
    X_test = get_features(test_data)
    y_test = get_labels(test_data)
    results = eval_models(X_test, y_test, models, features)

    fig, ax = plt.subplots()
    graph = ax.bar(names, [results['all'], results['causal']], color=['red', 'green'])
    ax.bar_label(graph)
    ax.set_ylabel("OOD accuracy")
    plt.close(fig)

    return fig

def run(event):
    for s in sliders:
        s.value = 0

g_fig = show_img()
pn.Row(pn.pane.Matplotlib(g_fig)).servable(target='graph_img')

pn.Column(pn.pane.Matplotlib(pn.bind(plot, *sliders))).servable(target='plot')