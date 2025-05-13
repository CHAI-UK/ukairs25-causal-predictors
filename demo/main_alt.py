import warnings
warnings.filterwarnings("ignore")

import panel as pn
import matplotlib.pyplot as plt

from my_utils import define_setup, load_models, sample, get_features, get_labels, eval_models, init_features

pn.extension(template='fast', sizing_mode="stretch_width")

sliders = []
sliders.append(pn.widgets.FloatSlider(start=-10, end=10, value=0, name='T shift'))
sliders.append(pn.widgets.FloatSlider(start=-10, end=10, value=0, name='C shift'))
sliders.append(pn.widgets.FloatSlider(start=-10, end=10, value=0, name='AC shift'))
sliders.append(pn.widgets.FloatSlider(start=-10, end=10, value=0, name='S shift'))
sliders.append(pn.widgets.FloatSlider(start=-10, end=10, value=0, name='TCC shift'))
sliders.append(pn.widgets.FloatSlider(start=-10, end=10, value=0, name='TACC shift'))
sliders.append(pn.widgets.FloatSlider(start=-10, end=10, value=0, name='TSC shift'))


case = '3-conf'
conf = 'observed'
n_noise = 'normal'
scm = 'linear'
n = 1000

adj, noise, mechanism, _  = define_setup(case, n_noise, scm)
models = load_models(case, conf, n_noise, scm, 'lr', n)
features = init_features(case, conf)

names = ['all-features', 'causal']

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
    ax.set_ylim(top=1.0)
    plt.close(fig)

    return fig

def run(event):
    for s in sliders:
        s.value = 0


pn.Row(pn.pane.PNG('graph.png')).servable(target='graph_img')
pn.Column(*sliders, height=400).servable(target='simple_app')
pn.Row(pn.pane.Matplotlib(pn.bind(plot, *sliders)), height=450).servable(target='plot')