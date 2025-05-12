import warnings
warnings.filterwarnings("ignore")

import panel as pn
import matplotlib.pyplot as plt

from pyscript import document, display
from my_utils import define_setup, load_models, sample, get_features, get_labels, eval_models, init_features

pn.extension(sizing_mode="stretch_width")

slider = pn.widgets.FloatSlider(start=-10, end=10, name='Shift')

plt.ion()

def callback(new):
    results = refresh_models()
    output_div = document.querySelector("#output")
    output_div.innerText = f"all-features = {results['all']:.3f}, causal = {results['causal']:.3f}"
    refresh_plot(results)
    return ''

def run(event):
    results = refresh_models()
    output_div = document.querySelector("#output")
    output_div.innerText = f"all-features = {results['all']:.3f}, causal = {results['causal']:.3f}"
    refresh_plot(results)

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

def refresh_plot(results):
    global graph

    graph.remove()
    graph = ax.bar(names, [results['all'], results['causal']], color=['red', 'green'])
    plt_pane.param.trigger("object")


case = 'TC-conf'
conf = 'observed'
n_noise = 'normal'
scm = 'linear'
n = 1000

adj, noise, mechanism, _  = define_setup(case, n_noise, scm)
models = load_models(case, conf, n_noise, scm, 'lr', n)
features = init_features(case, conf)

results = refresh_models()
output_div = document.querySelector("#output")
output_div.innerText = f"all-features = {results['all']:.3f}, causal = {results['causal']:.3f}"

names = ['all-features', 'causal']
fig, ax = plt.subplots()

graph = ax.bar(names, [results['all'], results['causal']], color=['red', 'green'])

ax.set_ylabel("OOD accuracy")
ax.set_title("intervention: anti-causal")

plt.close(fig)
plt_pane = pn.pane.Matplotlib(fig)

pn.Row(slider, pn.bind(callback, slider)).servable(target='simple_app')

pn.Row(plt_pane).servable(target='plot')