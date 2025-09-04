import warnings
warnings.filterwarnings("ignore")

import panel as pn
import matplotlib.pyplot as plt

from my_utils import define_setup, load_models, sample, get_features, get_labels, eval_models, init_features

pn.extension(template='fast', sizing_mode="stretch_width")

sliders = []
sliders.append(pn.widgets.FloatSlider(start=0, end=10, value=0, name='Shift strength'))

case = '3-conf'
conf = 'observed'
n_noise = 'normal'
scm = 'linear'
n = 1000

adj, noise, mechanism, _  = define_setup(case, n_noise, scm)
models = load_models(case, conf, n_noise, scm, 'lr', n)
features = init_features(case, conf)

names = ['All-Features', 'Causal']

def plot(*shifts):
    # anti-causal shift
    interventions = {2:shifts[0]}

    test_data = sample(n, adj, mechanism, noise, interventions)
    X_test = get_features(test_data)
    y_test = get_labels(test_data)
    results = eval_models(X_test, y_test, models, features)

    fig, ax = plt.subplots()
    graph = ax.bar(names, [results['all'], results['causal']], color=['#8f1402', '#95d0fc'])
    ax.bar_label(graph)
    ax.set_ylabel("OOD Accuracy")
    ax.set_ylim(top=1.0)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    plt.tight_layout()
    plt.close(fig)

    return fig

def reset_smoking(event):
    for s in sliders:
        s.value = 0


pn.Row(pn.pane.PNG('Smoking_Example_new.png', height=300)).servable(target='smoking_img')
pn.Column(*sliders, width=450).servable(target='smoking_app')
pn.Row(pn.pane.Matplotlib(pn.bind(plot, *sliders)), height=350).servable(target='smoking_plot')