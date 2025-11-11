"""Script to generate the miniatures used for the website homepage grid.

Due to the html structure of the page, the 6 figures below should aim for a 2/1 ratio.
Their width should be roughtly twice their height.
"""
from pathlib import Path
import arviz as az
from cycler import cycler
import matplotlib.pyplot as plt
import numpy as np


az.rcParams["plot.backend"] = "matplotlib"
az.style.use("arviz-variat")
# modiry color cycle to match doc website colorscheme
plt.rcParams["axes.prop_cycle"] = cycler(
    color=['#107591','#00c0bf','#f69a48','#fdcd49','#8da798','#a19368','#525252','#a6761d','#7035b7','#cf166e']
)

doc_source_dir = Path(__file__).parent.parent / "source" / "_miniatures"
doc_source_dir.mkdir(exist_ok=True)

idata = az.load_arviz_data("non_centered_eight")
idata_centered = az.load_arviz_data("centered_eight")

# 1 - plot rank dist
pc = az.plot_rank_dist(idata, var_names=["tau", "mu"], compact=False, figure_kwargs={"figsize": (10, 5)})
pc.savefig(doc_source_dir / "plot_rank_dist.png")

# 2 - plot forest with ESS
aux_dataset = (
    idata_centered["posterior"]
    .dataset.expand_dims(column=3)
    .assign_coords(column=["labels", "forest", "ess"])
)
pc = az.plot_forest(
    aux_dataset,
    var_names=["mu", "theta", "tau"],
    aes={"color": ["school"]},
)
pc.map(
    az.visuals.scatter_x,
    "ess",
    data=idata_centered.posterior.dataset.azstats.ess(sample_dims="draw"),
    coords={"column": "ess"},
)
pc.get_viz("plot", column="forest").set_xlabel("Posterior estimate")
pc.get_viz("plot", column="ess").set_xlabel("ESS")
pc.savefig(doc_source_dir / "plot_forest_ess.png")

# 3 - plot dist comparison
pc = az.plot_dist(
    {"Centered": idata_centered, "Non Centered": idata},
    var_names=["theta", "tau", "theta_t"],
    coords={"school": ["Choate", "Deerfield"]},
    col_wrap=2,
    figure_kwargs={"figsize": (17, 8)},
)
pc.savefig(doc_source_dir / "plot_dist_models.png")

# 4 - plot pair
pc = az.plot_pair(
    idata_centered,
    var_names=["theta", "tau"],
    coords= {"school": ["Choate", "Deerfield"]},
    visuals={"divergence": {"color": "C3"}},
    figure_kwargs={"figsize": (15, 7)},
)
pc.savefig(doc_source_dir / "plot_pair.png")

# 5 - plot ppc rootogram
dt = az.load_arviz_data("rugby")
pc = az.plot_ppc_rootogram(
    dt,
    var_names=["home_points", "away_points"],
    aes={"color": ["__variable__"]},
    aes_by_visuals={"title": ["color"]},
    figure_kwargs={"sharex": True, "sharey": True, "figsize": (8, 5)},
    col_wrap=1,
    visuals={"ylabel": False},
)
pc.get_viz("plot", "home_points").set_ylabel("Frequency")
pc.savefig(doc_source_dir / "plot_ppc_rootogram.png")

# 6 - plot dist hist & kde
az.rcParams["data.sample_dims"] = "sample"
rng = np.random.default_rng(94)
ds = az.dict_to_dataset({"Poisson": rng.poisson(4, size=1000), "Gaussian": rng.normal(size=1000)})
pc = az.plot_dist(
    ds,
    aes={"color": ["__variable__"]},
    color=["C1", "C2"],
    visuals={"dist": False},
    figure_kwargs={"figsize": (13, 6)},
)
alloff = {"credible_interval": False, "point_estimate": False, "point_estimate_text": False, "title": False}
az.plot_dist(ds[["Poisson"]], kind="hist", visuals={"dist": False, "face": True, **alloff}, plot_collection=pc)
az.plot_dist(ds[["Gaussian"]], kind="kde", visuals={"dist": True, **alloff}, plot_collection=pc)
pc.add_legend("__variable__")
pc.savefig(doc_source_dir / "plot_dist_hist_kde.png")
