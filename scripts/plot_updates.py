import pypsa
import pandas as pd
import matplotlib.pyplot as plt

from helper import override_component_attrs

def plot_renewables(n,axes,label):
    CF_s = n.generators_t.p_max_pu[n.generators.query('carrier == "solar"').index].sum(axis=1)
    (CF_s/CF_s.max()).plot(ax=axes[0],
                            lw=1,
                            alpha=1,
                            label=label)

    CF_ow = n.generators_t.p_max_pu[n.generators.query('carrier == "onwind"').index].sum(axis=1)
    (CF_ow/CF_ow.max()).plot(ax=axes[1],
                            lw=1,
                            alpha=1,
                            label=label)

    inflow = n.storage_units_t.inflow[n.storage_units_t.inflow.columns[n.storage_units_t.inflow.columns.str.contains('hydro')]]
    inflow.sum(axis=1).plot(ax=axes[2],
                            lw=1,
                            alpha=1,
                            label=label)

def plot_heat_load(n,ax,label):
    A = n.loads.index[n.loads.index.str.contains('heat')]
    B = n.loads_t.p_set.columns
    shared_indices = [i for i, item in enumerate(A) if item in B]
    print("Warning! ", n.loads.loc[A].drop(index=A[shared_indices]).index, " did not have any heat demand")
    loads = n.loads_t.p_set[A[shared_indices]].sum(axis=1)
    ax.plot(loads,alpha=0.5,label=label)

def plot_updated_heat_load(networks_dict,design_network):
    
    overrides = override_component_attrs(snakemake.input.overrides)
    n_design = pypsa.Network(design_network, override_component_attrs=overrides)
    
    fig,ax = plt.subplots(figsize=(10,5))
    plot_heat_load(n_design,ax,'design')
    
    for label, filename in networks_dict.items():
        n = pypsa.Network(filename, override_component_attrs=overrides)
        plot_heat_load(n,ax,label[1])
    
    ax.legend(frameon=True)

    return fig

def plot_updated_renewable_profiles(networks_dict,design_network):

    fig1,ax1 = plt.subplots(figsize=(10,5))
    fig2,ax2 = plt.subplots(figsize=(10,5))
    fig3,ax3 = plt.subplots(figsize=(10,5))

    axes = [ax1,ax2,ax3]

    overrides = override_component_attrs(snakemake.input.overrides)
    n_design = pypsa.Network(design_network, override_component_attrs=overrides)
    plot_renewables(n_design,axes,"design")

    for label, filename in networks_dict.items():
        n = pypsa.Network(filename, override_component_attrs=overrides)

        plot_renewables(n,axes,label[1])

    ax1.legend(frameon=True)
    ax2.legend(frameon=True)
    ax3.legend(frameon=True)

    return fig1, fig2, fig3

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from helper import mock_snakemake
        snakemake = mock_snakemake(
            'make_summary',
            design_year="2013", 
            weather_year="2008",
        )
    
    overrides = override_component_attrs(snakemake.input.overrides)
    nodes = snakemake.config['model']['nodes']
    tres = snakemake.config['model']['tres']

    networks_dict = {
    (design_year, weather_year): snakemake.config[
        "results_dir"
    ]
    + snakemake.config["run"]
   + "/prenetworks/base_n"+ nodes + "_" + tres + f"h_renewables_dy{design_year}_wy{weather_year}_heat.nc"
    for design_year in snakemake.config["scenario"]["design_year"]
    for weather_year in snakemake.config["scenario"]["weather_year"]
    }

    print(networks_dict)
    design_network = snakemake.input.design_network
    fig1,fig2,fig3 = plot_updated_renewable_profiles(networks_dict,design_network)

    fig1.savefig(
                snakemake.output.plot_solar,
                bbox_inches="tight"
                )

    fig2.savefig(
                snakemake.output.plot_wind,
                bbox_inches="tight"
                )

    fig3.savefig(
                snakemake.output.plot_hydro,
                bbox_inches="tight"
                )

    fig4 = plot_updated_heat_load(networks_dict,design_network)
    fig4.savefig(
                snakemake.output.plot_heat,
                bbox_inches="tight"
                )