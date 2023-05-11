import pypsa
import pandas as pd
import matplotlib.pyplot as plt

from helper import override_component_attrs

def update_heat(n,n_weather):
    heat_loads = n.loads_t.p_set.columns[n.loads_t.p_set.columns.str.contains('heat')]
    heat_p_set = n.loads_t.p_set[heat_loads]
    new_heat_p_set = n_weather.loads_t.p_set[heat_loads]
            
    df_loads_p_set = n.loads_t.p_set
    df_loads_p_set.loc[df_loads_p_set.index,heat_loads] = new_heat_p_set
    
    n.loads_t.p_set = df_loads_p_set

    return n 

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from helper import mock_snakemake
        snakemake = mock_snakemake(
            'solve_network',
            design_year="2013" 
            weather_year="2008",
        )
    
    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input.network, 
                        override_component_attrs=overrides)

    # Read network with heat demand from a different weather year
    weather_network = snakemake.input.weather_network 
    n_weather = pypsa.Network(weather_network, 
                                override_component_attrs=overrides)

    fig,ax = plt.subplots(figsize=(10,5))
    loads = n.loads_t.p[n.loads.index[n.loads.index.str.contains('heat')]].sum(axis=1)
    #loads_w = n_weather.loads_t.p[n_weather.loads.index[n_weather.loads.index.str.contains('heat')]].sum(axis=1)
    ax.plot(loads,alpha=0.5,label='old')

    update_heat(n,n_weather)
    n.export_to_netcdf(snakemake.output.network)

    ax.plot(loads,alpha=0.5,label='new')
    
    ax.legend(frameon=True)

    fig.savefig(
                snakemake.output.plot_heat,
                bbox_inches="tight"
                )