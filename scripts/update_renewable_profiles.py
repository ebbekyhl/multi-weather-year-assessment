import pypsa
import pandas as pd
import matplotlib.pyplot as plt

from helper import override_component_attrs

def update_renewables(n,n_weather):
    generators = ['solar','solar rooftop',  # CSP
                'onwind','offwind-ac','offwind-dc',
                'ror','hydro']

    for gen in generators:
        if gen == 'hydro':
            new_CF = n_weather.storage_units_t.inflow[n_weather.storage_units.query('carrier == @gen').index]
            
            df_hydro_t = n.storage_units_t.inflow
            df_hydro_t.loc[df_hydro_t.index,n.storage_units.query('carrier == @gen').index] = new_CF
            
            n.storage_units_t.inflow = df_hydro_t

        else:
            new_CF = n_weather.generators_t.p_max_pu[n_weather.generators.query('carrier == @gen').index]
            
            df_generators_t = n.generators_t.p_max_pu.copy()
            df_generators_t.loc[df_generators_t.index,n.generators.query('carrier == @gen').index] = new_CF
            
            n.generators_t.p_max_pu = df_generators_t

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

    # Read design network
    #design_year = snakemake.config['design_year']
    design_network = snakemake.input.design_network 
    n = pypsa.Network(design_network,
                        override_component_attrs=overrides)

    # Read network with capacity factors from a different weather year
    #weather_year = snakemake.config['weather_year']
    weather_network = snakemake.input.weather_network 
    n_weather = pypsa.Network(weather_network, 
                                override_component_attrs=overrides)

    bus = 'ES0 0'

    fig1,ax1 = plt.subplots(figsize=(10,5))
    fig2,ax2 = plt.subplots(figsize=(10,5))
    fig3,ax3 = plt.subplots(figsize=(10,5))

    n.generators_t.p_max_pu[bus + ' solar'].plot(ax=ax1,
                                                 lw=1,
                                                 alpha=1,
                                                 label='old')

    n.generators_t.p_max_pu[bus + ' onwind'].plot(ax=ax2,
                                                 lw=1,
                                                 alpha=1,
                                                 label='old')

    n.storage_units_t.inflow[bus + ' hydro'].plot(ax=ax3,
                                                 lw=1,
                                                 alpha=1,
                                                 label='old')

    update_renewables(n,n_weather)
    n.export_to_netcdf(snakemake.output.network)

    n.generators_t.p_max_pu[bus + ' solar'].plot(ax=ax1,
                                                 lw=1,
                                                 alpha=1,
                                                 label='new')

    n.generators_t.p_max_pu[bus + ' onwind'].plot(ax=ax2,
                                                 lw=1,
                                                 alpha=1,
                                                 label='new')

    n.storage_units_t.inflow[bus + ' hydro'].plot(ax=ax3,
                                                 lw=1,
                                                 alpha=1,
                                                 label='new')


    #n.generators_t.p_max_pu[bus + ' solar'].rolling(int(168/3)).mean().plot(ax=ax,lw=1,alpha=1,label='new')
    ax1.legend(frameon=True)
    ax2.legend(frameon=True)
    ax3.legend(frameon=True)

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