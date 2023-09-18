import pypsa
import pandas as pd
import matplotlib.pyplot as plt

from helper import override_component_attrs

def update_renewables(n,n_weather):
    generators = ['solar','solar rooftop',  # CSP
                'onwind','offwind-ac','offwind-dc',
                'ror']

    dic = {'solar':'solar',
           'solar rooftop':'solar',
           'onwind':'wind',
           'offwind-ac':'wind',
           'offwind-dc':'wind',
           'ror':'hydro'}

    ###################################
    ### Update VRE capacity factors ###
    ###################################
    for gen in generators:
        new_CF = n_weather.generators_t.p_max_pu[n_weather.generators.query('carrier == @gen').index]
        
        df_generators_t = n.generators_t.p_max_pu.copy()
        df_generators_t.loc[df_generators_t.index,n.generators.query('carrier == @gen').index] = new_CF
        
        if dic[gen] in snakemake.wildcards.opts:
            print('correcting ' + gen)
            n.generators_t.p_max_pu = df_generators_t

    ###################################
    ## Update hydro reservoir inflow ##
    ###################################
    new_inflow = n_weather.storage_units_t.inflow #[n_weather.storage_units.query('carrier == @gen').index]
    old_inflow = n_weather.storage_units_t.inflow

    A = new_inflow.columns
    B = old_inflow.columns
    shared_indices = [i for i, item in enumerate(A) if item in B] # columns of "new_inflow" contained also in "old_inflow"
    df_hydro_t = n.storage_units_t.inflow
    df_hydro_t.loc[df_hydro_t.index,A[shared_indices]] = old_inflow[A[shared_indices]]

    if 'hydro' in snakemake.wildcards.opts:
        print('correcting hydro')
        n.storage_units_t.inflow = df_hydro_t

    return n 

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from helper import mock_snakemake
        snakemake = mock_snakemake(
            'update_renewable_profiles',
            design_year="2013", 
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


    # Visualize update in the following figures
    # fig1,ax1 = plt.subplots(figsize=(10,5))
    # fig2,ax2 = plt.subplots(figsize=(10,5))
    # fig3,ax3 = plt.subplots(figsize=(10,5))

    # CF_s = n.generators_t.p_max_pu[n.generators.query('carrier == "solar"').index].sum(axis=1)
    # (CF_s/CF_s.max()).plot(ax=ax1,
    #                         lw=1,
    #                         alpha=1,
    #                         label='old')

    # CF_ow = n.generators_t.p_max_pu[n.generators.query('carrier == "onwind"').index].sum(axis=1)
    # (CF_ow/CF_ow.max()).plot(ax=ax2,
    #                         lw=1,
    #                         alpha=1,
    #                         label='old')

    # inflow = n.storage_units_t.inflow[n.storage_units_t.inflow.columns[n.storage_units_t.inflow.columns.str.contains('hydro')]]
    # inflow.sum(axis=1).plot(ax=ax3,
    #                         lw=1,
    #                         alpha=1,
    #                         label='old')

    update_renewables(n,n_weather)
    n.export_to_netcdf(snakemake.output.network)

    # CF_s_update = n.generators_t.p_max_pu[n.generators.query('carrier == "solar"').index].sum(axis=1)
    # (CF_s_update/CF_s_update.max()).plot(ax=ax1,
    #                                     lw=1,
    #                                     alpha=1,
    #                                     label='new')


    # CF_ow_update = n.generators_t.p_max_pu[n.generators.query('carrier == "onwind"').index].sum(axis=1)
    
    # (CF_ow_update/CF_ow_update.max()).plot(ax=ax2,
    #                                         lw=1,
    #                                         alpha=1,
    #                                         label='new')

    # inflow_update = n.storage_units_t.inflow[n.storage_units_t.inflow.columns[n.storage_units_t.inflow.columns.str.contains('hydro')]]
    # inflow_update.sum(axis=1).plot(ax=ax3,
    #                                 lw=1,
    #                                 alpha=1,
    #                                 label='new')

    # ax1.legend(frameon=True)
    # ax2.legend(frameon=True)
    # ax3.legend(frameon=True)

    # fig1.savefig(
    #             snakemake.output.plot_solar,
    #             bbox_inches="tight"
    #             )

    # fig2.savefig(
    #             snakemake.output.plot_wind,
    #             bbox_inches="tight"
    #             )

    # fig3.savefig(
    #             snakemake.output.plot_hydro,
    #             bbox_inches="tight"
    #             )