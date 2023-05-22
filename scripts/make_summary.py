import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from helper import override_component_attrs

fs = 16
plt.style.use('seaborn-ticks')
plt.rcParams['axes.labelsize'] = fs
plt.rcParams['xtick.labelsize'] = fs
plt.rcParams['ytick.labelsize'] = fs
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['legend.title_fontsize'] = fs
plt.rcParams['legend.fontsize'] = fs

def plot_lost_load(networks_dict):
    fig,ax = plt.subplots(figsize=(10,5))
    lost_load_duration = []
    for label, filename in networks_dict.items():
        overrides = override_component_attrs(snakemake.input.overrides)
        n = pypsa.Network(filename, override_component_attrs=overrides)

        lost_load = n.generators_t.p[n.generators.query('carrier == "load_el"').index].sum(axis=1)/1e3
        ax.plot(np.arange(len(lost_load))*(8760/len(n.snapshots)),lost_load.sort_values(ascending=False),label=label[1])

        lost_load_duration.append(len(lost_load[lost_load > 0.1])*(8760/len(n.snapshots)))
    
    ax.set_ylabel('GW')
    ax.set_xlabel('Hours')
    ax.set_xlim([0,max(lost_load_duration)])
    fig.legend(ncol=3,
                bbox_to_anchor=(0.7, -0.02),
                borderaxespad=0,
                fontsize=fs,
                frameon = True)

    return fig

def assign_carriers(n):
    if "carrier" not in n.lines:
        n.lines["carrier"] = "AC"

def assign_locations(n):
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        ifind = pd.Series(c.df.index.str.find(" ", start=4), c.df.index)
        for i in ifind.unique():
            names = ifind.index[ifind == i]
            if i == -1:
                c.df.loc[names, "location"] = ""
            else:
                c.df.loc[names, "location"] = names.str[:i]

def calculate_summary(n,label,df):

    sig_dig = 1

    tres_factor = 8760/len(n.snapshots)
    
    try:
        co2_cap = round(n.global_constraints.loc['CO2Limit'].constant,sig_dig)
        co2_price = round(n.global_constraints.loc['CO2Limit'].mu,sig_dig)

    except:
        co2_cap = np.inf
        co2_price = np.inf
    
    df.loc['co2_cap', label] = co2_cap
    df.loc['co2_price',label] = co2_price

    load_shedders_el = n.generators.query('carrier == "load_el"')
    lost_load_el = n.generators_t.p[load_shedders_el.index].sum().sum()
    df.loc['lost_load_el', label] = round(lost_load_el,sig_dig)

    load_shedders_heat = n.generators.query('carrier == "load_heat"')
    lost_load_heat = n.generators_t.p[load_shedders_heat.index].sum().sum()
    df.loc['lost_load_heat', label] = round(lost_load_heat,sig_dig)
    
    ###############################################################
    ######################## CO2 emissions ########################
    ###############################################################
    # CO2 emitters from power or heat production
    co2_emittors = n.links.query('bus2 == "co2 atmosphere"') # links going from fuel buses (e.g., gas, coal, lignite etc.) to "CO2 atmosphere" bus
    co2_emittors = co2_emittors.query('efficiency2 > 0') # excluding links with no CO2 emissions (e.g., nuclear)
    co2_t = -n.links_t.p2[co2_emittors.index]*tres_factor

    # CO2 emittors and capturing facilities from power or heat production
    chp = n.links.query('bus3 == "co2 atmosphere"') # links going from CHP fuel buses (e.g., biomass or natural gas) to "CO2 atmosphere" bus
    co2_chp_t = -n.links_t.p3[chp.index] *tres_factor
    # NB! Biomass w.o. CC has no emissions. Biomass w. CC then have negative emissions.

    # process emissions
    co2_process = n.links.query('bus1 == "co2 atmosphere"').index # links going from "process emissions" to "CO2 atmosphere" bus
    co2_process_t = -n.links_t.p1[co2_process]*tres_factor

    # direct air capture
    dac = n.links.index[n.links.index.str.contains('DAC')] # links going from "CO2 atmosphere" to "CO2 stored" (sequestration)
    co2_dac_t = -n.links_t.p1[dac]*tres_factor 

    # CO2 balance
    co2_tot = sum([co2_t.sum().sum(),
                   co2_chp_t.sum().sum(),
                   co2_process_t.sum().sum(),
                   -co2_dac_t.sum().sum()])
    
    df.loc['co2_electricity_mill_t', label] = round(co2_t.sum().sum()/1e6,sig_dig)
    df.loc['co2_chp_mill_t', label] = round(co2_chp_t.sum().sum()/1e6,sig_dig)
    df.loc['co2_process_mill_t', label] = round(co2_process_t.sum().sum()/1e6,sig_dig)
    df.loc['co2_dac_mill_t', label] = round(co2_dac_t.sum().sum()/1e6,sig_dig)
    df.loc['co2_net', label] = round(co2_tot/1e6,sig_dig)
    ########################################################################
    ########################################################################


    ########################################################################
    ######################## Electricity generation ########################
    ########################################################################
    A_g = n.generators.bus.values
    A_l = n.links.bus1
    B_HV = n.buses.query('carrier == "AC"').index
    B_LV = n.buses.query('carrier == "low voltage"').index
    
    shared_indices_HV = [i for i, item in enumerate(A_g) if item in B_HV]
    shared_indices_LV = [i for i, item in enumerate(A_g) if item in B_LV]
    shared_indices_l_HV = [i for i, item in enumerate(A_l) if item in B_HV]

    power_generators_HV = n.generators.iloc[shared_indices_HV]
    power_generators_HV_t = n.generators_t.p[power_generators_HV.index]

    power_generators_l_HV = n.links.iloc[shared_indices_l_HV]
    power_generators_l_HV_t = -n.links_t.p1[power_generators_l_HV.index]
    power_generators_l_HV_t = power_generators_l_HV_t.drop(columns=n.links.query('carrier == "DC"').index) # dropping transmission DC links
    power_generators_l_HV_t = power_generators_l_HV_t.drop(columns = power_generators_l_HV_t.columns[power_generators_l_HV_t.columns.str.contains('discharger')]) # dropping storage dischargers
    power_generators_l_HV_t = power_generators_l_HV_t.drop(columns = power_generators_l_HV_t.columns[power_generators_l_HV_t.columns.str.contains('Fuel Cell')]) # dropping H2 fuel cells

    power_generators_LV = n.generators.iloc[shared_indices_LV]
    power_generators_LV_t = n.generators_t.p[power_generators_LV.index]

    power_generators_hydro = n.storage_units.query('carrier == "hydro"')
    power_generators_hydro_t = n.storage_units_t.p[power_generators_hydro.index]

    power_generators_HV_dict = power_generators_HV.carrier.to_dict()
    power_generators_l_HV_dict = power_generators_l_HV.carrier.to_dict()
    power_generators_LV_dict = power_generators_LV.carrier.to_dict()
    power_generators_hydro_dict = power_generators_hydro.carrier.to_dict()

    power_generators_HV_t_rn = power_generators_HV_t.rename(columns=power_generators_HV_dict)
    power_generators_l_HV_t_rn = power_generators_l_HV_t.rename(columns=power_generators_l_HV_dict)
    power_generators_LV_t_rn = power_generators_LV_t.rename(columns=power_generators_LV_dict)
    power_generators_hydro_t_rn = power_generators_hydro_t.rename(columns=power_generators_hydro_dict)
    
    power_generators_t = pd.concat([power_generators_LV_t_rn, 
                                    power_generators_HV_t_rn,
                                    power_generators_l_HV_t_rn,
                                    power_generators_hydro_t_rn], axis=1)

    add_dict = {'offwind-ac':'offwind',
                'offwind-dc':'offwind',
                'solar rooftop':'solar'}

    power_generators_t_rn = power_generators_t.rename(columns=add_dict)
    df_power_generation = power_generators_t_rn.groupby(by=power_generators_t_rn.columns, axis=1).sum()
    df_shares = df_power_generation.sum()/(df_power_generation.sum().sum())*100

    for ind in df_shares.index:
        df.loc[ind + ' share', label] = round(df_shares.loc[ind],sig_dig)
    ########################################################################
    ########################################################################


    return df

def make_summaries(networks_dict,design_network):
    outputs = [
                "summary",
              ]

    columns = pd.MultiIndex.from_tuples(
        networks_dict.keys(), names=["design_year","weather_year"]
    )

    df = {}

    for output in outputs:
        df[output] = pd.DataFrame(columns=columns, dtype=float)

    for label, filename in networks_dict.items():
        print('label: ', label)
        print('filename: ', filename)
        logger.info(f"Make summary for scenario {label}, using {filename}")

        overrides = override_component_attrs(snakemake.input.overrides)
        n = pypsa.Network(filename, override_component_attrs=overrides)

        assign_carriers(n)
        assign_locations(n)

        for output in outputs:
            df[output] = globals()["calculate_" + output](n, label, df[output])

    # Include design network results as reference
    print(df)
    n_design = pypsa.Network(design_network,override_component_attrs=overrides)
    for output in outputs:
        df[output] = globals()["calculate_" + output](n_design, ('design','design'), df[output])

    return df

def to_csv(df):
    for key in df:
        df[key].to_csv(snakemake.output[key])

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
   + "/postnetworks/resolved_n"+ nodes + "_" + tres + f"h_dy{design_year}_wy{weather_year}.nc"
    for design_year in snakemake.config["scenario"]["design_year"]
    for weather_year in snakemake.config["scenario"]["weather_year"]
    }

    print(networks_dict)

    design_network = snakemake.input.design_network
    df = make_summaries(networks_dict,design_network)

    to_csv(df)

    fig = plot_lost_load(networks_dict)
    
    fig.savefig(snakemake.output.lost_load_plot, bbox_inches = 'tight')