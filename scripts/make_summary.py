import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from helper import override_component_attrs
import yaml

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

with open('scripts/tech_colors.yaml') as file:
    tech_colors = yaml.safe_load(file)['tech_colors']
tech_colors['urban central gas CHP'] = tech_colors['gas']
tech_colors['urban central gas CHP CC'] = tech_colors['gas']
tech_colors['urban central solid biomass CHP'] = tech_colors['biomass']
tech_colors['urban central solid biomass CHP CC'] = tech_colors['biomass']
tech_colors['lost load el.'] = 'k'
tech_colors['biogas to gas'] = tech_colors['biogas']
tech_colors['process'] = tech_colors['process emissions']
tech_colors['oil load'] = tech_colors['oil']
tech_colors['agriculture machinery'] = 'k'
tech_colors['dac'] = tech_colors['DAC']

def plot_lost_load(n,ax,lost_load_duration,label):
    lost_load = n.generators_t.p[n.generators.query('carrier == "load_el"').index].sum(axis=1)/1e3
    ax.plot(np.arange(len(lost_load))*(8760/len(n.snapshots)),lost_load.sort_values(ascending=False),label=label[1:])
    lost_load_duration.append(len(lost_load[lost_load > 0.1])*(8760/len(n.snapshots)))
    return ax,lost_load_duration

def make_heatmap_time_df(n,load_shedding_t,label):
    load_shedding_t[label[1:]] = n.generators_t.p[n.generators.query('carrier == "load_el"').index].sum(axis=1)/1e3 # GW
    return load_shedding_t

def make_heatmap_space_df(n,load_shedding_s,label):
    load_shedding_s[label[1:]] = n.generators_t.p[n.generators.query('carrier == "load_el"').index].max(axis=0)/1e3 # GW
    return load_shedding_s

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

    df.loc['peak_heat_demand_GW',label] = n.loads_t.p[n.loads.index[n.loads.carrier.str.contains('heat')]].sum(axis=1).max()/1e3

    df.loc['annual_inflow_TWh',label] = n.storage_units_t.inflow.sum().sum()/1e6

    load_shedders_el = n.generators.query('carrier == "load_el"')
    lost_load_el = n.generators_t.p[load_shedders_el.index].sum().sum()
    df.loc['lost_load_el', label] = round(lost_load_el,sig_dig)
    df.loc['capacity deficit el. GW',label] = round(n.generators_t.p[load_shedders_el.index].sum(axis=1).max()/1e3,sig_dig)

    load_shedders_heat = n.generators.query('carrier == "load_heat"')
    lost_load_heat = n.generators_t.p[load_shedders_heat.index].sum().sum()
    df.loc['lost_load_heat', label] = round(lost_load_heat,sig_dig)
    df.loc['capacity deficit heat GW',label] = round(n.generators_t.p[load_shedders_heat.index].sum(axis=1).max()/1e3,sig_dig)

    ###############################################################
    ######################## CO2 emissions ########################
    ###############################################################
    # CO2 emittors and capturing facilities from power, heat and fuel production
    co2_emittors = n.links.query('bus2 == "co2 atmosphere"') # links going from fuel buses (e.g., gas, coal, lignite etc.) to "CO2 atmosphere" bus
    co2_emittors = co2_emittors.query('efficiency2 != 0') # excluding links with no CO2 emissions (e.g., nuclear)
    co2_t = -n.links_t.p2[co2_emittors.index]*tres_factor

    co2_t_renamed = co2_t.rename(columns=co2_emittors.carrier.to_dict())
    co2_t_grouped = co2_t_renamed.groupby(by=co2_t_renamed.columns,axis=1).sum().sum()
    for i in range(len(co2_t_grouped.index)):
        if 'gas boiler' not in co2_t_grouped.index[i]:
            co2_t_i = round(co2_t_grouped.iloc[i]/1e6,sig_dig)
            df.loc['co2 emissions ' + co2_t_grouped.index[i],label] = co2_t_i

    co2_t_gas_boiler = round(co2_t_grouped.loc[co2_t_grouped.index[co2_t_grouped.index.str.contains('gas boiler')]].sum()/1e6,sig_dig)

    df.loc['co2 emissions gas boiler',label] = co2_t_gas_boiler
    ###############################################################

    # CO2 emissions and capturing from chp plants
    chp = n.links.query('bus3 == "co2 atmosphere"') # links going from CHP fuel buses (e.g., biomass or natural gas) to "CO2 atmosphere" bus
    co2_chp_t = -n.links_t.p3[chp.index]*tres_factor
    # NB! Biomass w.o. CC has no emissions. For this reason, Biomass w. CC has negative emissions.

    co2_chp_t_renamed = co2_chp_t.rename(columns=chp.carrier.to_dict())
    co2_chp_t_renamed_grouped = co2_chp_t_renamed.groupby(by=co2_chp_t_renamed .columns,axis=1).sum().sum()

    co2_chp_t_renamed_grouped_gas = round(co2_chp_t_renamed_grouped.loc[co2_chp_t_renamed_grouped.index[co2_chp_t_renamed_grouped.index.str.contains('gas CHP')]].sum()/1e6,sig_dig)
    co2_chp_t_renamed_grouped_biomass = round(co2_chp_t_renamed_grouped.loc[co2_chp_t_renamed_grouped.index[co2_chp_t_renamed_grouped.index.str.contains('solid biomass CHP')]].sum()/1e6,sig_dig)

    df.loc['co2 emissions gas CHP',label] = co2_chp_t_renamed_grouped_gas
    df.loc['co2 emissions biomass CHP',label] = co2_chp_t_renamed_grouped_biomass 
    ###############################################################

    # process emissions
    co2_process = n.links.query('bus1 == "co2 atmosphere"').index # links going from "process emissions" to "CO2 atmosphere" bus
    co2_process_t = -n.links_t.p1[co2_process]*tres_factor
    # process emissions have CC which captures 90 % of emissions. Here, we only consider the 10 % being emitted.
    # to include the 90% capture in the balance, call: -n.links_t.p2["EU process emissions CC"]
    co2_process_t_sum = round(co2_process_t.sum().sum()/1e6,sig_dig)

    df.loc['co2 emissions process', label] = co2_process_t_sum 
    ###############################################################

    # load emissions (e.g., land transport or agriculture)
    loads_co2 = n.loads # .query('bus == "co2 atmosphere"')
    load_emissions_index = loads_co2.index[loads_co2.index.str.contains('emissions')]
    load_emissions = n.loads.loc[load_emissions_index]
    load_emissions_t = -n.loads_t.p[load_emissions_index]*tres_factor

    load_emissions_t_sum_oil = round(load_emissions_t['oil emissions'].sum()/1e6,sig_dig)
    load_emissions_t_sum_agriculture = round(load_emissions_t['agriculture machinery oil emissions'].sum()/1e6,sig_dig)

    df.loc['co2 emissions oil load', label] = load_emissions_t_sum_oil
    df.loc['co2 emissions agriculture machinery', label] = load_emissions_t_sum_agriculture
    ###############################################################

    # direct air capture
    dac = n.links.index[n.links.index.str.contains('DAC')] # links going from "CO2 atmosphere" to "CO2 stored" (sequestration)
    co2_dac_t = -n.links_t.p1[dac]*tres_factor 
    co2_dac_t_sum = -round(co2_dac_t.sum().sum()/1e6,sig_dig) # i.e., negative emissions

    df.loc['co2 emissions dac', label] = co2_dac_t_sum
    ###############################################################

    # CO2 balance
    co2_tot = df.loc[df.index[df.index.str.contains('co2 emissions')],label].sum().sum()
    df.loc['net emissions', label] = round(co2_tot,sig_dig)
    ###############################################################
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
                'solar rooftop':'solar',
                'load_el':'lost load el.'}

    power_generators_t_rn = power_generators_t.rename(columns=add_dict)
    df_power_generation = power_generators_t_rn.groupby(by=power_generators_t_rn.columns, axis=1).sum()
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ### Plot
    freq = 'd'
    timescales = {'h':'hourly',
                  'd':'daily',
                  'w':'weekly',
                  'm':'monthly'}
    
    carriers_list = list(n.carriers.index).copy()

    df_plot = df_power_generation
    
    preferred_order = pd.Index([
                            "onwind",
                            "offwind",
                            "solar",
                            "hydro",
                            "ror",
                            ])
    
    new_index = preferred_order.intersection(df_plot.columns).append(df_plot.columns.difference(preferred_order))

    df_plot = df_plot.T.loc[new_index].T

    # full year
    fig,ax = plt.subplots(figsize=(10,5))
    df_plot_full_year = df_plot.rolling(8).mean()/1e3 # daily moving averages
    df_plot_full_year.plot.area(ax=ax,
                                stacked=True,
                                color=[tech_colors[i] for i in df_plot_full_year.columns],lw=0,
                                legend=False)
    ax.set_ylabel('Electricity supply [GWh]')
    fig.legend(bbox_to_anchor=(0.7, -0.15),
                borderaxespad=0,
                fontsize=fs,
                frameon = True)
    fig.savefig(snakemake.output.lost_load_plot[:-29] + 'electricity_supply_' + str(label[1:]) + '.png', bbox_inches = 'tight',dpi=300)
    
    # critical period 
    fig_critical_period,ax_critical_period = plt.subplots(figsize=(10,5))
    df_critical_period = df_plot
    df_critical_period.plot.area(ax=ax_critical_period,
                                stacked=True,
                                color=[tech_colors[i] for i in df_critical_period.columns],lw=0,
                                legend=False)
    ax_critical_period.set_ylabel('Electricity supply [GWh]')
    ax_critical_period.set_xlim([pd.to_datetime('1/1/2013'),pd.to_datetime('2/15/2013')])
    fig_critical_period.legend(bbox_to_anchor=(0.7, -0.15),
                                borderaxespad=0,
                                fontsize=fs,
                                frameon = True)
    fig_critical_period.savefig(snakemake.output.lost_load_plot[:-29] + 'electricity_supply_' + str(label[1:]) + '_critical_period.png', bbox_inches = 'tight',dpi=300)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
        networks_dict.keys(), names=["design_year","weather_year","opts"]
    )

    df = {}

    for output in outputs:
        df[output] = pd.DataFrame(columns=columns, dtype=float)

    fig_ll,ax_ll = plt.subplots(figsize=(10,5))
    
    ll_duration = []
    load_shedding_t = pd.DataFrame()
    load_shedding_s = pd.DataFrame()

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
        
        ####### PLOTTING
        load_shedding_t = make_heatmap_time_df(n,load_shedding_t,label)
        load_shedding_s = make_heatmap_space_df(n,load_shedding_s,label)
        ax_ll, ll_duration = plot_lost_load(n,ax_ll,ll_duration,label)
        ##################

    # Include design network results as reference
    print(df)
    n_design = pypsa.Network(design_network,override_component_attrs=overrides)
    for output in outputs:
        df[output] = globals()["calculate_" + output](n_design, ('design','design','design'), df[output])

    fig_heatmap_time,ax_heatmap_time = plt.subplots(figsize=(10,5))
    sns.heatmap(load_shedding_t, cmap='summer_r',ax=ax_heatmap_time,cbar_kws={'label': 'GW'})
    y_dates = load_shedding_t.index.strftime('%b').unique()
    ax_heatmap_time.set_yticks(np.linspace(0,len(n.snapshots),12))
    ax_heatmap_time.set_yticklabels(labels=y_dates, rotation=30, ha='right')
    ax_heatmap_time.set_ylabel('')

    fig_heatmap_space,ax_heatmap_space = plt.subplots(figsize=(5,10))
    sns.heatmap(load_shedding_s, cmap='summer_r',ax=ax_heatmap_space,cbar_kws={'label': 'GW'})
    load_shedding_s['index'] = load_shedding_s.index
    y_ticklabels = load_shedding_s['index'].str.split(' low',1,expand=True)[0]
    load_shedding_s.drop(columns='index',inplace=True)
    ax_heatmap_space.set_yticks(np.arange(len(load_shedding_s.index))+0.5)
    ax_heatmap_space.set_yticklabels(labels=y_ticklabels, rotation=0, ha='right')

    ax_ll.set_ylabel('GW')
    ax_ll.set_xlabel('Hours')
    ax_ll.set_xlim([0,max(ll_duration)])
    fig_ll.legend(ncol=3,
                    bbox_to_anchor=(0.7, -0.02),
                    borderaxespad=0,
                    fontsize=fs,
                    frameon = True)
    return df, fig_ll, fig_heatmap_time, fig_heatmap_space

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
            opts = "",
        )
    
    overrides = override_component_attrs(snakemake.input.overrides)
    nodes = snakemake.config['model']['nodes']
    tres = snakemake.config['model']['tres']

    networks_dict = {
    (design_year, weather_year, opts): snakemake.config[
        "results_dir"
    ]
    + snakemake.config["run"]
   + "/postnetworks/resolved_n"+ nodes + "_" + tres + f"h_dy{design_year}_wy{weather_year}_{opts}.nc"
    for design_year in snakemake.config["scenario"]["design_year"]
    for weather_year in snakemake.config["scenario"]["weather_year"]
    for opts in snakemake.config["scenario"]["opts"]
    }

    print(networks_dict)

    design_network = snakemake.input.design_network
    df, fig_ll, fig_heatmap_time, fig_heatmap_space = make_summaries(networks_dict,design_network)

    to_csv(df)

    fig_ll.savefig(snakemake.output.lost_load_plot, bbox_inches = 'tight')
    fig_heatmap_time.savefig(snakemake.output.load_shedding_heatmap_time, bbox_inches = 'tight')
    fig_heatmap_space.savefig(snakemake.output.load_shedding_heatmap_space, bbox_inches = 'tight')