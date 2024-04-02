import pandas as pd
from vresutils.costdata import annuity
import numpy as np

def calculate_curtailment(n,denominator_category="load"):
    total_inflexible_load = n.loads_t.p[n.loads.query('carrier == "electricity"').index].sum(axis=1).sum()
    total_wind_generation_actual = n.generators_t.p[n.generators.index[n.generators.index.str.contains('wind')]].sum().sum()
    total_wind_generation_theoretical = (n.generators_t.p_max_pu[n.generators.index[n.generators.index.str.contains('wind')]]*n.generators.loc[n.generators.index[n.generators.index.str.contains('wind')]].p_nom_opt).sum().sum()
    total_solar_generation_actual = n.generators_t.p[pd.concat([n.generators.query("carrier == 'solar'"),n.generators.query("carrier == 'solar rooftop'")]).index].sum().sum()
    total_solar_generation_theoretical = (n.generators_t.p_max_pu[pd.concat([n.generators.query("carrier == 'solar'"),n.generators.query("carrier == 'solar rooftop'")]).index]*n.generators.loc[pd.concat([n.generators.query("carrier == 'solar'"),n.generators.query("carrier == 'solar rooftop'")]).index].p_nom_opt).sum().sum()
    
    wind_abs_curt = []
    generator_total_curtailment_percentage_wind = []
    solar_abs_curt = []
    generator_total_curtailment_percentage_solar = []
    
    generators = ['solar','solar rooftop','onwind','offwind-ac','offwind-dc']
    for generator in generators:
        generator_index = n.generators.query('carrier == @generator').index
        capacity = n.generators.loc[generator_index].p_nom_opt
        
        nodal_production = n.generators_t.p[generator_index].sum()
        nodal_potential = (n.generators_t.p_max_pu[generator_index]*n.generators.query('carrier == @generator').p_nom_opt).sum()

        if "wind" in generator:
            denominator_dict = {"load":total_inflexible_load,
                                "gen_actual":total_wind_generation_actual,
                                "gen_theoretical":total_wind_generation_theoretical}
    
            denominator = denominator_dict[denominator_category]
            
            if nodal_potential.sum() > 0:
                abs_diff = nodal_potential.sum() - nodal_production.sum()
                wind_abs_curt.append(abs_diff)
                generator_total_curtailment_percentage_wind.append(((abs_diff)/denominator*100).round(3))
            else:
                wind_abs_curt.append(0)
                generator_total_curtailment_percentage_wind.append(0)
                
        elif "solar" in generator:
            denominator_dict = {"load":total_inflexible_load,
                                "gen_actual":total_solar_generation_actual,
                                "gen_theoretical":total_solar_generation_theoretical}
    
            denominator = denominator_dict[denominator_category]
        
            if nodal_potential.sum() > 0:
                abs_diff = nodal_potential.sum() - nodal_production.sum()
                solar_abs_curt.append(abs_diff)
                generator_total_curtailment_percentage_solar.append(((abs_diff)/denominator*100).round(3))
            else:
                solar_abs_curt.append(0)
                generator_total_curtailment_percentage_solar.append(0)
            
    return solar_abs_curt, generator_total_curtailment_percentage_solar, wind_abs_curt, generator_total_curtailment_percentage_wind

def calculate_storage_dispatch(n):
    #tot_load = n.loads_t.p_set[n.loads.query('carrier == "electricity"').index].sum().sum()
    
    bat_discharge = -n.links_t.p1[n.links.index[n.links.index.str.contains('battery discharge')]] # discharged electricity
    LDES_discharge = -n.links_t.p1[n.links.query('carrier == "LDES discharger"').index]
    H2_discharge = -n.links_t.p1[n.links.query('carrier == "H2 Fuel Cell"').index] # discharged electricity
    
    wind_generators = n.generators.index[n.generators.index.str.contains('wind')]
    solar_generators = pd.concat([n.generators.query('carrier == "solar"'),n.generators.query('carrier == "solar rooftop"')]).index
    VRE_generation = n.generators_t.p[wind_generators].sum().sum() + n.generators_t.p[solar_generators].sum().sum()
    
    # SDES
    storage_dispatch_sdes = bat_discharge.sum().sum()
        
    # LDES
    storage_dispatch_ldes = LDES_discharge.sum().sum()
    
    # H2
    storage_dispatch_H2 = H2_discharge.sum().sum()
    
    return storage_dispatch_sdes, storage_dispatch_ldes, storage_dispatch_H2

def calculate_backup_generation(n):
    links_wo_transmission = n.links.drop(n.links.query("carrier == 'DC'").index)
    electricity_buses = list(n.buses.query('carrier == "AC"').index) + list(
        n.buses.query('carrier == "low voltage"').index
        )
    
    buses = n.links.columns[n.links.columns.str.contains("bus")]
    backup_capacity = {}
    backup_capacity_factor_low = {}
    backup_capacity_factor_high = {}
    for bus in ["bus1"]: #buses:
        if bus != "bus0":
            boolean_elec_demand_via_links = [
                links_wo_transmission.bus1[i] in electricity_buses
                for i in range(len(links_wo_transmission.bus1))
                ]
            
            boolean_elec_demand_via_links_series = pd.Series(boolean_elec_demand_via_links)
            links = links_wo_transmission.iloc[
                        boolean_elec_demand_via_links_series[boolean_elec_demand_via_links_series ].index
                        ]

            # Drop batteries, LDES, and distribution
            links = links.drop(
                                links.index[links.index.str.contains("battery")]
                              )

            links = links.drop(
                                links.index[links.index.str.contains("LDES")]
                              )

            links = links.drop(
                                links.index[links.index.str.contains("distribution")]
                              )
            
            links = links.drop(
                                links.index[links.index.str.contains("V2G")]
                              )

            # Calculate technology-aggregated backup capacities
            bus_no = bus[-1]
            if bus_no == "1":
                cap = n.links.loc[links.index].p_nom_opt*n.links.loc[links.index].efficiency
            else:
                cap = n.links.loc[links.index].p_nom_opt*eval("n.links.loc[links.index].efficiency" + bus_no)

            cap_grouped = cap.groupby(links.carrier).sum()
            cap_grouped[cap_grouped < 1] = np.nan # drop technologies with zero capacities
            tech_capacity_factor = -eval("n.links_t.p" + bus_no)[links.index].groupby(links.carrier,axis=1).sum().sum()/(cap_grouped*len(n.snapshots))*100

            backup_capacity[bus] = cap_grouped/1e3 # GW
            backup_capacity_factor_low[bus] = tech_capacity_factor.min() # % min average capacity factor
            backup_capacity_factor_high[bus] = tech_capacity_factor.max() # % max average capacity factor
    
    df_backup_capacity = round(pd.DataFrame.from_dict(backup_capacity).sum().item(),2) #.dropna()
    
    df_backup_capacity_factor_low = round(backup_capacity_factor_low["bus1"],2)
    df_backup_capacity_factor_high = round(backup_capacity_factor_high["bus1"],2)
    
    return df_backup_capacity, df_backup_capacity_factor_low, df_backup_capacity_factor_high

def calculate_endogenous_demand(n):
    links_wo_transmission = n.links.drop(n.links.query("carrier == 'DC'").index)
    electricity_buses = list(n.buses.query('carrier == "AC"').index) + list(
        n.buses.query('carrier == "low voltage"').index
    )
    boolean_elec_demand_via_links = [
        links_wo_transmission.bus0[i] in electricity_buses
        for i in range(len(links_wo_transmission.bus0))
    ]
    boolean_elec_demand_via_links_series = pd.Series(boolean_elec_demand_via_links)
    elec_demand_via_links = links_wo_transmission.iloc[
        boolean_elec_demand_via_links_series[boolean_elec_demand_via_links_series].index
    ]

    # Drop batteries
    elec_demand_via_links = elec_demand_via_links.drop(
        elec_demand_via_links.index[elec_demand_via_links.index.str.contains("battery")]
    )

    # Drop LDES
    elec_demand_via_links = elec_demand_via_links.drop(
        elec_demand_via_links.index[elec_demand_via_links.index.str.contains("LDES")]
    )

    # Drop distribution links
    elec_demand_via_links = elec_demand_via_links.drop(
        elec_demand_via_links.index[
            elec_demand_via_links.index.str.contains("distribution")
        ]
    )
    
    endogenous_demand = n.links_t.p0[elec_demand_via_links.index]

    return endogenous_demand

def calculate_renewable_penetration(n):
    
    # exogenous demand
    exo_demand = load = (n.loads_t.p_set[n.loads.query("carrier == 'electricity'").index].sum().sum() + 
        (n.loads.query("carrier == 'industry electricity'").p_set*len(n.snapshots)).sum()
    )
    
    # endogenous demand
    endo_demand_i = calculate_endogenous_demand(n)
    endo_demand = endo_demand_i.sum().sum()
    
    # total electricity demand
    tot_load = exo_demand + endo_demand
    
    # solar generation
    solar_generators = pd.concat([n.generators.query('carrier == "solar"'),n.generators.query('carrier == "solar rooftop"')]).index
    solar_generation_capacity = n.generators.loc[solar_generators].p_nom_opt
    solar_generation = n.generators_t.p[solar_generators]
    solar_cap_factor_mean_act = solar_generation.sum().sum()/(solar_generation_capacity.sum()*len(n.snapshots))
    solar_cap_ES = solar_generation_capacity[solar_generation_capacity.index[solar_generation_capacity.index.str.contains("ES")]]
    solar_generation_ES = solar_generation[solar_generation.columns[solar_generation.columns.str.contains("ES")]].sum()
    solar_cap_factor_mean_ES = solar_generation_ES.sum()/(solar_cap_ES.sum()*len(n.snapshots))
    solar_cap_factor_var = (solar_generation.sum()/(solar_generation_capacity*len(n.snapshots))).var()    
    solar_potential = (n.generators_t.p_max_pu[solar_generators]*n.generators.p_nom_opt.loc[solar_generators]).sum().sum()
    solar_share = solar_generation.sum().sum()/tot_load
    solar_theo_share = solar_potential/tot_load
    solar_cap_factor_mean_theo = solar_potential/(solar_generation_capacity.sum()*len(n.snapshots))
    
    # wind generation
    wind_generators = n.generators.index[n.generators.index.str.contains('wind')]
    wind_generation_capacity = n.generators.loc[wind_generators].p_nom_opt
    wind_generation = n.generators_t.p[wind_generators]
    wind_cap_factor_mean_act = wind_generation.sum().sum()/(wind_generation_capacity.sum()*len(n.snapshots))
    wind_cap_DE = wind_generation_capacity[wind_generation_capacity.index[wind_generation_capacity.index.str.contains("DE")]]
    wind_generation_DE = wind_generation[wind_generation.columns[wind_generation.columns.str.contains("DE")]].sum()
    wind_cap_factor_mean_DE = wind_generation_DE.sum()/(wind_cap_DE.sum()*len(n.snapshots))
    wind_cap_factor_var = (wind_generation.sum()/(wind_generation_capacity*len(n.snapshots))).var() 
    wind_potential = (n.generators_t.p_max_pu[wind_generators]*n.generators.p_nom_opt.loc[wind_generators]).sum().sum()
    wind_cap_factor_mean_theo = wind_potential/(wind_generation_capacity.sum()*len(n.snapshots))

    wind_share = wind_generation.sum().sum()/tot_load
    wind_theo_share = wind_potential/tot_load
    
    solar_share = round(solar_share*100,1) # % 
    wind_share = round(wind_share*100,1) # %
    solar_theo_share = round(solar_theo_share*100,1) # %
    wind_theo_share = round(wind_theo_share*100,1) # %
    wind_potential = round(wind_potential/1e6,1) # TWh
    solar_potential = round(solar_potential/1e6,1) # TWh
    solar_cap_factor_mean_act = round(solar_cap_factor_mean_act*100,1) # %
    wind_cap_factor_mean_act = round(wind_cap_factor_mean_act*100,1) # %
    solar_cap_factor_mean_theo = round(solar_cap_factor_mean_theo*100,1) # %
    wind_cap_factor_mean_theo = round(wind_cap_factor_mean_theo*100,1) # %
    solar_cap_factor_var = round(solar_cap_factor_var*100,3) # %
    wind_cap_factor_var = round(wind_cap_factor_var*100,3) # %

    wind_cap_factor_mean_DE = round(wind_cap_factor_mean_DE*100,1) # %
    solar_cap_factor_mean_ES = round(solar_cap_factor_mean_ES*100,1) # %
    
    return solar_share, wind_share, solar_theo_share, wind_theo_share, solar_cap_factor_mean_act, wind_cap_factor_mean_act, solar_cap_factor_mean_theo, wind_cap_factor_mean_theo, solar_cap_factor_mean_ES, wind_cap_factor_mean_DE, solar_cap_factor_var, wind_cap_factor_var, wind_potential, solar_potential

def groupbycountry(df, component):
    column_names_df = pd.DataFrame(df.columns)
    column_names_df["country"] = column_names_df[component].str.split("0",expand=True)[0]
    column_names_df["country"] = column_names_df["country"] .str.split("1",expand=True)[0]
    column_names_df["country"] = column_names_df["country"] .str.split("2",expand=True)[0]
    column_names_df["country"] = column_names_df["country"] .str.split("3",expand=True)[0]
    column_names_df["country"] = column_names_df["country"] .str.split("4",expand=True)[0]
    column_names_df["country"] = column_names_df["country"] .str.split("5",expand=True)[0]
    column_names_df["country"] = column_names_df["country"] .str.split("6",expand=True)[0]

    df.columns = column_names_df["country"]
    df_country = df.T.groupby(df.T.index).sum().T

    return df_country

def swap_values_from_dfs(index, str, df):
    df_subset = pd.DataFrame(index = index)
    df_subset["values"] = df.loc[index]
    df_subset["new_ind"] = df_subset.index + str
    df_subset.set_index("new_ind", inplace=True)

    return df_subset

def rename_subset_of_df(index, str, df):
    df_subset = pd.DataFrame(index = index)
    df_subset["new_ind"] = df_subset.index + str
    subset_dict = df_subset["new_ind"].to_dict()
    df.rename(index=subset_dict, inplace=True)

    return df


def calculate_co2_emissions(n, networks_dict, label):
    ###############################################################
    ######################## CO2 emissions ########################
    ###############################################################
    
    outputs = [
                "summary",
              ]

    columns = pd.MultiIndex.from_tuples(
        networks_dict.keys(), names=["design_year","weather_year"]
    )
    
    tres_factor = 8760/len(n.snapshots)

    df = {}

    for output in outputs:
        df[output] = pd.DataFrame(columns=columns, dtype=float)
    
    # CO2 emittors and capturing facilities from power, heat and fuel production
    co2_emittors = n.links.query('bus2 == "co2 atmosphere"') # links going from fuel buses (e.g., gas, coal, lignite etc.) to "CO2 atmosphere" bus
    co2_emittors = co2_emittors.query('efficiency2 != 0') # excluding links with no CO2 emissions (e.g., nuclear)
    co2_t = -n.links_t.p2[co2_emittors.index]*tres_factor

    sig_dig = 3
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
    
    return df

def calculate_capacity_and_generation(n):

    # returns capacity in MW and generation in MWh

    weighting = 8760/len(n.snapshots)

    # generators
    df_generators = n.generators.p_nom_opt.groupby(n.generators.carrier).sum()
    df_mining = df_generators[["coal","gas","lignite","oil","uranium"]]
    df_generators.drop(["coal","gas","lignite","oil","uranium"], inplace=True) # remove "mining" from generators (coal, gas, oil, uranium)

    df_generators_t = n.generators_t.p.sum().groupby(n.generators.carrier).sum()
    df_mining_t = df_generators_t[["coal","gas","lignite","oil","uranium"]]
    df_generators_t.drop(["coal","gas","lignite","oil","uranium"], inplace=True) # remove "mining" from generators (coal, gas, oil, uranium)

    # links
    df_links = n.links.p_nom_opt.groupby(n.links.carrier).sum()
    df_links.drop(["DC"], inplace=True) # remove transmissions links (DC)

    df_links_t1 = -n.links_t.p1.sum().groupby(n.links.carrier).sum()
    chp_index = df_links_t1.loc[df_links_t1.index.str.contains("CHP")].index
    df_links_t1 = rename_subset_of_df(chp_index, " (el)", df_links_t1)
    
    df_links_t2 = -n.links_t.p2.sum().groupby(n.links.carrier).sum()
    chp_index = df_links_t2.loc[df_links_t2.index.str.contains("CHP")].index
    df_subset = swap_values_from_dfs(chp_index, " (heat)", df_links_t2)
    df_links_t = df_links_t1.append(df_subset["values"])

    df_efficiency = pd.DataFrame()
    df_efficiency["efficiency1"] = n.links.efficiency # if CHP, this is the electricity
    df_efficiency["efficiency2"] = n.links.efficiency2 # if CHP, this is the heat. If OCGT, this is the CO2
    df_efficiency["carrier"] = n.links.carrier
    df_efficiency = df_efficiency.drop_duplicates()
    df_efficiency.set_index("carrier", inplace=True)
    df_efficiency.drop(["DC"], inplace=True) 

    df_links_eff1 = df_links*df_efficiency.loc[df_links.index]["efficiency1"] # accounting for energy losses via the main link
    chp_index = df_links_eff1.loc[df_links_eff1.index.str.contains("CHP")].index
    df_links_eff1 = rename_subset_of_df(chp_index, " (el)", df_links_eff1)

    # account for heat pump COP
    heat_pump_index = df_links_eff1.index.str.contains("heat pump")
    avg_cop = n.links_t.efficiency[n.links.loc[n.links.index.str.contains("heat pump")].index].mean().mean()
    df_links_eff1.loc[heat_pump_index] *= avg_cop

    # accounting for the energy losses via the secondary link
    df_links_eff2 = df_links*df_efficiency.loc[df_links.index]["efficiency2"] 
    chp_index = df_links_eff2.loc[df_links_eff2.index.str.contains("CHP")].index
    df_subset = swap_values_from_dfs(chp_index, " (heat)", df_links_eff2)
    # add df_subset to df_links_eff1
    df_links = df_links_eff1.append(df_subset["values"])

    # stores
    df_stores = n.stores
    df_stores.loc[df_stores.query("carrier == 'H2'").query("capital_cost < 1000").index, "carrier"] = "H2 underground"
    df_stores.loc[df_stores.query("carrier == 'H2'").query("capital_cost > 1000").index, "carrier"] = "H2 overground"
    df_stores_sum = df_stores.e_nom_opt.groupby(df_stores.carrier).sum()
    df_stores_sum.drop(["coal","gas","lignite","oil","uranium"], inplace=True) # remove stores related to mining
    df_stores_sum.index = df_stores_sum.index + " storage"

    # storage units
    df_storage_units = n.storage_units.p_nom_opt.groupby(n.storage_units.carrier).sum()
    df_storage_units_t = pd.Series(index=["hydro"], data =[n.storage_units_t.p[n.storage_units.query("carrier == 'hydro'").index].sum().sum()])

    # concatenate all
    df_capacity = pd.concat([df_generators, df_links, df_stores_sum, df_storage_units])
    df_generation = pd.concat([df_generators_t, df_links_t, df_storage_units_t])*weighting

    return df_capacity, df_generation, df_mining, df_mining_t

def prepare_costs(nyears):
    
    fill_values = {"FOM": 0,
                    "VOM": 0,
                    "efficiency": 1,
                    "fuel": 0,
                    "investment": 0,
                    "lifetime": 25,
                    "CO2 intensity": 0,
                    "discount rate": 0.07}
    
    # set all asset costs and other parameters
    costs = pd.read_csv("costs_2030.csv", index_col=[0, 1]).sort_index()

    # correct units to MW and EUR
    costs.loc[costs.unit.str.contains("/kW"), "value"] *= 1e3

    # min_count=1 is important to generate NaNs which are then filled by fillna
    costs = (
        costs.loc[:, "value"].unstack(level=1).groupby("technology").sum(min_count=1)
    )

    costs = costs.fillna(fill_values)

    def annuity_factor(v):
        return annuity(v["lifetime"], v["discount rate"]) + v["FOM"] / 100

    costs["fixed"] = [
        annuity_factor(v) * v["investment"] * nyears for i, v in costs.iterrows()
    ]

    return costs


def calculate_costs(n, label, costs):

    opt_name = {
        "Store": "e",
        "Line": "s",
        "Transformer": "s"
        }

    for c in n.iterate_components(n.branch_components|n.controllable_one_port_components^{"Load"}):
        capital_costs = c.df.capital_cost*c.df[opt_name.get(c.name,"p") + "_nom_opt"]
        capital_costs_grouped = capital_costs.groupby(c.df.carrier).sum()

        capital_costs_grouped = pd.concat([capital_costs_grouped], keys=["capital"])
        capital_costs_grouped = pd.concat([capital_costs_grouped], keys=[c.list_name])

        costs = costs.reindex(capital_costs_grouped.index.union(costs.index))

        costs.loc[capital_costs_grouped.index, label] = capital_costs_grouped

        if c.name == "Link":
            p = c.pnl.p0.multiply(n.snapshot_weightings.generators, axis=0).sum()

        elif c.name == "Line":
            continue
        
        elif c.name == "StorageUnit":
            p_all = c.pnl.p.multiply(n.snapshot_weightings.generators, axis=0)
            p_all[p_all < 0.] = 0.
            p = p_all.sum()
        
        else:
            p = c.pnl.p.multiply(n.snapshot_weightings.generators, axis=0).sum()

        #correct sequestration cost
        if c.name == "Store":
            items = c.df.index[(c.df.carrier == "co2 stored") & (c.df.marginal_cost <= -100.)]
            c.df.loc[items, "marginal_cost"] = -20.

        marginal_costs = p*c.df.marginal_cost

        marginal_costs_grouped = marginal_costs.groupby(c.df.carrier).sum()

        marginal_costs_grouped = pd.concat([marginal_costs_grouped], keys=["marginal"])
        marginal_costs_grouped = pd.concat([marginal_costs_grouped], keys=[c.list_name])

        costs = costs.reindex(marginal_costs_grouped.index.union(costs.index))

        costs.loc[marginal_costs_grouped.index,label] = marginal_costs_grouped

    return costs


def calculate_capacity_deficit(n,loads, loads_t, generators, bus):
    exogenous_demand_t = loads_t[loads.query("carrier == 'electricity'").index].sum(axis=1) + loads_t[loads.query("carrier == 'industry electricity'").index].sum(axis=1)
    
    ed = calculate_endogenous_demand(n)
    if bus != "Europe":
        endogenous_demand_t = ed[ed.columns[ed.columns.str.contains(bus)]]
    else:
        endogenous_demand_t = ed
        
    total_demand_t = exogenous_demand_t + endogenous_demand_t.sum(axis=1)
    load_shedders_t = n.generators_t.p[generators.query('carrier == "load_el"').index]
    load_shedders_t_bus = load_shedders_t.sum(axis=1)
    peak_load = total_demand_t.max()
    capacity_deficit = (load_shedders_t_bus/peak_load*100)

    return capacity_deficit

def append_metrics(AA,BB,CC,DD,A,B,C,D,ii,n):
    AA_sum = sum(AA)*8760/len(n.snapshots)
    for AA_index in range(ii):
        AA[AA_index] = AA_sum
    
    CC_sum = sum(CC)/(24*(len(n.snapshots)/8760))
    for CC_index in range(ii):
        CC[CC_index] = CC_sum

    DD_sum = sum(DD)*8760/len(n.snapshots)
    for DD_index in range(ii):
        DD[DD_index] = DD_sum

    if len(AA) > 1:
        A.append(AA)
        B.append(BB)
        C.append(CC)
        D.append(DD)

    AA = []
    BB = []
    CC = []
    DD = []
    ii = 0
    return AA, BB, CC, DD, A, B, C, D, ii

# function to calculate the maximum unserved energy
def calculate_unserved_energy(n,
                              loads,
                              loads_t,
                              generators,
                              bus,
                              n_days_threshold=1,
                              ):
    
    exogenous_demand_t = loads_t[loads.query("carrier == 'electricity'").index].sum(axis=1) + loads_t[loads.query("carrier == 'industry electricity'").index].sum(axis=1)
    
    ed = calculate_endogenous_demand(n)
    
    if bus != "Europe":
        endogenous_demand_t = ed[ed.columns[ed.columns.str.contains(bus)]]
    else:
        endogenous_demand_t = ed
    
    total_demand_t = exogenous_demand_t + endogenous_demand_t.sum(axis=1)

    load_shedders_t = n.generators_t.p[generators.query('carrier == "load_el"').index]
    load_shedders_t_bus = load_shedders_t.sum(axis=1)

    load_shedding_t_binary = load_shedders_t_bus.copy()
    # count the consecutive hours where load shedding occurs
    load_shedding_t_binary[load_shedding_t_binary < 1] = 0
    load_shedding_t_binary[load_shedding_t_binary > 0] = 1

    i = 0
    ii = 0
    A = [] # unserved energy 
    B = [] # time index
    C = [] # number of days with unsered energy
    D = [] # concurrent demand (we could also pick peak load or average demand, 
           # depending on whether we are looking at capacity deficit or unserved energy)
    AA = [] # sublist of A
    BB = [] # sublist of B
    CC = [] # sublist of C
    DD = [] # sublist of D
    length = len(load_shedding_t_binary)
    before = np.zeros(length)
    for i in range(length+1):
        if i < length:
            if load_shedding_t_binary.iloc[i] == 1:
                AA.append(load_shedders_t_bus.iloc[i])
                BB.append(load_shedders_t_bus.index[i])
                CC.append(1)
                DD.append(total_demand_t.iloc[i])
                ii += 1 
                before[i] = 1

            elif i > 0:
                if (before[i-1]) == 1 and (len(AA) > 0) and (ii > n_days_threshold*24*len(n.snapshots)/8760):
                    AA, BB, CC, DD, A, B, C, D, ii = append_metrics(AA, BB, CC, DD, A, B, C, D, ii, n)
                else:
                    AA = []
                    BB = []
                    CC = []
                    DD = []
                    ii = 0
                    
        else:
            if (before[i-1]) == 1 and (len(AA) > 0) and (ii > n_days_threshold*24*len(n.snapshots)/8760):   
                AA, BB, CC, DD, A, B, C, D, ii = append_metrics(AA, BB, CC, DD, A, B, C, D, ii, n)

    unserved_energy = load_shedders_t_bus.copy()
    unserved_energy.iloc[:] = 0
    for b in range(len(B)):
        t_ind = B[b]
        unserved_energy.loc[t_ind] = np.array(A[b])/np.array(D[b])*100

    return unserved_energy