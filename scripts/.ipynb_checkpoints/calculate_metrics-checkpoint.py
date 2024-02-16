import os
import glob
import pypsa
import pandas as pd
import numpy as np
from pypsa_metrics import calculate_curtailment, calculate_costs, calculate_endogenous_demand, calculate_renewable_penetration, groupbycountry
from helper import override_component_attrs
overrides = override_component_attrs("data/override_component_attrs")

dyear = "1990"
RDIR = "results/simulation_3h_dy1990_and_2017/csvs/" + dyear + "/"
path = "results/simulation_3h_dy1990_and_2017/postnetworks/" + dyear + "/"

if not os.path.isdir(RDIR):
    os.mkdir(RDIR)
    print("creating new csv directory")
else:
    print("csv directory already exists - existing files are overwritten")
    
network_names = glob.glob(path + "*.nc")
no_networks = len(network_names)

# wind & solar    
wind_cap = {}
solar_cap = {}
wind_potential = {}
solar_potential = {}

wind_cap_factor_mean = {}
solar_cap_factor_mean = {}

wind_cap_factor_mean_theo = {}
solar_cap_factor_mean_theo = {}

wind_cap_factor_mean_DE = {}
solar_cap_factor_mean_ES = {}

wind_cap_factor_var = {}
solar_cap_factor_var = {}

solar_curt = {}
wind_curt = {}
wind_share = {}
solar_share = {}
wind_theo_share = {}
solar_theo_share = {}
VRE_share = {}
ren_curt = {}

# hydro
hydro_inflow_mean = {}
hydro_inflow_var = {}

# system
system_cost = {}
total_demand = {}

heat_demand_tot = {}
heat_demand_var = {}

# country specific data
wind_CF_c = {}
solar_CF_c = {}
hydro_inflow_c = {}
heat_demand_c = {}

COP_residential_rural_ground_c = {}
COP_services_rural_ground_c = {}
COP_residential_urban_decentral_air_c = {}
COP_services_urban_decentral_air_c = {}
COP_urban_central_air_c = {}

capacity_deficit_c = {}
unserved_energy_c_avlh = {}
unserved_energy_EU_TWh = {}

def group_per_country(df, bus_country_dict, group_option="sum"):
    df["ind"] = df.index
    df["bus"] = df["ind"].str.split(" ",expand=True)[0] + " " + df["ind"].str.split(" ",expand=True)[1]
    df.replace({"bus":bus_country_dict},inplace=True)
    if group_option == "sum":
        df = df.drop(columns=["ind"]).groupby("bus").sum()[0]
    elif group_option == "mean":
        df = df.drop(columns=["ind"]).groupby("bus").mean()[0]
    return df

df = {}
df["cost"] = pd.DataFrame(columns=[""], dtype=float)

# curtailment setting:
denominator = "gen_theoretical" 

for j in range(no_networks):
    n = pypsa.Network(network_names[j],override_component_attrs=overrides)

    AC_buses = n.buses.query("carrier == 'AC'")

    if j == 0:
        bus_country_dict = AC_buses.country.to_dict()

    try:
        n.objective
    except:
        continue
    opts = network_names[j].split('/')[4].split('_')
    year = opts[1][2:] if "capacity" in path else opts[4][2:]
    print(year)
            
    weighting = 8760/len(n.snapshots)
    exogenous_demand = (n.loads_t.p_set[n.loads.query("carrier == 'electricity'").index].sum().sum() + 
    (n.loads.query("carrier == 'industry electricity'").p_set*len(n.snapshots)).sum()
    )
    endogenous_demand = calculate_endogenous_demand(n)
    total_demand_j = exogenous_demand + endogenous_demand.sum().sum()
    total_demand[year] = round(total_demand_j*weighting/1e6,3) # TWh
    heat_loads = n.loads_t.p_set.columns[n.loads_t.p_set.columns.str.contains('heat')]
    heat_demand_j = n.loads_t.p_set[heat_loads].sum().sum()
    heat_demand_tot[year] = round(heat_demand_j*weighting/1e6,3) # TWh
    solar_share_j, wind_share_j, solar_theo_share_j, wind_theo_share_j, solar_cap_factor_mean_act_j, wind_cap_factor_mean_act_j, solar_cap_factor_mean_theo_j, wind_cap_factor_mean_theo_j,  solar_cap_factor_mean_ES_j, wind_cap_factor_mean_DE_j,solar_cap_factor_var_j, wind_cap_factor_var_j, wind_potential_j, solar_potential_j = calculate_renewable_penetration(n)
    
    solar_share[year] = round(solar_share_j,1)
    wind_share[year] = round(wind_share_j,1)
    solar_theo_share[year] = round(solar_theo_share_j,1)
    wind_theo_share[year] = round(wind_theo_share_j,1)
    solar_potential[year] = round(solar_potential_j*weighting,1)
    wind_potential[year] = round(wind_potential_j*weighting,1)
    
    solar_cap_factor_mean[year] = round(solar_cap_factor_mean_act_j,1)
    wind_cap_factor_mean[year] = round(wind_cap_factor_mean_act_j,1)
    solar_cap_factor_mean_theo[year] = round(solar_cap_factor_mean_theo_j,1)
    wind_cap_factor_mean_theo[year] = round(wind_cap_factor_mean_theo_j,1)

    solar_cap_factor_mean_ES[year] = round(solar_cap_factor_mean_ES_j,1)
    wind_cap_factor_mean_DE[year] = round(wind_cap_factor_mean_DE_j,1)

    solar_cap_factor_var[year] = round(solar_cap_factor_var_j,3)
    wind_cap_factor_var[year] = round(wind_cap_factor_var_j,3)
    VRE_share[year] = round(solar_share_j + wind_share_j,1)

    # unserved energy and capacity deficits
    ed = endogenous_demand.copy()
    generators = n.generators
    hydro_units = n.storage_units.query("carrier == 'hydro'")
    loads = n.loads
    loads_t = n.loads_t.p
    exogenous_demand_t_1 = loads_t[loads.query("carrier == 'electricity'").index]
    exogenous_demand_t_2 = loads_t[loads.query("carrier == 'industry electricity'").index]
    exogenous_demand_t_1_country = groupbycountry(exogenous_demand_t_1, "Load")
    exogenous_demand_t_2_country = groupbycountry(exogenous_demand_t_2, "Load") 
    exogenous_demand_t_country = exogenous_demand_t_1_country + exogenous_demand_t_2_country

    endogenous_demand_t_country = groupbycountry(ed, "Link")
    total_demand_t_country = exogenous_demand_t_country + endogenous_demand_t_country
    
    load_shedders_t = n.generators_t.p[generators.query('carrier == "load_el"').index]
    load_shedders_t_country = groupbycountry(load_shedders_t, "Generator")
    peak_load = total_demand_t_country.max()
    
    peak_capacity_deficit = (load_shedders_t_country/peak_load*100).max() # % of peak load
    unserved_energy = load_shedders_t_country.sum()*weighting # MWh

    capacity_deficit_c[year] = peak_capacity_deficit
    unserved_energy_c_avlh[year] = unserved_energy/(total_demand_t_country.sum()*weighting/8760) # % of total demand
    unserved_energy_EU_TWh[year] = (n.generators_t.p[n.generators.query("carrier == 'load_el'").index].sum().sum()*weighting)/1e6

    # Calculate capacities (in GW)        
    solar_cap[year] = round(n.generators.loc[pd.concat([n.generators.query("carrier == 'solar'"),n.generators.query("carrier == 'solar rooftop'")]).index].p_nom_opt.sum().sum()/1e3,1)
    wind_cap[year] = round(n.generators[n.generators.index.str.contains("wind")].p_nom_opt.sum()/1e3,1)
    
    # Calculate curtailment
    solar_abs_curt_j, solar_curt_j, wind_abs_curt_j, wind_curt_j = calculate_curtailment(n,denominator_category=denominator)
    solar_curt[year] = round(sum(solar_curt_j),1)
    wind_curt[year] = round(sum(wind_curt_j),1)
    
    solar_share_j = solar_share[year]/100
    wind_share_j = wind_share[year]/100
    solar_curt_j_norm = sum(solar_curt_j)*solar_share_j
    wind_curt_j_norm = sum(wind_curt_j)*wind_share_j
    if denominator == "gen_theoretical":
        ren_curt_j = (solar_curt_j_norm + wind_curt_j_norm)/(solar_share_j+wind_share_j)
    else:
        ren_curt_j = solar_curt_j + wind_curt_j    
    ren_curt[year] = round(ren_curt_j,1)

    # Calculate hydro inflow
    hydro_inflow_j = n.storage_units_t.inflow.sum().sum()/1e6 # converting units from MWh to TWh
    hydro_inflow_mean[year] = round(hydro_inflow_j*weighting,3)

    # Calculate system cost
    system_cost[year] = round(calculate_costs(n, "", df["cost"]).sum().item()/1e9,1)

    ################ country specific data ################
    # wind & solar
    wind_generators = n.generators.index[n.generators.index.str.contains('wind')]
    wind_CF_bus_t = n.generators_t.p_max_pu[wind_generators]
    wind_CF_bus = pd.DataFrame(wind_CF_bus_t.mean())
    wind_CF_c[year] = group_per_country(wind_CF_bus, bus_country_dict, group_option="mean")
    
    solar_generators = pd.concat([n.generators.query('carrier == "solar"'),n.generators.query('carrier == "solar rooftop"')]).index
    solar_CF_bus_t = n.generators_t.p_max_pu[solar_generators]
    solar_CF_bus = pd.DataFrame(solar_CF_bus_t.mean())
    solar_CF_c[year] = group_per_country(solar_CF_bus, bus_country_dict, group_option="mean")

    # hydro
    hydro_inflow_bus = pd.DataFrame((n.storage_units_t.inflow.sum()/1e6*weighting).round(3))
    hydro_inflow_c[year] = group_per_country(hydro_inflow_bus, bus_country_dict, group_option="sum")

    # heating
    heat_demand_bus = pd.DataFrame(n.loads_t.p_set[heat_loads].sum()*weighting/1e6) # TWh
    heat_demand_c[year] = group_per_country(heat_demand_bus, bus_country_dict, group_option="sum")

    # COP
    heat_pumps = n.links.loc[n.links.carrier[n.links.carrier.str.contains("heat pump")].index]
    heat_pump_types = heat_pumps.carrier.unique()
     
    summer_index = n.snapshots[n.snapshots > "2013/5/1"][n.snapshots[n.snapshots > "2013/5/1"] < "2013/10/1"]

    # first divide COP into individual dataframes for each heat pump type
    COP_dict = {}
    for t in heat_pump_types:
        COP_df = n.links_t.efficiency[heat_pumps.query("carrier == @t").index]
        COP_df.columns = [i[0:2] for i in COP_df.columns]
        
        # group by country
        COP_df = COP_df.T.groupby(COP_df.T.index).mean().T
        
        # omit summer months assuming heating season from 1st of October to 30th of April
        COP_df.loc[summer_index] = np.nan

        COP_dict[t] = COP_df.mean()

    COP_residential_rural_ground_c[year] = COP_dict['residential rural ground heat pump']
    COP_services_rural_ground_c[year] = COP_dict['services rural ground heat pump']
    COP_residential_urban_decentral_air_c[year] = COP_dict['residential urban decentral air heat pump']
    COP_services_urban_decentral_air_c[year] = COP_dict['services urban decentral air heat pump']
    COP_urban_central_air_c[year] = COP_dict['urban central air heat pump']
    ######################################################

    print(j)

variables_dict = {
                # Capacities 
                    "wind_cap":wind_cap,
                    "solar_cap":solar_cap,   
                    "wind_potential":wind_potential,
                    "solar_potential":solar_potential,                   
                    "wind_capacity_factor_mean":wind_cap_factor_mean,
                    "solar_capacity_factor_mean":solar_cap_factor_mean,
                    "wind_capacity_factor_mean_theo":wind_cap_factor_mean_theo,
                    "solar_capacity_factor_mean_theo":solar_cap_factor_mean_theo,
                    "wind_capacity_factor_var":wind_cap_factor_var,
                    "solar_capacity_factor_var":solar_cap_factor_var,
                    "wind_capacity_factor_mean_DE":wind_cap_factor_mean_DE,
                    "solar_capacity_factor_mean_ES":solar_cap_factor_mean_ES,
                    
                # electricity shares
                    "wind_share":wind_share,
                    "solar_share":solar_share,
                    "wind_theo_share":wind_theo_share,
                    "solar_theo_share":solar_theo_share,
                    "VRE_share":VRE_share,
                    "VRE_curt":ren_curt,
                
                # curtailment 
                    "wind_curt":wind_curt,
                    "solar_curt":solar_curt,

                # hydro inflow
                    "hydro_inflow":hydro_inflow_mean,
                    
                # system cost
                    "system_cost":system_cost,
                    
                # energy demand
                    "total_demand":total_demand,
                    "heat_demand":heat_demand_tot,

                # unserved energy 
                    "unserved_energy_TWh":unserved_energy_EU_TWh,

                # country specific data
                    # renewable resources and energy demand
                    "wind_CF_country":wind_CF_c,
                    "solar_CF_country":solar_CF_c,
                    "hydro_inflow_country":hydro_inflow_c,
                    "heat_demand_country":heat_demand_c,

                    # Coefficient of Performance
                    "COP_residential_rural_ground":COP_residential_rural_ground_c,
                    "COP_services_rural_ground":COP_services_rural_ground_c,
                    "COP_residential_urban_decentral_air":COP_residential_urban_decentral_air_c,
                    "COP_services_urban_decentral_air":COP_services_urban_decentral_air_c,
                    "COP_urban_central_air":COP_urban_central_air_c,

                    # load shedding
                    "unserved_energy_avlh_country":unserved_energy_c_avlh,
                    "capacity_deficit_country":capacity_deficit_c,
                    }

for key in variables_dict.keys():
    df_name = " ".join(key.split("_"))

    if "country" in key or "COP" in key: 
        df = pd.DataFrame(variables_dict[key])

    else:
        df = pd.DataFrame(variables_dict[key].values(),index=variables_dict[key].keys(),columns=[df_name])
        df.index.set_names("year",inplace=True)

    df.to_csv(RDIR + "/" + key + ".csv")