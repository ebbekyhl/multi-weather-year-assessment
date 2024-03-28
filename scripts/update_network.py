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

def update_heat(n,n_weather,weather_year):
    heat_loads = n.loads_t.p_set.columns[n.loads_t.p_set.columns.str.contains('heat')]
    new_heat_p_set = n_weather.loads_t.p_set[heat_loads]
            
    df_loads_p_set = n.loads_t.p_set
    df_loads_p_set.loc[df_loads_p_set.index,heat_loads] = new_heat_p_set
    
    n.loads_t.p_set = df_loads_p_set

    return n 


def freeze_network(n):
    # generators
    df = n.generators
    df_real = n.generators.query('capital_cost > 0') # "Real" technologies
    df.loc[df_real.index,"p_nom"] = df_real.p_nom_opt.values
    df.loc[df_real.index,'p_nom_extendable'] = snakemake.config['scenario']['generators_extendable']
    n.generators = df
    print(n.generators.p_nom_extendable[0:5])

    # lines
    lines = n.lines
    lines.loc[lines.index, "s_nom"] = lines.s_nom_opt.values
    lines.loc[lines.index, "s_nom_extendable"] = False
    n.lines = lines
    print(n.lines.s_nom_extendable[0:5])

    # links
    df = n.links
    df_real = n.links.query('capital_cost > 0') # "Real" technologies 
    df.loc[df_real.index,'p_nom'] = df_real.p_nom_opt.values
    df.loc[df_real.index,'p_nom_extendable'] = snakemake.config['scenario']['links_extendable']
    n.links = df
    print(n.links.p_nom_extendable[0:5])

    # battery discharger (special case)
    df = n.links
    df_bat = n.links[n.links.index.str.contains('battery discharger')]
    df.loc[df_bat.index,'p_nom'] = df_bat.p_nom_opt.values
    df.loc[df_bat.index,'p_nom_extendable'] = snakemake.config['scenario']['links_extendable']
    n.links = df

    # stores
    df = n.stores
    df_real = n.stores.query('capital_cost > 0')
    df.loc[df_real.index,'e_nom'] = df_real.e_nom_opt.values
    df.loc[df_real.index,'e_nom_extendable'] = snakemake.config['scenario']['stores_extendable'] 
    n.stores = df
    print(n.stores.e_nom_extendable[0:5])

    return n 

def add_load_shedding(n):
    nodes_LV = n.buses.query('carrier == "low voltage"').index

    nodes_heat1= n.buses.query('carrier == "residential rural heat"').index
    nodes_heat2 = n.buses.query('carrier == "services rural heat"').index
    nodes_heat3 = n.buses.query('carrier == "residential urban decentral heat"').index
    nodes_heat4 = n.buses.query('carrier == "services urban decentral heat"').index
    nodes_heat5 = n.buses.query('carrier == "urban central heat"').index
    
    n.add("Carrier", "load_el")
    n.add("Carrier", "load_heat")

    # Low-voltage electricity grid
    n.madd("Generator", 
            nodes_LV + " load shedding",
            bus=nodes_LV,
            carrier='load_el',
            marginal_cost=1e5, # Eur/MWh
            # intersect between macroeconomic and surveybased willingness to pay
            # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
            #p_nom=1e6, # MW
            p_nom_extendable = True,
            capital_cost = 0)

    # Heat buses
    heat_nodes_dict = {'1':nodes_heat1,
                        '2':nodes_heat2,
                        '3':nodes_heat3,
                        '4':nodes_heat4,
                        '5':nodes_heat5,}

    for heat_nodes in heat_nodes_dict.keys():
        n.madd("Generator", 
                heat_nodes_dict[heat_nodes] + " load shedding",
                bus=heat_nodes_dict[heat_nodes] ,
                carrier='load_heat',
                marginal_cost=1e5, # Eur/MWh
                # intersect between macroeconomic and surveybased willingness to pay
                # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
                #p_nom=1e6, # MW
                p_nom_extendable = True,
                capital_cost = 0)
    
    print('Load shedding was added')

    return n 

def add_co2_price(n,n_design):
    # Adding CO2 price to positive emittors as an extra cost to produce an additional unit of energy
    # and
    # Adding CO2 price to negative emittors as an extra incentive to capture one unit more of CO2)

    if snakemake.config['scenario']['custom_co2_price']:
        # Custom co2 price:
        co2_price = snakemake.config['scenario']['co2_price']
    else:
        # Lagrange multiplier for CO2 of design network:
        co2_price = n_design.global_constraints.loc['CO2Limit']['mu'] # EUR/tCO2

    ########################################################################
    ######## Electricity, heating, hydrogen production, etc. ###############
    ########################################################################
    # Links with "p2" corresponding to the CO2 emissions: 
    bus2_emittors_ind = list(n.links.query('bus2 == "co2 atmosphere"').index)
    bus2_emittors = n.links.loc[bus2_emittors_ind]
    bus2_emittors = bus2_emittors.query('efficiency2 != 0') # omit every link with zero emissions, e.g., nuclear. 
    # Negative emissions links such as "biogas to gas" are still included by allowing "efficiency2" to be negative.

    # Unit conversion
    # efficiency2 = tonne CO2 emitted per unit of inlet consumed (e.g., fuel burned. For gas, 0.2 tCO2/MWh_th) 
    # mwh_inlet_per_tco2 = 1/(bus2_emittors.efficiency2) # converting "tCO2/MWh_inlet" to "MWh_inlet/tCO2"
    # eur_per_MWh_inlet = co2_price/mwh_inlet_per_tco2
    bus2_emittors_co2_price = co2_price*bus2_emittors.efficiency2

    # UPDATE MARGINAL COST
    new_marginal_price_bus2_emittors = bus2_emittors.marginal_cost + bus2_emittors_co2_price
    bus2_emittors['marginal_cost'] = new_marginal_price_bus2_emittors
    n.links.loc[bus2_emittors.index] = bus2_emittors
    ########################################################################

    ########################################################################
    ################################## CHP #################################
    ########################################################################
    # Links with "p3" corresponding to CO2 emissions (CHP plants):
    chp_ind = list(n.links.query('bus3 == "co2 atmosphere"').index)
    chp = n.links.loc[chp_ind]
    
    # The following three lines are not used but given as an overview of the chp plants:
    chp_gas = chp.query('carrier == "urban central gas CHP"') # positive emissions 
    chp_gas_CC = chp.query('carrier == "urban central gas CHP CC"') # positive emissions (but reduced with CC)
    chp_biomass_cc = chp.query('carrier == "urban central solid biomass CHP CC"') # net-negative emissions

    # Unit conversion
    # efficiency3 = tonne CO2 emitted per unit of inlet consumed (e.g., fuel burned. For gas, 0.2 tCO2/MWh_th) 
    # mwh_th_per_tco2 = 1/(chp.efficiency3) # converting "tCO2/MWh_inlet" to "MWh_inlet/tCO2"
    # eur_per_MWh_th = co2_price/mwh_th_per_tco2 # converting "EUR/tCO2" to "EUR/MWh_inlet" (accounts for sign differences)
    chp_co2_price = co2_price*chp.efficiency3

    # UPDATE MARGINAL COST
    new_marginal_price_CHP = chp.marginal_cost + chp_co2_price
    chp['marginal_cost'] = new_marginal_price_CHP
    n.links.loc[chp.index] = chp
    ########################################################################

    ########################################################################
    ############################ Process emissions #########################
    ########################################################################
    # w/ and w/o CC! 
    process_index = n.links.query('bus1 == "co2 atmosphere"').index
    process = n.links.loc[process_index]

    # Unit conversion 
    # efficiency = tonne CO2 emitted per unit of inlet
    # tco2_per_tco2 = 1/(process.efficiency) # in case of CC, efficiency < 1, otherwise 1.
    # co2_eur_per_tco2 = co2_price/tco2_per_tco2
    process_co2_price = co2_price*process.efficiency

    # UPDATE MARGINAL COST
    new_marginal_price_process = process.marginal_cost + process_co2_price
    process['marginal_cost'] = new_marginal_price_process
    n.links.loc[process.index] = process
    ########################################################################

    ########################################################################
    ################################# DAC ##################################
    ########################################################################
    dac_index = n.links.index[n.links.index.str.contains('DAC')] # links going from "CO2 atmosphere" to "CO2 stored" (sequestration)
    dac = n.links.loc[dac_index]

    # Unit conversion 
    # efficiency = tonne CO2 emitted per unit of inlet
    # tco2_per_tco2 = 1/(dac.efficiency)
    # eur_per_tco2 = co2_price/tco2_per_tco2
    dac_co2_price = -co2_price*dac.efficiency

    # UPDATE MARGINAL COST
    new_marginal_price_dac = dac.marginal_cost + dac_co2_price # co2 price is subtracted from the marginal cost!
    dac['marginal_cost'] = new_marginal_price_dac
    n.links.loc[dac.index] = dac
    ########################################################################

def split_horizon(n, networks, weather_year):
    #################################### Copy network structure ##############################################
    # Here, we consider a reference network solely to copy the data strucure and format for later usagee
    freq = 8760/len(n.snapshots) # sample frequency

    # get the old time indices (January to December):
    df_index_old = pd.DataFrame(n.snapshots,columns=["snapshot"])
    df_index_old["month"] = df_index_old.snapshot.dt.month
    df_index_old["day"] = df_index_old.snapshot.dt.day
    df_index_old["hour"] = df_index_old.snapshot.dt.hour
    df_index_old.set_index(["month","day","hour"],inplace=True)

    # create new time indices (July to June):
    new_time_index = pd.date_range("2012-07-01", "2013-07-01", freq=freq)[:-1]
    df_index_new = pd.DataFrame(new_time_index, columns=["snapshot"])
    df_index_new["month"] = df_index_new.snapshot.dt.month
    df_index_new["day"] = df_index_new.snapshot.dt.day
    df_index_new["hour"] = df_index_new.snapshot.dt.hour
    df_index_new.set_index(["month","day","hour"],inplace=True)

    # order old time indices:
    old_time_index_ordered = pd.DatetimeIndex(df_index_old.loc[df_index_new.index].snapshot)
    # insert new time index
    n.snapshots = new_time_index

    # get all time-dependent variables
    temporal_data = {"links_t":n.links_t, 
                    "generators_t":n.generators_t,
                    "storage_units_t":n.storage_units_t,
                    "stores_t":n.stores_t,
                    "loads_t":n.loads_t,}

    variables = {}
    for comp in temporal_data.keys():
        n_comp_t = temporal_data[comp]
        keys = []
        for key in n_comp_t.keys():
            if not n_comp_t[key].empty:
                keys.append(key)        
        variables[comp] = keys

    # split indices in "Fall" and "Spring":
    index_fall = n.snapshots[n.snapshots >= pd.to_datetime("2013-07-01")]
    index_spring = n.snapshots[n.snapshots < pd.to_datetime("2013-07-01")]

    ################################# Acquire data from target network ###########################################
    label = {weather_year-1:"fall", weather_year:"spring"}
    indices = {"fall":index_fall, 
               "spring":index_spring}

    for comp in temporal_data.keys(): # loop over components, e.g., links, generators, etc.
        variables_c = variables[comp]

        for i in variables_c: # loop over variables in component, e.g., p_max_pu, efficiency, etc.

            for y in label.keys(): # loop over "fall" and "spring"
                category = label[y]

                n_ref = networks[y]
                df = getattr(n_ref, comp)[i]

                if label[y]== "fall":
                    df_filtered = df.loc[indices[category]]
                    
                elif label[y] == "spring":
                    df_filtered2 = pd.concat([df_filtered,df.loc[indices[category]]])
                    df_filtered2.index = new_time_index

                    # start adding components (e.g., available renewable resources or heating demand)
                    if comp == "loads_t":
                        n.loads_t[i]  = df_filtered2
                    
                    elif comp == "links_t":
                        n.links_t[i]  = df_filtered2

                    elif comp == "generators_t":
                        n.generators_t[i]  = df_filtered2

                    elif comp == "storage_units_t":
                        n.storage_units_t[i]  = df_filtered2

                    elif comp == "stores_t":
                        n.stores_t[i]  = df_filtered2


if __name__ == "__main__":
    if 'snakemake' not in globals():
        from helper import mock_snakemake
        snakemake = mock_snakemake(
            'update_renewable_profiles',
            design_year="2013", 
            weather_year="2008",
        )
    
    overrides = override_component_attrs(snakemake.input.overrides)
    weather_year = snakemake.wildcards.weather_year

    # Read design network
    n_design = pypsa.Network(snakemake.input.design_network, 
                    override_component_attrs=overrides)

    # Read network obtained with different weather year
    n_weather = pypsa.Network(snakemake.input.weather_network , 
                                override_component_attrs=overrides)

    # copy design network
    n = n_design.copy()
    
    update_renewables(n,n_weather)

    if 'heat' in snakemake.wildcards.opts:
        print('correcting heat demand')
        update_heat(n,n_weather,weather_year)

    freeze_network(n)

    add_load_shedding(n)

    if snakemake.config['scenario']['add_co2_price']:
        add_co2_price(n,n_design)

    if snakemake.config['scenario']['add_co2_lim']:
        n.global_constraints.loc['CO2Limit','constant'] = snakemake.config['scenario']['new_co2_lim']
    else:
        # Remove global CO2 constraint since it is being replaced by a CO2 price
        n.remove('GlobalConstraint',"CO2Limit")

    if snakemake.config['scenario']['split_horizon'] and weather_year > 1960:
        wyears = [weather_year-1, weather_year]
        config = snakemake.config
        nodes = config["model"]["nodes"]
        tres = config["model"]["tres"]
        co2_lvl = config["model"]["co2"]
        sectors = config["model"]["sectors"]
        networks = {}
        for wyear in wyears:
            networks[wyear] = pypsa.Network("networks/networks_n37_" + tres + "h/elec_wy" + str(wyear) + "_s370_" + nodes + "_lv1.0__Co2L" + co2_lvl + "-" + tres + "h-" + sectors + "-solar+p3-dist1_2050.nc")
        split_horizon(n, networks, weather_year)

    print("Network fully updated!")
    n.export_to_netcdf(snakemake.output.network)
