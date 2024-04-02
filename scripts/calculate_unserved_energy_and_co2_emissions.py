import pypsa
import glob
import numpy as np
import pandas as pd    
import os
from pypsa_metrics import calculate_endogenous_demand

# Overview of metrics
# - loss of load expectance LOLE (frequency of >3 hours events) 
# - unserved energy 
# - additional cost (operation) and potential cost from compensative capacity from storage/power plants)
# - additional CO2 emissions 

def calculate_capacity_deficit(n,loads, loads_t, generators, bus, unit="GW"):

    load_shedders_t = n.generators_t.p[generators.query('carrier == "load_el"').index]
    load_shedders_t_bus = load_shedders_t.sum(axis=1)

    if unit == "GW":
        capacity_deficit = load_shedders_t_bus/1e3

    else:
        exogenous_demand_t = loads_t[loads.query("carrier == 'electricity'").index].sum(axis=1) + loads_t[loads.query("carrier == 'industry electricity'").index].sum(axis=1)
        ed = calculate_endogenous_demand(n)
        if bus != "Europe":
            endogenous_demand_t = ed[ed.columns[ed.columns.str.contains(bus)]]
        else:
            endogenous_demand_t = ed
            
        total_demand_t = exogenous_demand_t + endogenous_demand_t.sum(axis=1)
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

    # DD_sum = sum(DD)*8760/len(n.snapshots)
    # for DD_index in range(ii):
    #     DD[DD_index] = DD_sum

    if len(AA) > 1:
        A.append(AA)
        B.append(BB)
        C.append(CC)
        D.append(DD)

    # reset sublists
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
    
    # exogenous_demand_t = loads_t[loads.query("carrier == 'electricity'").index].sum(axis=1) + loads_t[loads.query("carrier == 'industry electricity'").index].sum(axis=1)
    exogenous_demand_t_1 = loads_t[loads.query("carrier == 'electricity'").index]
    exogenous_demand_t_2 = loads_t[loads.query("carrier == 'industry electricity'").index]
    column_names_df = pd.DataFrame(exogenous_demand_t_2.columns)
    column_names_df["country"] = column_names_df["Load"].str.split(" ",expand=True)[0] + " " + column_names_df["Load"].str.split(" ",expand=True)[1]
    exogenous_demand_t_2.columns = column_names_df["country"]
    exogenous_demand_t = exogenous_demand_t_1 + exogenous_demand_t_2

    exogenous_demand_t = exogenous_demand_t[bus] if bus != "Europe" else exogenous_demand_t.sum(axis=1)

    ed = calculate_endogenous_demand(n)
    
    if bus != "Europe":
        endogenous_demand_t = ed[ed.columns[ed.columns.str.contains(bus)]]
    else:
        endogenous_demand_t = ed
    
    total_demand_t = exogenous_demand_t + endogenous_demand_t.sum(axis=1)
    total_demand_avg = (exogenous_demand_t + endogenous_demand_t.sum(axis=1)).mean()

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
    C = [] # number of days with unserved energy in the considered event
    D = [] # concurrent demand (we could also pick peak load or average demand, 
           # depending on whether we are looking at capacity deficit or unserved energy)
    
    AA = [] # sublist of A 
    BB = [] # sublist of B
    CC = [] # sublist of C
    DD = [] # sublist of D
    length = len(load_shedding_t_binary)
    before = np.zeros(length)

    for i in range(length+1): # looping over all hours with load shedding

        if i < length:

            if load_shedding_t_binary.iloc[i] == 1: # 1 indicates that it is an hour with load shedding
                AA.append(load_shedders_t_bus.iloc[i]) # append the load shedding in the given hour
                BB.append(load_shedders_t_bus.index[i]) # append the index of the given hour
                CC.append(1) # counting the number of hours in the event
                DD.append(total_demand_avg) # DD.append(total_demand_t.iloc[i])
                ii += 1 

                before[i] = 1 # to indicate for later use that we are still in the same event

            elif i > 0: 

                # if we are in a consective hour an event
                if (before[i-1]) == 1 and (len(AA) > 0) and (ii > n_days_threshold*24*len(n.snapshots)/8760):
                    AA, BB, CC, DD, A, B, C, D, ii = append_metrics(AA, BB, CC, DD, A, B, C, D, ii, n)
                
                else: # in case that we encounter a new event (which is not connected to the previous one)
                    AA = [] # reset sublist AA
                    BB = [] # reset sublist BB
                    CC = [] # reset sublist CC
                    DD = [] # reset sublist DD
                    ii = 0 # reset counter
                    
        else:

            if (before[i-1]) == 1 and (len(AA) > 0) and (ii > n_days_threshold*24*len(n.snapshots)/8760):   

                AA, BB, CC, DD, A, B, C, D, ii = append_metrics(AA, BB, CC, DD, A, B, C, D, ii, n)

    unserved_energy = load_shedders_t_bus.copy() # copying dataframe to keep the same structure
    unserved_energy.iloc[:] = 0 # setting all values to zero
    for b in range(len(B)):
        t_ind = B[b]
        unserved_energy.loc[t_ind] = np.array(A[b])/np.array(D[b]) # unserved energy in units of average load hours
        
    return unserved_energy, A, B, C, D, total_demand_t

def calculate_emissions(n,label,df):
    sig_dig = 3
    tres_factor = 8760/len(n.snapshots)

    # CO2 emittors and capturing facilities from power, heat and fuel production
    co2_emittors = n.links.query('bus2 == "co2 atmosphere"') # links going from fuel buses (e.g., gas, coal, lignite etc.) to "CO2 atmosphere" bus
    co2_emittors = co2_emittors.query('efficiency2 != 0') # excluding links with no CO2 emissions (e.g., nuclear)
    co2_emittors_index = co2_emittors.index
    n_links_t_p2 = n.links_t.p2
    idx = co2_emittors.index.intersection(n_links_t_p2.columns)
    
    co2_t = -n.links_t.p2[idx]*tres_factor # co2 emissions in every time step 
        
    co2_t_renamed = co2_t.rename(columns=co2_emittors.carrier.to_dict())
    co2_t_grouped = co2_t_renamed.groupby(by=co2_t_renamed.columns,axis=1).sum().sum()
    for i in range(len(co2_t_grouped.index)):
        if 'gas boiler' not in co2_t_grouped.index[i]:
            co2_t_i = round(co2_t_grouped.iloc[i]/1e6,sig_dig)
            df.loc[co2_t_grouped.index[i],label] = co2_t_i

    co2_t_gas_boiler = round(co2_t_grouped.loc[co2_t_grouped.index[co2_t_grouped.index.str.contains('gas boiler')]].sum()/1e6,sig_dig)

    df.loc['gas boiler',label] = co2_t_gas_boiler
    ###############################################################

    # CO2 emissions and capturing from chp plants
    chp = n.links.query('bus3 == "co2 atmosphere"') # links going from CHP fuel buses (e.g., biomass or natural gas) to "CO2 atmosphere" bus
    co2_chp_t = -n.links_t.p3[chp.index]*tres_factor
    # NB! Biomass w.o. CC has no emissions. For this reason, Biomass w. CC has negative emissions.

    co2_chp_t_renamed = co2_chp_t.rename(columns=chp.carrier.to_dict())
    co2_chp_t_renamed_grouped = co2_chp_t_renamed.groupby(by=co2_chp_t_renamed .columns,axis=1).sum().sum()

    co2_chp_t_renamed_grouped_gas = round(co2_chp_t_renamed_grouped.loc[co2_chp_t_renamed_grouped.index[co2_chp_t_renamed_grouped.index.str.contains('gas CHP')]].sum()/1e6,sig_dig)
    co2_chp_t_renamed_grouped_biomass = round(co2_chp_t_renamed_grouped.loc[co2_chp_t_renamed_grouped.index[co2_chp_t_renamed_grouped.index.str.contains('solid biomass CHP')]].sum()/1e6,sig_dig)

    df.loc['gas CHP',label] = co2_chp_t_renamed_grouped_gas
    df.loc['biomass CHP CC',label] = co2_chp_t_renamed_grouped_biomass 
    ###############################################################

    # process emissions CC
    #co2_process = n.links.query('bus1 == "co2 atmosphere"').index # links going from "process emissions" to "CO2 atmosphere" bus
    #co2_process_t = -n.links_t.p1[co2_process]*tres_factor
    # Process emissions can have CC which captures 90 % of the emissions. 
    # To include the 90% capture in the balance, call: -n.links_t.p2["EU process emissions CC"]
    #co2_process_t_sum = round(co2_process_t.sum().sum()/1e6,sig_dig)
    co2_process_capture_t_sum = round(n.links_t.p2["EU process emissions CC"].sum().sum()/1e6,sig_dig)*tres_factor

    df.loc['process emissions CC', label] = co2_process_capture_t_sum
    ###############################################################

    # load emissions (e.g., land transport or agriculture)
    loads_co2 = n.loads # .query('bus == "co2 atmosphere"')
    load_emissions_index = loads_co2.index[loads_co2.index.str.contains('emissions')]
    load_emissions_t = -n.loads_t.p[load_emissions_index]*tres_factor
    load_emissions_t = load_emissions_t.rename(columns={"oil emissions":"plastic decay and kerosene combustion",
                                                        "process emissions":"process emissions", 
                                                        "agriculture machinery oil emissions":"agriculture oil emissions"})
    
    for col in load_emissions_t.columns:
        load_emissions_t_sum = round(load_emissions_t[col].sum()/1e6,sig_dig)
        df.loc[col, label] = load_emissions_t_sum

    ###############################################################

    # direct air capture
    dac = n.links.index[n.links.index.str.contains('DAC')] # links going from "CO2 atmosphere" to "CO2 stored" (sequestration)
    co2_dac_t = -n.links_t.p1[dac]*tres_factor 
    co2_dac_t_sum = -round(co2_dac_t.sum().sum()/1e6,sig_dig) # i.e., negative emissions

    df.loc['dac', label] = co2_dac_t_sum
    ###############################################################

    # CO2 balance
    co2_tot = df[label].sum().sum()
    df.loc['net co2 emissions', label] = round(co2_tot,sig_dig)

    return df

bus = "Europe"

unit_capacity_deficit = "GW" if bus == "Europe" else "% of peak load"

dyears = [1968, 2013]

strings = ["opt", "1.3", "1.5","1.7", "2.0"]

for dyear in dyears:

    directory = "results/sensitivity_transmission_dy1968/postnetworks/" + str(dyear) + "/"

    for str_i in strings:

        path = directory + "lv" + str(str_i) + "/"

        RDIR = path + "csvs"

        file1 = RDIR + '/emissions_full_horizon_dy' + str(dyear) + '.csv'
        file2  = RDIR + '/net_co2_emissions_full_horizon_dy' + str(dyear) + '.csv'

        As = {} # unserved energy 
        Bs = {} # time index
        Cs = {} # number of days with unsered energy in the considered event
        Ds = {} # concurrent demand (we could also pick peak load or average demand, 
                # depending on whether we are looking at capacity deficit or unserved energy)
        total_demands = {}
        capacity_deficits_t = {} #pd.DataFrame()
        unserved_energy_t = {} #pd.DataFrame()

        df = pd.DataFrame()

        network_names = glob.glob(path + "*.nc")
        no_networks = len(network_names)

        for network_name in network_names:
            n = pypsa.Network(network_name)

            try:
                n.objective 
            except: 
                continue # if no objective, it did not solve, and for this reason, we skip it.

            opts = network_name.split('/')[5].split('_')

            dyear = opts[3][2:] # design year
            wyear = opts[4][2:] # operational year
            
            print(wyear)                  
            if bus != "Europe":
                generators_hv = n.generators.query("bus == @bus")
                bus_lv = bus + " low voltage"
                generators_lv = n.generators.query("bus == @bus_lv")
                generators = pd.concat([generators_hv, generators_lv])

                hydro_units = n.storage_units.query("bus == @bus").query("carrier == 'hydro'")
                loads_index = n.loads.bus.str.contains(bus)[n.loads.bus.str.contains(bus)].index
                loads = n.loads.loc[loads_index]
                loads_t_p_set = n.loads_t.p_set[loads.query("p_set == 0").index]
                loads_t_p = n.loads_t.p[loads.query("p_set > 0").index]
                loads_t = loads_t_p_set.merge(loads_t_p, how='outer', left_index=True, right_index=True)
            else:
                generators = n.generators
                hydro_units = n.storage_units.query("carrier == 'hydro'")
                loads = n.loads
                loads_t = n.loads_t.p

            capacity_deficits_t_i = calculate_capacity_deficit(n,loads, loads_t, generators, bus, unit=unit_capacity_deficit)
            capacity_deficits_t[wyear] = capacity_deficits_t_i

            unserved_energy_t[wyear], As[wyear], Bs[wyear], Cs[wyear], Ds[wyear], total_demands[wyear] = calculate_unserved_energy(n,
                                                                                                                                    loads,
                                                                                                                                    loads_t,
                                                                                                                                    generators,
                                                                                                                                    bus,
                                                                                                                                    n_days_threshold=0,
                                                                                                                                    )

            df = calculate_emissions(n, wyear, df)

        net_co2_emissions = df.loc['net co2 emissions',:]
        df.drop(index=['net co2 emissions'], inplace=True)
        df_full_horizon = df.copy()
        net_co2_emissions_full_horizon = net_co2_emissions.copy()
        df_full_horizon.to_csv(file1)
        net_co2_emissions_full_horizon.to_csv(file2)

        capacity_deficits_t_df = pd.DataFrame(capacity_deficits_t)
        capacity_deficits_t_df.to_csv(RDIR + "/capacity_deficits_t_" + bus[0:2] + ".csv")

        unserved_energy_t_df = pd.DataFrame(unserved_energy_t)
        unserved_energy_t_df.to_csv(RDIR + "/unserved_energy_t_" + bus[0:2] + ".csv")