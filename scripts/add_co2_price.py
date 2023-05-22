import pypsa
import pandas as pd
import matplotlib.pyplot as plt

from helper import override_component_attrs

def add_co2_price(n,n_design):
    # Adding CO2 price to positive emittors as an extra cost to produce an additional unit of energy
    # and
    # Adding CO2 price to negative emittors as an extra incentive to capture one unit more of CO2)

    # Lagrange multiplier for CO2 of design network
    co2_price = n_design.global_constraints.loc['CO2Limit']['mu'] # EUR/tCO2

    ########################################################################
    ######## Electricity, heating, hydrogen production, etc. ###############
    ########################################################################
    # Links with "p2" corresponding to the CO2 emissions: 
    bus2_emittors_ind = list(n.links.query('bus2 == "co2 atmosphere"').index)
    bus2_emittors = n.links.loc[bus2_emittors_ind]
    bus2_emittors = bus2_emittors.query('efficiency2 > 0') # omit every link with zero emissions, e.g., nuclear

    # Unit conversion
    # efficiency2 = tonne CO2 emitted per unit of inlet consumed (e.g., fuel burned. For gas, 0.2 tCO2/MWh_th) 
    mwh_inlet_per_tco2 = 1/(bus2_emittors.efficiency2) # converting "tCO2/MWh_inlet" to "MWh_inlet/tCO2"
    co2_eur_per_MWh_inlet = co2_price/mwh_inlet_per_tco2 # converting "EUR/tCO2" to "EUR/MWh_inlet" 

    # UPDATE MARGINAL PRICE BASED ON CO2 LAGRANGE MULTIPLIER
    new_marginal_price_bus2_emittors = bus2_emittors.marginal_cost + co2_eur_per_MWh_inlet
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
    mwh_th_per_tco2 = 1/(chp.efficiency3) # converting "tCO2/MWh_inlet" to "MWh_inlet/tCO2"
    co2_eur_per_MWh_th = co2_price/mwh_th_per_tco2 # converting "EUR/tCO2" to "EUR/MWh_inlet" (accounts for sign differences)

    # UPDATE MARGINAL PRICE BASED ON CO2 LAGRANGE MULTIPLIER
    new_marginal_price_CHP = chp.marginal_cost + co2_eur_per_MWh_th
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
    tco2_per_tco2 = 1/(process.efficiency) # in case of CC, efficiency < 1, otherwise 1.
    co2_eur_per_tco2 = co2_price/tco2_per_tco2

    # UPDATE MARGINAL PRICE BASED ON CO2 LAGRANGE MULTIPLIER
    new_marginal_price_process = process.marginal_cost + co2_eur_per_tco2
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
    tco2_per_tco2 = 1/(dac.efficiency) # in case of CC, efficiency < 1, otherwise 1.
    co2_eur_per_tco2 = co2_price/tco2_per_tco2

    # UPDATE MARGINAL PRICE BASED ON CO2 LAGRANGE MULTIPLIER
    new_marginal_price_dac = dac.marginal_cost - co2_eur_per_tco2 # co2 price is subtracted from the marginal cost!
    dac['marginal_cost'] = new_marginal_price_dac
    n.links.loc[dac.index] = dac
    ########################################################################

    ########################################################################
    #################### Remove global CO2 constraint ######################
    ########################################################################
    # Remove global CO2 constraint since it is being replaced by a CO2 price
    n.remove('GlobalConstraint',"CO2Limit")
    #n.global_constraints.loc['CO2Limit','constant'] = 1e10 # set co2 constraint very high (non-binding) when co2 price is set

    return n
    
if __name__ == "__main__":
    if 'snakemake' not in globals():
        from helper import mock_snakemake
        snakemake = mock_snakemake(
            'add_co2_price',
            design_year="2013", 
            weather_year="2008",
        )
    
    overrides = override_component_attrs(snakemake.input.overrides)
    
    n = pypsa.Network(snakemake.input.network, 
                        override_component_attrs=overrides)

    n_design = pypsa.Network(snakemake.input.design_network, 
                        override_component_attrs=overrides)

    if snakemake.config['scenario']['add_co2_price']:
        add_co2_price(n,n_design)

    n.export_to_netcdf(snakemake.output.network)