import pypsa
import pandas as pd
import matplotlib.pyplot as plt

from helper import override_component_attrs

def add_co2_price(n):
    # CO2 price of design network
    co2_price = n.global_constraints.loc['CO2Limit']['mu']

    # CHP plants with emissions or carbon capture:
    chp_ind = list(n.links.query('bus3 == "co2 atmosphere"').index)
    chp = n.links.loc[chp_ind]

    # Other links with direct CO2 emissions:
    direct_emittors_ind = list(n.links.query('bus2 == "co2 atmosphere"').index)
    direct_emittors = n.links.loc[direct_emittors_ind]
    direct_emittors = direct_emittors.query('efficiency2 > 0')

    # Unit conversion CHP
    mwh_el_per_tco2 = 1/(chp.efficiency3/chp.efficiency)
    mwh_th_per_tco2 = 1/(chp.efficiency3)
    #co2_eur_per_MWh_el = co2_price/mwh_el_per_tco2
    co2_eur_per_MWh_th = co2_price/mwh_th_per_tco2

    # Unit conversion direct emittors
    mwh_exit_per_tco2 = 1/(direct_emittors.efficiency2/direct_emittors.efficiency)
    mwh_inlet_per_tco2 = 1/(direct_emittors.efficiency2)
    co2_eur_per_MWh_exit = co2_price/mwh_exit_per_tco2
    co2_eur_per_MWh_inlet = co2_price/mwh_inlet_per_tco2

    # New marginal price including CO2 price
    new_marginal_price_CHP = chp.marginal_cost + co2_eur_per_MWh_th
    chp['marginal_cost'] = new_marginal_price_CHP
    n.links.loc[chp.index] = chp

    new_marginal_price_direct_emittors = direct_emittors.marginal_cost + co2_eur_per_MWh_inlet
    direct_emittors['marginal_cost'] = new_marginal_price_direct_emittors
    n.links.loc[direct_emittors.index] = direct_emittors

    if snakemake.config['add_co2_price']:
        n.global_constraints.loc['CO2Limit','constant'] = 1e10 # set co2 constraint very high (non-binding) when co2 price is set

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

    if snakemake.config['scenario']['add_co2_price']:
        add_co2_price(n)

    n.export_to_netcdf(snakemake.output.network)


