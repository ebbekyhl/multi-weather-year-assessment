import pypsa
import pandas as pd
import matplotlib.pyplot as plt

from helper import override_component_attrs

def freeze_network(n):
    # generators
    df = n.generators
    df_real = n.generators.query('capital_cost > 0') # "Real" technologies
    df.loc[df_real.index,"p_nom"] = df_real.p_nom_opt.values
    df.loc[df_real.index,'p_nom_extendable'] = snakemake.config['scenario']['generators_extendable']
    n.generators = df
    print(n.generators.p_nom_extendable[0:5])

    # lines
    n.lines.s_nom_extendable = False
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

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from helper import mock_snakemake
        snakemake = mock_snakemake(
            'solve_network',
            design_year="2013", 
            weather_year="2008",
            )
    
    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input.network, 
                        override_component_attrs=overrides)

    freeze_network(n)

    add_load_shedding(n)

    n.export_to_netcdf(snakemake.output.network)