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

    n.export_to_netcdf(snakemake.output.network)