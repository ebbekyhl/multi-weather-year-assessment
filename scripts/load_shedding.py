import pypsa
import pandas as pd
import matplotlib.pyplot as plt

from helper import override_component_attrs

def add_load_shedding(n):
    

    return n 

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from helper import mock_snakemake
        snakemake = mock_snakemake(
            'update_renewable_profiles',
            #wildcard='',
        )
    
    overrides = override_component_attrs(snakemake.input.overrides)
    
    n = pypsa.Network(snakemake.input.network, 
                        override_component_attrs=overrides)

    add_load_shedding(n)

    n.export_to_netcdf(snakemake.output.network)