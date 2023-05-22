import pypsa
import pandas as pd
import matplotlib.pyplot as plt

from helper import override_component_attrs

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from helper import mock_snakemake
        snakemake = mock_snakemake(
            'update_renewable_profiles',
            design_year="2013", 
            weather_year="2008",
        )
    
    overrides = override_component_attrs(snakemake.input.overrides)

    # Read design network
    design_network = snakemake.input.design_network 
    n = pypsa.Network(design_network,
                        override_component_attrs=overrides)

    # Save network
    n.export_to_netcdf(snakemake.output.network)