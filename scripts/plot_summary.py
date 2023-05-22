import pypsa
import pandas as pd
import matplotlib.pyplot as plt

from helper import override_component_attrs

def plot(n):
    

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from helper import mock_snakemake
        snakemake = mock_snakemake(
            'make_summary',
            design_year="2013", 
            weather_year="2008",
        )
    
    overrides = override_component_attrs(snakemake.input.overrides)

    networks_dict = {
    (design_year, weather_year): snakemake.config[
        "results_dir"
    ]
    + snakemake.config["run"]
    + f"/resolved_dy{design_year}_wy{weather_year}.nc"
    for design_year in snakemake.config["scenario"]["design_year"]
    for weather_year in snakemake.config["scenario"]["weather_year"]
    }

    df = plot(networks_dict)

    # savefig