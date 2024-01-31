import pypsa
import xarray as xr

from helper import override_component_attrs

def update_heat(n,n_weather,weather_year):
    heat_loads = n.loads_t.p_set.columns[n.loads_t.p_set.columns.str.contains('heat')]
    new_heat_p_set = n_weather.loads_t.p_set[heat_loads]
            
    df_loads_p_set = n.loads_t.p_set
    df_loads_p_set.loc[df_loads_p_set.index,heat_loads] = new_heat_p_set
    
    n.loads_t.p_set = df_loads_p_set

    return n 

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from helper import mock_snakemake
        snakemake = mock_snakemake(
            'solve_network',
            design_year="2013", 
            weather_year="2008",
        )
    
    weather_year = snakemake.wildcards.weather_year

    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input.network, 
                        override_component_attrs=overrides)

    # Read network with heat demand from a different weather year
    weather_network = snakemake.input.weather_network 
    n_weather = pypsa.Network(weather_network, 
                                override_component_attrs=overrides)

    if 'heat' in snakemake.wildcards.opts:
        print('correcting heat demand')
        update_heat(n,n_weather,weather_year)
    n.export_to_netcdf(snakemake.output.network)