import pypsa
import xarray as xr

from helper import override_component_attrs

def update_heat(n,n_weather):
    heat_loads = n.loads_t.p_set.columns[n.loads_t.p_set.columns.str.contains('heat')]
    new_heat_p_set = n_weather.loads_t.p_set[heat_loads]

    weather_year = snakemake.wildcards.wyear

    filename_design_year = "../data/modeled_heat_demands/heat_demand_total_elec_wy2013_s370_37.nc"
    filename_new_year = "../data/modeled_heat_demands/heat_demand_total_elec_wy" + weather_year + "_s370_37.nc"
    modeled_heat_demand_design_year = xr.open_dataarray(filename_design_year, engine="netcdf4").to_pandas()
    modeled_heat_demand_new_year = xr.open_dataarray(filename_new_year, engine="netcdf4").to_pandas()
    interannual_variability_factor = modeled_heat_demand_design_year.sum().sum()/modeled_heat_demand_new_year.sum().sum()
            
    df_loads_p_set = n.loads_t.p_set
    df_loads_p_set.loc[df_loads_p_set.index,heat_loads] = new_heat_p_set*interannual_variability_factor
    
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
    
    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input.network, 
                        override_component_attrs=overrides)

    # Read network with heat demand from a different weather year
    weather_network = snakemake.input.weather_network 
    n_weather = pypsa.Network(weather_network, 
                                override_component_attrs=overrides)

    if 'heat' in snakemake.wildcards.opts:
        print('correcting heat demand')
        update_heat(n,n_weather)
    n.export_to_netcdf(snakemake.output.network)