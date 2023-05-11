configfile: "config.yaml"

RDIR = config['results_dir'] + config['run']
CNDIR = config['cap_network_dir']
PNDIR = config['pre_network_dir']

wildcard_constraints:
    design_year="[0-9]+m?",
    weather_year="[0-9]+m?",

rule all:
    input:
        expand(RDIR + "resolved_dy{design_year}_wy{weather_year}.nc")


rule update_renewable_profiles:
    input:
        design_network = CNDIR + "elec_wy{design_year}_37_Co2L0-168H.nc",
        weather_network = CNDIR + "elec_wy{weather_year}_37_Co2L0-168H.nc",
        overrides = "data/override_component_attrs",
    output: 
        network = PNDIR + "base_renewables_dy{design_year}_wy{weather_year}.nc",
        plot_solar = "pregraphs/capacity_factors_solar_dy{design_year}_wy{weather_year}.pdf",
        plot_wind = "pregraphs/capacity_factors_wind_dy{design_year}_wy{weather_year}.pdf",
        plot_hydro = "pregraphs/inflow_hydro_dy{design_year}_wy{weather_year}.pdf",
    threads: 1
    resources: mem_mb=1000 # check jobinfo to see how much memory is used
    script: 'scripts/update_renewable_profiles.py'
    

rule update_heat_demand:
    input:
        overrides = "data/override_component_attrs",
        network = PNDIR + "base_renewables_dy{design_year}_wy{weather_year}.nc",
        weather_network = CNDIR + "elec_wy{weather_year}_37_Co2L0-168H.nc",
    output: 
        network = PNDIR + "base_renewables_dy{design_year}_wy{weather_year}_heat.nc",
        plot_heat = "pregraphs/heat_demand_dy{design_year}_wy{weather_year}.pdf",
    threads: 1
    resources: mem_mb=1000 # check jobinfo to see how much memory is used
    script: 'scripts/update_heat_demand.py'


rule freeze_capacities:
    input:
        overrides = "data/override_component_attrs",
        network = PNDIR + "base_renewables_dy{design_year}_wy{weather_year}_heat.nc",
    output: 
        network = PNDIR + "base_renewables_dy{design_year}_wy{weather_year}_heat_capacitylock.nc",
    threads: 1
    resources: mem_mb=1000 # check jobinfo to see how much memory is used
    script: 'scripts/freeze_capacities.py'

    
# rule add_co2_prices:
#     input:
#         overrides = "data/override_component_attrs",
#         network = "prenetworks/base_renewables_heat_capacitylock.nc"
#     output: 
#         network = "prenetworks/base_renewables_heat_capacitylock_co2price.nc"
#     threads: 1
#     resources: mem_mb=10000 # check jobinfo to see how much memory is used
#     script: 'scripts/add_co2_prices.py'

# rule change_temporal_resolution:
#
#
#
#

rule add_load_shedding:
    input:
        overrides = "data/override_component_attrs",
        network = PNDIR + "base_renewables_dy{design_year}_wy{weather_year}_heat_capacitylock.nc",
    output: 
        network = PNDIR + "base_renewables_dy{design_year}_wy{weather_year}_heat_capacitylock_ls.nc",
    threads: 1
    resources: mem_mb=10000 # check jobinfo to see how much memory is used
    script: 'scripts/add_load_shedding.py'


rule resolve_network:
    input:
        overrides = "data/override_component_attrs",
        network = PNDIR + "base_renewables_dy{design_year}_wy{weather_year}_heat_capacitylock_ls.nc"
    output: 
        network = RDIR + "resolved_dy{design_year}_wy{weather_year}.nc"
    log:
        solver="logs/resolved_dy{design_year}_wy{weather_year}_solver.log",
        python="logs/resolved_dy{design_year}_wy{weather_year}_python.log",
        memory="logs/resolved_dy{design_year}_wy{weather_year}_memory.log"
    threads: 4
    resources: 100000
    benchmark: RDIR + "/benchmarks/resolve_network/resolved_dy{design_year}_wy{weather_year}"
    script: "scripts/resolve_network.py"


# rule make_summary:
#     input:
#         overrides = "data/override_component_attrs",
#         network = "networks_resolved/dispatch_optimized.nc"
#     output: 
#         sspace = "summary.csv"
#     threads: 1
#     resources: mem_mb=10000 # check jobinfo to see how much memory is used
#     script: 'scripts/make_summary.py'


# rule dag:
#     message:
#         "Creating DAG of workflow."
#     output:
#         png="dag.png",
#     shell:
#         """
#         snakemake --rulegraph all | dot -Tpng > dag.png
#         """
