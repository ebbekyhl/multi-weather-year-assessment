configfile: "config.yaml"

tres = config['model']['tres']
nodes = config['model']['nodes']
co2 = config['model']['co2']
sectors = config['model']['sectors']
dyear = str(config['scenario']['design_year'][0])

RDIR = config['results_dir'] + config['run'] + '/'
PNDIR = config['results_dir'] + config['run'] + '/prenetworks/'
CNDIR = 'networks/networks_n' + nodes + '_' + tres + 'h/'


wildcard_constraints:
    design_year="[0-9]+m?",
    weather_year="[0-9]+m?",

rule all:
    input: RDIR + 'csvs/summary.csv'


rule copy_config:
    output: RDIR + 'configs/config.yaml'
    threads: 1
    resources: mem_mb=1000
    benchmark: RDIR + "benchmarks/copy_config"
    script: "scripts/copy_config.py"

rule copy_design_network:
    input:
        design_network = CNDIR + "elec_wy{design_year}_s370_" + nodes + "_lv1.0__Co2L" + co2 + "-" + tres +"H-" + sectors + "-solar+p3-dist1_2050.nc",
        overrides = "data/override_component_attrs",
    output: 
        network = PNDIR + "base_n" + nodes + "_" + tres + "h_dy{design_year}.nc",
    threads: 1
    resources: mem_mb=1000 
    script: 'scripts/copy_design_network.py'

rule update_renewable_profiles:
    input:
        design_network = PNDIR + "base_n" + nodes + "_" + tres + "h_dy{design_year}.nc",
        weather_network = CNDIR + "elec_wy{weather_year}_s370_" + nodes + "_lv1.0__Co2L" + co2 + "-" + tres + "H-" + sectors + "-solar+p3-dist1_2050.nc",
        overrides = "data/override_component_attrs",
    output: 
        network = PNDIR + "base_n" + nodes + "_" + tres + "h_renewables_dy{design_year}_wy{weather_year}.nc",
        plot_solar = "graphs/capacity_factors_solar_dy{design_year}_wy{weather_year}.pdf",
        plot_wind = "graphs/capacity_factors_wind_dy{design_year}_wy{weather_year}.pdf",
        plot_hydro = "graphs/inflow_hydro_dy{design_year}_wy{weather_year}.pdf",
    threads: 1
    resources: mem_mb=1000 
    script: 'scripts/update_renewable_profiles.py'
    

rule update_heat_demand:
    input:
        overrides = "data/override_component_attrs",
        network = PNDIR + "base_n" + nodes + "_" + tres + "h_renewables_dy{design_year}_wy{weather_year}.nc",
        weather_network = CNDIR + "elec_wy{weather_year}_s370_" + nodes + "_lv1.0__Co2L" + co2 + "-" + tres + "H-" + sectors + "-solar+p3-dist1_2050.nc",
    output: 
        network = PNDIR + "base_n" + nodes + "_" + tres + "h_renewables_dy{design_year}_wy{weather_year}_heat.nc",
        plot_heat = "graphs/heat_demand_dy{design_year}_wy{weather_year}.pdf",
    threads: 1
    resources: mem_mb=1000
    script: 'scripts/update_heat_demand.py'


rule freeze_capacities:
    input:
        overrides = "data/override_component_attrs",
        network = PNDIR + "base_n" + nodes + "_" + tres + "h_renewables_dy{design_year}_wy{weather_year}_heat.nc",
    output: 
        network = PNDIR + "base_n" + nodes + "_" + tres + "h_renewables_dy{design_year}_wy{weather_year}_heat_capacitylock.nc",
    threads: 1
    resources: mem_mb=1000 
    script: 'scripts/freeze_capacities.py'

    
rule add_co2_price:
    input:
        overrides = "data/override_component_attrs",
        network =  PNDIR + "base_n"+ nodes + "_" + tres +"h_renewables_dy{design_year}_wy{weather_year}_heat_capacitylock.nc"
    output: 
        network =  PNDIR + "base_n"+ nodes + "_" + tres +"h_renewables_dy{design_year}_wy{weather_year}_heat_capacitylock_co2price.nc"
    threads: 1
    resources: mem_mb=1000 
    script: 'scripts/add_co2_price.py'

# rule change_temporal_resolution: reoptimization should be carried out in (preferably) hourly res!
#
# # script that downsamples from x-hourly to 1-hourly
#
#

rule resolve_network:
    input:
        overrides = "data/override_component_attrs",
        config= RDIR + 'configs/config.yaml',
        network = PNDIR + "base_n"+ nodes + "_" + tres + "h_renewables_dy{design_year}_wy{weather_year}_heat_capacitylock_co2price.nc",
    output: 
        network = RDIR + "resolved_n"+ nodes + "_" + tres +"h_dy{design_year}_wy{weather_year}.nc"
    log:
        solver="logs/resolved_n"+ nodes + "_" + tres +"h_dy{design_year}_wy{weather_year}_solver.log",
        python="logs/resolved_n"+ nodes + "_" + tres +"h_dy{design_year}_wy{weather_year}_python.log",
        memory="logs/resolved_n"+ nodes + "_" + tres +"h_dy{design_year}_wy{weather_year}_memory.log"
    threads: 4
    resources: mem_mb=config['solving']['mem']
    benchmark: RDIR + "benchmarks/resolve_network/resolved_dy{design_year}_wy{weather_year}"
    script: "scripts/resolve_network.py"


rule make_summary:
    input:
        overrides = "data/override_component_attrs",
        design_network = PNDIR + "base_n" + nodes + "_" + tres + "h_dy" + dyear + ".nc", 
        networks=expand(
            RDIR + "resolved_n"+ nodes + "_" + tres +"h_dy{design_year}_wy{weather_year}.nc",
            **config['scenario']
        ),
        #network = RDIR + "resolved_dy{design_year}_wy{weather_year}.nc"
    output: 
        summary = RDIR + "csvs/summary.csv"
    threads: 1
    resources: mem_mb=5000 # check jobinfo to see how much memory is used
    script: 'scripts/make_summary.py'


# rule plot_summary:
#     input:

#     output:

#     threads: 1
#     resources: mem_mb=1000 # check jobinfo to see how much memory is used
#     script: 'scripts/plotting.py'



# rule dag:
#     message:
#         "Creating DAG of workflow."
#     output:
#         png="dag.png",
#     shell:
#         """
#         snakemake --rulegraph all | dot -Tpng > dag.png
#         """
