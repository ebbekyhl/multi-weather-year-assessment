configfile: "config.yaml"

tres = config['model']['tres'] # temporal resolution of capacity-optimized networks
hour = 'h' #'h' or 'H'
nodes = config['model']['nodes'] # number of nodes in the network
co2 = config['model']['co2'] # co2 cap
sectors = config['model']['sectors'] # sectors included
dyear = str(config['scenario']['design_year'][0]) # design year

RDIR = config['results_dir'] + config['run'] + '/' # results directory
PNDIR = config['results_dir'] + config['run'] + '/prenetworks/' # pre-networks directory
CNDIR = 'networks/networks_n' + nodes + '_' + tres + 'h/' # capacity-optimized networks directory
        

wildcard_constraints:
    design_year="[0-9]+m?",
    weather_year="[0-9]+m?",
    opts="[-+a-zA-Z0-9\.\s]*"

rule all:
    input: RDIR + "graphs/energy.pdf"

    
rule resolve_all_networks:
    input:
        expand(RDIR + "postnetworks/resolved_n"+ nodes + "_" + tres +"h_dy{design_year}_wy{weather_year}_{opts}.nc",
               **config['scenario'])


rule copy_config:
    output: RDIR + 'configs/config.yaml'
    threads: 1
    resources: mem_mb=1000
    benchmark: RDIR + "benchmarks/copy_config"
    script: "scripts/copy_config.py"

rule copy_design_network:
    input:
        design_network = CNDIR + "elec_wy{design_year}_s370_" + nodes + "_lv1.0__Co2L" + co2 + "-" + tres + hour + "-" + sectors + "-solar+p3-dist1_2050.nc",
        overrides = "data/override_component_attrs",
    output: 
        network = PNDIR + "base_n" + nodes + "_" + tres + "h_dy{design_year}.nc",
    threads: 1
    resources: mem_mb=10000 
    script: 'scripts/copy_design_network.py'

rule update_renewable_profiles:
    input:
        design_network = PNDIR + "base_n" + nodes + "_" + tres + "h_dy{design_year}.nc",
        weather_network = CNDIR + "elec_wy{weather_year}_s370_" + nodes + "_lv1.0__Co2L" + co2 + "-" + tres + hour + "-" + sectors + "-solar+p3-dist1_2050.nc",
        overrides = "data/override_component_attrs",
    output: 
        network = PNDIR + "base_n" + nodes + "_" + tres + "h_renewables_dy{design_year}_wy{weather_year}_{opts}.nc",
    threads: 1
    resources: mem_mb=10000 
    script: 'scripts/update_renewable_profiles.py'
    

rule update_heat_demand:
    input:
        overrides = "data/override_component_attrs",
        network = PNDIR + "base_n" + nodes + "_" + tres + "h_renewables_dy{design_year}_wy{weather_year}_{opts}.nc",
        weather_network = CNDIR + "elec_wy{weather_year}_s370_" + nodes + "_lv1.0__Co2L" + co2 + "-" + tres + hour + "-" + sectors + "-solar+p3-dist1_2050.nc",
    output: 
        network = PNDIR + "base_n" + nodes + "_" + tres + "h_renewables_dy{design_year}_wy{weather_year}_{opts}_heat.nc",
    threads: 1
    resources: mem_mb=10000
    script: 'scripts/update_heat_demand.py'


rule plot_updated_renewable_profiles:
    input:
        overrides = "data/override_component_attrs",
        design_network = PNDIR + "base_n" + nodes + "_" + tres + "h_dy" + dyear + ".nc", 
        networks=expand(
                        PNDIR + "base_n" + nodes + "_" + tres + "h_renewables_dy{design_year}_wy{weather_year}_{opts}_heat.nc",
                        **config['scenario']
                        ),
    output: 
        plot_solar = RDIR + "graphs/capacity_factors_solar.pdf",
        plot_wind = RDIR + "graphs/capacity_factors_wind.pdf",
        plot_hydro = RDIR + "graphs/inflow_hydro.pdf",
        plot_heat = RDIR + "graphs/heat_load.pdf",
    threads: 1
    resources: mem_mb=10000 # check jobinfo to see how much memory is used
    script: 'scripts/plot_updates.py'


rule freeze_capacities:
    input:
        overrides = "data/override_component_attrs",
        network = PNDIR + "base_n" + nodes + "_" + tres + "h_renewables_dy{design_year}_wy{weather_year}_{opts}_heat.nc",
    output: 
        network = PNDIR + "base_n" + nodes + "_" + tres + "h_renewables_dy{design_year}_wy{weather_year}_{opts}_heat_capacitylock.nc",
    threads: 1
    resources: mem_mb=10000 
    script: 'scripts/freeze_capacities.py'

    
rule add_co2_price:
    input:
        overrides = "data/override_component_attrs",
        design_network = PNDIR + "base_n" + nodes + "_" + tres + "h_dy{design_year}.nc",
        network =  PNDIR + "base_n"+ nodes + "_" + tres +"h_renewables_dy{design_year}_wy{weather_year}_{opts}_heat_capacitylock.nc"
    output: 
        network =  PNDIR + "base_n"+ nodes + "_" + tres +"h_renewables_dy{design_year}_wy{weather_year}_{opts}_heat_capacitylock_co2price.nc"
    threads: 1
    resources: mem_mb=10000 
    script: 'scripts/add_co2_price.py'


rule split_horizon:
    input:
        overrides = "data/override_component_attrs",
        network =  PNDIR + "base_n"+ nodes + "_" + tres + "h_renewables_dy{design_year}_wy{weather_year}_{opts}_heat_capacitylock_co2price.nc"
    output: 
        network =  PNDIR + "base_n"+ nodes + "_" + tres + "h_renewables_dy{design_year}_wy{weather_year}_{opts}_heat_capacitylock_co2price_split.nc"
    threads: 1
    resources: mem_mb=10000 
    script: 'scripts/split_horizon.py'


rule resolve_network:
    input:
        overrides = "data/override_component_attrs",
        plot_hydro = RDIR + "graphs/inflow_hydro.pdf",
        config= RDIR + 'configs/config.yaml',
        network = PNDIR + "base_n"+ nodes + "_" + tres + "h_renewables_dy{design_year}_wy{weather_year}_{opts}_heat_capacitylock_co2price_split.nc"
    output: 
        network = RDIR + "postnetworks/resolved_n"+ nodes + "_" + tres +"h_dy{design_year}_wy{weather_year}_{opts}.nc"
    log:
        solver="logs/resolved_n"+ nodes + "_" + tres +"h_dy{design_year}_wy{weather_year}_{opts}_solver.log",
        python="logs/resolved_n"+ nodes + "_" + tres +"h_dy{design_year}_wy{weather_year}_{opts}_python.log",
        memory="logs/resolved_n"+ nodes + "_" + tres +"h_dy{design_year}_wy{weather_year}_{opts}_memory.log"
    threads: 4
    resources: mem_mb=config['solving']['mem']
    benchmark: RDIR + "benchmarks/resolve_network/resolved_dy{design_year}_wy{weather_year}_{opts}"
    script: "scripts/resolve_network.py"


rule make_summary:
    input:
        overrides = "data/override_component_attrs",
        design_network = PNDIR + "base_n" + nodes + "_" + tres + "h_dy" + dyear + ".nc", 
        networks=expand(
                        RDIR + "postnetworks/resolved_n"+ nodes + "_" + tres +"h_dy{design_year}_wy{weather_year}_{opts}.nc",
                        **config['scenario']
                        ),
    output: 
        summary = RDIR + "csvs/summary.csv",
        lost_load_plot = RDIR + "graphs/lost_load_duration_curves.pdf",
        load_shedding_heatmap_time = RDIR + "graphs/load_shedding_heatmap_time.pdf",
        load_shedding_heatmap_space = RDIR + "graphs/load_shedding_heatmap_space.pdf",
    threads: 1
    resources: mem_mb=10000 # check jobinfo to see how much memory is used
    script: 'scripts/make_summary.py'


rule plot_summary:
    input:
        summary_csv = RDIR + "csvs/summary.csv",
    output:
        energy = RDIR + "graphs/energy.pdf",
        lost_load = RDIR + "graphs/lost_load.pdf",
        co2_balance = RDIR + "graphs/co2_balance.pdf",
    threads: 1
    resources: mem_mb=1000 # check jobinfo to see how much memory is used
    script: 'scripts/plot_summary.py'


# rule dag:
#     message:
#         "Creating DAG of workflow."
#     output:
#         png="dag.png",
#     shell:
#         """
#         snakemake --rulegraph all | dot -Tpng > dag.png
#         """
