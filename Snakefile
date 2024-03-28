configfile: "config.yaml"

tres = config['model']['tres'] # temporal resolution of capacity-optimized networks
hour = 'h' #'h' or 'H'
nodes = config['model']['nodes'] # number of nodes in the network
co2 = config['model']['co2'] # co2 cap
sectors = config['model']['sectors'] # sectors included
dyear = str(config['scenario']['design_year'][0]) # design year

RDIR = config['results_dir'] + config['run'] + '/' # results directory
PNDIR = config['results_dir'] + config['run'] + '/prenetworks/' # pre-networks directory
WNDIR = 'networks/networks_n37_3h/' # weather networks directory
CNDIR = 'networks/sensitivity_transmission/' # capacity-optimized networks directory

wildcard_constraints:
    design_year="[0-9]+m?",
    weather_year="[0-9]+m?",
    opts="[-+a-zA-Z0-9\.\s]*"

rule all:
    input: RDIR + "graphs/energy.pdf"

    
rule resolve_all_networks:
    input:
        expand(RDIR + "postnetworks/resolved_n"+ nodes + "_" + tres +"h_dy{design_year}_wy{weather_year}_{opts}_lv{lvl}.nc",
               **config['scenario'])


rule copy_config:
    output: RDIR + 'configs/config.yaml'
    threads: 1
    resources: mem_mb=1000
    benchmark: RDIR + "benchmarks/copy_config"
    script: "scripts/copy_config.py"

rule update_network:
    input:
        design_network = CNDIR + "elec_wy{design_year}_s370_" + nodes + "_lv{lvl}__Co2L" + co2 + "-" + tres + hour + "-" + sectors + "-solar+p3-dist1_2050.nc",
        weather_network = WNDIR + "elec_wy{weather_year}_s370_" + nodes + "_lv1.0__Co2L" + co2 + "-" + tres + hour + "-" + sectors + "-solar+p3-dist1_2050.nc",
        overrides = "data/override_component_attrs",
    output: 
        network = PNDIR + "base_n" + nodes + "_" + tres + "h_updated_dy{design_year}_wy{weather_year}_{opts}_lv{lvl}.nc",
    threads: 1
    resources: mem_mb=10000 
    script: 'scripts/update_network.py'

rule resolve_network:
    input:
        overrides = "data/override_component_attrs",
        config= RDIR + 'configs/config.yaml',
        network = PNDIR + "base_n" + nodes + "_" + tres + "h_updated_dy{design_year}_wy{weather_year}_{opts}_lv{lvl}.nc",
    output: 
        network = RDIR + "postnetworks/resolved_n"+ nodes + "_" + tres +"h_dy{design_year}_wy{weather_year}_{opts}_lv{lvl}.nc"
    log:
        solver="logs/resolved_n"+ nodes + "_" + tres +"h_dy{design_year}_wy{weather_year}_{opts}_lv{lvl}_solver.log",
        python="logs/resolved_n"+ nodes + "_" + tres +"h_dy{design_year}_wy{weather_year}_{opts}_lv{lvl}_python.log",
        memory="logs/resolved_n"+ nodes + "_" + tres +"h_dy{design_year}_wy{weather_year}_{opts}_lv{lvl}_memory.log"
    threads: 4
    resources: mem_mb=config['solving']['mem']
    benchmark: RDIR + "benchmarks/resolve_network/resolved_dy{design_year}_wy{weather_year}_{opts}_lv{lvl}"
    script: "scripts/resolve_network.py"


rule make_summary:
    input:
        overrides = "data/override_component_attrs",
        networks=expand(
                        RDIR + "postnetworks/resolved_n"+ nodes + "_" + tres +"h_dy{design_year}_wy{weather_year}_{opts}_lv{lvl}.nc",
                        **config['scenario']
                        ),
    output: 
        summary = RDIR + "csvs/summary.csv",
        #lost_load_plot = RDIR + "graphs/lost_load_duration_curves.pdf",
        #load_shedding_heatmap_time = RDIR + "graphs/load_shedding_heatmap_time.pdf",
        #load_shedding_heatmap_space = RDIR + "graphs/load_shedding_heatmap_space.pdf",
    threads: 1
    resources: mem_mb=40000 # check jobinfo to see how much memory is used
    script: 'scripts/make_summary.py'


rule plot_summary:
    input:
        summary_csv = RDIR + "csvs/summary.csv",
    output:
        energy = RDIR + "graphs/energy.pdf",
        # lost_load = RDIR + "graphs/lost_load. pdf",
        # co2_balance = RDIR + "graphs/co2_balance.pdf",
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
