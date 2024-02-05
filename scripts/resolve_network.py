import pypsa
import os 

import numpy as np
import pandas as pd

from pypsa.linopt import get_var, linexpr, define_constraints, join_exprs

from pypsa.linopf import network_lopf, ilopf

from vresutils.benchmark import memory_logger

from helper import override_component_attrs, update_config_with_sector_opts

import logging
logger = logging.getLogger(__name__)
pypsa.pf.logger.setLevel(logging.WARNING)

def reset_temp_information(n):
    n.generators_t.p[n.generators_t.p.columns] = 0
    n.links_t.p0[n.links_t.p0.columns] = 0
    n.links_t.p1[n.links_t.p1.columns] = 0
    n.links_t.p2[n.links_t.p2.columns] = 0
    n.links_t.p3[n.links_t.p3.columns] = 0
    n.links_t.p4[n.links_t.p4.columns] = 0
    n.stores_t.e[n.stores_t.e.columns] = 0
    n.stores_t.p[n.stores_t.p.columns] = 0
    n.stores_t.q[n.stores_t.q.columns] = 0
    n.storage_units_t.p[n.storage_units_t.p.columns] = 0

    return n

def prepare_network(n, solve_opts=None):

    if 'clip_p_max_pu' in solve_opts:
        for df in (n.generators_t.p_max_pu, n.generators_t.p_min_pu, n.storage_units_t.inflow):
            df.where(df>solve_opts['clip_p_max_pu'], other=0., inplace=True)

    if solve_opts.get('noisy_costs'):
        for t in n.iterate_components():
            if 'marginal_cost' in t.df:
                np.random.seed(174)
                t.df['marginal_cost'] += 1e-2 + 2e-3 * (np.random.random(len(t.df)) - 0.5)

        for t in n.iterate_components(['Line', 'Link']):
            np.random.seed(123)
            t.df['capital_cost'] += (1e-1 + 2e-2 * (np.random.random(len(t.df)) - 0.5)) * t.df['length']

    if solve_opts.get('nhours'):
        nhours = solve_opts['nhours']
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = 8760./nhours

    return n

def solve_network(n, cf_solving, solver_options, **kwargs):
    
    solver_name = solver_options['name']

    track_iterations = cf_solving.get('track_iterations', False)
    min_iterations = cf_solving.get('min_iterations', 4)
    max_iterations = cf_solving.get('max_iterations', 6)
    keep_shadowprices = cf_solving.get('keep_shadowprices', True)

    if cf_solving.get('skip_iterations', False):
        network_lopf(n, solver_name=solver_name, solver_options=solver_options,
                     extra_functionality=extra_functionality,
                     keep_shadowprices=keep_shadowprices, **kwargs)
    else:
        ilopf(n, solver_name=solver_name, solver_options=solver_options,
              track_iterations=track_iterations,
              min_iterations=min_iterations,
              max_iterations=max_iterations,
              extra_functionality=extra_functionality,
              keep_shadowprices=keep_shadowprices,
              **kwargs)
    return n

def read_dispatch(country):
    df = pd.DataFrame(index=pd.date_range('1/1/2013','1/1/2014',freq='h',inclusive='left'))
    
    df1 = pd.DataFrame(historical_dispatch)
    years_to_drop = df1.index.year.value_counts()[df1.index.year.value_counts() < 8700].index
    index_to_drop = [df1.index[df1.index.year == years_to_drop[i]] for i in range(len(years_to_drop))]
    for i in range(len(index_to_drop)):
        df1 = df1.drop(index_to_drop[i])
    df1['year'] = df1.index.year
    for year in df1.index.year.unique():
        df1_year = df1.query('year == @year')
        df1_year = df1_year[~df1_year.index.duplicated(keep='first')]

        year_datetime = pd.date_range(start = str(df1_year.index[0])[0:11], 
                                      end = str(int(str(df1_year.index[0])[0:4])+1) + str(df1_year.index[0])[4:11],
                                      freq='h',tz='UTC')[0:-1]

        missing_dates = year_datetime.difference(df1_year.index)
        add_data = pd.DataFrame(columns=['Hydro Water Reservoir','year'],index=missing_dates)
        add_data['year'] = missing_dates.year.values
        df1_year = pd.concat([df1_year,add_data]).sort_index()
        df1_year.interpolate().loc[missing_dates]

        if (year % 4 == 0) and (year % 100 != 0): # leap year    
            day_29 = df1_year.index.day == 29
            feb_29 = df1_year[day_29][df1_year[day_29].index.month == 2]
            df1_year = df1_year.drop(index=feb_29.index)

        df[year] = df1_year.T.iloc[0].values
        
    return df

def read_hydro_soc(country, historical_soc):
    df_soc = pd.DataFrame(index=pd.date_range('1/1/2015','1/1/2016',freq='w',inclusive='left'))
    df1 = pd.DataFrame(historical_soc[country])

    # dropping years that do not have sufficient data reported
    minimum_no_of_weeks = 40
    years_to_drop = df1.index.year.value_counts()[df1.index.year.value_counts() < minimum_no_of_weeks].index
    index_to_drop = [df1.index[df1.index.year == years_to_drop[i]] for i in range(len(years_to_drop))] 
    for i in range(len(index_to_drop)):
        df1 = df1.drop(index_to_drop[i])
    
    # indexing by year
    df1['year'] = df1.index.year
    for year in df1.index.year.unique():
        print(year)
        df1_year = df1.query('year == @year')
        df1_year = df1_year[~df1_year.index.duplicated(keep='first')]
        df_soc[year] = df1_year.T.iloc[0].values[0:52]
    df_soc = df_soc.interpolate() # impute missing values using linear interpolation

    return df_soc

def add_hydro_constraint_soc(n):
    tres_factor = 8760/len(n.snapshots)
    # enforce model to reduce its deviation from the historical soc median
    # Read historical data
    # (Path must be changed to an external repository)
    historical_dispatch = pd.read_csv('data/hydro/hydro_res_dispatch_' + country + '_ENTSOE.csv',index_col=0)
    historical_dispatch.index = pd.to_datetime(historical_dispatch.index,utc=True)
    historical_dispatch = historical_dispatch.resample('h').sum()

    historical_soc = pd.read_csv('data/hydro/hydro_res_soc_ENTSOE.csv',index_col=0)
    historical_soc.index = pd.to_datetime(historical_soc.index,utc=True)
    
    constraint_countries = historical_soc.columns
    
    ######################## Calculate median SOC ############################
    soc_median = {}
    p_median = {}
    for c in constraint_countries:

        df_soc = read_hydro_soc(c, historical_soc) # historical state-of-charge (filling level) of hydropower reservoirs [MWh]

        p = read_dispatch(c) # historical dispatch from hydropower reservoires acc. to ENTSO-E [MWh]

        df_median = df_soc.median(axis=1) # calculate the median SOC
        df_median_added = pd.Series(df_median.values[0],index=[df_median.index[-1] + pd.Timedelta('1w')])
        df_median_collect = pd.concat([df_median,df_median_added])
        df_median_collect = df_median_collect.groupby(df_median_collect.index.week).sum()

        df_min = df_soc.min(axis=1) # calculate the min SOC
        df_min_added = pd.Series(df_min.values[0],index=[df_min.index[-1] + pd.Timedelta('1w')])
        df_min_collect = pd.concat([df_min,df_min_added])
        df_min_collect = df_min_collect.groupby(df_min_collect.index.week).sum()

        df_max = df_soc.max(axis=1) # calculate the max SOC
        df_max_added = pd.Series(df_max.values[0],index=[df_max.index[-1] + pd.Timedelta('1w')])
        df_max_collect = pd.concat([df_max,df_max_added])
        df_max_collect = df_max_collect.groupby(df_max_collect.index.week).sum()
        
        soc_median[c] = df_median_collect # median SOC for country c
        soc_max[c] = df_max_collect # max SOC for country c
        soc_min[c] = df_min_collect # min SOC for country c
        p_median[c] = p.median(axis=1) # median dispatch for country c

    # hydropower in PyPSA-Eur
    hydro = n.storage_units.query('carrier == "hydro"')
    hydro_units = hydro.index
    # find countries with hydropower that have data available from ENTSO-E (otherwise, we cannot impose a constraint)
    hydro_units_constrained = []
    for c in constraint_countries:
        hydro_units_constrained.append(list(hydro_units[hydro_units.str.contains(c)]))
    hydro_units_constrained = [item for sublist in hydro_units_constrained for item in sublist]
    
    # loop over bus "i" containing hydropower with historical data available
    for i in hydro_units_constrained:
        country = i[0:2]

        # total hydropower capacity in country
        country_capacity = hydro.loc[hydro.index[hydro.index.str.contains(country)]].p_nom.sum()
        
        # if countries have multiple buses, we need to split the historical data (which is aggregated by country) 
        # by hydropower bus. We do this by weighting according to the nominal hydropower capacity.
        nodal_share = n.storage_units.loc[i].p_nom/country_capacity
        
        min_series = nodal_share*soc_min[country] # capacity-weighted min SOC
        median_series = nodal_share*soc_median[country] # capacity-weighted median SOC
        max_series = nodal_share*soc_max[country] # capacity-weighted max SOC 

        ######################## SCALE CONSTRAINT ACC. TO INFLOW ############################
        inflow = n.storage_units_t.inflow[i]

        A = inflow # inflow in PyPSA-Eur network
        B = nodal_share*p_median # capacity-weighted hydropower production from ENTSO-E

        # scale SOC constraint
        scaling = A.sum()/B.sum()
        # median_series *= scaling 

        max_series *= scaling

        # if the available inflow in the PyPSA-Eur network is lower than the historical level, 
        # we need to relax the requirement for the minimum SOC level:
        if B.sum() > A.sum(): 
            min_series *= scaling

        minimum = list(min_series)
        # median = list(median_series)
        maximum = list(max_series)
    
        ########################## DEFINE LEFT HAND SIDE ###################################
        # BELOW LINES SHOULD BE UPDATED ACCORDING TO PYPSA VERSION!
        soc = get_var(n, 'StorageUnit', 'state_of_charge')
        lhs_var = soc[i]
        lhs = linexpr((1, lhs_var)).groupby(soc.index.week).apply(join_exprs)

        ############################ DEFINE LOWER LIMIT ###################################
        # RHS 1
        lower = pd.DataFrame(minimum, index=np.arange(1,13), columns=['hydro_soc']) 
        rhs1 = lower['hydro_soc']*(168/tres_factor) # since "lhs" is the sum of SOC for one week, we scale the rhs as well
        define_constraints(n, lhs, '>', rhs1, 'StorageUnit', 'hydro soc lower limit constraint ' + i)
        
        ########################### DEFINE UPPER LIMIT ####################################
        # RHS 2
        upper = pd.DataFrame(maximum, index=np.arange(1,13), columns=['hydro_soc'])
        rhs2 = upper['hydro_soc']*(168/tres_factor) # since "lhs" is the sum of SOC for one week, we scale the rhs as well
        define_constraints(n, lhs, '<', rhs2, 'StorageUnit', 'hydro soc upper limit constraint ' + i)

def extra_functionality(n, snapshots):
    if snakemake.config['scenario']["hydroconstrained"]:
        add_hydropower_constraint_soc(n)
        
if __name__ == "__main__":
    if 'snakemake' not in globals():
        from helper import mock_snakemake
        snakemake = mock_snakemake(
            'solve_network',
            design_year="2013", 
            weather_year="2008",
        )

    logging.basicConfig(filename=snakemake.log.python,
                        level=snakemake.config['logging_level'])

    # update_config_with_sector_opts(snakemake.config, snakemake.wildcards.sector_opts)

    tmpdir = '/scratch/' + os.environ['SLURM_JOB_ID']
    if tmpdir is not None:
        from pathlib import Path
        Path(tmpdir).mkdir(parents=True, exist_ok=True)

    fn = getattr(snakemake.log, 'memory', None)
    with memory_logger(filename=fn, interval=30.) as mem:

        overrides = override_component_attrs(snakemake.input.overrides)
        
        n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)

        reset_temp_information(n)

        solver_options = {'name':'gurobi',
                'threads': 4,
                'method': 2, # barrier    
                'crossover': 0, 
                'BarConvTol': 1.e-6, 
                'Seed': 123, 
                'AggFill': 0, 
                'PreDual': 0, 
                'GURO_PAR_BARDENSETHRESH': 200
                    }

        cf_solving = {'formulation': 'kirchhoff',
                    'clip_p_max_pu': 1.e-2,
                    'noisy_costs': True,
                    'skip_iterations': True,
                    'track_iterations': False,
                    'min_iterations': 4,
                    'max_iterations': 6,
                    'keep_shadowprices': ['Bus', 'Line', 'Link', 'Transformer', 
                                        'GlobalConstraint', 'Generator', 
                                        'Store', 'StorageUnit']
                    }

        n = prepare_network(n, cf_solving)

        # n.objective = 1e10

        n = solve_network(n,
                          cf_solving=cf_solving,
                          solver_options=solver_options,
                          solver_dir=tmpdir)

        n.export_to_netcdf(snakemake.output.network)

    logger.info("Maximum memory usage: {}".format(mem.mem_usage))
