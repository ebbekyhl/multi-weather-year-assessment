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
    # Path must be changed to an external repository!
    historical_dispatch = pd.read_csv('hydro_dispatch/hydro_res_dispatch_' + country + '_ENTSOE.csv',index_col=0)
    
    historical_dispatch.index = pd.to_datetime(historical_dispatch.index,utc=True)
    historical_dispatch = historical_dispatch.resample('h').sum()
    df = pd.DataFrame(index=pd.date_range('1/1/2013','1/1/2014',freq='h',inclusive='left'))
    
    df1 = pd.DataFrame(historical_dispatch)
    years_to_drop = df1.index.year.value_counts()[df1.index.year.value_counts() < 8700].index
    index_to_drop = [df1.index[df1.index.year == years_to_drop[i]] for i in range(len(years_to_drop))]
    for i in range(len(index_to_drop)):
        df1 = df1.drop(index_to_drop[i])
    df1['year'] = df1.index.year
    for year in df1.index.year.unique():
        #print(year)
        df1_year = df1.query('year == @year')
        df1_year = df1_year[~df1_year.index.duplicated(keep='first')]

        year_datetime = pd.date_range(start = str(df1_year.index[0])[0:11], 
                                      end = str(int(str(df1_year.index[0])[0:4])+1) + str(df1_year.index[0])[4:11],
                                      freq='h',tz='UTC')[0:-1]

        missing_dates = year_datetime.difference(df1_year.index)
        #if len(missing_dates) > 0:
        #    print('Imputing ',len(missing_dates), ' missing timestamp(s)!')
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

def add_hydropower_constraint(n):

    constraint_countries = ['AT', 'BA', 'BE', 'BG', 'CH', 'CZ', 'DE', 
                            'ES', 'FI', 'FR', 'GR', 'HR', 'HU', 
                            'IT', 'LU', 'LV', 'ME', 'MK', 'NO', 'SE', 'SI','GB']
    df_min = {}
    df_max = {}
    for c in constraint_countries:
        if c == 'GB':
            df = read_dispatch('UK')
        else:
            df = read_dispatch(c)
        df_min[c] = df.quantile(0.05,axis=1).resample('m').sum() # <------------------------- Define lower limit
        df_max[c] = df.quantile(0.95,axis=1).resample('m').sum() # <------------------------- Define upper limit

    hydro = n.storage_units.query('carrier == "hydro"')
    hydro_units = hydro.index
    hydro_units_constrained = []
    for c in constraint_countries:
        hydro_units_constrained.append(list(hydro_units[hydro_units.str.contains(c)]))
    hydro_units_constrained = [item for sublist in hydro_units_constrained for item in sublist]
    
    for i in hydro_units_constrained:
        if hydro.loc[i].max_hours > 6:
            ######################## READ HISTORICAL DISPATCH DATA ############################
            country = i[0:2]
            country_capacity = hydro.loc[hydro.index[hydro.index.str.contains(country)]].p_nom.sum()
            nodal_share = n.storage_units.loc[i].p_nom/country_capacity
            tres_factor = 8760/len(n.snapshots)
            limit_lower_series = nodal_share*df_min[country]/tres_factor
            limit_upper = list(nodal_share*df_max[country]/tres_factor)
            
            # Scale according to inflow
            inflow = n.storage_units_t.inflow[i].sum()
            
            if sum(limit_lower_series) > inflow:
                print('Scaling according to inflow')
                limit_lower_series *= inflow/sum(limit_lower)
                
            limit_lower = list(limit_lower_series)
        
            # BELOW LINES SHOULD BE UPDATED ACCORDING TO PYPSA VERSION!
            p = get_var(n, 'StorageUnit', 'p_dispatch')
            lhs_var = p[i]
            ############################ DEFINE LOWER LIMIT ###################################
            # LHS 1
            lhs1 = linexpr((1, lhs_var)).groupby(p.index.month).apply(join_exprs)

            # RHS 1
            lower_limit = pd.DataFrame(limit_lower, index=np.arange(1,13), columns=['hydro'])
            rhs1 = lower_limit['hydro'] 
            define_constraints(n, lhs1, '>', rhs1, 'StorageUnit', 'hydro lower limit constraint ' + i)
            
            ########################### DEFINE UPPER LIMIT ####################################
            # LHS 2
            lhs2 = linexpr((1, lhs_var)).groupby(p.index.month).apply(join_exprs)

            # RHS 2
            upper_limit = pd.DataFrame(limit_upper, index=np.arange(1,13), columns=['hydro'])
            rhs2 = upper_limit['hydro'] 
            define_constraints(n, lhs2, '<', rhs2, 'StorageUnit', 'hydro upper limit constraint ' + i)

        else:
            print('hydro unit ', i, ' is attributed with max hours = 6 which already constrains the seasonal hydropower operation.')

def extra_functionality(n, snapshots):
    if snakemake.config['scenario']["hydroconstrained"]:
        add_hydropower_constraint(n)
        
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
                #'load_shedding': False,
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

        n.objective = 1e10

        n = solve_network(n,
                          cf_solving=cf_solving,
                          solver_options=solver_options,
                          solver_dir=tmpdir)

        n.export_to_netcdf(snakemake.output.network)

    logger.info("Maximum memory usage: {}".format(mem.mem_usage))
