import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from helper import override_component_attrs

import yaml

with open('scripts/tech_colors.yaml') as file:
    tech_colors = yaml.safe_load(file)['tech_colors']
tech_colors['urban central gas CHP'] = tech_colors['gas']
tech_colors['urban central gas CHP CC'] = tech_colors['gas']
tech_colors['urban central solid biomass CHP'] = tech_colors['biomass']
tech_colors['urban central solid biomass CHP CC'] = tech_colors['biomass']
tech_colors['lost load el.'] = 'k'

fs = 16
plt.style.use('seaborn-ticks')
plt.rcParams['axes.labelsize'] = fs
plt.rcParams['xtick.labelsize'] = fs
plt.rcParams['ytick.labelsize'] = fs
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['legend.title_fontsize'] = fs
plt.rcParams['legend.fontsize'] = fs

dyear = str(snakemake.config['scenario']['design_year'][0])

def plot_lost_load(df):
    
    fig,ax = plt.subplots(figsize=(10,5))

    el = df[dyear].loc['lost_load_el'].copy()
    heat = df[dyear].loc['lost_load_heat'].copy()
    dic = {'lost load electricity':el,
           'lost load heat':heat}
    
    dic_to_df = pd.DataFrame.from_dict(dic)
    dic_to_df.plot.bar(ax=ax)
    fig.savefig(snakemake.output.lost_load, 
                bbox_inches = 'tight')

def plot_co2_balance(df):
    
    fig,ax = plt.subplots(figsize=(10,5))
    df_plot = df[dyear]

    df_plot['design'] = df['design'].loc[df_plot.index]
    
    # emissions sources
    electricity_supply_emissions = df_plot.loc['co2_electricity_mill_t'] # million tonne of CO2 emitted from plants purely producing electricity 
    CHP_net_emissions = df_plot.loc['co2_chp_mill_t'] # million tonne of CO2 emitted from combined heat and power plants
    # the "CHP" variable here is the net emissions of postive (gas) and negative (biomass CC) emittors
    process_emissions = df_plot.loc['co2_process_mill_t'] # million tonne of CO2 emitted from industrial processes
    dac_capture = -df_plot.loc['co2_dac_mill_t'] # million tonne of CO2 captured with direct air capture technology
    
    co2_net = df_plot.loc['co2_net']

    dic = {'Electricity':electricity_supply_emissions,
           'CHP':CHP_net_emissions,
           'Process_emissions':process_emissions,
           'DAC':dac_capture,
           }

    dic_to_df = pd.DataFrame.from_dict(dic)
    dic_to_df.plot.bar(ax=ax,
                       stacked=True,
                       legend=False)

    print(co2_net)
    ax.scatter(co2_net.index,co2_net,color='k',marker='_',s=5,label='Net emissions')

    ax.set_ylabel('Mill. tCO2')

    fig.legend(bbox_to_anchor=(0.7, -0.15),
                borderaxespad=0,
                fontsize=fs,
                frameon = True)

    fig.savefig(snakemake.output.co2_balance, 
                bbox_inches = 'tight')

def plot_energy_mix(df):
    fig,ax = plt.subplots(figsize=(10,5))

    energy_shares = df[dyear] 
    energy_shares = energy_shares.loc[df.index[df.index.str.contains('share')]]
    #energy_shares.drop(energy_shares.index[energy_shares.index.str.contains('load')],inplace=True)
    energy_shares_nz = energy_shares.loc[~(energy_shares==0).all(axis=1)]
    energy_shares_nz['design'] = df['design'].loc[energy_shares_nz.index]

    energy_shares_nz['carrier'] = energy_shares_nz.index
    energy_shares_nz['carrier'] = energy_shares_nz.carrier.str.split('share',1,expand=True)[0]

    s_i = 0
    for s in energy_shares_nz.index:
        energy_shares_nz.loc[s,'carrier'] = energy_shares_nz.carrier[s_i].rstrip()
        s_i += 1
        
    energy_shares_nz.index = energy_shares_nz.carrier
    energy_shares_nz.drop(columns='carrier',inplace=True)

    index_sorted = energy_shares_nz[energy_shares_nz.columns[0]].sort_values().index

    energy_shares_nz = energy_shares_nz.loc[index_sorted]
    
    #energy_shares_T = energy_shares_nz.T

    print(energy_shares_nz.index[energy_shares_nz.index.str.contains('load')])

    energy_shares_nz.rename(index={'load_el':'lost load el.'},inplace=True)

    print(energy_shares_nz.index[energy_shares_nz.index.str.contains('load')])

    energy_shares_nz.T.plot.bar(ax=ax,
                                stacked=True,
                                color=[tech_colors[t] for t in list(energy_shares_nz.index)],
                                legend=False)

    ax.set_ylabel('Percentage')

    fig.legend(bbox_to_anchor=(0.7, -0.15),
            borderaxespad=0,
            fontsize=fs,
            frameon = True)
    
    fig.savefig(snakemake.output.energy, bbox_inches = 'tight')

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from helper import mock_snakemake
        snakemake = mock_snakemake(
            'make_summary',
            design_year="2013", 
            weather_year="2008",
        )

    df = pd.read_csv(snakemake.input.summary_csv,
                index_col=list(range(1)),
                header=list(range(2)))

    plot_energy_mix(df)
    plot_lost_load(df)
    plot_co2_balance(df)