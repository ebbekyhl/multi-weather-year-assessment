import pandas as pd
import matplotlib.pyplot as plt
import yaml
import cartopy.crs as ccrs
from matplotlib.patches import Circle, Ellipse
from matplotlib.legend_handler import HandlerPatch
import numpy as np

def load_tech_colors():
    with open('scripts/tech_colors.yaml') as file:
        tech_colors = yaml.safe_load(file)['tech_colors']
    tech_colors["biomass boiler"] = "#A67B5B"
    tech_colors['gas CHP'] = tech_colors['gas']
    tech_colors['gas CHP CC'] = tech_colors['gas']
    tech_colors['biomass CHP'] = tech_colors['biomass']
    tech_colors['biomass CHP CC'] = tech_colors['biomass']
    tech_colors['lost load el.'] = 'k'
    tech_colors['biogas to gas'] = tech_colors['biogas']
    tech_colors['dac'] = tech_colors['DAC']
    tech_colors["el. transmission lines"] = '#6c9459'
    tech_colors["el. distribution grid"] = tech_colors["electricity distribution grid"]
    tech_colors["hot water storage"] = '#e28172'
    tech_colors["OCGT"] = '#ecc1a6'
    tech_colors["H2"] = tech_colors["H2 Electrolysis"]
    tech_colors["solar PV rooftop"] = tech_colors["solar rooftop"]
    tech_colors["H2 overground"] = tech_colors["H2"]
    tech_colors["H2 underground"] = "#ff6ee5"
    tech_colors['gas CHP'] = tech_colors['CHP']

    return tech_colors

tech_colors = load_tech_colors()

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

preferred_order = pd.Index([
    "el. transmission lines",
    "el. distribution grid",
    "hydroelectricity",
    "hydro reservoir",
    "run of river",
    "pumped hydro storage",
    "solid biomass",
    "biogas",
    "onshore wind",
    "offshore wind",
    "offshore wind (AC)",
    "offshore wind (DC)",
    "solar PV",
    "solar PV rooftop",
    "solar thermal",
    "solar",
    "battery storage",
    "battery",
    "BEV charger",
    "V2G",
    "H2 Electrolysis",
    "H2 pipeline",
    "H2 Fuel Cell",
    "H2 underground",
    "H2 overground",
    "helmeth",
    "methanation",
    "Fischer-Tropsch",
    "biomass boiler",
    "hot water storage",
    "ground heat pump",
    "air heat pump",
    "heat pump",
    "resistive heater",
    "gas boiler",
    "natural gas",
    "CHP",
    "gas CHP",
    "biomass CHP CC",
    "nuclear",
    "OCGT",
    "CCGT",
    "oil",
    "solid biomass for industry CC",
    "process emissions CC",
    "gas for industry CC",
    "CO2 sequestration",
    "DAC",
    ])

def rename_techs(label):

    prefix_to_remove = [
        "residential ",
        "services ",
        "urban ",
        "rural ",
        "central ",
        "decentral "
    ]

    rename_if_contains = [
        # "H2",
        "biomass CHP CC",
        "gas CHP",
        # "CHP",
        "gas boiler",
        "biogas",
        "solar thermal",
        "air heat pump",
        "ground heat pump",
        "resistive heater",
        "Fischer-Tropsch"
    ]

    rename_if_contains_dict = {
        "water tanks": "hot water storage",
        "retrofitting": "building retrofitting",
        "battery charger": "battery",
        "battery discharger": "battery",
        "home battery storage": "battery storage",
        "Li ion": "BEV",
        "home battery": "battery",
    }

    rename = {
        "electricity distribution grid": "el. distribution grid",
        "solar": "solar PV",
        "solar rooftop": "solar PV rooftop",
        "Sabatier": "methanation",
        "offwind": "offshore wind",
        "uranium": "nuclear",
        "offwind-ac": "offshore wind",
        "offwind-dc": "offshore wind",
        "onwind": "onshore wind",
        "ror": "hydroelectricity",
        "hydro": "hydroelectricity",
        "PHS": "hydroelectricity",
        "NH3": "ammonia",
        "co2 Store": "DAC",
        "co2 stored": "CO2 sequestration",
        "AC": "el. transmission lines",
        "DC": "el. transmission lines",
        "B2B": "el. transmission lines",
        "gas":"natural gas",
        "H2":"H2 Electrolysis",
    }

    for ptr in prefix_to_remove:
        if label[:len(ptr)] == ptr:
            label = label[len(ptr):]

    for rif in rename_if_contains:
        if rif in label:
            label = rif

    for old,new in rename_if_contains_dict.items():
        if old in label:
            label = new

    for old,new in rename.items():
        if old == label:
            label = new
    return label

def plot_costs(cost_df, figsize, ncols = 3, leg_y = -0.5, leg_x = 0.5, costs_threshold = 1, costs_max = 1000, legend=True):

    df = cost_df.groupby(cost_df.index.get_level_values(2)).sum()

    #convert to billions
    df = df / 1e9

    df = df.groupby(df.index.map(rename_techs)).sum()

    to_drop = df.index[df.max(axis=1) < costs_threshold]

    print("dropping:", to_drop)

    df = df.drop(to_drop)

    new_index = preferred_order.intersection(df.index).append(df.index.difference(preferred_order))

    new_columns = df.sum().sort_values().index

    fig, ax = plt.subplots(figsize=figsize)

    df.loc[new_index,new_columns].T.plot(
                                        kind="bar",
                                        ax=ax,
                                        stacked=True,
                                        width = 0.7,
                                        color=[tech_colors[i] for i in new_index],
                                        legend = False,
                                        )

    handles,labels = ax.get_legend_handles_labels()

    handles.reverse()
    labels.reverse()

    # ax.set_ylim([0,costs_max])

    ax.set_ylabel("Total System Cost (billion EUR)")

    ax.set_xlabel("")

    ax.grid(axis="both")

    # ax.legend(handles, labels, ncol=1, loc="upper left", bbox_to_anchor=[1,1], frameon=False)

    # legend below plot
    if legend:
        fig.legend(handles, labels, 
                ncol=ncols, 
                loc="lower center", 
                bbox_to_anchor=[leg_x,leg_y],
                prop={'size': fs-2}, 
                frameon=False)
        
    # add subfigure on right hande side (add_axes)
    # ax2 = fig.add_axes([0.9,0.11,0.1,0.77])

    return fig, ax#, ax2

def plot_stacked_bar(df,carrier=None):
    fig, ax = plt.subplots(figsize=(15,6))

    # only sort if carrier is not None
    if carrier is not None:
        df = df.sort_values(by=carrier,ascending=True)

        # make carrier as the first column in the dataframe
        cols = list(df.columns)
        cols.insert(0, cols.pop(cols.index(carrier)))
        df = df.loc[:, cols]

    # make bar plot without gaps between bars
    df.plot(kind="bar",
            ax=ax,
            stacked=True,
            color=[tech_colors[t] for t in list(df.columns)],
            legend=False,
            width=0.95)
    
    # add two horizontal lines indicating the maximum and minimum production of the considered carrier
    if carrier is not None:
        ax.axhline(y=df[carrier].max(), color='k', linestyle='--', linewidth=1)
        ax.axhline(y=df[carrier].min(), color='k', linestyle='--', linewidth=1)

        # calculate the unbiased standard deviation of the carrier
        std = df[carrier].std(ddof=1)
        carrier_range = (df[carrier] - df[carrier].mean()).max()/df[carrier].mean()
        
        # put the two labels outside the plot (just below the legend)
        # ax.text(1.02, 0.8,
        #         "σ = " + str(round(std/(df.sum(axis=1).mean())*100,1)) + "% of total \n 2013-renewable resources", 
        #         transform=ax.transAxes,
        #         fontsize=fs,
        #         #fontweight='bold',
        #         verticalalignment='top')
        
        # ax.text(1.02, 0.95,
        #         "σ = " + str(round(std/df[carrier].mean()*100,1)) + "% of mean \n" + carrier + " resources", 
        #         transform=ax.transAxes,
        #         fontsize=fs,
        #         #fontweight='bold',
        #         verticalalignment='top')

        ax.text(1.02, 0.95,
        "IV [%] = " + str(round(carrier_range*100,1)) + " %",
        transform=ax.transAxes,
        fontsize=fs,
        #fontweight='bold',
        verticalalignment='top')
        
                
    # include vertical arrow above the bar to indicate the 2013 supply
    # first, convert 2013 to x-axis coordinate
    x = df.index.get_loc(2013)
    # then, convert 2013 supply to y-axis coordinate
    y = df.loc[2013].sum()
    # finally, plot the arrow (make it grey so that it is not too distracting and without a border)
    ax.annotate('',
                xy=(x, y),
                xytext=(x, y*1.1),
                arrowprops=dict(facecolor='grey', edgecolor='none', shrink=0.05),
                horizontalalignment='center',
                verticalalignment='top',
                fontsize=fs)
    
    # write "design year" above the arrow in grey as well
    ax.text(x, y*1.15,
            'design year',
            horizontalalignment='center',
            verticalalignment='top',
            fontsize=fs,
            color='grey')
    
            
    ax.set_xticklabels([str(int(x)) for x in df.index],rotation=90)
    ax.set_ylabel("Renewable resources (TWh)", fontsize=fs)

    ax.set_ylim(0,12000)

    fig.legend(loc='center right', bbox_to_anchor=(1.02, 0.5), ncol=1)
    return fig, ax

def assign_location(n):
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        ifind = pd.Series(c.df.index.str.find(" ", start=4), c.df.index)
        for i in ifind.value_counts().index:
            # these have already been assigned defaults
            if i == -1: continue
            names = ifind.index[ifind == i]
            c.df.loc[names, 'location'] = names.str[:i]

def make_legend_circles_for(sizes, scale=1.0, **kw):
    return [Circle((0, 0), radius=(s / scale)**0.5, **kw) for s in sizes]

def make_handler_map_to_scale_circles_as_in(ax, dont_resize_actively=False):
    fig = ax.get_figure()

    def axes2pt():
        return np.diff(ax.transData.transform([(0, 0), (1, 1)]), axis=0)[
            0] * (72. / fig.dpi)

    ellipses = []
    if not dont_resize_actively:
        def update_width_height(event):
            dist = axes2pt()
            for e, radius in ellipses:
                e.width, e.height = 2. * radius * dist
        fig.canvas.mpl_connect('resize_event', update_width_height)
        ax.callbacks.connect('xlim_changed', update_width_height)
        ax.callbacks.connect('ylim_changed', update_width_height)

    def legend_circle_handler(legend, orig_handle, xdescent, ydescent,
                              width, height, fontsize):
        w, h = 2. * orig_handle.get_radius() * axes2pt()
        e = Ellipse(xy=(0.5 * width - 0.5 * xdescent, 0.5 *
                        height - 0.5 * ydescent), width=w, height=w)
        ellipses.append((e, orig_handle.get_radius()))
        return e
    return {Circle: HandlerPatch(patch_func=legend_circle_handler)}

def plot_capacity_map(networks, 
                      ax,
                      tech_colors, 
                      technology_list,
                      threshold=10,
                      reference=False,
                      category="",
                      components=["generators"],
                      bus_size_factor=1.5e5,
                      electricity_transmission=False,
                      H2_transmission=False):

    if not reference:
        n = networks.copy()
    else:
        n_ref = networks[0].copy()
        n = networks[1].copy()
    
    assign_location(n)
    if reference:
        assign_location(n_ref)

    # Drop non-electric buses so they don't clutter the plot
    n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)
    if reference:
        n_ref.buses.drop(n_ref.buses.index[n_ref.buses.carrier != "AC"], inplace=True)

    capacity = pd.DataFrame(index=n.buses.index)
    for comp in components:
        df_c = getattr(n, comp)
        if reference:
            df_c_ref = getattr(n_ref, comp)
            
        if len(df_c) == 0:
            continue # Some countries might not have e.g. storage_units
        
        attr = "e_nom_opt" if comp == "stores" else "p_nom_opt"

        df_c["nice_group"] = df_c.carrier.map(rename_techs)
        if reference:
            df_c_ref["nice_group"] = df_c_ref.carrier.map(rename_techs)

        if category == "generation" and comp == 'storage_units':
            df_c = df_c.drop(df_c.query('carrier == "PHS"').index)
            if reference:
                df_c_ref = df_c_ref.drop(df_c_ref.query('carrier == "PHS"').index)
        elif category == "storage" and comp == 'storage_units':
            df_c = df_c.drop(df_c.query('carrier == "hydro"').index)
            if reference:
                df_c_ref = df_c_ref.drop(df_c_ref.query('carrier == "hydro"').index)

        capacity_c = ((df_c[attr])
                      .groupby([df_c.location, df_c.nice_group]).sum()
                      .unstack().fillna(0.))
        if reference:
            capacity_c_ref = ((df_c_ref[attr])
                              .groupby([df_c_ref.location, df_c_ref.nice_group]).sum()
                              .unstack().fillna(0.))
            
            capacity_c = capacity_c - capacity_c_ref

        if category == "generation" and comp == 'storage_units':
            capacity_c = capacity_c[technology_list[-1]]

        elif category == "generation":
            capacity_c = capacity_c[technology_list[0:-1]]
            
        else:
            capacity_c = capacity_c[technology_list]
        
        capacity = pd.concat([capacity, capacity_c], axis=1)
 
    if comp == "links": # we need to multiply by the efficiency
        efficiencies = n.links.groupby(n.links.carrier).mean().efficiency # get efficiencies for every carrier
        efficiencies.rename(index=rename_techs, inplace=True) # rename carriers to grouped technology naming
        efficiency_dict = efficiencies.to_dict() # convert to dictionary
        
        for i in capacity.columns: # multiply capacities by efficiencies
            capacity[i] = capacity[i] * efficiency_dict[i]

    plot = capacity.groupby(capacity.columns, axis=1).sum()

    plot.drop(columns=plot.sum().loc[plot.sum().abs() < threshold].index,
              inplace=True)
    
    technologies = plot.columns

    plot.drop(list(plot.columns[(plot == 0.).all()]), axis=1, inplace=True)

    preferred_order = pd.Index(["domestic demand",
                                "industry demand",
                                "heat pump",
                                "resistive heater",
                                "BEV",
                                "H2 charging",
                                "nuclear",
                                "hydroelectricity",
                                "wind",
                                "solar PV",
                                "solar rooftop",
                                "CHP",
                                "CHP CC",
                                "biomass CHP CC",
                                "gas CHP",
                                "biomass",
                                "gas",
                                "home battery",
                                "battery",
                                "V2G",
                                "H2"
                                "solar thermal",
                                "Fischer-Tropsch",
                                "CO2 capture",
                                "CO2 sequestration",
                            ])
    
    new_columns = ((preferred_order & plot.columns)
                   .append(plot.columns.difference(preferred_order)))
    plot = plot[new_columns]
    for item in new_columns:
        if item not in tech_colors:
            print("Warning!",item,"not in config/plotting/tech_colors")
    plot = plot.stack()
    
    if 'stores' in components:
        n.buses.loc["EU gas", ["x", "y"]] = n.buses.loc["DE0 0", ["x", "y"]]
    to_drop = plot.index.levels[0] ^ n.buses.index
    if len(to_drop) != 0:
        # print("dropping non-buses", to_drop)
        plot.drop(to_drop, level=0, inplace=True, axis=0)

    plot.index = pd.MultiIndex.from_tuples(plot.index.values)

    if electricity_transmission:
        line_lower_threshold = 500.
        # line_upper_threshold = 2e4
        linewidth_factor = 2e3
        ac_color = "gray"
        dc_color = "m"
        
        links = n.links #[n.links.carrier == 'DC']
        lines = n.lines

        line_widths = lines.s_nom_opt - lines.s_nom
        link_widths = links.p_nom_opt - links.p_nom

        line_lower_threshold = 0.
        line_widths[line_widths < line_lower_threshold] = 0.
        link_widths[link_widths < line_lower_threshold] = 0.
        # line_widths[line_widths > line_upper_threshold] = line_upper_threshold
        # link_widths[link_widths > line_upper_threshold] = line_upper_threshold

    elif H2_transmission:
        line_lower_threshold = 0
        line_upper_threshold = 1e12
        linewidth_factor = 2e3
        ac_color = "pink"
        dc_color = "pink"

        pipelines = n.links[n.links.carrier == 'H2 pipeline']
        link_widths = pipelines.p_nom_opt 
        line_widths = 0
    else:
        ac_color = "gray"
        dc_color = "m"
        linewidth_factor = 2e3
        line_widths = 0
        link_widths = 0
    
    # fig.set_size_inches(16, 12)
    n.plot(bus_sizes=plot / bus_size_factor,
           bus_colors=tech_colors,
           line_colors=ac_color,
           link_colors=dc_color,
           line_widths=line_widths / linewidth_factor,
           link_widths=link_widths / linewidth_factor,
           ax=ax,
           add_map_features = False,
        #    color_geomap={'ocean': 'white', 'land': "whitesmoke"}
        )
    
    for i in technologies:
        ax.plot([0,0],[1,1],label=i,color=tech_colors[i],lw=5)
    
    handles = []
    labels = []
    for s in (20, 10):
        handles.append(plt.Line2D([0], [0], color=ac_color,
                                linewidth=s * 1e3 / linewidth_factor))
        labels.append("{} ".format(s))

    if electricity_transmission or H2_transmission:
        trans = "H2" if H2_transmission else "Electricity"
        l2 = ax.legend(handles, labels,
                        loc="upper left", bbox_to_anchor=(0.2, 0.98),
                        frameon=False,
                        fontsize=15,
                        title_fontsize = 15,
                        labelspacing=2, handletextpad=1.5,
                        title='    ' + trans + ' transmission')
        ax.add_artist(l2)

    # return capacity

def group_pipes(df, drop_direction=False):
    """Group pipes which connect same buses and return overall capacity.
    """
    if drop_direction:
        positive_order = df.bus0 < df.bus1
        df_p = df[positive_order]
        swap_buses = {"bus0": "bus1", "bus1": "bus0"}
        df_n = df[~positive_order].rename(columns=swap_buses)
        df = pd.concat([df_p, df_n])

    # there are pipes for each investment period rename to AC buses name for plotting
    df.index = df.apply(
        lambda x: f"H2 pipeline {x.bus0.replace(' H2', '')} -> {x.bus1.replace(' H2', '')}",
        axis=1
    )
    # group pipe lines connecting the same buses and rename them for plotting
    pipe_capacity = df["p_nom_opt"].groupby(level=0).sum()

    return pipe_capacity

def plot_h2_map(networks, 
                ax, 
                reference=False,
                linewidth_factor=1e4, 
                bus_size_factor = 1e5, 
                add_legend=False):

    if not reference:
        n = networks.copy()
        link_colors='#a2f0f2'
        # link_colors='#0000FF',
    else:
        n_ref = networks[0].copy()
        n = networks[1].copy()
        # link_colors='#a2f0f2',
        link_colors='#0000FF'

    if "H2 pipeline" not in n.links.carrier.unique():
        return

    assign_location(n)
    if reference:
        assign_location(n_ref)

    # Drop non-electric buses so they don't clutter the plot
    n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)
    if reference:
        n_ref.buses.drop(n_ref.buses.index[n_ref.buses.carrier != "AC"], inplace=True)

    stores = n.stores[n.stores.carrier.isin(["H2"])].index
    bus_sizes = n.stores.loc[stores,"e_nom_opt"]
    if reference:
        bus_sizes_ref = n_ref.stores.loc[stores,"e_nom_opt"]
        bus_sizes = bus_sizes - bus_sizes_ref

    bus_sizes_df = pd.DataFrame()
    bus_sizes_df["bus"] = bus_sizes.index
    bus_sizes_df["bus"] = bus_sizes_df.bus.str.split(" ", expand=True)[0] + " " + bus_sizes_df.bus.str.split(" ", expand=True)[1] + " " + bus_sizes_df.bus.str.split(" ", expand=True)[2]
    bus_sizes_df["tech"] = "H2 underground"
    bus_sizes_df["size"] = bus_sizes.values
    bus_sizes_df.set_index(["bus","tech"], inplace=True)
    bus_sizes = bus_sizes_df["size"]

    bus_sizes = bus_sizes / bus_size_factor # GWh

    # make a fake MultiIndex so that area is correct for legend
    bus_sizes.rename(index=lambda x: x.replace(" H2", ""), level=0, inplace=True)
    bus_sizes = bus_sizes.fillna(0)

    # drop all links which are not H2 pipelines
    n.links.drop(n.links.index[~n.links.carrier.str.contains("H2 pipeline")], inplace=True)
    if reference:
        n_ref.links.drop(n_ref.links.index[~n_ref.links.carrier.str.contains("H2 pipeline")], inplace=True)

    h2_new = n.links.loc[n.links.carrier=="H2 pipeline"]
    h2_retro = n.links.loc[n.links.carrier=='H2 pipeline retrofitted']
    h2_new = group_pipes(h2_new) # sum capacitiy for pipelines from different investment periods
    h2_retro = group_pipes(h2_retro, drop_direction=True).reindex(h2_new.index).fillna(0)

    if reference:
        h2_new_ref = n_ref.links.loc[n_ref.links.carrier=="H2 pipeline"]
        h2_retro_ref = n_ref.links.loc[n_ref.links.carrier=='H2 pipeline retrofitted']
        h2_new_ref = group_pipes(h2_new_ref)
        h2_retro_ref = group_pipes(h2_retro_ref, drop_direction=True).reindex(h2_new.index).fillna(0)
        
        h2_new = h2_new - h2_new_ref
        h2_retro = h2_retro - h2_retro_ref

    n.links.rename(index=lambda x: x.split("-2")[0], inplace=True)
    n.links = n.links.groupby(level=0).first()
    link_widths_total = (h2_new + h2_retro) / linewidth_factor
    link_widths_total = link_widths_total.reindex(n.links.index).fillna(0.)
    
    link_widths_total_negative = link_widths_total.copy()
    link_widths_total_negative[link_widths_total_negative > 0] = 0.
    link_widths_total_negative = link_widths_total_negative.abs()
    
    link_widths_total[link_widths_total < 0] = 0.

    retro = n.links.p_nom_opt.where(n.links.carrier=='H2 pipeline retrofitted', other=0.)
    link_widths_retro = retro / linewidth_factor
    link_widths_retro[link_widths_retro < 0] = 0.

    n.links.bus0 = n.links.bus0.str.replace(" H2", "")
    n.links.bus1 = n.links.bus1.str.replace(" H2", "")

    # fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})

    n.plot(
        bus_sizes=bus_sizes,
        bus_colors=tech_colors,
        link_colors=link_colors,
        link_widths=link_widths_total,
        branch_components=["Link"],
        ax=ax,
        add_map_features = False)

    n.plot(
        bus_sizes=0,
        link_colors='#72d3d6',
        link_widths=link_widths_retro,
        branch_components=["Link"],
        add_map_features = False,
        ax=ax)
    
    n.plot(
        bus_sizes=0,
        link_colors='#FAA0A0',
        link_widths=link_widths_total_negative,
        branch_components=["Link"],
        add_map_features = False,
        ax=ax)

    if add_legend:
        handles = make_legend_circles_for(
            [1e6, 1e7],
            scale=bus_size_factor,
            facecolor='grey'
        )

        labels = ["{} TWh".format(s) for s in (1, 10)]

        l2 = ax.legend(
            handles, labels,
            loc="upper left",
            bbox_to_anchor=(-0.03, 1.01),
            labelspacing=1.0,
            frameon=False,
            title='H2 storage capacity',
            handler_map=make_handler_map_to_scale_circles_as_in(ax)
        )

        ax.add_artist(l2)

        handles = []
        labels = []

        for s in (10, 20):
            handles.append(plt.Line2D([0], [0], color="grey",
                                    linewidth=s * 1e3 / linewidth_factor))
            labels.append("{} GW".format(s))

        l1_1 = ax.legend(
            handles, labels,
            loc="upper left",
            bbox_to_anchor=(0.28, 1.01),
            frameon=False,
            labelspacing=0.8,
            handletextpad=1.5,
            title='H2 pipeline capacity'
        )

        ax.add_artist(l1_1)

    # fig.set_size_inches(16, 12)

    # return fig