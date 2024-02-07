import pypsa
import pandas as pd

from helper import override_component_attrs

def split_horizon(n, networks, weather_year):
    #################################### Copy network structure ##############################################
    # Here, we consider a reference network solely to copy the data strucure and format for later usagee
    freq = 8760/len(n.snapshots) # sample frequency

    # get the old time indices (January to December):
    df_index_old = pd.DataFrame(n.snapshots,columns=["snapshot"])
    df_index_old["month"] = df_index_old.snapshot.dt.month
    df_index_old["day"] = df_index_old.snapshot.dt.day
    df_index_old["hour"] = df_index_old.snapshot.dt.hour
    df_index_old.set_index(["month","day","hour"],inplace=True)

    # create new time indices (July to June):
    new_time_index = pd.date_range("2012-07-01", "2013-07-01", freq=freq)[:-1]
    df_index_new = pd.DataFrame(new_time_index, columns=["snapshot"])
    df_index_new["month"] = df_index_new.snapshot.dt.month
    df_index_new["day"] = df_index_new.snapshot.dt.day
    df_index_new["hour"] = df_index_new.snapshot.dt.hour
    df_index_new.set_index(["month","day","hour"],inplace=True)

    # order old time indices:
    old_time_index_ordered = pd.DatetimeIndex(df_index_old.loc[df_index_new.index].snapshot)
    # insert new time index
    n.snapshots = new_time_index

    # get all time-dependent variables
    temporal_data = {"links_t":n.links_t, 
                    "generators_t":n.generators_t,
                    "storage_units_t":n.storage_units_t,
                    "stores_t":n.stores_t,
                    "loads_t":n.loads_t,}

    variables = {}
    for comp in temporal_data.keys():
        n_comp_t = temporal_data[comp]
        keys = []
        for key in n_comp_t.keys():
            if not n_comp_t[key].empty:
                keys.append(key)        
        variables[comp] = keys

    # split indices in "Fall" and "Spring":
    index_fall = n.snapshots[n.snapshots >= pd.to_datetime("2013-07-01")]
    index_spring = n.snapshots[n.snapshots < pd.to_datetime("2013-07-01")]

    ################################# Acquire data from target network ###########################################
    label = {weather_year-1:"fall", weather_year:"spring"}
    indices = {"fall":index_fall, 
               "spring":index_spring}

    for comp in temporal_data.keys(): # loop over components, e.g., links, generators, etc.
        variables_c = variables[comp]

        for i in variables_c: # loop over variables in component, e.g., p_max_pu, efficiency, etc.

            for y in label.keys(): # loop over "fall" and "spring"
                category = label[y]

                n_ref = networks[y]
                df = getattr(n_ref, comp)[i]

                if label[y]== "fall":
                    df_filtered = df.loc[indices[category]]
                    
                elif label[y] == "spring":
                    df_filtered2 = pd.concat([df_filtered,df.loc[indices[category]]])
                    df_filtered2.index = new_time_index

                    # start adding components (e.g., available renewable resources or heating demand)
                    if comp == "loads_t":
                        n.loads_t[i]  = df_filtered2
                    
                    elif comp == "links_t":
                        n.links_t[i]  = df_filtered2

                    elif comp == "generators_t":
                        n.generators_t[i]  = df_filtered2

                    elif comp == "storage_units_t":
                        n.storage_units_t[i]  = df_filtered2

                    elif comp == "stores_t":
                        n.stores_t[i]  = df_filtered2


if __name__ == "__main__":
    if 'snakemake' not in globals():
        from helper import mock_snakemake
        snakemake = mock_snakemake(
            'add_co2_price',
            design_year="2013", 
            weather_year="2008",
        )
    
    weather_year = int(snakemake.wildcards.weather_year)

    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input.network)

    if snakemake.config['scenario']['split_horizon'] and weather_year > 1960:
        
        wyears = [weather_year-1, weather_year]

        config = snakemake.config
        nodes = config["model"]["nodes"]
        tres = config["model"]["tres"]
        co2_lvl = config["model"]["co2"]
        sectors = config["model"]["sectors"]

        networks = {}
        for wyear in wyears:
            networks[wyear] = pypsa.Network("networks/networks_n37_" + tres + "h/elec_wy" + str(wyear) + "_s370_" + nodes + "_lv1.0__Co2L" + co2_lvl + "-" + tres + "h-" + sectors + "-solar+p3-dist1_2050.nc")

        split_horizon(n, networks, weather_year)

    n.export_to_netcdf(snakemake.output.network)