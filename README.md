# multi weather year assessment 
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This repository builds on top of the [PyPSA-Eur](https://github.com/PyPSA/pypsa-eur) framework to run dispatch optimization for weather years from 1960 to 2021, in order to evaluate energy adequacy and CO2 emissions abatement.

The input network files used in this framework is found in [this Zenodo repository](https://doi.org/10.5281/zenodo.10891263) containing 62 networks, resulting from a join capacity and dispatch optimization in PyPSA-Eur using this [branch](https://github.com/ebbekyhl/pypsa-eur/tree/multi_weather_year) together with PyPSA-Eur-Sec in this [branch](https://github.com/ebbekyhl/pypsa-eur-sec/tree/multi_weather_year).

Dependencies:
- python 3.9.13
- pandas 1.4.3
- numpy 1.23.1
- pypsa 0.20.0
- gurobi 9.5.2
- yaml 0.2.5
- pytz 2022.1
- snakemake-minimal 7.12.0
- matplotlib 3.5.2
