# multi weather year assessment 
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This repository builds on top of the [PyPSA-Eur](https://github.com/PyPSA/pypsa-eur) framework to run dispatch optimization for weather years from 1960 to 2021, in order to evaluate energy adequacy and CO2 emissions abatement.

It is not necessary to have run the PyPSA-Eur before running this assessment framework, since you can find input networks from [this Zenodo repository](https://doi.org/10.5281/zenodo.10891263) containing 62 networks, resulting from a join capacity and dispatch optimization in PyPSA-Eur using this [branch](https://github.com/ebbekyhl/pypsa-eur/tree/multi_weather_year) together with PyPSA-Eur-Sec in this [branch](https://github.com/ebbekyhl/pypsa-eur-sec/tree/multi_weather_year).