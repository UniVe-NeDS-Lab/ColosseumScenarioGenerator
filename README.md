# Colosseum Scenario Generator

This tool has been developed during the [OR²AN²G](https://or2an2g.dais.unive.it/) project to automatically build [Colosseum](https://www.northeastern.edu/colosseum/) RF Scenario for IAB networks starting from the placement of the gNBs generated with another [tool](https://github.com/UniVe-NeDS-Lab/TrueBS) previously developed.

## How to install:

Install the python3 requirements:

`pip install -r requirements.txt`

## How to use:

1. Copy the configuration file: `cp sim.yaml.example sim.yaml`
2. Edit the relevant parameters such as `frequency, area`
3. Generate the scenarios by running `python3 scenario_gen.py`
