# Analysing crossing behaviour of a pedestrian with an automated car and other pedestrian in the traffic scence

This project defines a framework for the analysis of crossing behaviour in the interaction between multiple pedestrians and an automated vehicle, from the perspective of one of the pedestrians using a crowdsourcing approach. The jsPsych framework is used to for the frontend. In the description below, it is assumed that the repo is stored in the folder multiped. Terminal commands lower assume macOS.

## Setup
Tested with Python 3.9.11. To setup the environment run these two commands in a parent folder of the downloaded repository (replace `/` with `\` and possibly add `--user` if on Windows:
- `pip install -e multiped` will setup the project as a package accessible in the environment.
- `pip install -r multiped/requirements.txt` will install required packages.

## Troubleshooting
### Troubleshooting setup
#### ERROR: multiped is not a valid editable requirement
Check that you are indeed in the parent folder for running command `pip install -e multiped`. This command will not work from inside of the folder containing the repo.
