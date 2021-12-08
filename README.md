# Analysing crossing behaviour of a pedestrian with an automated car and other pedestrian in the traffic scence

This project defines a framework for the analysis of crossing behaviour in the interaction between multiple pedestrians and an automated vehicle, from the perspective of one of the pedestrians using a crowdsourcing approach. The jsPsych framework is used to for the frontend. In the description below, it is assumed that the repo is stored in the folder multiped-crowdsourced. Terminal commands lower assume macOS.

## Setup
Tested with Python 3.8.5. To setup the environment run these two commands in a parent folder of the downloaded repository (replace `/` with `\` and possibly add `--user` if on Windows:
- `pip install -e multiped-crowdsourced` will setup the project as a package accessible in the environment.
- `pip install -r multiped-crowdsourced/requirements.txt` will install required packages.
