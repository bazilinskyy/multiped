# Multi-pedestrian interaction with automated vehicle
Framework for the analysis of crossing behaviour in the interaction between multiple pedestrians and an automated vehicle, from the perspective of one of the pedestrians using a crowdsourcing approach.

## Citation
If you use the simulator for academic work please cite the following papers:

>  Alam, M. S., Dey, D., Martens, M.H., & Bazilinskyy, P. (2026). You’ll never walk alone: Inter-pedestrian distance, eHMIs, and crossing decisions in virtual reality.


## Getting started
[![Python Version](https://img.shields.io/badge/python-3.9.11-blue.svg)](https://www.python.org/downloads/release/python-3919/)
[![Package Manager: uv](https://img.shields.io/badge/package%20manager-uv-green)](https://docs.astral.sh/uv/)

Tested with **Python 3.9.11** and the [`uv`](https://docs.astral.sh/uv/) package manager.  
Follow these steps to set up the project.

**Step 1:** Install `uv`. `uv` is a fast Python package and environment manager. Install it using one of the following methods:

**macOS / Linux (bash/zsh):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

**Alternative (if you already have Python and pip):**
```bash
pip install uv
```

**Step 2:** Fix permissions (if needed):t

Sometimes `uv` needs to create a folder under `~/.local/share/uv/python` (macOS/Linux) or `%LOCALAPPDATA%\uv\python` (Windows).  
If this folder was created by another tool (e.g. `sudo`), you may see an error like:
```lua
error: failed to create directory ... Permission denied (os error 13)
```

To fix it, ensure you own the directory:

### macOS / Linux
```bash
mkdir -p ~/.local/share/uv
chown -R "$(id -un)":"$(id -gn)" ~/.local/share/uv
chmod -R u+rwX ~/.local/share/uv
```

### Windows
```powershell
# Create directory if it doesn't exist
New-Item -ItemType Directory -Force "$env:LOCALAPPDATA\uv"

# Ensure you (the current user) own it
# (usually not needed, but if permissions are broken)
icacls "$env:LOCALAPPDATA\uv" /grant "$($env:UserName):(OI)(CI)F"
```

**Step 3:** After installing, verify:
```bash
uv --version
```

**Step 4:** Clone the repository:
```command line
git clone https://github.com/bazilinskyy/multiped
cd multiped
```

**Step 5:** Ensure correct Python version. If you don’t already have Python 3.9.11 installed, let `uv` fetch it:
```command line
uv python install 3.9.11
```
The repo should contain a .python-version file so `uv` will automatically use this version.

**Step 6:** Create and sync the virtual environment. This will create **.venv** in the project folder and install dependencies exactly as locked in **uv.lock**:
```command line
uv sync --frozen
```

**Step 7:** Activate the virtual environment:

**macOS / Linux (bash/zsh):**
```bash
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (cmd.exe):**
```bat
.\.venv\Scripts\activate.bat
```

**Step 8:** Ensure that dataset are present. Place required datasets (including **mapping.csv**) into the **data/** directory:


**Step 9:** Run the code:
```command line
python3 analysis.py
```

## Configuration of project
Configuration of the project needs to be defined in `multiped/config`. Please use the `default.config` file for the required structure of the file. If no custom config file is provided, `default.config` is used. The config file has the following parameters:
* `mapping`: CSV file that contains all data found in the videos.
* `plotly_template`: Template used to make graphs in the analysis.
* `output`: Directory where analysis results and intermediate output files will be saved.
* `figures`: Directory where final figures and plots are stored.
* `data`: Directory containing all raw and processed data files used in the analysis.
* `intake_questionnaire`: CSV file containing participant responses from the intake (pre-experiment) questionnaire.
* `post_experiment_questionnaire`: CSV file containing participant responses from the post-experiment questionnaire.
* `always_analyse`: Boolean toggle indicating whether existing cached analysis outputs should be ignored and regenerated.
* `trigger_threshold`: List of trigger thresholds used to binarise the pressure-sensitive controller trigger. Samples greater than the threshold are coded as unsafe, and samples at or below the threshold are coded as safe/no unsafe response.
* `compare_trial`: Reference trial against which all other trials are compared during t-tests in the analysis.
* `kp_resolution`: Time bin size, in milliseconds, used for storing keypress data, which controls the resolution of keypress event logs.
* `yaw_resolution`: Time bin size, in milliseconds, used for storing yaw (head rotation) data, controlling the resolution of HMD orientation data.
* `smoothen_signal`:  Boolean toggle to enable or disable signal smoothing for data analysis.
* `freq`: Frequency parameter used by the One Euro Filter for signal smoothing.
* `mincutoff`: Minimum cutoff frequency for the One Euro Filter.
* `beta`: Beta value controlling the speed-versus-smoothness tradeoff in the One Euro Filter.
* `font_family`: Font family to be used in all generated figures for visual consistency.
* `font_size`: Font size to be applied to all text in generated figures.
* `p_value`: p-value threshold to be used for statistical significance testing (e.g., in t-tests).



[![crossing_risk_vs_Q2_scatter](figures/crossing_risk_vs_Q2_scatter.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/crossing_risk_vs_Q2_scatter.html)
Condition-wise relationship between trigger-based perceived crossing risk and Q2 across all inter-pedestrian distances and experimental factors.

[![near_minus_far_crossing_risk_vs_Q123](figures/near_minus_far_crossing_risk_vs_Q123.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/near_minus_far_crossing_risk_vs_Q123.html)
Near (2–4 m) minus far (8–10 m) differences for perceived crossing risk and Q1–Q3 for each AV behaviour × eHMI status × co-pedestrian visibility context.

[![equivalence_near_far_crossing_risk](figures/equivalence_near_vs_far_crossing_risk.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/equivalence_near_vs_far_crossing_risk.html)
Equivalence test for the near (2–4 m) versus far (8–10 m) contrast in perceived crossing risk, shown overall and for each AV behaviour × eHMI status × co-pedestrian visibility context.

[![within_between_crossing_risk_coefficients](figures/within_between_crossing_risk_coefficients.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/within_between_crossing_risk_coefficients.html)
Within-participant and between-participant coefficients relating trigger-based perceived crossing risk to Q1, Q2, and Q3.

[![crossing_risk_full_factorial](figures/crossing_risk_full_factorial.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/crossing_risk_full_factorial.html)
Trigger-based perceived crossing risk across inter-pedestrian distance, split by AV behaviour, eHMI status, and co-pedestrian visibility.

[![Q2_full_factorial](figures/Q2_full_factorial.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/Q2_full_factorial.html)
Self-reported distance influence (Q2, 0–100) across inter-pedestrian distance, split by AV behaviour, eHMI status, and co-pedestrian visibility.

[![crossing_risk_full_factorial_legend_yielding](figures/crossing_risk_full_factorial_legend_yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/crossing_risk_full_factorial_legend_yielding.html)
Trigger-based perceived crossing risk across inter-pedestrian distance with AV behaviour shown in the legend and facets for eHMI status and co-pedestrian visibility.

[![crossing_risk_full_factorial_legend_eHMI](figures/crossing_risk_full_factorial_legend_eHMI.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/crossing_risk_full_factorial_legend_eHMI.html)
Trigger-based perceived crossing risk across inter-pedestrian distance with eHMI status shown in the legend and facets for AV behaviour and co-pedestrian visibility.

[![Q2_full_factorial_legend_yielding](figures/Q2_full_factorial_legend_yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/Q2_full_factorial_legend_yielding.html)
Q2 across inter-pedestrian distance with AV behaviour shown in the legend and facets for eHMI status and co-pedestrian visibility.

[![Q2_full_factorial_legend_eHMI](figures/Q2_full_factorial_legend_eHMI.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/Q2_full_factorial_legend_eHMI.html)
Q2 across inter-pedestrian distance with eHMI status shown in the legend and facets for AV behaviour and co-pedestrian visibility.

[![q1_behaviour_of_the_other_pedestrian](figures/behaviour_of_the_other_pedestrian.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/behaviour_of_the_other_pedestrian.html)
Q1: self-reported influence of the co-pedestrian's behaviour on participants' decision to cross (0–100), split by AV behaviour, eHMI status, and co-pedestrian visibility.

[![q2_distance_between_pedestrians](figures/distance_between_pedestrian.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/distance_between_pedestrian.html)
Q2: self-reported influence of inter-pedestrian distance on participants' decision to cross (0–100), split by AV behaviour, eHMI status, and co-pedestrian visibility.

[![q3_intention_of_the_vehicle](figures/intention_of_the_vehicle.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/intention_of_the_vehicle.html)
Q3: self-reported understanding of the vehicle's intention (0–100), split by AV behaviour, eHMI status, and co-pedestrian visibility.


### Trigger press from the participant

[![kp_all_y](figures/kp_all_y.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/kp_all_y.html)
Yielding trials.

[![kp_all_ny](figures/kp_all_ny.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/kp_all_ny.html)
Non-yielding trials.

[![kp_e1_y](figures/kp_e1_y.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/kp_e1_y.html)
Yielding trials with eHMI.

[![kp_e1_ny](figures/kp_e1_ny.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/kp_e1_ny.html)
Non-yielding trials with eHMI.

[![kp_e0_y](figures/kp_e0_y.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/kp_e0_y.html)
Yielding trials with no eHMI.

[![kp_e0_ny](figures/kp_e0_ny.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/kp_e0_ny.html)
Non-yielding trials with no eHMI.

[![kp_p1_e1_ny](figures/kp_p1_e1_ny.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/kp_p1_e1_ny.html)
Co-pedestrian visible, eHMI, non-yielding trials.

[![kp_p1_e1_y](figures/kp_p1_e1_y.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/kp_p1_e1_y.html)
Co-pedestrian visible, eHMI, yielding trials.

[![kp_p1_e0_ny](figures/kp_p1_e0_ny.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/kp_p1_e0_ny.html)
Co-pedestrian visible, no eHMI, non-yielding trials.

[![kp_p1_e0_y](figures/kp_p1_e0_y.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/kp_p1_e0_y.html)
Co-pedestrian visible, no eHMI, yielding trials.

[![kp_p2_e1_ny](figures/kp_p2_e1_ny.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/kp_p2_e1_ny.html)
Co-pedestrian not visible, eHMI, non-yielding trials.

[![kp_p2_e1_y](figures/kp_p2_e1_y.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/kp_p2_e1_y.html)
Co-pedestrian not visible, eHMI, yielding trials.

[![kp_p2_e0_ny](figures/kp_p2_e0_ny.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/kp_p2_e0_ny.html)
Co-pedestrian not visible, no eHMI, non-yielding trials.

[![kp_p2_e0_y](figures/kp_p2_e0_y.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/kp_p2_e0_y.html)
Co-pedestrian not visible, no eHMI, yielding trials.

[![heatmap](figures/heatmap.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/heatmap.html)
Ratio-based heatmap showing relationships between AV behaviour, eHMI status, and inter-pedestrian distance conditions.

[![trigger_feature_distance_profiles](figures/trigger_feature_distance_profiles.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/trigger_feature_distance_profiles.html)
Distance profiles for additional trigger-based features, including peak trigger, area under the curve, switch count, first press latency, and unsafe proportion, across the five inter-pedestrian distance levels.

[![trigger_feature_model_coefficients](figures/trigger_feature_model_coefficients.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/trigger_feature_model_coefficients.html)
Mixed model coefficients for the additional trigger features as a function of AV behaviour, eHMI status, co-pedestrian visibility, and inter-pedestrian distance.


### Head movement from the participant

[![all_yaw_values_with_yielding](figures/all_yaw_values_with_yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/all_yaw_values_with_yielding.html)
Head yaw in yielding trials.

[![all_yaw_values_without_yielding](figures/all_yaw_values_without_yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/all_yaw_values_without_yielding.html)
Head yaw in non-yielding trials.

[![yaw_no_ehmi_yielding](figures/yaw_eHMI_off_yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/yaw_eHMI_off_yielding.html)
Head yaw in yielding trials with no eHMI.

[![yaw_ehmi_yielding](figures/yaw_eHMI_on_yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/yaw_eHMI_on_yielding.html)
Head yaw in yielding trials with eHMI.

[![yaw_no_ehmi_non_yielding](figures/yaw_eHMI_off_non-yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/yaw_eHMI_off_non-yielding.html)
Head yaw in non-yielding trials with no eHMI.

[![yaw_ehmi_non_yielding](figures/yaw_eHMI_on_non-yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/yaw_eHMI_on_non-yielding.html)
Head yaw in non-yielding trials with eHMI.

[![yaw_visible_ehmi_non_yielding](figures/yaw_first_eHMI_on_non-yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/yaw_first_eHMI_on_non-yielding.html)
Co-pedestrian visible, eHMI, non-yielding trials.

[![yaw_visible_ehmi_yielding](figures/yaw_first_eHMI_on_yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/yaw_first_eHMI_on_yielding.html)
Co-pedestrian visible, eHMI, yielding trials.

[![yaw_visible_no_ehmi_non_yielding](figures/yaw_first_eHMI_off_non-yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/yaw_first_eHMI_off_non-yielding.html)
Co-pedestrian visible, no eHMI, non-yielding trials.

[![yaw_visible_no_ehmi_yielding](figures/yaw_first_eHMI_off_yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/yaw_first_eHMI_off_yielding.html)
Co-pedestrian visible, no eHMI, yielding trials.

[![yaw_not_visible_ehmi_non_yielding](figures/yaw_second_eHMI_on_non-yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/yaw_second_eHMI_on_non-yielding.html)
Co-pedestrian not visible, eHMI, non-yielding trials.

[![yaw_not_visible_ehmi_yielding](figures/yaw_second_eHMI_on_yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/yaw_second_eHMI_on_yielding.html)
Co-pedestrian not visible, eHMI, yielding trials.

[![yaw_not_visible_no_ehmi_non_yielding](figures/yaw_second_eHMI_off_non-yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/yaw_second_eHMI_off_non-yielding.html)
Co-pedestrian not visible, no eHMI, non-yielding trials.

[![yaw_not_visible_no_ehmi_yielding](figures/yaw_second_eHMI_off_yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/yaw_second_eHMI_off_yielding.html)
Co-pedestrian not visible, no eHMI, yielding trials.

[![yaw_hist_co_pedestrian_visible](figures/yaw_hist_can_see.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/yaw_hist_can_see.html)
Yaw distribution when the co-pedestrian was visible, across all AV behaviour and eHMI status conditions.

[![yaw_hist_co_pedestrian_not_visible](figures/yaw_hist_cannot_see.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/yaw_hist_cannot_see.html)
Yaw distribution when the co-pedestrian was not visible, across all AV behaviour and eHMI status conditions.

[![yaw_hist_visibility_distance](figures/yaw_hist_cam_dist.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/yaw_hist_cam_dist.html)
Yaw distributions by co-pedestrian visibility and inter-pedestrian distance; 2 × 5 grid of conditions.


### Participant responses

[![age_distribution](figures/age.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/age.html)
Age distribution of the participants.

[![gender_distribution](figures/gender_intake.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/gender_intake.html)
Gender distribution of the participants.

[![consent](figures/consent_intake.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/consent_intake.html)
Consent to participate in the study.

[![instructions](figures/instructions_intake.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/instructions_intake.html)
Understanding of instructions.

[![seeing_aids](figures/seeing_aids_intake.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/seeing_aids_intake.html)
Use of seeing aids.

[![hearing](figures/hearing_intake.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/hearing_intake.html)
Hearing problems.

[![vr_experience](figures/vr_exp_intake.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/vr_exp_intake.html)
Experience with virtual reality.

[![comfort_traffic](figures/comfort_traffic_intake.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/comfort_traffic_intake.html)
Comfort with walking in dense traffic.

[![pedestrian_presence_willingness](figures/ped_reduces_crossing_intake.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/ped_reduces_crossing_intake.html)
Effect of pedestrian presence on willingness to cross.

[![transport](figures/transport_intake.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/transport_intake.html)
Primary mode of transportation.

[![driving_frequency](figures/driving_freq_intake.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/driving_freq_intake.html)
Frequency of driving in the last 12 months.

[![driving_km](figures/driving_km_intake.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/driving_km_intake.html)
Kilometres driven in the last 12 months.

[![licence_age](figures/licence_age.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/licence_age.html)
Age at obtaining first driving licence.

[![accidents](figures/accidents_intake.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/accidents_intake.html)
Accidents in the last 3 years.

[![tailgating](figures/tailgating_intake.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/tailgating_intake.html)
Self-reported tailgating frequency.

[![road_user_communication](figures/road_user_comm_intake.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/road_user_comm_intake.html)
Willingness to communicate with other road users while crossing.

[![trust_av_intake](figures/trust_av_intake.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/trust_av_intake.html)
Trust in automated versus manually driven cars before the experiment.

[![pedestrian_influence_post](figures/ped_influence_post.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/ped_influence_post.html)
Post-experiment rating of the influence of another pedestrian on willingness to cross.

[![car_type_effect_post](figures/car_type_effect_post.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/car_type_effect_post.html)
Post-experiment rating of the effect of car type/eHMI on crossing decisions.

[![trust_av_post](figures/trust_av_post.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/trust_av_post.html)
Trust in automated versus manually driven cars after the experiment.

[![stress](figures/stress.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/stress.html)
Stress during the experiment.

[![anxiety](figures/anxiety.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/anxiety.html)
Anxiety during the experiment.

[![realism](figures/realism.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/realism.html)
Perceived realism of the experiment.

[![overall_experience](figures/overall_experience.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/overall_experience.html)
Overall experience rating.


## Contact
If you have any questions or suggestions, feel free to reach out to md_shadab_alam@outlook.com.