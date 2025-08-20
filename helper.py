import pandas as pd
import os
import glob
import plotly.graph_objects as go
import plotly as py
import plotly.io as pio
from plotly.subplots import make_subplots
# For OneEuroFilter, see https://github.com/casiez/OneEuroFilter
from OneEuroFilter import OneEuroFilter
import common
from custom_logger import CustomLogger
import re
import numpy as np
from scipy.stats import ttest_rel, ttest_ind
from scipy.stats import zscore
from utils.HMD_helper import HMD_yaw
from utils.tools import Tools
from tqdm import tqdm
from datetime import datetime


logger = CustomLogger(__name__)  # use custom logger
template = common.get_configs("plotly_template")

HMD_class = HMD_yaw()
extra_class = Tools()

# Consts
SAVE_PNG = True
SAVE_EPS = True


class HMD_helper:

    # set template for plotly output
    template = common.get_configs('plotly_template')
    smoothen_signal = common.get_configs('smoothen_signal')
    folder_figures = common.get_configs('figures')  # subdirectory to save figures
    folder_stats = 'statistics'  # subdirectory to save statistical output

    def __init__(self):
        self.test_trial = common.get_configs("compare_trial")

    def smoothen_filter(self, signal, type_flter='OneEuroFilter'):
        """Smoothen list with a filter.

        Args:
            signal (list): input signal to smoothen
            type_flter (str, optional): type_flter of filter to use.

        Returns:
            list: list with smoothened data.
        """
        if type_flter == 'OneEuroFilter':
            filter_kp = OneEuroFilter(freq=common.get_configs('freq'),            # frequency
                                      mincutoff=common.get_configs('mincutoff'),  # minimum cutoff frequency
                                      beta=common.get_configs('beta'))            # beta value
            return [filter_kp(value) for value in signal]
        else:
            logger.error('Specified filter {} not implemented.', type_flter)
            return -1

    def get_sound_clip_name(self, df, video_id_value):
        """
        Returns the display name for a given video_id from the provided DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing at least 'video_id' and 'display_name' columns.
            video_id_value (str or int): The video_id to search for.

        Returns:
            str or None: The corresponding display name, or None if not found.
        """
        # Filter DataFrame for the matching video_id and get the display_name
        result = df.loc[df["video_id"] == video_id_value, "display_name"]
        return result.iloc[0] if not result.empty else None

    def plot_column_distribution(self, df, columns, output_folder, save_file=True, tag=None):
        """
        Plots and prints distributions of specified survey columns.

        Parameters:
            df (DataFrame or str): DataFrame or path to CSV.
            columns (list): List of column names to analyse.
            output_folder (str): Folder where plots will be saved.
            save_file (bool): Whether to save plots or just show them.
        """
        if isinstance(df, str):
            df = pd.read_csv(df)

        for column in columns:
            if column not in df.columns:
                logger.error(f"Column not found: {column}")
                continue

            logger.info(f"Distribution for: '{column}'")
            # Drop missing
            data = df[column].dropna().astype(str).str.strip()
            value_counts = data.value_counts()

            # Print counts
            for value, count in value_counts.items():
                logger.info(f"{value}: {count}")

            # Create pie chart
            fig = go.Figure(data=[
                go.Pie(labels=value_counts.index, values=value_counts.values, hole=0.0)
            ])

            fig.update_layout(
                margin=dict(l=10, r=10, t=40, b=10)
            )

            # Save or display
            if save_file:
                # Replace spaces with underscores, remove question marks, strip final periods
                filename = column.strip()  # remove leading/trailing whitespace
                filename = re.sub(r"[^\w\s-]", "", filename)  # remove punctuation except underscores/hyphens
                filename = filename.replace(" ", "_").lower()
                if tag:
                    filename = f"{filename}_{tag}"
                self.save_plotly(fig, filename, save_final=True)
            else:
                fig.show()

    def distribution_plots(self, df, column_names, output_folder, save_file=True):

        if isinstance(df, str):
            df = pd.read_csv(df)

        current_year = datetime.now().year

        for column_name in column_names:
            if column_name not in df.columns:
                logger.warning(f"Column not found: {column_name}")
                continue

            # Try numeric conversion
            temp_series = pd.to_numeric(df[column_name], errors='coerce')
            is_numeric = pd.api.types.is_numeric_dtype(temp_series)

            # Drop NaNs
            df_clean = df.dropna(subset=[column_name]).copy()

            if df_clean.empty:
                logger.warning(f"No valid data in column: {column_name}")
                continue

            if is_numeric:
                # Numeric column processing
                df_clean[column_name] = pd.to_numeric(df_clean[column_name], errors='coerce')

                # Special cleaning for age column
                if column_name.strip().lower() == "what is your age (in years)?".lower():
                    cleaned_ages = []
                    for val in df_clean[column_name]:
                        if 18 <= val <= 99:  # Valid age
                            cleaned_ages.append(val)
                        elif 1900 <= val <= current_year:  # Looks like year of birth
                            age = current_year - val
                            if 18 <= age <= 99:
                                cleaned_ages.append(age)
                        # Else: ignore nonsensical values
                    df_clean[column_name] = cleaned_ages

                mean_val = df_clean[column_name].mean()
                std_val = df_clean[column_name].std()
                logger.info(f"{column_name} - Mean: {mean_val:.2f}, Std Dev: {std_val:.2f}")

                value_counts = df_clean[column_name].round().value_counts().sort_index()
                labels = [f"{int(v)}" for v in value_counts.index]
                values = value_counts.values
            else:
                # Categorical column processing
                df_clean[column_name] = df_clean[column_name].astype(str).str.strip()
                value_counts = df_clean[column_name].value_counts()
                labels = value_counts.index.tolist()
                values = value_counts.values.tolist()
                logger.info(f"{column_name} - Response counts: {dict(zip(labels, values))}")

            # Plotting
            fig = go.Figure(data=[
                go.Pie(labels=labels, values=values, hole=0.0, showlegend=True, sort=False)
            ])

            fig.update_layout(
                margin=dict(l=10, r=10, t=40, b=10)
            )

            # Save or display
            if save_file:
                # Replace spaces with underscores, remove question marks, strip final periods
                filename = column_name.strip()  # remove leading/trailing whitespace
                filename = re.sub(r"[^\w\s-]", "", filename)  # remove punctuation except underscores/hyphens
                filename = filename.replace(" ", "_").lower()
                self.save_plotly(fig, filename, save_final=True)
            else:
                fig.show()

    @staticmethod
    def read_slider_data(data_folder, mapping, output_folder):
        """
        Reads participant slider CSVs from all participant folders, aggregates the
        ratings (noticeability, informativeness, annoyance) for all trials,
        and saves a summary CSV per slider to the output folder.

        Args:
            data_folder (str): Path to the folder containing participant subfolders.
            mapping (pd.DataFrame): Mapping DataFrame with 'video_id' and 'sound_clip_name'.
            output_folder (str): Directory to save aggregated CSVs for each slider.
        """
        participant_data = {}  # Store per-participant DataFrames
        all_trials = set()  # Collect all unique trial IDs

        # Create a mapping from video_id to sound_clip_name for column renaming
        mapping_dict = dict(zip(mapping["video_id"], mapping["sound_clip_name"]))

        # Iterate over each participant's folder
        for folder in sorted(os.listdir(data_folder)):
            folder_path = os.path.join(data_folder, folder)
            if not os.path.isdir(folder_path):
                continue

            # Parse participant ID from folder name
            match = re.match(r'Participant_(\d+)_', folder)
            if not match:
                continue
            participant_id = int(match.group(1))

            # Find the CSV with slider data for this participant
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                # Expected pattern: Participant_[id]_[number]_[number].csv
                if re.match(rf'Participant_{participant_id}_\d+_\d+\.csv', file):
                    # Assume no header: columns are trial, noticeability, info, annoyance
                    df = pd.read_csv(file_path, header=None, names=["trial", "noticeability", "info", "annoyance"])
                    df.set_index("trial", inplace=True)
                    participant_data[participant_id] = df
                    all_trials.update(df.index)
                    break  # Stop at first valid slider CSV

        # Build a sorted trial list (with 'test' first if present)
        all_trials = sorted([t for t in all_trials if t != "test"],
                            key=lambda x: int(re.search(r'\d+', x).group()))  # type: ignore
        all_trials.insert(0, "test") if "test" in all_trials else None

        # Prepare dict to aggregate each slider rating across all participants
        slider_data = {"noticeability": [], "info": [], "annoyance": []}

        # For each participant, gather ratings for all trials, filling missing with None
        for participant_id, df in sorted(participant_data.items()):
            row = {"participant_id": participant_id}
            for trial in all_trials:
                if trial in df.index:
                    row[trial] = df.loc[trial].to_list()
                else:
                    row[trial] = [None, None, None]

            # Split values for each slider
            slider_data["noticeability"].append([participant_id] + [vals[0] for vals in row.values() if isinstance(vals, list)])  # noqa: E501
            slider_data["info"].append([participant_id] + [vals[1] for vals in row.values() if isinstance(vals, list)])
            slider_data["annoyance"].append([participant_id] + [vals[2] for vals in row.values() if isinstance(vals, list)])  # noqa: E501

        # Convert lists to DataFrames, rename columns, and add average row
        for slider, data in slider_data.items():
            df = pd.DataFrame(data, columns=["participant_id"] + all_trials)
            # Rename trial columns using mapping (video_id to sound_clip_name)
            df.rename(columns={trial: mapping_dict.get(trial, trial) for trial in all_trials}, inplace=True)

            # Add average row at the end (ignoring participant_id)
            avg_values = df.iloc[:, 1:].mean(skipna=True)
            avg_row = pd.DataFrame([["average"] + avg_values.tolist()], columns=df.columns)
            df = pd.concat([df, avg_row], ignore_index=True)

            # Save the aggregated slider data to CSV
            output_path = os.path.join(output_folder, f"slider_input_{slider}.csv")
            df.to_csv(output_path, index=False)

    def save_plotly(self, fig, name, remove_margins=False, width=1320, height=680, save_eps=True, save_png=True,
                    save_html=True, open_browser=True, save_mp4=False, save_final=False):
        """
        Helper function to save figure as html file.

        Args:
            fig (plotly figure): figure object.
            name (str): name of html file.
            path (str): folder for saving file.
            remove_margins (bool, optional): remove white margins around EPS figure.
            width (int, optional): width of figures to be saved.
            height (int, optional): height of figures to be saved.
            save_eps (bool, optional): save image as EPS file.
            save_png (bool, optional): save image as PNG file.
            save_html (bool, optional): save image as html file.
            open_browser (bool, optional): open figure in the browse.
            save_mp4 (bool, optional): save video as MP4 file.
            save_final (bool, optional): whether to save the "good" final figure.
        """
        # disable mathjax globally for Kaleido
        pio.kaleido.scope.mathjax = None
        # build path
        path = os.path.join(common.get_configs("output"))
        if not os.path.exists(path):
            os.makedirs(path)

        # build path for final figure
        path_final = self.folder_figures
        if save_final and not os.path.exists(path_final):
            os.makedirs(path_final)

        # limit name to max 200 char (for Windows)
        if len(path) + len(name) > 195 or len(path_final) + len(name) > 195:
            name = name[:200 - len(path) - 5]

        # save as html
        if save_html:
            if open_browser:
                # open in browser
                py.offline.plot(fig, filename=os.path.join(path, name + '.html'))
                # also save the final figure
                if save_final:
                    py.offline.plot(fig, filename=os.path.join(path_final, name + '.html'), auto_open=False)
            else:
                # do not open in browser
                py.offline.plot(fig, filename=os.path.join(path, name + '.html'), auto_open=False)
                # also save the final figure
                if save_final:
                    py.offline.plot(fig, filename=os.path.join(path_final, name + '.html'), auto_open=False)

        # remove white margins
        if remove_margins:
            fig.update_layout(margin=dict(l=100, r=2, t=20, b=12))

        # save as eps
        if save_eps:
            fig.write_image(os.path.join(path, name + '.eps'), width=width, height=height)

            # also save the final figure
            if save_final:
                fig.write_image(os.path.join(path_final, name + '.eps'), width=width, height=height)

        # save as png
        if save_png:
            fig.write_image(os.path.join(path, name + '.png'), width=width, height=height)

            # also save the final figure
            if save_final:
                fig.write_image(os.path.join(path_final, name + '.png'), width=width, height=height)

        # save as mp4
        if save_mp4:
            fig.write_image(os.path.join(path, name + '.mp4'), width=width, height=height)

    def plot_kp(self, df, y: list, y_legend_kp=None, x=None, events=None, events_width=1,
                events_dash='dot', events_colour='black', events_annotations_font_size=20,
                events_annotations_colour='black', xaxis_title='Time (s)',
                yaxis_title='Percentage of trials with response key pressed',
                xaxis_title_offset=0, yaxis_title_offset=0,
                xaxis_range=None, yaxis_range=None, stacked=False,
                pretty_text=False, orientation='v', show_text_labels=False,
                name_file='kp', save_file=False, save_final=False,
                fig_save_width=1320, fig_save_height=680, legend_x=0.7, legend_y=0.95, legend_columns=1,
                font_family=None, font_size=None, ttest_signals=None, ttest_marker='circle',
                ttest_marker_size=3, ttest_marker_colour='black', ttest_annotations_font_size=10,
                ttest_annotation_x=0, ttest_annotations_colour='black', anova_signals=None, anova_marker='cross',
                anova_marker_size=3, anova_marker_colour='black', anova_annotations_font_size=10,
                anova_annotations_colour='black', ttest_anova_row_height=0.5, xaxis_step=5,
                yaxis_step=5, y_legend_bar=None, line_width=1, bar_font_size=None,
                custom_line_colors=None, custom_line_dashes=None, flag_trigger=False, margin=None):
        """
        Plots keypress (response) data from a dataframe using Plotly, with options for custom lines,
        annotations, t-test and ANOVA result overlays, event markers, and customisable styling and saving.


        Args:
            df (dataframe): DataFrame with stimuli data.
            y (list): Column names of DataFrame to plot.
            y_legend_kp (list, optional): Names for variables for keypress data to be shown in the legend.
            x (list, optional): Values in index of DataFrame to plot for. If None, the index of df is used.
            events (list, optional): List of events to draw, formatted as values on x axis.
            events_width (int, optional): Thickness of the vertical lines.
            events_dash (str, optional): Style of the vertical lines (e.g., 'dot', 'dash').
            events_colour (str, optional): Colour of the vertical lines.
            events_annotations_font_size (int, optional): Font size for annotations on vertical lines.
            events_annotations_colour (str, optional): Colour for annotations on vertical lines.
            xaxis_title (str, optional): Title for x axis of the keypress plot.
            yaxis_title (str, optional): Title for y axis of the keypress plot.
            xaxis_title_offset (float, optional): Horizontal offset for x axis title of keypress plot.
            yaxis_title_offset (float, optional): Vertical offset for y axis title of keypress plot.
            xaxis_range (list or None, optional): Range of x axis in format [min, max] for keypress plot.
            yaxis_range (list or None, optional): Range of y axis in format [min, max] for keypress plot.
            stacked (bool, optional): Whether to show bars as stacked chart.
            pretty_text (bool, optional): Prettify tick labels by replacing underscores with spaces and capitalising.
            orientation (str, optional): Orientation of bars; 'v' = vertical, 'h' = horizontal.
            show_text_labels (bool, optional): Whether to output automatically positioned text labels.
            name_file (str, optional): Name of file to save.
            save_file (bool, optional): Whether to save the plot as an HTML file.
            save_final (bool, optional): Whether to save the figure as a final image in /figures.
            fig_save_width (int, optional): Width of the figure when saving.
            fig_save_height (int, optional): Height of the figure when saving.
            legend_x (float, optional): X location of legend as percentage of plot width.
            legend_y (float, optional): Y location of legend as percentage of plot height.
            legend_columns (int, optional): Number of columns in legend
            font_family (str, optional): Font family to use in the figure.
            font_size (int, optional): Font size to use in the figure.
            ttest_signals (list, optional): Signals to compare using t-test.
            ttest_marker (str, optional): Marker style for t-test points.
            ttest_marker_size (int, optional): Size of t-test markers.
            ttest_marker_colour (str, optional): Colour of t-test markers.
            ttest_annotations_font_size (int, optional): Font size of t-test annotations.
            ttest_annotations_colour (str, optional): Colour of t-test annotations.
            anova_signals (dict, optional): Signals to compare using ANOVA.
            anova_marker (str, optional): Marker style for ANOVA points.
            anova_marker_size (int, optional): Size of ANOVA markers.
            anova_marker_colour (str, optional): Colour of ANOVA markers.
            anova_annotations_font_size (int, optional): Font size of ANOVA annotations.
            anova_annotations_colour (str, optional): Colour of ANOVA annotations.
            ttest_anova_row_height (float, optional): Height per row for t-test/ANOVA marker rows.
            xaxis_step (int): Step between ticks on x axis.
            yaxis_step (float): Step between ticks on y axis.
            y_legend_bar (list, optional): Names for variables in bar data for legend.
            line_width (int): Line width for keypress data plot.
            bar_font_size (int, optional): Font size for bar plot text, if applicable.
            custom_line_colors (list, optional): List of custom colors for each line.
            custom_line_dashes (list, optional): List of custom dash styles for each line.
            flag_trigger (bool, optional): If True, scale y values to percentages (multiply by 100).
            margin (dict, optional): Plotly layout margin dict (e.g., {'l':40, 'r':40, ...}) for fine control.

        """
        # todo: update docstrings in all methods

        logger.info('Creating keypress figure.')
        # calculate times
        times = df['Timestamp'].values
        # plotly
        fig = go.Figure()

        # adjust ylim, if ttest results need to be plotted
        if ttest_signals:
            if yaxis_range[0] != 0:  # type: ignore
                # assume one row takes ttest_anova_row_height on y axis
                yaxis_range[0] = (yaxis_range[0] - len(ttest_signals) * ttest_anova_row_height - ttest_anova_row_height)  # noqa: E501  # type: ignore

        # adjust ylim, if anova results need to be plotted
        if anova_signals:
            # assume one row takes ttest_anova_row_height on y axis
            yaxis_range[0] = (yaxis_range[0] - len(anova_signals) * ttest_anova_row_height - ttest_anova_row_height)  # noqa: E501  # type: ignore

        # track plotted values to compute min/max for ticks
        all_values = []

        # plot keypress data
        for row_number, key in enumerate(y):
            values = df[key]  # or whatever logic fits
            if y_legend_kp:
                name = y_legend_kp[row_number]
            else:
                name = key

            # smoothen signal
            if self.smoothen_signal:
                if isinstance(values, pd.Series):
                    # Replace NaNs with 0 before smoothing
                    values = values.fillna(0).tolist()
                    values = self.smoothen_filter(values)
            else:
                # If not smoothing, ensure no NaNs anyway
                values = values.fillna(0).tolist() if isinstance(values, pd.Series) else [v if not pd.isna(v) else 0 for v in values]  # noqa: E501

            # convert to 0-100%
            if flag_trigger:
                values = [v * 100 for v in values]  # type: ignore
            else:
                values = [v for v in values]  # type: ignore

            # collect values for y-axis tick range
            all_values.extend(values)  # type: ignore

            name = y_legend_kp[row_number] if y_legend_kp else key

            # plot signal
            fig.add_trace(go.Scatter(y=values,
                                     mode='lines',
                                     x=times,
                                     line=dict(width=line_width,
                                               color=custom_line_colors[row_number] if custom_line_colors else None,
                                               dash=custom_line_dashes[row_number] if custom_line_dashes else None,
                                               ),
                                     name=name))

        # draw events
        HMD_helper.draw_events(fig=fig,
                               yaxis_range=yaxis_range,
                               events=events,
                               events_width=events_width,
                               events_dash=events_dash,
                               events_colour=events_colour,
                               events_annotations_font_size=events_annotations_font_size,
                               events_annotations_colour=events_annotations_colour)

        # update axis
        if xaxis_step:
            fig.update_xaxes(title_text=xaxis_title,
                             range=xaxis_range,
                             dtick=xaxis_step,
                             title_font=dict(family=font_family, size=font_size or common.get_configs('font_size'))
                             )
        else:
            fig.update_xaxes(title_text=xaxis_title,
                             range=xaxis_range,
                             title_font=dict(family=font_family, size=font_size or common.get_configs('font_size')))
        # Find actual y range across all series
        actual_ymin = min(all_values)
        actual_ymax = max(all_values)

        # Generate ticks from 0 up to actual_ymax
        positive_ticks = np.arange(0, actual_ymax + yaxis_step, yaxis_step)
        formatted_positive_ticks = [int(tick) if tick.is_integer() else tick for tick in positive_ticks]

        # Generate ticks from 0 down to actual_ymin (note: ymin is negative)
        negative_ticks = np.arange(0, actual_ymin - yaxis_step, -yaxis_step)
        formatted_negative_ticks = [int(tick) if tick.is_integer() else tick for tick in negative_ticks]

        # Combine and sort ticks
        visible_ticks = np.sort(np.unique(np.concatenate((formatted_negative_ticks, formatted_positive_ticks))))

        tick_labels = [str(int(t)) if t.is_integer() else f"{t:.2f}" for t in visible_ticks]

        # Update y-axis with only relevant tick marks
        fig.update_yaxes(
            showgrid=True,
            range=yaxis_range,
            tickvals=visible_ticks,  # only show ticks for data range
            ticktext=tick_labels,
            automargin=True,
            title=dict(
                        text="",
                        font=dict(family=font_family, size=font_size or common.get_configs('font_size')),
                        standoff=0
            )
        )

        fig.add_annotation(
            text=yaxis_title,
            xref='paper',
            yref='paper',
            x=xaxis_title_offset,     # still left side
            y=0.5 + yaxis_title_offset,  # push label higher (was 0.5 + offset)
            showarrow=False,
            textangle=-90,
            font=dict(family=font_family, size=font_size or common.get_configs('font_size')),
            xanchor='center',
            yanchor='middle'
        )

        # prettify text
        if pretty_text:
            for variable in y:
                # check if column contains strings
                if isinstance(df.iloc[0][variable], str):
                    # replace underscores with spaces
                    df[variable] = df[variable].str.replace('_', ' ')
                    # capitalise
                    df[variable] = df[variable].str.capitalize()

        # Plot slider data
        # use index of df if none is given
        if not x:
            x = df.index

        # draw ttest and anova rows
        self.draw_ttest_anova(fig=fig,
                              times=times,
                              name_file=name_file,
                              yaxis_range=yaxis_range,
                              yaxis_step=yaxis_step,
                              ttest_signals=ttest_signals,
                              ttest_marker=ttest_marker,
                              ttest_marker_size=ttest_marker_size,
                              ttest_marker_colour=ttest_marker_colour,
                              ttest_annotations_font_size=ttest_annotations_font_size,
                              ttest_annotations_colour=ttest_annotations_colour,
                              anova_signals=anova_signals,
                              anova_marker=anova_marker,
                              anova_marker_size=anova_marker_size,
                              anova_marker_colour=anova_marker_colour,
                              anova_annotations_font_size=anova_annotations_font_size,
                              anova_annotations_colour=anova_annotations_colour,
                              ttest_anova_row_height=ttest_anova_row_height,
                              ttest_annotation_x=ttest_annotation_x,
                              flag_trigger=flag_trigger)

        # update template
        fig.update_layout(template=self.template)

        # format text labels
        if show_text_labels:
            fig.update_traces(texttemplate='%{text:.2f}')

        # stacked bar chart
        if stacked:
            fig.update_layout(barmode='stack')

        # legend
        if legend_columns == 1:  # single column
            fig.update_layout(legend=dict(x=legend_x,
                                          y=legend_y,
                                          bgcolor='rgba(0,0,0,0)',
                                          font=dict(family=font_family,
                                                    size=font_size or common.get_configs('font_size') - 6)))

        # multiple columns
        elif legend_columns == 2:
            fig.update_layout(
                legend=dict(
                    x=legend_x,
                    y=legend_y,
                    # margin=dict(l=0, r=0, t=0, b=0, pad=0),
                    bgcolor='rgba(0,0,0,0)',
                    font=dict(size=font_size or common.get_configs('font_size')),
                    orientation='h',  # must be vertical
                    traceorder='normal',
                    itemwidth=30,  # fixed item width to ensure consistent wrapping
                    itemsizing='constant'
                ),
                legend_title_text='',
                legend_tracegroupgap=5,
                legend_groupclick='toggleitem',
                legend_itemclick='toggleothers',
                legend_itemdoubleclick='toggle',
            )

        # adjust margins because of hardcoded ylim axis
        if margin:
            fig.update_layout(margin=margin)

        # update font family
        if font_family:
            # use given value
            fig.update_layout(font=dict(family=font_family))
        else:
            # use value from config file
            fig.update_layout(font=dict(family=common.get_configs('font_family')))

        # update font size
        if font_size:
            # use given value
            fig.update_layout(font=dict(size=font_size))
        else:
            # use value from config file
            fig.update_layout(font=dict(size=common.get_configs('font_size')))

        # save file to local output folder
        if save_file:
            self.save_plotly(fig=fig,
                             name=name_file,
                             remove_margins=False,
                             width=fig_save_width,
                             height=fig_save_height,
                             save_final=save_final)  # also save as "final" figure
        # open it in localhost instead
        else:
            fig.show()

    def ttest(self, signal_1, signal_2, type='two-sided', paired=True):
        """
        Perform a t-test on two signals, computing p-values and significance.

        Args:
            signal_1 (list): First signal, a list of numeric values.
            signal_2 (list): Second signal, a list of numeric values.
            type (str, optional): Type of t-test to perform. Options are "two-sided",
                                  "greater", or "less". Defaults to "two-sided".
            paired (bool, optional): Indicates whether to perform a paired t-test
                                     (ttest_rel) or an independent t-test (ttest_ind).
                                     Defaults to True (paired).

        Returns:
            list: A list containing two elements:
                  - p_values (list): Raw p-values for each bin.
                  - significance (list): Binary flags (0 or 1) indicating whether
                    the p-value for each bin is below the threshold configured in
                    tr.common.get_configs('p_value').
        """
        # Check if the lengths of the two signals are the same
        if len(signal_1) != len(signal_2):
            logger.error('The lengths of signal_1 and signal_2 must be the same.')
            return -1

        p_values = []
        significance = []
        threshold = common.get_configs("p_value")

        for i in range(len(signal_1)):
            data1 = signal_1[i]
            data2 = signal_2[i]

            # Skip if data is empty
            if not data1 or not data2 or (paired and len(data1) != len(data2)):
                p_values.append(1.0)
                significance.append(0)
                continue

            try:
                if paired:
                    t_stat, p_val = ttest_rel(data1, data2, alternative=type)
                else:
                    t_stat, p_val = ttest_ind(data1, data2, equal_var=False, alternative=type)

                # Handles the nan cases
                if np.isnan(p_val):  # type: ignore
                    p_val = 1.0
            except Exception as e:
                logger.warning(f"Skipping t-test at time index {i} due to error: {e}")
                p_val = 1.0

            p_values.append(p_val)
            significance.append(int(p_val < threshold))

        return [p_values, significance]

    def avg_csv_files(self, data_folder, mapping):
        """
        Averages multiple CSV files corresponding to the same video ID. Each file is expected to contain
        time-series data, including quaternion rotations and potentially other columns. The output is a
        CSV file with averaged values for each timestamp across the files.

        Parameters:
            data_folder (str): Path to the folder containing input CSV files.
            mapping (pd.DataFrame): A DataFrame containing metadata, including 'video_id' and 'video_length'.

        Outputs:
            For each video_id, saves an averaged DataFrame as a CSV in the output directory.
            The output CSV is named as "<video_id>_avg_df.csv".
        """

        # Group file paths by video_id using a helper function
        grouped_data = HMD_class.group_files_by_video_id(data_folder, mapping)

        # calculate resolution based on the param in
        resolution = common.get_configs("yaw_resolution") / 1000.0

        # Process each video ID and its associated files
        logger.info("Exporting CSV files.")
        for video_id, file_locations in tqdm(grouped_data.items()):
            all_dfs = []

            # Retrieve the video length from the mapping DataFrame
            video_length_row = mapping.loc[mapping["video_id"] == video_id, "video_length"]
            if video_length_row.empty:
                logger.info(f"Video length not found for video_id: {video_id}")
                continue

            video_length = video_length_row.values[0] / 1000  # Convert milliseconds to seconds

            # Read and process each file associated with the video ID
            for file_location in file_locations:
                df = pd.read_csv(file_location)

                # Filter the DataFrame to only include rows where Timestamp >= 0 and <= video_length
                # todo: 0.01 hardcoded value does not work?
                df = df[(df["Timestamp"] >= 0) & (df["Timestamp"] <= video_length + 0.01)]

                # Round the Timestamp to the nearest multiple of resolution
                df["Timestamp"] = ((df["Timestamp"] / resolution).round() * resolution).astype(float)

                all_dfs.append(df)

            # Skip if no dataframes were collected
            if not all_dfs:
                continue

            # Concatenate all DataFrames row-wise
            combined_df = pd.concat(all_dfs, ignore_index=True)

            # Group by 'Timestamp'
            grouped = combined_df.groupby('Timestamp')

            avg_rows = []
            for timestamp, group in grouped:
                row = {'Timestamp': timestamp}

                # Perform SLERP-based quaternion averaging if quaternion columns are present
                if {"HMDRotationW", "HMDRotationX", "HMDRotationY", "HMDRotationZ"}.issubset(group.columns):
                    quats = group[["HMDRotationW", "HMDRotationX", "HMDRotationY", "HMDRotationZ"]].values.tolist()
                    avg_quat = HMD_class.average_quaternions_eigen(quats)
                    row.update({
                        "HMDRotationW": avg_quat[0],
                        "HMDRotationX": avg_quat[1],
                        "HMDRotationY": avg_quat[2],
                        "HMDRotationZ": avg_quat[3],
                    })

                # Average all remaining columns (excluding Timestamp and quaternion cols)
                other_cols = [col for col in group.columns if col not in ["Timestamp",
                                                                          "HMDRotationW",
                                                                          "HMDRotationX",
                                                                          "HMDRotationY",
                                                                          "HMDRotationZ"]]
                for col in other_cols:
                    row[col] = group[col].mean()

                avg_rows.append(row)

            # Create a new DataFrame from the averaged rows
            avg_df = pd.DataFrame(avg_rows)

            # Save dataframe in the output folder
            avg_df.to_csv(os.path.join(common.get_configs("output"), f"{video_id}_avg_df.csv"), index=False)

    def draw_ttest_anova(self, fig, times, name_file, yaxis_range, yaxis_step, ttest_signals, ttest_marker,
                         ttest_marker_size, ttest_marker_colour, ttest_annotations_font_size, ttest_annotations_colour,
                         anova_signals, anova_marker, anova_marker_size, anova_marker_colour,
                         anova_annotations_font_size, anova_annotations_colour, ttest_anova_row_height,
                         ttest_annotation_x, flag_trigger=False):
        """Draw ttest and anova test rows.

        Args:
            fig (figure): figure object.
            name_file (str): name of file to save.
            yaxis_range (list): range of x axis in format [min, max] for the keypress plot.
            yaxis_step (int): step between ticks on y axis.
            ttest_signals (list): signals to compare with ttest. None = do not compare.
            ttest_marker (str): symbol of markers for the ttest.
            ttest_marker_size (int): size of markers for the ttest.
            ttest_marker_colour (str): colour of markers for the ttest.
            ttest_annotations_font_size (int): font size of annotations for ttest.
            ttest_annotations_colour (str): colour of annotations for ttest.
            anova_signals (dict): signals to compare with ANOVA. None = do not compare.
            anova_marker (str): symbol of markers for the ANOVA.
            anova_marker_size (int): size of markers for the ANOVA.
            anova_marker_colour (str): colour of markers for the ANOVA.
            anova_annotations_font_size (int): font size of annotations for ANOVA.
            anova_annotations_colour (str): colour of annotations for ANOVA.
            ttest_anova_row_height (int): height of row of ttest/anova markers.
        """
        # todo: anova support is broken after migration from original code
        # todo: when no markers are to be shown, empty space is still added to the y axis
        # Save original axis limits
        original_min, original_max = yaxis_range
        # Counters for marker rows
        counter_ttest = 0
        counter_anova = 0

        # calculate resolution based on the param
        if flag_trigger:
            resolution = common.get_configs("kp_resolution") / 1000.0
        else:
            resolution = common.get_configs("yaw_resolution") / 1000.0

        # --- t-test markers ---
        if ttest_signals:
            for comp in ttest_signals:
                p_vals, sig = self.ttest(
                    signal_1=comp['signal_1'], signal_2=comp['signal_2'], paired=comp['paired']
                )  # type: ignore

                # Save csv
                # TODO: rounding to 2 is hardcoded and wrong?
                times_csv = [round(i * resolution, 2) for i in range(len(comp['signal_1']))]
                self.save_stats_csv(t=times_csv,
                                    p_values=p_vals,
                                    name_file=f"{comp['label']}_{name_file}.csv")

                if any(sig):
                    xs, ys = [], []
                    # hardcoding margin from the x
                    # todo: hardcoding value of margin
                    if flag_trigger:
                        y_offset = original_min - ttest_anova_row_height * (counter_ttest + 1) - 5
                    else:
                        y_offset = original_min - (ttest_anova_row_height * (counter_ttest + 1)) + 0.15

                    for i, s in enumerate(sig):
                        if s:
                            xs.append(times[i])
                            ys.append(y_offset)
                    # plot markers
                    for x, y, p_val in zip(xs, ys, p_vals):
                        fig.add_annotation(
                            x=x,
                            y=y,
                            text='*',  # TODO: use ttest_marker
                            showarrow=False,
                            yanchor='middle',
                            font=dict(family=common.get_configs("font_family"),
                                      size=ttest_marker_size,
                                      color=ttest_marker_colour),
                            hovertext=f"{comp['label']}: time={x}, p={p_val}",
                            hoverlabel=dict(bgcolor="white"),
                        )

                    # label row
                    fig.add_annotation(x=ttest_annotation_x,
                                       y=y_offset,
                                       text=comp['label'],
                                       xanchor='right',
                                       showarrow=False,
                                       font=dict(family=common.get_configs("font_family"),
                                                 size=ttest_annotations_font_size,
                                                 color=ttest_annotations_colour))
                    counter_ttest += 1

        # --- Adjust axis ---
        if counter_ttest:
            n_rows = counter_ttest + max(0, counter_anova - counter_ttest)
            min_y = original_min - ttest_anova_row_height * (n_rows + 1)
            # Use dtick + tickformat for float ticks
            fig.update_layout(yaxis=dict(
                range=[min_y, original_max],
                dtick=yaxis_step,
                tickformat='.2f'
            ))

    def save_stats_csv(self, t, p_values, name_file):
        """Save results of statistical test in csv.

        Args:
            t (list): list of time slices.
            p_values (list): list of p values.
            name_file (str): name of file.
        """
        path = os.path.join(common.get_configs("output"), self.folder_stats)  # where to save csv
        # build path
        if not os.path.exists(path):
            os.makedirs(path)
        df = pd.DataFrame(columns=['t', 'p-value'])  # dataframe to save to csv
        df['t'] = t
        df['p-value'] = p_values
        df.to_csv(os.path.join(path, name_file))

    @staticmethod
    def draw_events(fig, yaxis_range, events, events_width, events_dash, events_colour,
                    events_annotations_font_size, events_annotations_colour):
        """Draw lines and annotations of events.

        Args:
            fig (figure): figure object.
            yaxis_range (list): range of x axis in format [min, max] for the keypress plot.
            events (list): list of events to draw formatted as values on x axis.
            events_width (int): thickness of the vertical lines.
            events_dash (str): type of the vertical lines.
            events_colour (str): colour of the vertical lines.
            events_annotations_font_size (int): font size of annotations for the vertical lines.
            events_annotations_colour (str): colour of annotations for the vertical lines.
        """
        # count lines to calculate increase in coordinates of drawing
        counter_lines = 0
        # draw lines with annotations for events
        if events:
            for event in events:
                # draw start
                fig.add_shape(type='line',
                              x0=event['start'],
                              y0=yaxis_range[0],
                              x1=event['start'],
                              y1=yaxis_range[1],
                              line=dict(color=events_colour,
                                        dash=events_dash,
                                        width=events_width))
                # draw other elements only is start and finish are not the same
                if event['start'] != event['end']:
                    # draw finish
                    fig.add_shape(type='line',
                                  x0=event['end'],
                                  y0=yaxis_range[0],
                                  x1=event['end'],
                                  y1=yaxis_range[1],
                                  line=dict(color=events_colour,
                                            dash=events_dash,
                                            width=events_width))
                    # draw horizontal line
                    fig.add_annotation(ax=event['start'],
                                       axref='x',
                                       ay=yaxis_range[1] - counter_lines * 2 - 2,
                                       ayref='y',
                                       x=event['end'],
                                       arrowcolor='black',
                                       xref='x',
                                       y=yaxis_range[1] - counter_lines * 2 - 2,
                                       yref='y',
                                       arrowwidth=events_width,
                                       arrowside='end+start',
                                       arrowsize=1,
                                       arrowhead=2)
                    # draw text label
                    fig.add_annotation(text=event['annotation'],
                                       x=(event['end'] + event['start']) / 2,
                                       y=yaxis_range[1] - counter_lines * 2 - 1,  # use ylim value and draw lower
                                       showarrow=False,
                                       font=dict(family=common.get_configs("font_family"),
                                                 size=events_annotations_font_size,
                                                 color=events_annotations_colour))
                # just draw text label
                else:
                    fig.add_annotation(text=event['annotation'],
                                       x=event['start'] + 0.0,
                                       y=yaxis_range[1],
                                       # y=yaxis_range[1] - counter_lines * 2 - 0.2,  # use ylim value and draw lower
                                       showarrow=False,
                                       font=dict(family=common.get_configs("font_family"),
                                                 size=events_annotations_font_size,
                                                 color=events_annotations_colour))
                # increase counter of lines drawn
                counter_lines = counter_lines + 1

    def export_participant_trigger_matrix(self, data_folder, video_id, output_file, column_name, mapping):
        """
        Export a matrix of trigger (or other column) values per participant for a given video.

        Each cell contains a list of values (one per frame or timepoint) for that participant and timestamp.
        Missing data is left as NaN, not zero.

        Args:
            data_folder (str): Path to folder containing participant subfolders with CSVs.
            video_id (str): Target video identifier (e.g. '002', 'test', etc.).
            output_file (str): Path to output CSV file (e.g. '_output/participant_trigger_002.csv').
            column_name (str): Name of the column to export (e.g. 'TriggerValueRight').
            mapping (pd.DataFrame): Mapping DataFrame containing at least 'video_id' and 'video_length'.
        """
        participant_matrix = {}    # Store trigger value lists for each participant, keyed by timestamp
        all_timestamps = set()     # Collect all observed timestamps for alignment

        # Calculate time bin resolution (in seconds) from config
        resolution = common.get_configs("kp_resolution") / 1000.0

        # Iterate over participant folders
        for folder in sorted(os.listdir(data_folder)):
            folder_path = os.path.join(data_folder, folder)
            if not os.path.isdir(folder_path):
                continue  # Ignore files, only process directories

            # Extract participant ID from folder name (expecting "Participant_###_...")
            match = re.match(r'Participant_(\d+)_', folder)
            if not match:
                continue
            participant_id = int(match.group(1))

            # Search for this participant's file matching the video ID
            for file in os.listdir(folder_path):
                if f"_{video_id}.csv" in file:
                    file_path = os.path.join(folder_path, file)
                    df = pd.read_csv(file_path)

                    # Check required columns
                    if "Timestamp" not in df or column_name not in df:
                        continue

                    # Bin timestamps to specified resolution
                    df["Timestamp"] = ((df["Timestamp"] / resolution).round() * resolution).round(2)

                    # Group by timestamp, collect all values in a list per bin
                    grouped = df.groupby("Timestamp", as_index=True)[column_name].apply(list)

                    # Store the resulting dict: timestamp -> list of values
                    participant_matrix[f"P{participant_id}"] = grouped.to_dict()
                    all_timestamps.update(grouped.index)
                    break  # Only process the first matching file for this participant

        # Get the expected timeline from mapping for alignment (using video_length)
        video_length_row = mapping.loc[mapping["video_id"] == video_id, "video_length"]
        if not video_length_row.empty:
            video_length_sec = video_length_row.values[0] / 1000  # Convert ms to seconds
            all_timestamps = np.round(np.arange(0.0, video_length_sec + resolution, resolution), 2).tolist()

        else:
            logger.warning(f"Video length not found in mapping for video_id {video_id}")

        # Build DataFrame with one row per timestamp
        combined_df = pd.DataFrame({"Timestamp": all_timestamps})

        # For each participant, add a column: each entry is a list or NaN (if no data for that timestamp)
        for participant, values in participant_matrix.items():
            combined_df[participant] = combined_df["Timestamp"].map(values)

        # Save matrix to CSV (do NOT fill missing with zero; keep NaN for clarity)
        combined_df.to_csv(output_file, index=False)

    def export_participant_quaternion_matrix(self, data_folder, video_id, output_file, mapping):
        """
        Export a matrix of raw HMD quaternions per participant per timestamp for a given video.

        Each cell contains a stringified list of quaternion vectors for that participant and timestamp.
        This allows post-hoc reconstruction of head rotation trajectories per participant,
        aligned to a common set of timestamps for the given video.

        Args:
            data_folder (str): Folder containing all participant data folders.
            video_id (str): Video identifier (e.g. '002', 'test', etc.).
            output_file (str): Output CSV file path (e.g. '_output/participant_quat_002.csv').
            mapping (pd.DataFrame): DataFrame containing at least 'video_id' and 'video_length' columns.
        """
        participant_matrix = {}  # Store quaternions per participant per timestamp
        all_timestamps = set()  # Collect all timestamps observed

        # Bin size for timestamps (in seconds)
        resolution = common.get_configs("yaw_resolution") / 1000.0

        # Iterate over all participant folders in the data directory
        for folder in sorted(os.listdir(data_folder)):
            folder_path = os.path.join(data_folder, folder)
            if not os.path.isdir(folder_path):
                continue  # Skip files

            # Extract participant ID from folder name (expecting "Participant_###_...")
            match = re.match(r'Participant_(\d+)_', folder)
            if not match:
                continue
            participant_id = int(match.group(1))

            # Find file(s) for this participant matching the current video_id
            for file in os.listdir(folder_path):
                if f"_{video_id}.csv" in file:
                    file_path = os.path.join(folder_path, file)
                    df = pd.read_csv(file_path)

                    required_cols = {"Timestamp", "HMDRotationW", "HMDRotationX", "HMDRotationY", "HMDRotationZ"}
                    # Skip files not containing quaternion data
                    if not required_cols.issubset(df.columns):
                        continue

                    # Bin the timestamps to the specified resolution
                    df["Timestamp"] = ((df["Timestamp"] / resolution).round() * resolution).round(2)

                    # Group all quaternion lists by timestamp for this participant
                    quats_by_time = (
                        df.groupby("Timestamp")[["HMDRotationW", "HMDRotationX", "HMDRotationY", "HMDRotationZ"]]
                        .apply(lambda g: g.values.tolist()).to_dict()
                    )

                    participant_matrix[f"P{participant_id}"] = quats_by_time
                    all_timestamps.update(quats_by_time.keys())
                    break  # Only use the first matching file for this participant

        # Build the set of aligned timestamps using the expected video duration from mapping
        video_length_row = mapping.loc[mapping["video_id"] == video_id, "video_length"]
        if not video_length_row.empty:
            video_length_sec = video_length_row.values[0] / 1000  # Convert ms to seconds
            all_timestamps = np.round(np.arange(0, video_length_sec + resolution, resolution), 2).tolist()
        else:
            logger.warning(f"Video length not found in mapping for video_id {video_id}")

        # Create DataFrame with 'Timestamp' as the index column
        combined_df = pd.DataFrame({"Timestamp": all_timestamps})

        # For each participant, fill the column with the stringified list of quaternions at each timestamp
        for participant, values in participant_matrix.items():
            # If a timestamp is missing for a participant, use empty list
            combined_df[participant] = combined_df["Timestamp"].map(
                lambda ts: str(values.get(ts, []))
            )

        # Write the matrix to the specified CSV file
        combined_df.to_csv(output_file, index=False)

    def plot_column(self, mapping, column_name="TriggerValueRight", xaxis_title=None, yaxis_title=None,
                    xaxis_range=None, yaxis_range=None, margin=None):
        """
        Generate a comparison plot of keypress data (or other time-series columns) and subjective slider ratings
        across multiple video trials relative to a test/reference condition.

        This function processes participant trigger matrices for each trial,
        aligns timestamps, attaches slider-based subjective ratings (annoyance,
        informativeness, noticeability), and prepares data for visualization,
        including significance testing (paired t-tests) between the test trial and each comparison trial.

        Args:
            mapping (pd.DataFrame): DataFrame containing video metadata, including
                'video_id', 'sound_clip_name', 'display_name', and 'colour'.
            column_name (str): The column to extract for plotting (e.g., 'TriggerValueRight').
            xaxis_title (str, optional): Custom label for the x-axis.
            yaxis_title (str, optional): Custom label for the y-axis.
            xaxis_range (list, optional): x-axis [min, max] limits for the plot.
            yaxis_range (list, optional): y-axis [min, max] limits for the plot.
            margin (dict, optional): Custom plot margin dictionary.
        """
        # Filter out control/test video IDs for comparison
        video_id = mapping["video_id"]
        plot_videos = video_id[~video_id.isin(["test", "est"])]

        # Map display names to line colors for plotting
        color_dict = dict(zip(mapping['display_name'], mapping['colour']))

        # Prepare containers for results and stats
        all_dfs = []  # To store averaged time-series for each trial
        all_labels = []  # To store display names for legend
        ttest_signals = []  # For collecting signals for significance testing

        data_folder = common.get_configs("data")  # Get path to participant data

        # === Export trigger matrix for test (reference) trial ===
        self.export_participant_trigger_matrix(
            data_folder=data_folder,
            video_id=self.test_trial,
            output_file=f"_output/participant_{column_name}_{self.test_trial}.csv",
            column_name=column_name,
            mapping=mapping
        )

        # Read matrix and extract time-series for the test trial
        test_raw_df = pd.read_csv(f"_output/participant_{column_name}_{self.test_trial}.csv")
        test_matrix = extra_class.extract_time_series_values(test_raw_df)

        # === Loop through each comparison trial ===
        for video in plot_videos:
            # Get human-readable display name for this trial
            display_name = mapping.loc[mapping["video_id"] == video, "display_name"].values[0]

            # Export trigger matrix for this video
            self.export_participant_trigger_matrix(
                data_folder=data_folder,
                video_id=video,
                output_file=f"_output/participant_{column_name}_{video}.csv",
                column_name=column_name,
                mapping=mapping
            )

            # Read and process the trigger matrix to extract time series for this trial
            trial_raw_df = pd.read_csv(f"_output/participant_{column_name}_{video}.csv")
            trial_matrix = extra_class.extract_time_series_values(trial_raw_df)

            # Compute participant-averaged time series (by timestamp) for this trial
            avg_df = extra_class.average_dataframe_vectors_with_timestamp(trial_raw_df,
                                                                          column_name=f"{column_name}")

            all_dfs.append(avg_df)
            all_labels.append(display_name)

            # Prepare paired t-test between test trial and each comparison trial
            if video != "test":
                ttest_signals.append({
                    "signal_1": test_matrix,
                    "signal_2": trial_matrix,
                    "paired": True,
                    "label": f"{display_name}"
                })

        # === Combine all trial DataFrames for multi-trial plotting ===
        combined_df = pd.DataFrame()
        combined_df["Timestamp"] = all_dfs[0]["Timestamp"]

        for df, label in zip(all_dfs, all_labels):
            combined_df[label] = df[column_name]

        # Optional: add vertical event lines (e.g., stimulus onset) to plot
        events = []
        events.append({'id': 1,
                       'start': 8.7,  # type: ignore
                       'end': 8.7,  # type: ignore
                       'annotation': ''})

        # Set line style: dashed for test trial, solid for others
        custom_line_dashes = []
        for label in all_labels:
            vid = mapping.loc[mapping["display_name"] == label, "video_id"].values[0]
            if vid == self.test_trial:
                custom_line_dashes.append("dot")
            else:
                custom_line_dashes.append("solid")

        # === Generate the main plot (delegated to plot_kp helper) ===
        self.plot_kp(
            df=combined_df,
            y=all_labels,
            y_legend_kp=all_labels,
            yaxis_range=yaxis_range,
            xaxis_range=xaxis_range,
            xaxis_title=xaxis_title,  # type: ignore
            yaxis_title=yaxis_title,  # type: ignore
            xaxis_title_offset=-0.04,  # type: ignore
            yaxis_title_offset=0.18,  # type: ignore
            name_file=f"all_videos_kp_slider_plot_{column_name}",
            show_text_labels=True,
            pretty_text=True,
            events=events,
            events_width=2,
            events_annotations_font_size=common.get_configs("font_size") - 6,
            stacked=False,
            ttest_signals=ttest_signals,
            ttest_anova_row_height=4,
            ttest_annotations_font_size=common.get_configs("font_size") - 6,
            ttest_annotation_x=0.7,  # type: ignore
            ttest_marker="circle",
            ttest_marker_size=common.get_configs("font_size"),
            legend_x=0,
            legend_y=1.225,
            legend_columns=2,
            xaxis_step=1,
            yaxis_step=20,  # type: ignore
            line_width=3,
            font_size=common.get_configs("font_size"),
            fig_save_width=1470,
            fig_save_height=850,
            save_file=True,
            save_final=True,
            custom_line_colors=[color_dict.get(label, None) for label in all_labels],
            custom_line_dashes=custom_line_dashes,
            flag_trigger=True,
            margin=margin
        )

    def plot_yaw(self, mapping, column_name="Yaw", xaxis_title=None, yaxis_title=None,
                 xaxis_range=None, yaxis_range=None, margin=None):
        """
        Generate a comparison plot of keypress yaw data and subjective slider ratings
        for multiple video trials relative to a test condition.

        The function processes trigger matrices for each participant and trial,
        aligns time series data, attaches subjective slider-based ratings (annoyance,
        informativeness, noticeability), and prepares the data for visualization.
        Significance testing (paired t-tests) is performed between the test condition
        and each other trial.

        Args:
            mapping (pd.DataFrame): DataFrame with video metadata, including
                'video_id', 'sound_clip_name', 'display_name', and 'colour'.
            column_name (str, optional): The matrix column to process (default "Yaw").
            xaxis_title (str, optional): Custom label for the x-axis.
            yaxis_title (str, optional): Custom label for the y-axis.
            xaxis_range (list, optional): x-axis [min, max] limits for the plot.
            yaxis_range (list, optional): y-axis [min, max] limits for the plot.
            margin (dict, optional): Custom plot margin dictionary.
        """

        # Filter out any 'test' or 'est' control videos from the mapping
        video_id = mapping["video_id"]
        plot_videos = video_id[~video_id.isin(["test", "est"])]

        # Build a color dictionary for plotting
        color_dict = dict(zip(mapping['display_name'], mapping['colour']))

        all_dfs = []  # List to collect DataFrames for each trial
        all_labels = []  # Corresponding list of human-friendly trial labels
        ttest_signals = []  # Store t-test pairs for stats annotations

        data_folder = common.get_configs("data")  # Get path to raw data

        # === Reference (test) trial: export yaw matrix and compute average yaw per timestamp ===
        test_video = self.test_trial
        test_participant_csv = f"_output/participant_{column_name}_{test_video}.csv"
        self.export_participant_quaternion_matrix(
            data_folder=data_folder,
            video_id=test_video,
            output_file=test_participant_csv,
            mapping=mapping
        )

        # Compute average yaw for the reference (test) trial and save
        test_yaw_csv = f"_output/yaw_avg_{test_video}.csv"
        HMD_class.compute_avg_yaw_from_matrix_csv(
            input_csv=test_participant_csv,
            output_csv=test_yaw_csv
        )
        test_matrix = extra_class.all_yaws_per_bin(
            input_csv=f"_output/participant_{column_name}_{self.test_trial}.csv"
        )

        # === Iterate through each video trial (excluding control/test) ===
        for video in plot_videos:
            # Get display name for current trial
            display_name = mapping.loc[mapping["video_id"] == video, "display_name"].values[0]
            participant_csv = f"_output/participant_{column_name}_{video}.csv"

            # Export quaternion/yaw matrix for this trial
            self.export_participant_quaternion_matrix(
                data_folder=data_folder,
                video_id=video,
                output_file=participant_csv,
                mapping=mapping
            )

            # Compute avg yaw for this trial
            yaw_csv = f"_output/yaw_avg_{video}.csv"
            HMD_class.compute_avg_yaw_from_matrix_csv(
                input_csv=participant_csv,
                output_csv=yaw_csv
            )

            df = pd.read_csv(yaw_csv)
            all_dfs.append(df)
            all_labels.append(display_name)

            # Extract all per-bin yaw values (for saving and t-test)
            trial_matrix = extra_class.all_yaws_per_bin(
                input_csv=f"_output/participant_{column_name}_{video}.csv"
            )

            yaw_values = extra_class.flatten_trial_matrix(trial_matrix)
            yaw_values = yaw_values[~np.isnan(yaw_values)]  # Remove NaNs if present
            trial_txt_path = f"_output/yaw_values_{video}.txt"
            np.savetxt(trial_txt_path, yaw_values)

            # Prepare for t-test: compare each trial vs. test reference (exclude self-comparison)
            if video != test_video:
                ttest_signals.append({
                    "signal_1": test_matrix,
                    "signal_2": trial_matrix,
                    "paired": True,
                    "label": f"{display_name}"
                })

        # === Combine all trial DataFrames into a single one for plotting ===
        combined_df = pd.DataFrame()
        combined_df["Timestamp"] = all_dfs[0]["Timestamp"]

        # Add events if required (here, a placeholder event at t=8.7s)
        events = []
        events.append({'id': 1,
                       'start': 8.7,  # type: ignore
                       'end': 8.7,  # type: ignore
                       'annotation': ''})

        # Add trial average yaw series as columns
        for df, label in zip(all_dfs, all_labels):
            combined_df[label] = df["AvgYaw"]

        # Choose line style: dashed for test trial, solid for others
        custom_line_dashes = []
        # for label in all_labels:
        #     vid = mapping.loc[mapping["display_name"] == label, "video_id"].values[0]
        #     if vid == self.test_trial:
        #         custom_line_dashes.append("dot")
        #     else:
        #         custom_line_dashes.append("solid")

        # === Call central plotting function with all visualization & stats options ===
        self.plot_kp(
            df=combined_df,
            y=all_labels,
            y_legend_kp=all_labels,
            xaxis_range=xaxis_range,
            yaxis_range=yaxis_range,
            xaxis_title=xaxis_title,  # type: ignore
            yaxis_title=yaxis_title,  # type: ignore
            xaxis_title_offset=-0.047,  # type: ignore
            # yaxis_title_offset=0.17,  # type: ignore
            name_file=f"all_videos_yaw_angle_{column_name}",
            show_text_labels=True,
            pretty_text=True,
            events=events,
            events_width=2,
            events_annotations_font_size=common.get_configs("font_size") - 6,
            stacked=False,
            # ttest_signals=ttest_signals,
            ttest_anova_row_height=0.01,
            ttest_annotations_font_size=common.get_configs("font_size") - 6,
            ttest_annotation_x=0.8,  # type: ignore
            ttest_marker_size=common.get_configs("font_size") - 4,
            xaxis_step=1,
            yaxis_step=0.03,  # type: ignore
            legend_x=0,
            legend_y=1.225,
            legend_columns=2,
            line_width=3,
            fig_save_width=1470,
            fig_save_height=850,
            font_size=common.get_configs("font_size"),
            save_file=True,
            save_final=True,
            custom_line_colors=[color_dict.get(label, None) for label in all_labels],
            custom_line_dashes=custom_line_dashes,
            flag_trigger=False,
            margin=margin
        )

    def plot_individual_csvs(self, csv_paths, mapping_df, font_size=None, color_dict=None, vertical_spacing=0.25,
                             height=1500, width=1600, margin=dict(t=30, b=100, l=40, r=40)):  # noqa:E741
        """
        Plots individual participant responses from three CSV files as boxplots, with one subplot
        for each metric (Annoyance, Informativeness, Noticeability) and one for the composite score.

        Sound clip order and display names are taken from the mapping_df. Only clips found in all
        CSV files will be plotted. Colors are set based on a user-defined color_dict.

        Composite Score Calculation:
            The composite score is calculated for each sound clip and participant as follows:
            1. Invert the Annoyance score (higher Annoyance = worse) by subtracting each value
                from the maximum scale value (e.g., 10).
            2. Standardise each metric (inverted Annoyance, Informativeness, Noticeability) using z-scores.
            3. Average the three standardised values (equal weighting) to form a single composite score per response.
            This approach is based on the SUM (Single Usability Metric) methodology described
                by Sauro & Kindlund (2005) (https://doi.org/10.1145/1054972.1055028).

        Parameters:
            csv_paths (list of str): List of three CSV file paths in the order:
                                     [Annoyance, Informativeness, Noticeability].
            mapping_df (pd.DataFrame): DataFrame with columns 'sound_clip_name' and 'display_name'
                                       used to map internal sound clip names to human-readable labels.
            font_size (int, optional): Font size for figure text.
            color_dict (dict, optional): Dictionary mapping display names to specific plot colors.
            vertical_spacing (int, optional): Vertical spacing between subplots.
            height (int, optional): Height of plot.
            width (int, optional): Width of plot.
            margin (dict, optional): Custom plot margin dictionary.
        """
        # todo: copy changes in attributes to the barplot.

        if len(csv_paths) != 3:
            raise ValueError("Please provide exactly three CSV file paths.")

        color_dict = dict(zip(mapping_df['display_name'], mapping_df['colour']))

        # Load all data frames and get the set of common columns
        all_data = []
        all_columns_sets = []

        for path in csv_paths:
            df = pd.read_csv(path)
            df = df[df['participant_id'] != 'average']
            df_numeric = df.drop(columns='participant_id').astype(float)
            all_data.append(df_numeric)
            all_columns_sets.append(set(df_numeric.columns))

        # Compute the intersection of all column names across datasets
        common_cols = set.intersection(*all_columns_sets)

        # Filter mapping_df to only keep sound clips that exist in all datasets
        mapping_df = mapping_df[mapping_df['sound_clip_name'].isin(common_cols)]
        sorted_internal_names = mapping_df['sound_clip_name'].tolist()
        sorted_display_names = mapping_df['display_name'].tolist()

        # Reorder columns in all data frames
        all_data = [df[sorted_internal_names] for df in all_data]

        # Compute composite score from annoyance, informativeness, noticeability
        max_scale = 10
        annoyance = all_data[2]
        info = all_data[1]
        notice = all_data[0]

        non_annoyance = max_scale - annoyance
        z_annoyance = zscore(non_annoyance, axis=0)
        z_info = zscore(info, axis=0)
        z_notice = zscore(notice, axis=0)

        composite = (z_annoyance + z_info + z_notice) / 3
        composite = pd.DataFrame(composite, columns=sorted_internal_names)

        plot_data = all_data + [composite]

        # Define subplot layout and titles
        subplot_titles = ['Noticeability', 'Informativeness', 'Annoyance', 'Composite score']

        fig = make_subplots(rows=2, cols=2, subplot_titles=subplot_titles, vertical_spacing=vertical_spacing)

        # Set subplot title font sizes
        for annotation in fig['layout']['annotations']:  # type: ignore
            annotation['font'] = dict(size=font_size or common.get_configs('font_size'),  # type: ignore
                                      family=common.get_configs('font_family'))

        # Assign colors from color_dict
        all_colors = [color_dict.get(label, None) for label in sorted_display_names]

        # Plot each metric
        for i, df_metric in enumerate(plot_data):
            row = (i // 2) + 1
            col = (i % 2) + 1
            # y_max = 13 if i < 3 else df_metric.max().max() + 2

            # check for composite score
            if i < 3:
                for j, colname in enumerate(sorted_internal_names):
                    fig.add_trace(
                        go.Box(
                            y=df_metric[colname],
                            name=sorted_display_names[j],
                            boxpoints='outliers',
                            marker_color=all_colors[j],
                            line=dict(width=2),
                            showlegend=False
                        ),
                        row=row,
                        col=col
                    )

                    # Compute and plot mean markers
                    mean_y = [df_metric[col].mean() for col in sorted_internal_names]
                    fig.add_trace(
                        go.Scatter(
                            x=sorted_display_names,
                            y=mean_y,
                            mode='markers',
                            marker=dict(symbol='diamond', color='black', size=8),
                            name='Mean',
                            showlegend=(i == 0)  # Only show legend once
                        ),
                        row=row,
                        col=col
                    )

        # Create mapping from internal sound clip names to display names
        mapping_dict = dict(zip(mapping_df['sound_clip_name'], mapping_df['display_name']))

        avgs, stds, all_columns_sets = [], [], []

        # Read and process each CSV file
        for path in csv_paths:
            df = pd.read_csv(path)
            avg_row = df[df['participant_id'] == 'average']
            if avg_row.empty:
                raise ValueError(f"No 'average' row found in {path}")

            # Exclude 'average' row to compute per-sound-clip std deviation
            numeric_df = df[df['participant_id'] != 'average'].drop(columns='participant_id').astype(float)
            std_row = numeric_df.std()
            avg_row = avg_row.drop(columns='participant_id').iloc[0].astype(float)

            avgs.append(avg_row)
            stds.append(std_row)
            all_columns_sets.append(set(numeric_df.columns))

        # Find sound clips present in all three CSVs
        common_cols = set.intersection(*all_columns_sets)

        # Filter and sort mapping_df to retain and order only the common sound clips
        mapping_df = mapping_df[mapping_df['sound_clip_name'].isin(common_cols)]
        sorted_internal_names = mapping_df['sound_clip_name'].tolist()
        sorted_display_names = mapping_df['display_name'].tolist()
        mapping_dict = dict(zip(sorted_internal_names, sorted_display_names))

        # Reorder averages and stds to match the mapping order
        avgs = [avg[sorted_internal_names] for avg in avgs]
        stds = [std[sorted_internal_names] for std in stds]

        columns = avgs[0].index.tolist()
        display_names = [mapping_dict.get(col, col) for col in columns]

        # Compute Composite Score (z-score normalised average with inverted Annoyance)
        annoyance = avgs[2]  # First CSV = Annoyance
        info = avgs[1]  # Second CSV = Informativeness
        notice = avgs[0]  # Third CSV = Noticeability

        max_scale = 10  # Assumed survey/rating scale maximum
        non_annoyance = max_scale - annoyance

        z_annoyance = zscore(non_annoyance)
        z_info = zscore(info)
        z_notice = zscore(notice)

        composite = (z_annoyance + z_info + z_notice) / 3
        composite_std = ((stds[0] + stds[1] + stds[2]) / 3).fillna(0)  # Optional, just for label

        # Prepare data for each subplot: means and standard deviations
        data_to_plot = [
            (avgs[0], stds[0]),
            (avgs[1], stds[1]),
            (avgs[2], stds[2]),
            (composite, composite_std)
        ]

        # Add each subplot's barplot
        for i, (means, deviations) in enumerate(data_to_plot):
            if i == 3:
                fig.add_trace(
                    go.Bar(
                        x=display_names,
                        y=means,
                        # name=subplot_titles[i],
                        showlegend=False,
                        marker_color=all_colors,
                    ),
                    row=row,
                    col=col
                )

                # Annotate each bar with mean and std (as text)
                for x_val, y_val, m, d in zip(display_names, means, means, deviations):
                    fig.add_annotation(
                        text=f"{m:.2f} ({d:.2f})",
                        x=x_val,
                        y=y_val + 0.05,
                        showarrow=False,
                        textangle=-90,
                        font=dict(size=20),
                        xanchor='center',
                        yanchor='bottom',
                        row=row,
                        col=col
                    )

        # Layout settings
        fig.update_layout(
            font=dict(size=font_size or common.get_configs('font_size'), family=common.get_configs('font_family')),
            height=height,
            width=width,
            margin=margin,
            showlegend=False
        )
        fig.update_xaxes(tickangle=45)

        # Save plot
        self.save_plotly(
            fig,
            'boxplot_response',
            height=height,
            width=width,
            save_final=True
        )

    def plot_individual_csvs_barplot(self, csv_paths, mapping_df, font_size=None):
        """
        Reads three CSV files, extracts the 'average' row from each, and generates a 2x2 grid of bar plots.
        Each bar plot displays the average and standard deviation of 15 sound clips for a different response
        metric (Noticeability, Informativeness, Annoyance), plus a composite score plot.

        Composite Score Calculation:
            The composite score is calculated for each sound clip as follows:
            1. Invert the Annoyance mean (higher Annoyance = worse) by subtracting each value from the
                maximum scale value (e.g., 10).
            2. Standardise the mean values of each metric (inverted Annoyance, Informativeness, Noticeability)
                using z-scores.
            3. Average the three standardised values (equal weighting) to form a single composite score per sound clip.
            This process follows the Single Usability Metric (SUM) methodology by Sauro & Kindlund (2005).

        Parameters:
            csv_paths (list of str): List of three file paths to CSVs. Each CSV must have a row labeled
                'average' and per-participant rows.
            mapping_df (pd.DataFrame): DataFrame mapping 'sound_clip_name' to human-readable 'display_name'.
            font_size (int, optional): Font size for plot labels and titles.
        """

        # Ensure exactly three CSVs are provided
        if len(csv_paths) != 3:
            raise ValueError("Please provide exactly three CSV file paths.")

        # Create mapping from internal sound clip names to display names
        mapping_dict = dict(zip(mapping_df['sound_clip_name'], mapping_df['display_name']))

        avgs, stds, all_columns_sets = [], [], []

        # Read and process each CSV file
        for path in csv_paths:
            df = pd.read_csv(path)
            avg_row = df[df['participant_id'] == 'average']
            if avg_row.empty:
                raise ValueError(f"No 'average' row found in {path}")

            # Exclude 'average' row to compute per-sound-clip std deviation
            numeric_df = df[df['participant_id'] != 'average'].drop(columns='participant_id').astype(float)
            std_row = numeric_df.std()
            avg_row = avg_row.drop(columns='participant_id').iloc[0].astype(float)

            avgs.append(avg_row)
            stds.append(std_row)
            all_columns_sets.append(set(numeric_df.columns))

        # Find sound clips present in all three CSVs
        common_cols = set.intersection(*all_columns_sets)

        # Filter and sort mapping_df to retain and order only the common sound clips
        mapping_df = mapping_df[mapping_df['sound_clip_name'].isin(common_cols)]
        sorted_internal_names = mapping_df['sound_clip_name'].tolist()
        sorted_display_names = mapping_df['display_name'].tolist()
        mapping_dict = dict(zip(sorted_internal_names, sorted_display_names))

        # Reorder averages and stds to match the mapping order
        avgs = [avg[sorted_internal_names] for avg in avgs]
        stds = [std[sorted_internal_names] for std in stds]

        columns = avgs[0].index.tolist()
        display_names = [mapping_dict.get(col, col) for col in columns]

        # Compute Composite Score (z-score normalised average with inverted Annoyance)
        annoyance = avgs[2]  # First CSV = Annoyance
        info = avgs[1]  # Second CSV = Informativeness
        notice = avgs[0]  # Third CSV = Noticeability

        max_scale = 10  # Assumed survey/rating scale maximum
        non_annoyance = max_scale - annoyance

        z_annoyance = zscore(non_annoyance)
        z_info = zscore(info)
        z_notice = zscore(notice)

        composite = (z_annoyance + z_info + z_notice) / 3
        composite_std = ((stds[0] + stds[1] + stds[2]) / 3).fillna(0)  # Optional, just for label

        # Prepare plot titles
        subplot_titles = ['Noticeability', 'Informativeness', 'Annoyance', 'Composite score']

        # Create 2x2 subplots
        fig = make_subplots(rows=2, cols=2, subplot_titles=subplot_titles, vertical_spacing=0.3)

        # Adjust subplot title font sizes
        for annotation in fig['layout']['annotations']:  # type: ignore
            annotation['font'] = dict(size=font_size or common.get_configs('font_size'),  # type: ignore
                                      family=common.get_configs('font_family'))

        # Prepare data for each subplot: means and standard deviations
        data_to_plot = [
            (avgs[0], stds[0]),
            (avgs[1], stds[1]),
            (avgs[2], stds[2]),
            (composite, composite_std)
        ]

        # Add each subplot's barplot
        for i, (means, deviations) in enumerate(data_to_plot):
            row = (i // 2) + 1
            col = (i % 2) + 1

            fig.add_trace(
                go.Bar(
                    x=display_names,
                    y=means,
                    # name=subplot_titles[i],
                    showlegend=False
                ),
                row=row,
                col=col
            )

            # Annotate each bar with mean and std (as text)
            for x_val, y_val, m, d in zip(display_names, means, means, deviations):
                fig.add_annotation(
                    text=f"{m:.2f} ({d:.2f})",
                    x=x_val,
                    y=y_val + 0.15,
                    showarrow=False,
                    textangle=-90,
                    font=dict(size=15),
                    xanchor='center',
                    yanchor='bottom',
                    row=row,
                    col=col
                )
            # Fix y-axis range for the first 3 subplots
            if i < 3:
                fig.update_yaxes(range=[0, 12], row=row, col=col)

        # Update figure layout and style
        fig.update_layout(
            font=dict(size=font_size or common.get_configs('font_size'), family=common.get_configs('font_family')),
            height=1200,
            width=1600,
            margin=dict(t=20, b=120, l=40, r=40),
            showlegend=False
        )

        fig.update_xaxes(tickangle=45)

        # Save the resulting figure
        self.save_plotly(fig,
                         'bar_response',
                         height=1200,
                         width=1600,
                         save_final=True)

    def plot_yaw_histogram(self, mapping, angle=180, data_folder='_output', num_bins=None,
                           smoothen_filter_param=False):
        """
        Plots a histogram of average yaw angles for each trial across participants.

        This function loads yaw angle data from text files for different trials, optionally smooths the data,
        and plots histograms for each trial on a shared figure. Each line represents the distribution of yaw
        angles for a single trial.

        Parameters:
            mapping (DataFrame): A data structure (e.g., pandas DataFrame) mapping trial identifiers to display names.
            angle (int, optional): The yaw angle range considered for the histogram, from -angle to +angle.
                Default is 180.
            data_folder (str, optional): Path to the folder containing the yaw angle data files. Default is '_output'.
            num_bins (int, optional): Number of bins for the histogram. If None, defaults to 2 * angle.
            smoothen_filter_param (bool, optional): Whether to smooth the yaw data using a filter
                (e.g., OneEuroFilter).
        """

        # Find all yaw angle text files in the data folder
        txt_files = glob.glob(os.path.join(data_folder, 'yaw_values_*.txt'))
        txt_files = sorted(txt_files)  # Ensure a consistent order

        fig = go.Figure()  # Initialise the plotly figure
        if num_bins is None:
            num_bins = 2 * angle  # Default bin count

        for file_path in txt_files:
            # Extract the trial id from the filename
            match = re.search(r'yaw_values_(.+)\.txt', os.path.basename(file_path))
            if not match:
                continue
            trial_id = match.group(1)

            # Get the human-readable trial name using the mapping DataFrame
            display_name = self.get_sound_clip_name(df=mapping, video_id_value=trial_id)

            # Load all yaw values for the trial from file
            yaw_values = np.loadtxt(file_path)

            # Convert to degrees (if not already in deg)
            yaw_deg = np.degrees(yaw_values)

            # Smoothing (optional)
            if smoothen_filter_param:
                yaw_deg = self.smoothen_filter(yaw_deg.tolist(), type_flter='OneEuroFilter')
                yaw_deg = np.array(yaw_deg)

            # Keep only angles within the specified range
            filtered = yaw_deg[(yaw_deg >= -angle) & (yaw_deg <= angle)]
            if len(filtered) == 0:
                continue  # Skip if no data falls within the range

            # Compute histogram for the current trial
            hist, bins = np.histogram(filtered, bins=num_bins, range=(-angle, angle), density=True)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])

            # Add the histogram as a line to the plot
            fig.add_trace(go.Scatter(
                x=bin_centers,
                y=hist,
                mode='lines',
                name=display_name,
                line=dict(width=2)
            ))

        # Add a vertical dashed line at 0 degrees (center)
        fig.add_vline(x=0, line=dict(dash='dash',
                                     color='gray'),
                      annotation_text="0°",
                      annotation_position="top")

        # Set up axis labels, tick values, legend, and overall plot style
        fig.update_layout(
            xaxis_title='Yaw angle (deg)',
            yaxis_title='Frequency',
            xaxis=dict(
                tickmode='array',
                tickvals=[-angle, -(2*angle/3), -(angle/3), 0, (angle/3), ((2*angle/3)), angle]
            ),
            legend=dict(font=dict(size=20)),
            width=1400,
            height=800,
            font=dict(size=common.get_configs('font_size'),
                      family=common.get_configs('font_family')),
            margin=dict(t=60, b=60, l=60, r=60)
        )

        # Save the generated plot using a helper function
        self.save_plotly(fig, 'yaw_histogram', save_final=True)