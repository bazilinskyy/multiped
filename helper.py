import pandas as pd
import os
import glob
import plotly.graph_objects as go
import plotly as py
import plotly.io as pio
import plotly.express as px
# For OneEuroFilter, see https://github.com/casiez/OneEuroFilter
from OneEuroFilter import OneEuroFilter
import common
from custom_logger import CustomLogger
import re
import numpy as np
from scipy.stats import ttest_rel, ttest_ind
from utils.HMD_helper import HMD_yaw
from utils.tools import Tools
from tqdm import tqdm
from datetime import datetime
import ast


logger = CustomLogger(__name__)  # use custom logger

HMD_class = HMD_yaw()
extra_class = Tools()

# Consts
SAVE_PNG = True
SAVE_EPS = True
output = common.get_configs("output")


class HMD_helper:

    def __init__(self):
        self.template = common.get_configs('plotly_template')
        self.smoothen_signal = common.get_configs('smoothen_signal')
        self.folder_figures = common.get_configs('figures')  # subdirectory to save figures
        self.folder_stats = 'statistics'  # subdirectory to save statistical output
        self.data_folder = common.get_configs("data")  # Get path to participant data

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

        # Iterate over each participant's folder
        for folder in sorted(os.listdir(data_folder)):
            folder_path = os.path.join(data_folder, folder)
            if not os.path.isdir(folder_path):
                continue

            # Parse participant ID from folder name
            match = re.match(r'Participant_(\d+)', folder)
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
            # df.rename(columns={trial: mapping_dict.get(trial, trial) for trial in all_trials}, inplace=True)

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
            match = re.match(r'Participant_(\d+)', folder)
            if not match:
                continue
            participant_id = int(match.group(1))

            # Search for this participant's file matching the video ID
            for file in os.listdir(folder_path):
                if f"{video_id}.csv" in file:
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

    def plot_column(self, mapping, column_name="TriggerValueRight", parameter=None, parameter_value=None,
                    additional_parameter=None, additional_parameter_value=None,
                    compare_trial="video_1", xaxis_title=None, yaxis_title=None, xaxis_range=None,
                    yaxis_range=[0, 100], margin=None, name=None):
        """
        Generate a comparison plot of keypress data (or other time-series columns) and subjective slider ratings
        across multiple video trials relative to a test/reference condition.

        This function processes participant trigger matrices for each trial,
        aligns timestamps, attaches slider-based subjective ratings (annoyance,
        informativeness, noticeability), and prepares data for visualisation,
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
        # Find the video_length for the given video_id
        lens = mapping.loc[mapping["video_id"].eq(compare_trial), "video_length"].unique()

        if len(lens) == 0:
            raise ValueError(f"No rows found for video_id='{compare_trial}'")
        elif len(lens) > 1:
            # If the same video_id appears with different lengths, keep all matching lengths
            mapping_filtered = mapping[mapping["video_length"].isin(lens)].copy()
        else:
            # Typical case: one length
            mapping_filtered = mapping[mapping["video_length"].eq(lens[0])].copy()

        if parameter is not None:
            mapping_filtered = mapping_filtered[mapping_filtered[parameter] == parameter_value]

        if additional_parameter is not None:
            mapping_filtered = mapping_filtered[mapping_filtered[additional_parameter] == additional_parameter_value]

        # Filter out control/test video IDs for comparison
        video_id = mapping_filtered["video_id"]
        plot_videos = video_id[~video_id.isin(["baseline_1", "baseline_2"])]

        # Prepare containers for results and stats
        all_dfs = []  # To store averaged time-series for each trial
        all_labels = []  # To store display names for legend
        ttest_signals = []  # For collecting signals for significance testing

        # === Export trigger matrix for test (reference) trial ===
        self.export_participant_trigger_matrix(
            data_folder=self.data_folder,
            video_id=compare_trial,
            output_file=os.path.join(common.get_configs("output"), f"participant_{column_name}_{compare_trial}.csv"),
            column_name=column_name,
            mapping=mapping_filtered
        )

        # Read matrix and extract time-series for the test trial
        test_raw_df = pd.read_csv(os.path.join(common.get_configs("output"),
                                               f"participant_{column_name}_{compare_trial}.csv"))
        test_matrix = extra_class.extract_time_series_values(test_raw_df)

        # === Loop through each comparison trial ===
        for video in plot_videos:
            # Get human-readable display name for this trial
            display_name = mapping_filtered.loc[mapping_filtered["video_id"] == video, "video_id"].values[0]

            # Export trigger matrix for this video
            self.export_participant_trigger_matrix(
                data_folder=self.data_folder,
                video_id=video,
                output_file=os.path.join(common.get_configs("output"), f"participant_{column_name}_{video}.csv"),
                column_name=column_name,
                mapping=mapping_filtered
            )

            # Read and process the trigger matrix to extract time series for this trial
            trial_raw_df = pd.read_csv(os.path.join(common.get_configs("output"),
                                                    f"participant_{column_name}_{video}.csv"))
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
            vid = mapping_filtered.loc[mapping_filtered["video_id"] == label, "video_id"].values[0]
            if vid == compare_trial:
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
            name_file=f"all_videos_kp_slider_plot_{name}",
            show_text_labels=True,
            pretty_text=True,
            events=events,
            events_width=2,
            events_annotations_font_size=common.get_configs("font_size") - 6,
            stacked=False,
            ttest_signals=ttest_signals,
            ttest_anova_row_height=6,
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
            # custom_line_colors=[color_dict.get(label, None) for label in all_labels],
            custom_line_dashes=custom_line_dashes,
            flag_trigger=True,
            margin=margin
        )

    def plot_gender_by_nationality(self, csv_path,
                                   gender_col="What is your gender?",
                                   nationality_col="Nationality"):
        """
        Reads a CSV file and generates an interactive Plotly bar chart
        showing gender distribution for each nationality.
        """

        df = pd.read_csv(csv_path)

        nationality_map = {
            "Pakistani": "Pakistan",
            "Yemini": "Yemen",
            "Yemeni": "Yemen",
            "Nepalese": "Nepal",
            "Chinese": "China",
            "chinese": "China",
            " Chinese": "China",
            "Polish": "Poland",
            "Indian ": "India",
            "Dutch ": "Netherlands",
            "Iranian": "Iran",
            "Romanian": "Romania",
            "Spanish": "Spain",
            "Colombian": "Colombia",
            "portuguese": "Portugal",
            "Taiwanese": "Taiwan",
            "German": "Germany"
        }

        # Apply mapping
        df[nationality_col] = df[nationality_col].map(nationality_map).fillna(
            df[nationality_col].str.capitalize()
        )

        # --- Group data ---
        s = df.groupby([nationality_col, gender_col]).size()
        s.name = "Count"
        grouped = s.reset_index()

        # --- Plot ---
        fig = px.bar(
            grouped,
            x=nationality_col,
            y="Count",
            color=gender_col,
            barmode="group",
            title=""
        )

        fig.show()

    def heat_plot(self, folder_path: str, mapping_df: pd.DataFrame, relation: str = "ratio",
                  colorscale: str = "Viridis", summary_func=np.mean):
        """
        Compute one summary value per video CSV, rename axes using `mapping_df`,
        and show a Plotly heatmap of pairwise relations (no numbers, no colorbar).

        Parameters
        ----------
        folder_path : str
            Folder with 'participant_TriggerValueRight_video_*.csv'.
        mapping_df : pd.DataFrame
            Must include columns: 'video_id', 'condition_name', 'camera',
            'cross_p1_time_s', 'cross_p2_time_s'.
            The cutoff time used per video is cross_p1_time_s if camera==0,
            otherwise cross_p2_time_s. Only rows with Timestamp <= cutoff are used.
        relation : {'ratio','diff'}
            'ratio' uses vb/va; 'diff' uses vb - va.
        colorscale : str
            Plotly colorscale name.
        summary_func : callable
            Aggregator applied to all numeric values per video (default: np.mean).

        Returns
        -------
        averages : dict
            {mapped_label: summary_value}
        relation_df : pd.DataFrame
            Pairwise matrix (mapped_label × mapped_label)
        fig : plotly.graph_objects.Figure
            The heatmap figure.
        """
        # ---- Validate mapping_df ----
        required_cols = {"video_id", "condition_name", "camera", "cross_p1_time_s", "cross_p2_time_s"}
        if not required_cols.issubset(set(mapping_df.columns)):
            raise ValueError(f"mapping_df must have columns {required_cols}, got {list(mapping_df.columns)}")

        # Normalize dtypes
        md = mapping_df.copy()
        md["video_id"] = md["video_id"].astype(str)
        # Force numerics for times; invalids -> NaN
        md["cross_p1_time_s"] = pd.to_numeric(md["cross_p1_time_s"], errors="coerce")
        md["cross_p2_time_s"] = pd.to_numeric(md["cross_p2_time_s"], errors="coerce")
        md["camera"] = pd.to_numeric(md["camera"], errors="coerce").astype("Int64")

        # Build lookup: video_id -> {label, camera, p1, p2}
        mapping_info = {
            row["video_id"]: {
                "label": str(row["condition_name"]),
                "camera": int(row["camera"]) if pd.notna(row["camera"]) else None,
                "p1": float(row["cross_p1_time_s"]) if pd.notna(row["cross_p1_time_s"]) else None,
                "p2": float(row["cross_p2_time_s"]) if pd.notna(row["cross_p2_time_s"]) else None,
            }
            for _, row in md.iterrows()
        }

        # ---- Find files ----
        pattern = os.path.join(folder_path, "participant_TriggerValueRight_video_*.csv")
        file_list = glob.glob(pattern)
        if not file_list:
            raise FileNotFoundError(f"No files matched pattern: {pattern}")

        # Sort by numeric video index if present (video_123), else alphabetically
        def _video_key(fn):
            m = re.search(r"video_(\d+)\.csv$", os.path.basename(fn), flags=re.IGNORECASE)
            return (0, int(m.group(1))) if m else (1, os.path.basename(fn).lower())
        file_list = sorted(file_list, key=_video_key)

        # ---- Helper: extract video_id from filename ----
        def _extract_video_id(filename: str) -> str:
            base = os.path.basename(filename)
            m = re.search(r"(video_\d+)\.csv$", base, flags=re.IGNORECASE)
            return m.group(1) if m else base

        # ---- Compute per-video summary (respecting time cutoff) ----
        averages = {}
        for file_path in file_list:
            video_id = _extract_video_id(file_path)
            info = mapping_info.get(video_id)

            # Determine cutoff (if mapping missing or incomplete, we'll fall back to no cutoff)
            cutoff = None
            label = video_id
            if info:
                label = info["label"]
                cam = info["camera"]
                if cam == 0:
                    cutoff = info["p1"]
                elif cam == 1:
                    cutoff = info["p2"]
                else:
                    logger.warning(f"⚠️ camera not 0/1 for {video_id}; using full data (no cutoff).")

            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                logger.error(f"⚠️ Error reading {file_path}: {e}")
                continue

            if "Timestamp" not in df.columns:
                logger.warning(f"⚠️ 'Timestamp' column missing in {file_path}; using full data.")
                ts_filtered = df
            else:
                # Coerce Timestamp to numeric seconds, drop rows with invalid timestamps
                ts = pd.to_numeric(df["Timestamp"], errors="coerce")
                valid = ts.notna()
                df = df.loc[valid].copy()
                ts = ts.loc[valid]

                if cutoff is not None and np.isfinite(cutoff):
                    mask = ts <= float(cutoff)
                    ts_filtered = df.loc[mask].copy()
                else:
                    ts_filtered = df

            # If nothing remains after filtering, skip
            if ts_filtered.empty:
                logger.warning(f"⚠️ No rows after time filtering for {file_path}; skipping.")
                continue

            # Drop Timestamp before aggregating values
            ts_filtered = ts_filtered.drop(columns=["Timestamp"], errors="ignore")

            # Parse list-like cells and collect finite numerics
            all_values = []
            for col in ts_filtered.columns:
                for val in ts_filtered[col]:
                    try:
                        nums = ast.literal_eval(val) if isinstance(val, str) else val
                        if isinstance(nums, list):
                            all_values.extend(
                                float(x) for x in nums
                                if isinstance(x, (int, float)) and np.isfinite(x)
                            )
                    except Exception:
                        # skip unparseable cells
                        continue

            if not all_values:
                logger.warning(f"⚠️ No numeric values found (after cutoff) in {file_path}; skipping.")
                continue

            averages[label] = float(summary_func(all_values))

        if not averages:
            raise ValueError("No valid numeric data found in videos (after applying time cutoffs).")

        # ---- Build pairwise relation matrix ----
        labels = list(averages.keys())
        n = len(labels)
        mat = np.full((n, n), np.nan, dtype=float)

        for i, a in enumerate(labels):
            va = averages[a]
            for j, b in enumerate(labels):
                vb = averages[b]
                if relation == "ratio":
                    mat[i, j] = (vb / va) if va not in (0, None) else np.nan
                elif relation == "diff":
                    mat[i, j] = vb - va
                else:
                    raise ValueError("relation must be 'ratio' or 'diff'")

        relation_df = pd.DataFrame(mat, index=labels, columns=labels)

        # ---- Add formatted decimal text for cells ----
        text_values = np.where(
            np.isnan(relation_df.values),
            "",
            np.round(relation_df.values, 2).astype(str)
        )

        # ---- Plotly heatmap (no cell text, no colorbar) ----
        fig = go.Figure(
            data=go.Heatmap(
                z=relation_df.values,
                x=relation_df.columns,
                y=relation_df.index,
                text=text_values,           # show decimals in cells
                texttemplate="%{text}",     # ensures text is rendered
                colorscale=colorscale,
                showscale=False,  # remove colorbar
                hovertemplate=(
                    "<b>%{y}</b> → <b>%{x}</b><br>"
                    f"{relation}: %{{z:.6f}}<extra></extra>"
                )
            )
        )

        fig.update_layout(
            title="",
            xaxis=dict(
                title="",
                tickangle=45,
                tickfont=dict(size=18, color="black"),
                titlefont=dict(size=16, color="black")
            ),
            yaxis=dict(
                title="",
                autorange="reversed",
                tickfont=dict(size=18, color="black"),
                titlefont=dict(size=16, color="black")
            ),
            margin=dict(l=80, r=40, t=60, b=90),
            width=2400,
            height=2400,
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        self.save_plotly(fig, 'heatmap', height=2400, width=2400, save_final=True)

        # One row per condition / label with the average trigger value
        summary_df = pd.DataFrame({
            "label": list(averages.keys()),
            "avg_trigger": list(averages.values())
        }).sort_values("label")

        trigger_summary_path = os.path.join(folder_path, "trigger_summary.csv")
        summary_df.to_csv(trigger_summary_path, index=False)
        logger.info(f"Saved trigger averages to: {trigger_summary_path}")

        # Optional: also save the pairwise relation matrix
        relation_path = os.path.join(folder_path, "trigger_relation_matrix.csv")
        relation_df.to_csv(relation_path)
        logger.info(f"Saved trigger relation matrix to: {relation_path}")

    def load_and_average_Q2(self, trigger_summary_csv: str, responses_root: str, mapping_df: pd.DataFrame,
                            n_participants: int = 50, response_col_index: int = 2, save_combined: bool = True):
        """
        Combine per-condition trigger averages with averaged Q2 (distance question)
        per condition.

        Parameters
        ----------
        trigger_summary_csv : str
            Path to 'trigger_summary.csv' saved by `heat_plot`.
            Must have columns: ['label', 'avg_trigger'] where 'label' matches
            mapping_df['condition_name'] (or video_id if you used that as label).
        responses_root : str
            Root folder containing Participant_1, Participant_2, ... subfolders.
        mapping_df : pd.DataFrame
            Must contain at least ['video_id', 'condition_name'].
        n_participants : int
            Total number of participants (default 50).
        response_col_index : int
            Zero-based index of the column containing the distance question (Q2).
            With lines like 'video_25,0,41,71', index 2 corresponds to 41.
        save_combined : bool
            If True, save condition-level summary to
            responses_root/condition_level_trigger_Q2.csv

        Returns
        -------
        trial_df : pd.DataFrame
            Trial-level data with columns:
            ['participant', 'video_id', 'condition_name', 'Q_distance', 'avg_trigger']

        condition_df : pd.DataFrame
            Condition-level summary with columns:
            ['condition_name', 'avg_trigger', 'mean_Q2', 'std_Q2', 'n_trials']
        """
        # --- Load trigger summary (condition -> avg_trigger) --- #
        trigger_df = pd.read_csv(trigger_summary_csv)

        if "label" not in trigger_df.columns or "avg_trigger" not in trigger_df.columns:
            raise ValueError(
                f"trigger_summary_csv must have columns ['label', 'avg_trigger'], "
                f"got {trigger_df.columns.tolist()}"
            )

        # Rename 'label' to 'condition_name' to match mapping_df
        trigger_df = trigger_df.rename(columns={"label": "condition_name"})
        trigger_df["condition_name"] = trigger_df["condition_name"].astype(str)

        # --- Prepare mapping: video_id -> condition_name --- #
        map_df = mapping_df[["video_id", "condition_name"]].copy()
        map_df["video_id"] = map_df["video_id"].astype(str)
        map_df["condition_name"] = map_df["condition_name"].astype(str)

        # --- Read all participant response CSVs (no headers) --- #
        all_records = []

        for pid in range(1, n_participants + 1):
            participant_folder = os.path.join(responses_root, f"Participant_{pid}")
            if not os.path.isdir(participant_folder):
                logger.warning(f"Participant folder not found: {participant_folder}")
                continue

            pattern = os.path.join(participant_folder, f"Participant_{pid}_*.csv")
            files = glob.glob(pattern)
            if not files:
                logger.warning(f"No response CSVs for Participant {pid} with pattern {pattern}")
                continue

            for fp in files:
                df = pd.read_csv(fp, header=None)

                # Need at least: trial_name (col 0) + Q2 column
                if df.shape[1] <= response_col_index:
                    logger.warning(
                        f"File {fp} has only {df.shape[1]} columns, "
                        f"cannot take column index {response_col_index}."
                    )
                    continue

                tmp = df[[0, response_col_index]].copy()
                tmp.columns = ["video_id", "Q_distance"]
                tmp["participant"] = pid

                # Keep only actual trials (video_*) and drop baselines
                mask = tmp["video_id"].astype(str).str.startswith("video_")
                tmp = tmp.loc[mask]

                all_records.append(tmp)

        if not all_records:
            raise ValueError("No participant response data found. Check paths / patterns.")

        trial_df = pd.concat(all_records, ignore_index=True)

        # --- Attach condition_name via mapping_df (video_id -> condition_name) --- #
        trial_df["video_id"] = trial_df["video_id"].astype(str)
        trial_df = trial_df.merge(
            map_df,
            on="video_id",
            how="left",
        )

        # --- Attach avg_trigger per condition_name --- #
        trial_df = trial_df.merge(
            trigger_df,
            on="condition_name",
            how="left",
        )

        # Now trial_df has:
        # participant, video_id, Q_distance, condition_name, avg_trigger

        # --- Average Q2 per condition (and pick avg_trigger per condition) --- #
        condition_df = (
            trial_df
            .groupby("condition_name", as_index=False)
            .agg(
                avg_trigger=("avg_trigger", "mean"),  # same value repeated; mean is fine
                mean_Q2=("Q_distance", "mean"),
                std_Q2=("Q_distance", "std"),
                n_trials=("Q_distance", "size"),
            )
            .sort_values("condition_name")
        )

        if save_combined:
            out_path = os.path.join(responses_root, "condition_level_trigger_Q2.csv")
            condition_df.to_csv(out_path, index=False)
            logger.info(f"Saved condition-level trigger + Q2 data to: {out_path}")

    def analyze_and_plot_distance_effect_plotly(self, condition_df, mapping_df, out_dir=None):
        """
        Merge condition-level averages with distance, yielding, eHMI, and camera,
        compute full-factorial summaries, and create Plotly figures.

        Assumes
        -------
        condition_df has at least:
            ['condition_name', 'avg_trigger', 'mean_Q2']
            - avg_trigger: proportion of time trigger was held (unsafe) per condition.
        mapping_df has at least:
            ['condition_name', 'distPed', 'yielding', 'eHMIOn', 'camera']

        Interpretation
        --------------
        - avg_trigger is scaled to 0–100 and interpreted as:
            "Mean perceived crossing risk (0–100)".
        - mean_Q2 is interpreted as:
            "Self-reported influence of other pedestrian (0–100)".
        """

        # Helper: turn "var=value" facet titles into just "value"
        def _strip_facet_equals(fig):
            fig.for_each_annotation(
                lambda a: a.update(text=a.text.split("=", 1)[-1].strip())
            )

        # --- REQUIRED columns ---
        required_cols = ["condition_name", "distPed", "yielding", "eHMIOn", "camera"]
        missing = [c for c in required_cols if c not in mapping_df.columns]
        if missing:
            raise ValueError(f"mapping_df missing: {missing}")

        needed_cond_cols = ["condition_name", "avg_trigger", "mean_Q2"]
        missing2 = [c for c in needed_cond_cols if c not in condition_df.columns]
        if missing2:
            raise ValueError(f"condition_df missing: {missing2}")

        # --- Prepare mapping info ---
        mapping_df = mapping_df.copy()
        mapping_df["condition_name"] = mapping_df["condition_name"].astype(str)

        cond_plot_df = condition_df.copy()
        cond_plot_df["condition_name"] = cond_plot_df["condition_name"].astype(str)

        # Merge mapping info onto condition-level data
        cond_plot_df = cond_plot_df.merge(
            mapping_df[required_cols],
            on="condition_name",
            how="left",
        )

        # Convert coded distance values to actual meters (1→2 m, 2→4 m, ...)
        cond_plot_df["distPed_m"] = cond_plot_df["distPed"] * 2

        # Scale trigger to 0–100: "Mean perceived crossing risk (0–100)"
        cond_plot_df["crossing_risk"] = cond_plot_df["avg_trigger"] * 100.0

        # Drop if some factors missing
        cond_plot_df = cond_plot_df.dropna(
            subset=["distPed", "yielding", "eHMIOn", "camera"]
        )

        # Label maps for binary factors (0/1 → text)
        label_map_yield = {0: "Not yielding", 1: "Yielding"}
        label_map_ehmi = {0: "eHMI off", 1: "eHMI on"}
        label_map_cam = {0: "Can see other person", 1: "Cannot see other person"}

        cond_plot_df["yielding_label"] = cond_plot_df["yielding"].map(label_map_yield)
        cond_plot_df["eHMI_label"] = cond_plot_df["eHMIOn"].map(label_map_ehmi)
        cond_plot_df["camera_label"] = cond_plot_df["camera"].map(label_map_cam)

        # ============================
        # Full-factorial summary (means only, no SE)
        # ============================
        group_cols = ["distPed_m", "yielding", "eHMIOn", "camera"]

        by_cond = (
            cond_plot_df
            .groupby(group_cols, as_index=False)
            .agg(
                mean_crossing_risk=("crossing_risk", "mean"),  # Mean perceived crossing risk
                Q2_mean=("mean_Q2", "mean"),                   # Self-report (Q2)
            )
            .sort_values(group_cols)
        )

        # Add label columns for plotting facets
        by_cond["yielding_label"] = by_cond["yielding"].map(label_map_yield)
        by_cond["eHMI_label"] = by_cond["eHMIOn"].map(label_map_ehmi)
        by_cond["camera_label"] = by_cond["camera"].map(label_map_cam)

        logger.info("\n=== Full-factorial condition table (NO SE, 0–100 scales) ===")
        logger.info(
            "\n=== Full-factorial condition table (NO SE, 0–100 scales) ===\n{}",
            by_cond.to_string(index=False),
        )
        logger.info("===========================================================\n")

        # Common label mapping for all figures
        base_labels = {
            "distPed_m": "Distance between pedestrians (m)",
            "crossing_risk": "Mean perceived crossing risk (0–100)",
            "mean_crossing_risk": "Mean perceived crossing risk (0–100)",
            "avg_trigger": "Mean perceived crossing risk (0–100)",
            "Q2_mean": "Self-reported influence of other pedestrian (0–100)",
            "mean_Q2": "Self-reported influence of other pedestrian (0–100)",
            "camera_label": "Camera",
            "yielding_label": "Yielding",
            "eHMI_label": "eHMI",
            "context": "Context (Y = yielding, H = eHMI, C = camera)",
            "delta": "Near–far difference (0–100)",
            "measure": "Measure",
        }

        # Category ordering for cleaner facets / legends
        category_orders = {
            "eHMI_label": ["eHMI off", "eHMI on"],
            "yielding_label": ["Not yielding", "Yielding"],
            "camera_label": ["Can see other person", "Cannot see other person"],
        }

        # ============================
        # Figure 1 — Mean crossing risk vs Distance (legend = camera)
        # ============================
        fig_beh = px.line(
            by_cond,
            x="distPed_m",
            y="mean_crossing_risk",
            color="camera_label",
            facet_col="eHMI_label",
            facet_row="yielding_label",
            markers=True,
            labels=base_labels,
            category_orders=category_orders,
            title="",
        )
        fig_beh.update_layout(template="plotly_white")
        _strip_facet_equals(fig_beh)
        fig_beh.update_layout(legend_title_text="")

        # ============================
        # Figure 2 — Q2 vs Distance (legend = camera)
        # ============================
        fig_q2 = px.line(
            by_cond,
            x="distPed_m",
            y="Q2_mean",
            color="camera_label",
            facet_col="eHMI_label",
            facet_row="yielding_label",
            markers=True,
            labels=base_labels,
            category_orders=category_orders,
            title="",
        )
        fig_q2.update_layout(template="plotly_white")
        _strip_facet_equals(fig_q2)
        fig_q2.update_layout(legend_title_text="")

        # ============================
        # EXTRA Figure A — Mean crossing risk vs distance, legend = yielding
        # ============================
        fig_beh_yield = px.line(
            by_cond,
            x="distPed_m",
            y="mean_crossing_risk",
            color="yielding_label",
            facet_col="eHMI_label",
            facet_row="camera_label",
            markers=True,
            labels=base_labels,
            category_orders=category_orders,
            title="",
        )
        fig_beh_yield.update_layout(template="plotly_white")
        _strip_facet_equals(fig_beh_yield)
        fig_beh_yield.update_layout(legend_title_text="")

        # ============================
        # EXTRA Figure B — Mean crossing risk vs distance, legend = eHMI
        # ============================
        fig_beh_ehmi = px.line(
            by_cond,
            x="distPed_m",
            y="mean_crossing_risk",
            color="eHMI_label",
            facet_col="yielding_label",
            facet_row="camera_label",
            markers=True,
            labels=base_labels,
            category_orders=category_orders,
            title="",
        )
        fig_beh_ehmi.update_layout(template="plotly_white")
        _strip_facet_equals(fig_beh_ehmi)
        fig_beh_ehmi.update_layout(legend_title_text="")

        # ============================
        # EXTRA Figure C — Q2 vs distance, legend = yielding
        # ============================
        fig_q2_yield = px.line(
            by_cond,
            x="distPed_m",
            y="Q2_mean",
            color="yielding_label",
            facet_col="eHMI_label",
            facet_row="camera_label",
            markers=True,
            labels=base_labels,
            category_orders=category_orders,
            title="",
        )
        fig_q2_yield.update_layout(template="plotly_white")
        _strip_facet_equals(fig_q2_yield)
        fig_q2_yield.update_layout(legend_title_text="")

        # ============================
        # EXTRA Figure D — Q2 vs distance, legend = eHMI
        # ============================
        fig_q2_ehmi = px.line(
            by_cond,
            x="distPed_m",
            y="Q2_mean",
            color="eHMI_label",
            facet_col="yielding_label",
            facet_row="camera_label",
            markers=True,
            labels=base_labels,
            category_orders=category_orders,
            title="",
        )
        fig_q2_ehmi.update_layout(template="plotly_white")
        _strip_facet_equals(fig_q2_ehmi)
        fig_q2_ehmi.update_layout(legend_title_text="")

        # ============================
        # Figure 3 — Mean crossing risk vs Q2 scatter (condition-level)
        # ============================
        fig_scatter = px.scatter(
            cond_plot_df,
            x="crossing_risk",
            y="mean_Q2",
            color="distPed_m",
            labels=base_labels,
            title="",
        )

        x_vals = cond_plot_df["crossing_risk"].values
        y_vals = cond_plot_df["mean_Q2"].values
        if len(x_vals) >= 2 and np.isfinite(x_vals).all() and np.isfinite(y_vals).all():
            b1, b0 = np.polyfit(x_vals, y_vals, 1)
            xs = np.linspace(x_vals.min(), x_vals.max(), 100)
            ys = b0 + b1 * xs
            fig_scatter.add_trace(
                go.Scatter(x=xs, y=ys, mode="lines", name="Trend")
            )

        fig_scatter.update_layout(template="plotly_white")

        # ============================
        # Figure 4 — NEAR (1–2 m) minus FAR (4–5 m) per context
        # ============================
        ctx_cols = ["yielding", "eHMIOn", "camera"]

        near = (
            by_cond[by_cond["distPed_m"].isin([1, 2])]
            .groupby(ctx_cols, as_index=False)
            .agg(
                crossing_risk_near=("mean_crossing_risk", "mean"),
                Q2_near=("Q2_mean", "mean"),
            )
        )
        far = (
            by_cond[by_cond["distPed_m"].isin([4, 5])]
            .groupby(ctx_cols, as_index=False)
            .agg(
                crossing_risk_far=("mean_crossing_risk", "mean"),
                Q2_far=("Q2_mean", "mean"),
            )
        )

        diff_df = near.merge(far, on=ctx_cols, how="inner")
        diff_df["delta_crossing_risk"] = (
            diff_df["crossing_risk_near"] - diff_df["crossing_risk_far"]
        )
        diff_df["delta_Q2"] = diff_df["Q2_near"] - diff_df["Q2_far"]

        # Context string with on/off text instead of 0/1
        diff_df["context"] = diff_df.apply(
            lambda r: (
                f"Y{'on' if r['yielding'] == 1 else 'off'}_"
                f"H{'on' if r['eHMIOn'] == 1 else 'off'}_"
                f"C{int(r['camera'])}"
            ),
            axis=1,
        )

        logger.info("\n=== NEAR–FAR differences per context ===")
        logger.info(
            "\n=== NEAR–FAR differences per context ===\n{}",
            diff_df[["context", "delta_crossing_risk", "delta_Q2"]].to_string(
                index=False
            ),
        )
        logger.info("========================================\n")

        long_diff = diff_df.melt(
            id_vars=["context"],
            value_vars=["delta_crossing_risk", "delta_Q2"],
            var_name="measure",
            value_name="delta",
        )

        long_diff["measure"] = long_diff["measure"].map({
            "delta_crossing_risk": "Mean perceived crossing risk (0–100)",
            "delta_Q2": "Self-reported influence of other pedestrian (0–100)",
        })

        fig_diff = px.bar(
            long_diff,
            x="context",
            y="delta",
            color="measure",
            barmode="group",
            labels={**base_labels, "delta": "Near–far difference (0–100)"},
            title="",
        )
        fig_diff.update_layout(template="plotly_white")
        fig_diff.add_hline(y=0, line_dash="dash", line_color="black")

        # ============================
        # Stats summary
        # ============================
        corr = cond_plot_df["crossing_risk"].corr(cond_plot_df["mean_Q2"])
        logger.info(
            f"Correlation (Mean perceived crossing risk, Q2): r = {corr:.3f}"
        )

        xd = by_cond["distPed_m"].values
        yd_risk = by_cond["mean_crossing_risk"].values
        slope_risk, intercept_risk = np.polyfit(xd, yd_risk, 1)
        logger.info(
            "Overall Mean perceived crossing risk vs distance: "
            f"slope = {slope_risk:.4f} (risk units per 1 m)"
        )

        yd_q2 = by_cond["Q2_mean"].values
        slope_q2, intercept_q2 = np.polyfit(xd, yd_q2, 1)
        logger.info(
            "Overall Q2 vs distance: "
            f"slope = {slope_q2:.4f} (Q2 units per 1 m)"
        )

        # ============================
        # Save figures
        # ============================
        self.save_plotly(fig_beh, "crossing_risk_full_factorial", save_final=True)
        self.save_plotly(fig_q2, "Q2_full_factorial", save_final=True)
        self.save_plotly(fig_scatter, "crossing_risk_vs_Q2_scatter", save_final=True)
        self.save_plotly(
            fig_diff, "near_minus_far_crossing_risk_vs_Q2", save_final=True
        )
        self.save_plotly(
            fig_beh_yield, "crossing_risk_full_factorial_legend_yielding", save_final=True
        )
        self.save_plotly(
            fig_beh_ehmi, "crossing_risk_full_factorial_legend_eHMI", save_final=True
        )
        self.save_plotly(
            fig_q2_yield, "Q2_full_factorial_legend_yielding", save_final=True
        )
        self.save_plotly(
            fig_q2_ehmi, "Q2_full_factorial_legend_eHMI", save_final=True
        )

        # Optionally return data and figs if you want to use them later
        figs = {
            "crossing_risk_full_factorial": fig_beh,
            "Q2_full_factorial": fig_q2,
            "crossing_risk_vs_Q2_scatter": fig_scatter,
            "near_minus_far_crossing_risk_vs_Q2": fig_diff,
            "crossing_risk_full_factorial_legend_yielding": fig_beh_yield,
            "crossing_risk_full_factorial_legend_eHMI": fig_beh_ehmi,
            "Q2_full_factorial_legend_yielding": fig_q2_yield,
            "Q2_full_factorial_legend_eHMI": fig_q2_ehmi,
        }
        return cond_plot_df, by_cond, figs
