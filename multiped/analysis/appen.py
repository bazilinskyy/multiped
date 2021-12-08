# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import pandas as pd
import numpy as np
import datetime as dt
from collections import Counter

import crossing as cs

logger = cs.CustomLogger(__name__)  # use custom logger


class Appen:
    # pandas dataframe with extracted data
    appen_data = pd.DataFrame()
    # pandas dataframe with data per country
    countries_data = pd.DataFrame()
    # pickle file for saving data
    file_p = 'appen_data.p'
    # csv file for saving data
    file_csv = 'appen_data.csv'
    # csv file for saving country data
    file_country_csv = 'country_data.csv'
    # csv file for saving list of cheaters
    file_cheaters_csv = 'cheaters.csv'
    # mapping between appen column names and readable names
    columns_mapping = {'_started_at': 'start',
                       '_created_at': 'end',
                       'about_how_many_kilometers_miles_did_you_drive_in_the_last_12_months': 'milage',  # noqa: E501
                       'at_which_age_did_you_obtain_your_first_license_for_driving_a_car_or_motorcycle': 'year_license',  # noqa: E501
                       'have_you_read_and_understood_the_above_instructions': 'instructions',  # noqa: E501
                       'how_many_accidents_were_you_involved_in_when_driving_a_car_in_the_last_3_years_please_include_all_accidents_regardless_of_how_they_were_caused_how_slight_they_were_or_where_they_happened': 'accidents',  # noqa: E501
                       'how_often_do_you_do_the_following_becoming_angered_by_a_particular_type_of_driver_and_indicate_your_hostility_by_whatever_means_you_can': 'dbq1_anger',  # noqa: E501
                       'how_often_do_you_do_the_following_disregarding_the_speed_limit_on_a_motorway': 'dbq2_speed_motorway',  # noqa: E501
                       'how_often_do_you_do_the_following_disregarding_the_speed_limit_on_a_residential_road': 'dbq3_speed_residential',  # noqa: E501
                       'how_often_do_you_do_the_following_driving_so_close_to_the_car_in_front_that_it_would_be_difficult_to_stop_in_an_emergency': 'dbq4_headway',  # noqa: E501
                       'how_often_do_you_do_the_following_racing_away_from_traffic_lights_with_the_intention_of_beating_the_driver_next_to_you': 'dbq5_traffic_lights',  # noqa: E501
                       'how_often_do_you_do_the_following_sounding_your_horn_to_indicate_your_annoyance_with_another_road_user': 'dbq6_horn',  # noqa: E501
                       'how_often_do_you_do_the_following_using_a_mobile_phone_without_a_hands_free_kit': 'dbq7_mobile',  # noqa: E501
                       'if_you_answered_other_in_the_previous_question_please_decribe_the_place_where_you_located_now_below': 'place_other',  # noqa: E501
                       'if_you_answered_other_in_the_previous_question_please_decribe_your_input_device_below': 'device_other',  # noqa: E501
                       'in_which_type_of_place_are_you_located_now': 'place',
                       'if_you_answered_other_in_the_previous_question_please_describe_the_place_where_you_are_located_now_below': 'place_other',  # noqa: E501
                       'in_which_year_do_you_think_that_most_cars_will_be_able_to_drive_fully_automatically_in_your_country_of_residence': 'year_ad',  # noqa: E501
                       'on_average_how_often_did_you_drive_a_vehicle_in_the_last_12_months': 'driving_freq',  # noqa: E501
                       'please_provide_any_suggestions_that_could_help_engineers_to_build_safe_and_enjoyable_automated_cars': 'suggestions_ad',  # noqa: E501
                       'type_the_code_that_you_received_at_the_end_of_the_experiment': 'worker_code',  # noqa: E501
                       'what_is_your_age': 'age',
                       'what_is_your_gender': 'gender',
                       'what_is_your_primary_mode_of_transportation': 'mode_transportation',  # noqa: E501
                       'which_input_device_are_you_using_now': 'device',
                       'if_you_answered_other_in_the_previous_question_please_describe_your_input_device_below': 'device_other',  # noqa: E501
                       'as_a_driver_what_does_it_mean_to_you_when_a_pedestrian_makes_eye_contact_with_you': 'ec_driver',  # noqa: E501
                       'as_a_pedestrian_what_does_it_mean_to_you_when_a_driver_makes_eye_contact_with_you': 'ec_pedestrian',  # noqa: E501
                       'how_do_you_feel_about_the_following_communication_between_driver_and_pedestrian_is_important_for_road_safety': 'communication_importance'}  # noqa: E501

    def __init__(self,
                 file_data: list,
                 save_p: bool,
                 load_p: bool,
                 save_csv: bool):
        # file with raw data
        self.file_data = file_data
        # save data as pickle file
        self.save_p = save_p
        # load data as pickle file
        self.load_p = load_p
        # save data as csv file
        self.save_csv = save_csv

    def set_data(self, appen_data):
        """Setter for the data object.
        """
        old_shape = self.appen_data.shape  # store old shape for logging
        self.appen_data = appen_data
        logger.info('Updated appen_data. Old shape: {}. New shape: {}.',
                    old_shape,
                    self.appen_data.shape)

    def read_data(self, filter_data=True, clean_data=True):
        """Read data into an attribute.

        Args:
            filter_data (bool, optional): flag for filtering data.
            clean_data (bool, optional): clean data.

        Returns:
            dataframe: udpated dataframe.
        """
        # load data
        if self.load_p:
            df = cs.common.load_from_p(self.file_p,
                                       'appen data')
        # process data
        else:
            logger.info('Reading appen data from {}.', self.file_data)
            # load from csv
            df = pd.read_csv(self.file_data)
            # drop legcy worker code column
            df = df.drop('worker_code', axis=1)
            # drop _gold columns
            df = df.drop((x for x in df.columns.tolist() if '_gold' in x),
                         axis=1)
            # replace linebreaks
            df = df.replace('\n', '', regex=True)
            # rename columns to readable names
            df.rename(columns=self.columns_mapping, inplace=True)
            # convert to time
            df['start'] = pd.to_datetime(df['start'])
            df['end'] = pd.to_datetime(df['end'])
            df['time'] = (df['end'] - df['start']) / pd.Timedelta(seconds=1)
            # remove underscores in the beginning of column name
            df.columns = df.columns.str.lstrip('_')
            # clean data
            if clean_data:
                df = self.clean_data(df)
            # filter data
            if filter_data:
                df = self.filter_data(df)
            # mask IDs and IPs
            df = self.mask_ips_ids(df)
            # move worker_code to the front
            worker_code_col = df['worker_code']
            df.drop(labels=['worker_code'], axis=1, inplace=True)
            df.insert(0, 'worker_code', worker_code_col)
        # save to pickle
        if self.save_p:
            cs.common.save_to_p(self.file_p, df, 'appen data')
        # save to csv
        if self.save_csv:
            df.to_csv(cs.settings.output_dir + '/' + self.file_csv)
            logger.info('Saved appen data to csv file {}', self.file_csv)
        # assign to attribute
        self.appen_data = df
        # return df with data
        return df

    def filter_data(self, df):
        """Filter data based on the folllowing criteria:
            1. People who did not read instructions.
            2. People that are under 18 years of age.
            3. People who completed the study in under 5 min.
            4. People who completed the study from the same IP more than once
               (the 1st data entry is retained).
            5. People who used the same `worker_code` multiple times.
        """
        logger.info('Filtering appen data.')
        # people that did not read instructions
        df_1 = df.loc[df['instructions'] == 'no']
        logger.info('Filter-a1. People who did not read instructions: {}',
                    df_1.shape[0])
        # people that are underages
        df_2 = df.loc[df['age'] < 18]
        logger.info('Filter-a2. People that are under 18 years of age: {}',
                    df_2.shape[0])
        # People that took less than cs.common.get_configs('allowed_min_time')
        # minutes to complete the study
        df_3 = df.loc[df['time'] < cs.common.get_configs('allowed_min_time')]
        logger.info('Filter-a3. People who completed the study in under ' +
                    str(cs.common.get_configs('allowed_min_time')) +
                    ' sec: {}',
                    df_3.shape[0])
        # people that completed the study from the same IP address
        df_4 = df[df['ip'].duplicated(keep='first')]
        logger.info('Filter-a4. People who completed the study from the ' +
                    'same IP: {}',
                    df_4.shape[0])
        # people that entered the same worker_code more than once
        df_5 = df[df['worker_code'].duplicated(keep='first')]
        logger.info('Filter-a5. People who used the same worker_code: {}',
                    df_5.shape[0])
        # save to csv
        if self.save_csv:
            df_5 = df_5.reset_index()
            df_5.to_csv(cs.settings.output_dir + '/' + self.file_cheaters_csv)
            logger.info('Filter-a5. Saved list of cheaters to csv file {}',
                        self.file_cheaters_csv)
        # concatanate dfs with filtered data
        old_size = df.shape[0]
        df_filtered = pd.concat([df_1, df_2, df_3, df_4, df_5])
        # check if there are people to filter
        if not df_filtered.empty:
            # drop rows with filtered data
            unique_worker_codes = df_filtered['worker_code'].drop_duplicates()
            df = df[~df['worker_code'].isin(unique_worker_codes)]
            # reset index in dataframe
            df = df.reset_index()
        logger.info('Filtered in total in appen data: {}',
                    old_size - df.shape[0])
        # assign to attribute
        self.appen_data = df
        # return df with data
        return df

    def clean_data(self, df, clean_years=True):
        """Clean data from unexpected values.

        Args:
            df (dataframe): dataframe with data.
            clean_years (bool, optional): clean years question by removing
                                          unrealistic answers.

        Returns:
            dataframe: updated dataframe.
        """
        logger.info('Cleaning appen data.')
        if clean_years:
            # get current number of nans
            nans_before = np.zeros(3, dtype=np.int8)
            nans_before[0] = df['year_ad'].isnull().sum()
            nans_before[1] = df['year_license'].isnull().sum()
            nans_before[2] = df['age'].isnull().sum()
            # replace all non-numeric values to nan for questions invlolving
            # years
            df['year_ad'] = df['year_ad'].apply(
                lambda x: pd.to_numeric(x, errors='coerce'))
            df['year_license'] = df['year_license'].apply(
                lambda x: pd.to_numeric(x, errors='coerce'))
            df['age'] = df['age'].apply(
                lambda x: pd.to_numeric(x, errors='coerce'))
            logger.info('Clean-a1. Replaced {} non-numeric values in columns'
                        + ' year_ad, {} non-numeric values in column'
                        + ' year_license, {} non-numeric values in column'
                        + ' age.',
                        df['year_ad'].isnull().sum() - nans_before[0],
                        df['year_license'].isnull().sum() - nans_before[1],
                        df['age'].isnull().sum() - nans_before[2])
            # replace number of nans
            nans_before[0] = df['year_ad'].isnull().sum()
            nans_before[1] = df['year_license'].isnull().sum()
            nans_before[2] = df['age'].isnull().sum()
            # get current year
            now = dt.datetime.now()
            # year of introduction of automated driving cannot be in the past
            # and unrealistically large values are removed
            df.loc[df['year_ad'] < now.year, 'year_ad'] = np.nan
            df.loc[df['year_ad'] > 2300,  'year_ad'] = np.nan
            # year of obtaining driver's license is assumed to be always < 70
            df.loc[df['year_license'] >= 70] = np.nan
            # age is assumed to be always < 100
            df.loc[df['age'] >= 100] = np.nan
            logger.info('Clean-a2. Cleaned {} values of years in column'
                        + ' year_ad, {} values of years in column year_license'
                        + ' , {} values in column age.',
                        df['year_ad'].isnull().sum() - nans_before[0],
                        df['year_license'].isnull().sum() - nans_before[1],
                        df['age'].isnull().sum() - nans_before[2])
        # assign to attribute
        self.appen_data = df
        # return df with data
        return df

    def mask_ips_ids(self, df, mask_ip=True, mask_id=True):
        """Anonymyse IPs and IDs. IDs are anonymised by subtracting the
        given ID from cs.common.get_configs('mask_id').
        """
        # loop through rows of the file
        if mask_ip:
            proc_ips = []  # store masked IP's here
            logger.info('Replacing IPs in appen data.')
        if mask_id:
            proc_ids = []  # store masked ID's here
            logger.info('Replacing IDs in appen data.')
        for i in range(len(df['ip'])):  # loop through ips
            # anonymise IPs
            if mask_ip:
                # IP address
                # new IP
                if not any(d['o'] == df['ip'][i] for d in proc_ips):
                    # mask in format 0.0.0.ID
                    masked_ip = '0.0.0.' + str(len(proc_ips))
                    # record IP as already replaced
                    # o=original; m=masked
                    proc_ips.append({'o': df['ip'][i], 'm': masked_ip})
                    df.at[i, 'ip'] = masked_ip
                    logger.debug('{}: replaced IP {} with {}.',
                                 df['worker_code'][i],
                                 proc_ips[-1]['o'],
                                 proc_ips[-1]['m'])
                else:  # already replaced
                    for item in proc_ips:
                        if item['o'] == df['ip'][i]:

                            # fetch previously used mask for the IP
                            df.at[i, 'ip'] = item['m']
                            logger.debug('{}: replaced repeating IP {} with ' +
                                         '{}.',
                                         df['worker_code'][i],
                                         item['o'],
                                         item['m'])
            # anonymise worker IDs
            if mask_id:
                # new worker ID
                if not any(d['o'] == df['worker_id'][i] for d in proc_ids):
                    # mask in format random_int - worker_id
                    masked_id = (str(cs.common.get_configs('mask_id') -
                                     df['worker_id'][i]))
                    # record IP as already replaced
                    proc_ids.append({'o': df['worker_id'][i],
                                     'm': masked_id})
                    df.at[i, 'worker_id'] = masked_id
                    logger.debug('{}: replaced ID {} with {}.',
                                 df['worker_code'][i],
                                 proc_ids[-1]['o'],
                                 proc_ids[-1]['m'])
                # already replaced
                else:
                    for item in proc_ids:
                        if item['o'] == df['worker_id'][i]:
                            # fetch previously used mask for the ID
                            df.at[i, 'worker_id'] = item['m']
                            logger.debug('{}: replaced repeating ID {} '
                                         + 'with {}.',
                                         df['worker_code'][i],
                                         item['o'],
                                         item['m'])
        # output for checking
        if mask_ip:
            logger.info('Finished replacement of IPs in appen data.')
            logger.info('Unique IPs detected: {}', str(len(proc_ips)))
        if mask_id:
            logger.info('Finished replacement of IDs in appen data.')
            logger.info('Unique IDs detected: {}', str(len(proc_ids)))
        # return dataframe with replaced values
        return df

    def process_countries(self):
        # todo: map textual questions to int
        # df for reassignment of textual values
        df = self.appen_data
        # df for storing counts
        df_counts = pd.DataFrame()
        # get countries and counts of participants
        df_counts['counts'] = df['country'].value_counts()
        # set i_prefer_not_to_respond as nan
        df[df == 'i_prefer_not_to_respond'] = np.nan
        # map gender
        di = {'female': 0, 'male': 1}
        df = df.replace({'gender': di})
        # get mean values for countries
        df_country = df.groupby('country').mean(
            numeric_only=True).reset_index()
        # use median for year
        df_country['year_ad'] = df.groupby('country').median(
            numeric_only=True).reset_index()['year_ad']
        df_country['year_license'] = df.groupby('country').median(
            numeric_only=True).reset_index()['year_license']
        # assign counts after manipulations
        df_country = df_country.set_index('country', drop=False)
        df_country = df_country.merge(df_counts,
                                      left_index=True,
                                      right_index=True,
                                      how='left')
        # drop not needed columns
        df_country.drop(['unit_id', 'id', 'tainted', 'worker_id'], 1)
        # assign to attribute
        self.countries_data = df_country
        # save to csv
        if self.save_csv:
            df_country.to_csv(cs.settings.output_dir + '/'
                              + self.file_country_csv)
            logger.info('Saved country data to csv file {}', self.file_csv)
        # return df with data
        return df_country

    def show_info(self):
        """Output info for data in object.
        """
        # info on age
        logger.info('Age: mean={:,.2f}, std={:,.2f}',
                    self.appen_data['age'].mean(),
                    self.appen_data['age'].std())
        # info on gender
        count = Counter(self.appen_data['gender'])
        logger.info('Gender: {}', count.most_common())
        # info on most represted countries in minutes
        count = Counter(self.appen_data['country'])
        logger.info('Countires: {}', count.most_common())
        # info on duration in minutes
        logger.info('Time of participation: mean={:,.2f} min, '
                    + 'median={:,.2f} min, std={:,.2f} min.',
                    self.appen_data['time'].mean() / 60,
                    self.appen_data['time'].median() / 60,
                    self.appen_data['time'].std() / 60)
        logger.info('oldest timestamp={}, newest timestamp={}.',
                    self.appen_data['start'].min(),
                    self.appen_data['start'].max())
