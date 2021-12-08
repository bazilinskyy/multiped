# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import ast
from statistics import mean
import warnings

import crossing as cs

# warning about partial assignment
pd.options.mode.chained_assignment = None  # default='warn'

logger = cs.CustomLogger(__name__)  # use custom logger


# todo: parse browser interactions
class Heroku:
    # pandas dataframe with extracted data
    heroku_data = pd.DataFrame()
    # pandas dataframe with mapping
    mapping = pd.read_csv(cs.common.get_configs('mapping_stimuli'))
    # resolution for keypress data
    res = cs.common.get_configs('kp_resolution')
    # number of stimuli
    num_stimuli = cs.common.get_configs('num_stimuli')
    # number of stimuli shown for each participant
    num_stimuli_participant = cs.common.get_configs('num_stimuli_participant')
    # number of repeats for each stimulus
    num_repeat = cs.common.get_configs('num_repeat')
    # allowed number of stimuli with detected wrong duration
    allowed_length = cs.common.get_configs('allowed_stimuli_wrong_duration')
    # allowed number of mistakes for questions with signs
    allowed_signs = cs.common.get_configs('allowed_mistakes_signs')
    # pickle file for saving data
    file_p = 'heroku_data.p'
    # csv file for saving data
    file_data_csv = 'heroku_data'
    # csv file for mapping of stimuli
    file_mapping_csv = 'mapping'
    # keys with meta information
    meta_keys = ['worker_code',
                 'browser_user_agent',
                 'browser_app_name',
                 'browser_major_version',
                 'browser_full_version',
                 'browser_name',
                 'window_height',
                 'window_width',
                 'video_ids']
    # prefixes used for files in node.js implementation
    prefixes = {'stimulus': 'video_'}
    # stimulus duration
    default_dur = 0

    def __init__(self,
                 files_data: list,
                 save_p: bool,
                 load_p: bool,
                 save_csv: bool):
        # list of files with raw data
        self.files_data = files_data
        # save data as pickle file
        self.save_p = save_p
        # load data as pickle file
        self.load_p = load_p
        # save data as csv file
        self.save_csv = save_csv

    def set_data(self, heroku_data):
        """Setter for the data object.
        """
        old_shape = self.heroku_data.shape  # store old shape for logging
        self.heroku_data = heroku_data
        logger.info('Updated heroku_data. Old shape: {}. New shape: {}.',
                    old_shape,
                    self.heroku_data.shape)

    def read_data(self, filter_data=True):
        """Read data into an attribute.

        Args:
            filter_data (bool, optional): flag for filtering data.

        Returns:
            dataframe: udpated dataframe.
        """
        # load data
        if self.load_p:
            df = cs.common.load_from_p(self.file_p,
                                       'heroku data')
        # process data
        else:
            # read files with heroku data one by one
            data_list = []
            data_dict = {}  # dictionary with data
            for file in self.files_data:
                logger.info('Reading heroku data from {}.', file)
                f = open(file, 'r')
                # add data from the file to the dictionary
                data_list += f.readlines()
                f.close()
            # hold info on previous row for worker
            prev_row_info = pd.DataFrame(columns=['worker_code',
                                                  'time_elapsed'])
            prev_row_info.set_index('worker_code', inplace=True)
            # read rows in data
            for row in tqdm(data_list):  # tqdm adds progress bar
                # use dict to store data
                dict_row = {}
                # load data from a single row into a list
                list_row = json.loads(row)
                # last found stimulus
                stim_name = ''
                # trial last found stimulus
                stim_trial = -1
                # last time_elapsed for logging duration of trial
                elapsed_l = 0
                # record worker_code in the row. assuming that each row has at
                # least one worker_code
                worker_code = [d['worker_code'] for d in list_row['data'] if 'worker_code' in d][0]  # noqa: E501
                # go over cells in the row with data
                for data_cell in list_row['data']:
                    # extract meta info form the call
                    for key in self.meta_keys:
                        if key in data_cell.keys():
                            # piece of meta data found, update dictionary
                            dict_row[key] = data_cell[key]
                            if key == 'worker_code':
                                logger.debug('{}: working with row with data.',
                                             data_cell['worker_code'])
                    # check if stimulus data is present
                    if 'stimulus' in data_cell.keys():
                        # extract name of stimulus after last slash
                        # list of stimuli. use 1st
                        if isinstance(data_cell['stimulus'], list):
                            stim_no_path = data_cell['stimulus'][0].rsplit('/', 1)[-1]  # noqa: E501
                        # single stimulus
                        else:
                            stim_no_path = data_cell['stimulus'].rsplit('/', 1)[-1]  # noqa: E501
                        # remove extension
                        stim_no_path = os.path.splitext(stim_no_path)[0]
                        # Check if it is a block with stimulus and not an
                        # instructions block
                        if (cs.common.search_dict(self.prefixes, stim_no_path)
                                is not None):
                            # stimulus is found
                            logger.debug('Found stimulus {}.', stim_no_path)
                            if self.prefixes['stimulus'] in stim_no_path:
                                # Record that stimulus was detected for the
                                # cells to follow
                                stim_name = stim_no_path
                                # record trial of stimulus
                                stim_trial = data_cell['trial_index']
                                # add trial duration
                                if 'time_elapsed' in data_cell.keys():
                                    # positive time elapsed from las cell
                                    if elapsed_l:
                                        time = elapsed_l
                                    # non-positive time elapsed. use value from
                                    # the known cell for worker
                                    else:
                                        time = prev_row_info.loc[worker_code, 'time_elapsed']  # noqa: E501
                                    # calculate duration
                                    dur = float(data_cell['time_elapsed']) - time  # noqa: E501
                                    if stim_name + '-dur' not in dict_row.keys() and dur > 0:  # noqa: E501
                                        # first value
                                        dict_row[stim_name + '-dur'] = dur
                    # keypresses
                    if 'rts' in data_cell.keys() and stim_name != '':
                        # record given keypresses
                        responses = data_cell['rts']
                        logger.debug('Found {} points in keypress data.',
                                     len(responses))
                        # extract pressed keys and rt values
                        key = [point['key'] for point in responses]
                        rt = [point['rt'] for point in responses]
                        # check if values were recorded previously
                        if stim_name + '-key' not in dict_row.keys():
                            # first value
                            dict_row[stim_name + '-key'] = key
                        else:
                            # previous values found
                            dict_row[stim_name + '-key'].extend(key)
                        # check if values were recorded previously
                        if stim_name + '-rt' not in dict_row.keys():
                            # first value
                            dict_row[stim_name + '-rt'] = rt
                        else:
                            # previous values found
                            dict_row[stim_name + '-rt'].extend(rt)
                    # questions after stimulus
                    if 'responses' in data_cell.keys() and stim_name != '':
                        # record given keypresses
                        responses = data_cell['responses']
                        logger.debug('Found responses to questions {}.',
                                     responses)
                        # extract pressed keys and rt values
                        responses = ast.literal_eval(re.search('({.+})',
                                                               responses).group(0))  # noqa: E501
                        # unpack questions and answers
                        questions = []
                        answers = []
                        for key, value in responses.items():
                            questions.append(key)
                            answers.append(value)
                        # check if values were recorded previously
                        if stim_name + '-qs' not in dict_row.keys():
                            # first value
                            dict_row[stim_name + '-qs'] = questions
                        else:
                            # previous values found
                            dict_row[stim_name + '-qs'].extend(questions)
                        # Check if time spent values were recorded
                        # previously
                        if stim_name + '-as' not in dict_row.keys():
                            # first value
                            dict_row[stim_name + '-as'] = answers
                        else:
                            # previous values found
                            dict_row[stim_name + '-as'].extend(answers)
                    # browser interaction events
                    if 'interactions' in data_cell.keys() and stim_name != '':
                        interactions = data_cell['interactions']
                        logger.debug('Found {} browser interactions.',
                                     len(interactions))
                        # extract events and timestamps
                        event = []
                        time = []
                        for interation in interactions:
                            if interation['trial'] == stim_trial:
                                event.append(interation['event'])
                                time.append(interation['time'])
                        # Check if inputted values were recorded previously
                        if stim_name + '-event' not in dict_row.keys():
                            # first value
                            dict_row[stim_name + '-event'] = event
                        else:
                            # previous values found
                            dict_row[stim_name + '-event'].extend(event)
                        # check if values were recorded previously
                        if stim_name + '-time' not in dict_row.keys():
                            # first value
                            dict_row[stim_name + '-time'] = time
                        else:
                            # previous values found
                            dict_row[stim_name + '-time'].extend(time)
                    # questions in the end
                    if 'responses' in data_cell.keys() and stim_name == '':
                        # record given keypresses
                        responses_end = data_cell['responses']
                        logger.debug('Found responses to final questions {}.',
                                     responses_end)
                        # extract pressed keys and rt values
                        responses_end = ast.literal_eval(re.search('({.+})',
                                                         responses_end).group(0))  # noqa: E501
                        # unpack questions and answers
                        questions = []
                        answers = []
                        for key, value in responses_end.items():
                            questions.append(key)
                            answers.append(value)
                        # Check if inputted values were recorded previously
                        if 'end-qs' not in dict_row.keys():
                            dict_row['end-qs'] = questions
                            dict_row['end-as'] = answers
                        else:
                            # previous values found
                            dict_row['end-qs'].extend(questions)
                            dict_row['end-as'].extend(answers)
                    # question order
                    if 'question_order' in data_cell.keys() \
                       and stim_name == '':
                        # unpack question order
                        qo_str = data_cell['question_order']
                        # remove brackets []
                        qo_str = qo_str[1:]
                        qo_str = qo_str[:-1]
                        # unpack to int
                        question_order = [int(x) for x in qo_str.split(',')]
                        logger.debug('Found question order for final ' +
                                     'questions {}.',
                                     question_order)
                        # Check if inputted values were recorded previously
                        if 'end-qo' not in dict_row.keys():
                            dict_row['end-qo'] = question_order
                        else:
                            # previous values found
                            dict_row['end-qo'].extend(question_order)
                    # record last time_elapsed
                    if 'time_elapsed' in data_cell.keys():
                        elapsed_l = float(data_cell['time_elapsed'])
                # update last time_elapsed for worker
                prev_row_info.loc[dict_row['worker_code'], 'time_elapsed'] = elapsed_l  # noqa: E501
                # worker_code was encountered before
                if dict_row['worker_code'] in data_dict.keys():
                    # iterate over items in the data dictionary
                    for key, value in dict_row.items():
                        # worker_code does not need to be added
                        if key in self.meta_keys:
                            continue
                        # new value
                        if key + '-0' not in data_dict[dict_row['worker_code']].keys():  # noqa: E501
                            data_dict[dict_row['worker_code']][key + '-0'] = value  # noqa: E501
                        # update old value
                        else:
                            # traverse repetition ids untill get new repetition
                            for rep in range(0, self.num_repeat):
                                # build new key with id of repetition
                                new_key = key + '-' + str(rep)
                                if new_key not in data_dict[dict_row['worker_code']].keys():  # noqa: E501
                                    data_dict[dict_row['worker_code']][new_key] = value  # noqa: E501
                                    break
                # worker_code is encountered for the first time
                else:
                    # iterate over items in the data dictionary and add -0
                    for key, value in list(dict_row.items()):
                        # worker_code does not need to be added
                        if key in self.meta_keys:
                            continue
                        # new value
                        dict_row[key + '-0'] = dict_row.pop(key)
                    # add row of data
                    data_dict[dict_row['worker_code']] = dict_row
            # turn into pandas dataframe
            df = pd.DataFrame(data_dict)
            df = df.transpose()
            # report people that attempted study
            unique_worker_codes = df['worker_code'].drop_duplicates()
            logger.info('People who attempted to participate: {}',
                        unique_worker_codes.shape[0])
            # filter data
            if filter_data:
                df = self.filter_data(df)
            # sort columns alphabetically
            df = df.reindex(sorted(df.columns), axis=1)
            # move worker_code to the front
            worker_code_col = df['worker_code']
            df.drop(labels=['worker_code'], axis=1, inplace=True)
            df.insert(0, 'worker_code', worker_code_col)
        # save to pickle
        if self.save_p:
            cs.common.save_to_p(self.file_p, df, 'heroku data')
        # save to csv
        if self.save_csv:
            # todo: check whith index=False is needed here
            df.to_csv(cs.settings.output_dir + '/' + self.file_data_csv +
                      '.csv', index=False)
            logger.info('Saved heroku data to csv file {}',
                        self.file_data_csv + '.csv')
        # update attribute
        self.heroku_data = df
        # return df with data
        return df

    def read_mapping(self):
        """
        Read mapping.
        """
        # read mapping from a csv file
        df = pd.read_csv(cs.common.get_configs('mapping_stimuli'))
        # set index as stimulus_id
        df.set_index('video_id', inplace=True)
        # update attribute
        self.mapping = df
        # return mapping as a dataframe
        return df

    def process_kp(self, filter_length=True):
        """Process keypresses for resolution self.res.

        Returns:
            mapping: updated mapping df.

        Args:
            filter_length (bool, optional): filter out stimuli with unexpected
                                            length.
        """
        logger.info('Processing keypress data with res={} ms.', self.res)
        # array to store all binned rt data in
        mapping_rt = []
        # counter of videos filtered because of length
        counter_filtered = 0
        # loop through all stimuli
        for num in tqdm(range(self.num_stimuli)):
            video_kp = []
            # video ID
            video_id = 'video_' + str(num)
            for rep in range(self.num_repeat):
                # add suffix with repetition ID
                video_rt = 'video_' + str(num) + '-rt-' + str(rep)
                video_dur = 'video_' + str(num) + '-dur-' + str(rep)
                # extract video length
                video_len = self.mapping.loc[video_id]['video_length']
                rt_data = []
                counter_data = 0
                for (col_name, col_data) in self.heroku_data.iteritems():
                    # find the right column to loop through
                    if video_rt == col_name:
                        # loop through rows in column
                        for row_index, row in enumerate(col_data):
                            # consider only videos of allowed length
                            if (video_dur in self.heroku_data.keys()
                                    and filter_length):
                                # extract recorded duration
                                dur = self.heroku_data.iloc[row_index][video_dur]  # noqa: E501
                                # check if duration is within limits
                                if (dur < self.mapping['min_dur'][video_id]
                                        or dur > self.mapping['max_dur'][video_id]):  # noqa: E501
                                    # increase counter of filtered videos
                                    logger.debug('Filtered keypress data from '
                                                 + 'video {} of detected '
                                                 + 'duration of {} for '
                                                 + 'worker {}.',
                                                 video_id, dur,
                                                 self.heroku_data.index[row_index])  # noqa: E501
                                    # increase counter of filtered videos
                                    counter_filtered = counter_filtered + 1
                                    continue
                            # check if data is string to filter out nan data
                            if type(row) == list:
                                # saving amount of times the video has been
                                # watched
                                counter_data = counter_data + 1
                                # if list contains only one value, append to
                                # rt_data
                                if len(row) == 1:
                                    rt_data.append(row[0])
                                # if list contains more then one value, go
                                # through list to remove keyholds
                                elif len(row) > 1:
                                    for j in range(1, len(row)):
                                        # if time between 2 stimuli is more
                                        # than 35 ms, add to array (no hold)
                                        if row[j] - row[j - 1] > 35:
                                            # append buttonpress data to rt
                                            # array
                                            rt_data.append(row[j])
                        # if all data for one video was found, divide them in
                        # bins
                        kp = []
                        # loop over all bins, dependent on resolution
                        for rt in range(self.res, video_len + self.res,
                                        self.res):
                            bin_counter = 0
                            for data in rt_data:
                                # go through all video data to find all data
                                # within specific bin
                                if rt - self.res < data <= rt:
                                    # if data is found, up bin counter
                                    bin_counter = bin_counter + 1
                            if counter_data:
                                percentage = bin_counter / counter_data
                                kp.append(round(percentage * 100))
                            else:
                                kp.append(0)
                        # store keypresses from repetition
                        video_kp.append(kp)
                        break
            # calculate mean keypresses from all repetitions
            kp_mean = [*map(mean, zip(*video_kp))]
            # append data from one video to the mapping array
            mapping_rt.append(kp_mean)
        logger.info('Filtered out keypress data from {} videos with '
                    + 'unexpected length.', counter_filtered)
        # update own mapping to include keypress data
        self.mapping['kp'] = mapping_rt
        # save to csv
        if self.save_csv:
            # save to csv
            self.mapping.to_csv(cs.settings.output_dir + '/' +
                                self.file_mapping_csv + '.csv')
        # return new mapping
        return self.mapping

    def process_stimulus_questions(self, questions):
        """Process questions that follow each stimulus.

        Args:
            questions (list): list of questions with types of possible values
                              as int or str.

        Returns:
            dataframe: updated mapping dataframe.
        """
        logger.info('Processing post-stimulus questions')
        # array in which arrays of video_as data is stored
        mapping_as = []
        # loop through all stimuli
        for num in tqdm(range(self.num_stimuli)):
            # calculate length of of array with answers
            length = 0
            for q in questions:
                # 1 column required for numeric data
                # numberic answer, create 1 column to store mean value
                if q['type'] == 'num':
                    length = length + 1
                # strings as answers, create columns to store counts
                elif q['type'] == 'str':
                    length = length + len(q['options'])
                else:
                    logger.error('Wrong type of data {} in question {}' +
                                 'provided.', q['type'], q['question'])
                    return -1
            # array in which data of a single stimulus is stored
            answers = [[[] for i in range(self.heroku_data.shape[0])]
                       for i in range(len(questions))]
            # for number of repetitions in survey, add extra number
            for rep in range(self.num_repeat):
                # add suffix with repetition ID
                video_as = 'video_' + str(num) + '-as-' + str(rep)
                video_order = 'video_' + str(num) + '-qs-' + str(rep)
                # loop over columns
                for col_name, col_data in self.heroku_data.iteritems():
                    # when col_name equals video, then check
                    if col_name == video_as:
                        # loop over rows in column
                        for pp, row in enumerate(col_data):
                            # filter out empty values
                            if type(row) == list:
                                order = self.heroku_data.iloc[pp][video_order]  # noqa: E501
                                # check if injection question is present
                                if 'injection' in order:
                                    # delete injection
                                    del row[order.index('injection')]
                                    del order[order.index('injection')]
                                # loop through questions
                                for i, q in enumerate(questions):
                                    # extract answer
                                    ans = row[order.index(q['question'])]
                                    # store answer from repetition
                                    answers[i][pp].append(ans)
            # calculate mean answers from all repetitions for numeric questions
            for i, q in enumerate(questions):
                if q['type'] == 'num' and answers[i]:
                    # convert to float
                    answers[i] = [list(map(float, sublist))
                                  for sublist in answers[i]]
                    # calculate mean of mean of responses of each participant
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore',
                                              category=RuntimeWarning)
                        answers[i] = np.nanmean([np.nanmean(j)
                                                 for j in answers[i]])
            # save question data in array
            mapping_as.append(answers)
        # add column with data to current mapping file
        for i, q in enumerate(questions):
            # extract answers for the given question
            q_ans = [item[i] for item in mapping_as]
            # for numeric question, add column with mean values
            if q['type'] == 'num':
                self.mapping[q['question']] = q_ans
            # for textual question, add columns with counts of each value
            else:
                # go over options and count answers with the option for each
                # stimulus
                for option in q['options']:
                    # store counts in list
                    count_option = []
                    # go over each answer
                    for ans in q_ans:
                        # flatten list of answers
                        ans = [item for sublist in ans for item in sublist]
                        # add count for answers for the given option
                        count_option.append(ans.count(option))
                    # build name of column
                    col_name = q['question'] + '-' + option.replace(' ', '_')
                    col_name = col_name.lower()
                    # add to mapping
                    self.mapping[col_name] = count_option
        # save to csv
        if self.save_csv:
            # save to csv
            self.mapping.to_csv(cs.settings.output_dir + '/' +
                                self.file_mapping_csv + '.csv')
        # return new mapping
        return self.mapping

    def filter_data(self, df):
        """
        Filter data based on the folllowing criteria:
            1. People who had more than allowed_stimuli share of stimuli of
               unexpected length.
            2. People who made more than allowed_mistakes_signs mistakes with
               questions of traffic sign.

        Args:
            df (dataframe): dataframe with data.

        Returns:
            dataframe: updated dataframe.
        """
        # more than allowed number of mistake with codes for sentinel images
        # load mapping of codes and coordinates
        logger.info('Filtering heroku data.')
        # 1. People who made mistakes in injected questions
        logger.info('Filter-h1. People who had too many stimuli of unexpected'
                    + ' length.')
        # df to store data to filter out
        df_1 = pd.DataFrame()
        # array to store in video names
        video_dur = []
        for i in range(0, self.num_stimuli):
            for rep in range(0, self.num_repeat):
                video_dur.append('video_' + str(i) + '-dur-' + str(rep))
        # tqdm adds progress bar
        # loop over participants in data
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            data_count = 0
            counter_filtered = 0
            for i in range(self.num_stimuli):
                for rep in range(self.num_repeat):
                    # add suffix with repetition ID
                    video_dur = 'video_' + str(i) + '-dur-' + str(rep)
                    # check id value is present
                    if video_dur not in row.keys():
                        continue
                    # check for nan values
                    if pd.isna(row[video_dur]):
                        continue
                    else:
                        # up data count when data is found
                        data_count = data_count + 1
                        if (row[video_dur] < (self.mapping['min_dur'].iloc[i])  # noqa: E501
                           or row[video_dur] > (self.mapping['max_dur'].iloc[i])):  # noqa: E501
                            # up counter if data with wrong length is found
                            counter_filtered = counter_filtered + 1
            # Only check for participants that watched all videos
            if data_count >= self.num_stimuli_participant * self.num_repeat:
                # check threshold ratio
                if counter_filtered / data_count > self.allowed_length:
                    # if threshold reached, append data of this participant to
                    # df_1
                    df_1 = df_1.append(row)
        logger.info('Filter-h1. People who had more than {} share of stimuli'
                    + ' of unexpected length: {}.',
                    self.allowed_length,
                    df_1.shape[0])
        # 2. People that made too many mistakes with questions with traffic
        # signs
        logger.info('Filter-h2. People who made too many mistakes with '
                    + 'questions of traffic signs.')
        # df to store data to filter out
        df_2 = pd.DataFrame()
        # answers to injected questions
        signs_answers = cs.common.get_configs('signs_answers')
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            counter_filtered = 0
            # get array of data about traffic signs for each pedestrian
            # check if value is list, so no nan
            if type(row['end-as-0']) == list:
                for count, data in enumerate(row['end-as-0']):
                    # answer-data starts at 5th element
                    if count > 4:
                        if data != signs_answers[count-5]:
                            # if wrong answer, up counter
                            counter_filtered = counter_filtered + 1
            if counter_filtered > self.allowed_signs:
                # append participant if too much mistakes in signs_answers
                df_2 = df_2.append(row)
        # people that made too many mistakes with questions with traffic signs
        logger.info('Filter-h2. People who made more than {} mistakes with '
                    + 'questions of traffic signs: {}',
                    self.allowed_signs,
                    df_2.shape[0])
        # concatanate dfs with filtered data
        old_size = df.shape[0]
        df_filtered = pd.concat([df_1, df_2])
        # check if there are people to filter
        if not df_filtered.empty:
            # drop rows with filtered data
            unique_worker_codes = df_filtered['worker_code'].drop_duplicates()
            df = df[~df['worker_code'].isin(unique_worker_codes)]
            # reset index in dataframe
            df = df.reset_index()
        logger.info('Filtered in total in heroku data: {}',
                    old_size - df.shape[0])
        return df

    def evaluate_bins(self, df, col_name):
        """
        Update bins to specific bin size
        Args:
            df (dataframe): dataframe with data.
            col_name: nmae of column to update
        Returns:
            dataframe: updated dataframe.
        """
        # arrays to store row arrays with velocities
        updated_col = []
        # current length of velocity bins
        bin_length = 100
        # check if averaging of data is needed
        new_bin_length = int(self.res/bin_length)
        # iterate over videos
        for index, row in df.iterrows():
            # arrays to store in data of one video
            bin_data = []
            # convert data from string to array
            if type(row[col_name]) != list:
                data = ast.literal_eval(row[col_name])
            else:
                data = row[col_name]
            # check if there is actually data to process
            if len(data) > 1:
                bin_count = 0
                bin_sum = 0
                # iterate over all data to divide in bins
                for i in range(len(data)):
                    # save sum of the data + count
                    bin_sum = bin_sum + data[i]
                    bin_count = bin_count + 1
                    # when all data in 1 bin is found, append to new bin
                    if bin_count >= new_bin_length:
                        bin_data.append(bin_sum/new_bin_length)
                        bin_sum = 0
                        bin_count = 0
                        # append average of the data of new bin size
                updated_col.append(bin_data)
            else:
                updated_col.append(['no data found'])
        # add data to mapping
        df[col_name] = updated_col
        return df

    def process_velocity_risk(self, df):
        """
        add extra column to dataframe, which is the quantification
        of velocity to risk per video

        Args:
            df (dataframe): dataframe with data.

        Returns:
            dataframe: updated dataframe.
        """
        vel_risk = []

        for index, row in df.iterrows():
            # change character array to normal array
            tolist = row['vehicle_velocity_GPS']
            if type(tolist) == list:
                if len(tolist) > 2:
                    # loop to counter weird data
                    for velocity in tolist:
                        if velocity > 100:
                            tolist = row['vehicle_velocity_OBD']
                    # change integers to float
                    vel = np.array([float(i) for i in tolist])
                    kp = np.array([float(i) for i in row['kp']])
                    # append dot product of vel divided by sum of % keypresses
                    vel_risk.append(np.dot(vel, kp)/np.sum(kp))
                else:
                    vel_risk.append(['no data found'])
            else:
                # if no velocity data was present, append this string
                vel_risk.append(['no data found'])

        df['velocity_risk'] = vel_risk
        return df

    def add_object_count(self, df, object_name):
        """
        add extra column to dataframe counting the frequency
        of occurance of an object in each frame

        Args:
            df (dataframe): dataframe with data.
            object_name: name of the object to count

        Returns:
            dataframe: updated dataframe.
        """
        col_data = []
        # loop over dataframe
        for index, row in df.iterrows():
            entity_count_array = []
            # loop over entity data of one video
            for frame in ast.literal_eval(row['object_entities']):
                entity_count = 0
                # find all object data in each frame
                for entity in frame.split('_'):
                    # if entity is the requested object, up counter
                    if entity == object_name:
                        entity_count = entity_count + 1
                # after going through one frame, append frame count
                entity_count_array.append(entity_count)
            # after going through one video, append array with all data
            col_data.append(entity_count_array)

        # add data to dataframe
        col_name = object_name + '_count'
        df[col_name] = col_data
        return df

    def verify_looking(self, df):
        """Check if looking is correctly perceived by the participants

        Args:
            df (dataframe): dataframe with keypress data.
        """
        failedarray = []
        for index, row in df.iterrows():
            totalvalue = row['EC-yes'] + \
                         row['EC-yes_but_too_late'] + \
                         row['EC-no']

            if re.search('_Looking', row['cross_look']) is not None:
                failed = row['EC-no']/totalvalue
            elif re.search('notLooking', row['cross_look']) is not None:
                failed = (row['EC-yes'] +
                          row['EC-yes_but_too_late'])/totalvalue
            else:
                failed = 0
            failedarray.append(failed)

        df['looking_fails'] = failedarray
        return df

    def add_binary_data(self, df, col_name, key, new_col_name):
        """Check for a certain key in rows

        Args:
            df: dataframe of mapping file
            row: Which row to retrieve from
            key: keyword of which to retrieve

        Returns:
            df: new dataframe with binary data of requested key
        """
        binary_data = []
        # loop over rows in dataframe
        for index, row in df.iterrows():
            # if key is found in rows, then
            if re.search(key, row[col_name]) is not None:
                value = 0
            else:
                value = 1
            binary_data.append(value)
        # add data to mapping
        df[new_col_name] = binary_data
        return df

    def add_data_at_time(self, df, col, time_array):

        """retrieve column with data at a certain time.

        Args:
            df: dataframe of mapping file
            col: column of which data you want to process
            time (s): Array of data on which time you want to receive

        Returns:
            df: containing an extra column with requested data at time
        """
        for time in time_array:

            data_array = []
            # time from s to ms
            time = time*1000
            # get index of the array to find the velocity
            vel_index = int(time/self.res)
            for index, row in df.iterrows():
                # convert data from string to array
                array = row[col]
                # check if data is correctly perceived
                if type(array) == list:
                    # check if correct data is contained in list
                    if len(array) < 2:
                        data_array.append(['no data found'])
                    else:
                        data_array.append(array[vel_index])
                # create array if data is perceived as string
                elif type(array) == str:
                    array = ast.literal_eval(array)
                    # Check if values exist in array
                    if len(array) < 2:
                        data_array.append(['no data found'])
                    else:
                        data_array.append(array[vel_index])

            name = col + '_at_' + str(time/1000)
            df[name] = data_array

        return df

    def add_velocity_at_time(self, df, time_array):
        """retrieve column with velocity data at a certain time.

        Args:
            time (s): The speed on which time you want to receive

        Returns:
            df: containing an extra column with speed data at specific time.
        """
        # time from s to ms
        for time in time_array:
            velocity_data = []
            time = time*1000
            # get index of the array to find the velocity
            vel_index = int(time/self.res)
            for index, row in df.iterrows():
                # convert data from string to array
                array = row['vehicle_velocity_GPS']
                # check if there is actually data to process
                if type(array) == list:
                    # check if no errors in data by thresholding
                    if array[vel_index] > 70:
                        array = row['vehicle_velocity_OBD']
                    velocity_data.append(array[vel_index])
                else:
                    velocity_data.append('no data found')

            name = 'velocity_at_' + str(time/1000)
            df[name] = velocity_data
        return df

    def show_info(self):
        """
        Output info for data in object.
        """
        logger.info('No info to show.')
