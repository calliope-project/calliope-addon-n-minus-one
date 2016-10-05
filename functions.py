"""
Copyright (C) 2016 Christoph Thormeyer.
Licensed under the Apache 2.0 License.

functions.py
~~~~~~~

Subfunctions definitions to allow overall analysis.

"""
import os
import shutil
import time
import yaml
import calliope
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from calliope import utils


def clean_temp(path):
    """Deletes allchanged and new files and copies the raw data."""
    if os.path.exists(path + '/example_model'):
        shutil.rmtree(path + '/example_model')
    if os.path.exists(path + '/temp'):
        shutil.rmtree(path + '/temp')
    shutil.copytree(path+'/paste-folder/example_model',
                    path + '/example_model')


def get_time(t_start):
    """Returns the runtime since the input time."""
    t_end = time.time()
    t_running = t_end - t_start
    time_string = 'Process executed after ' + str(round(t_running, 1)) \
                  + ' seconds (i.e. ' + str(int(t_running/60)) + ':' \
                  + str(int(round(t_running % 60, 0))).zfill(2) + ').'
    return time_string


def get_stuff(solution):
    """
    Gets used information from calliope.solution into more handy form.
    Returns calliope.utils.AttrDict with information.
    """
    info_attrdict = utils.AttrDict()
    # Get model config (nodes, edges, capacities, ...)
    info_attrdict['nodepower'] = solution['node']['e:power']
    # Node_power
    info_attrdict['e_cap'] = solution['parameters']['e_cap']
    # Parameter_e_cap
    info_attrdict['config_model'] = solution['config_model']
    # Locations & metadata
    info_attrdict['costs'] = solution['costs']
    # Costs for each y in each x
    info_attrdict['node_power'] = solution['node']['e:power']
    # Node power for all times
    info_attrdict['totals'] = solution['totals']
    # Totals
    frame_index = info_attrdict['costs']
    location_list = list(frame_index.axes[1])
    techs_list = list(frame_index.axes[2])
    i = 0
    for each_link in info_attrdict['config_model']['links'].keys():
        for each_tech in \
                info_attrdict['config_model']['links'][each_link].keys():
            i += 1
    frame_index = frame_index.to_frame()

    info_attrdict['labels'] = {'frame': {'index': frame_index.axes,
                                         'shape': frame_index.shape},
                               'techs': techs_list,
                               'locations': location_list,
                               'connections': i}
    return info_attrdict


def get_stuff_from_hdf(path_to_solution):
    """
    Gets used information from calliope solution.hdf into more
    handy form.  Returns calliope.utils.AttrDict with information.
    """
    info_attrdict = utils.AttrDict()
    info_compleate = calliope.read.read_hdf(path_to_solution)
    # Get model config (nodes, edges, capacities, ...)
    info_attrdict['nodepower'] = info_compleate['node']['e:power']
    # Node_power
    info_attrdict['e_cap'] = info_compleate['parameters']['e_cap']
    # Parameter_e_cap
    info_attrdict['config_model'] = info_compleate['config_model']
    # Locations & metadata
    info_attrdict['costs'] = info_compleate['costs']
    # Costs for each y in each x
    info_attrdict['node_power'] = info_compleate['node']['e:power']
    # Node power for all times
    info_attrdict['totals'] = info_compleate['totals']
    # Totals
    frame_index = info_attrdict['costs']
    location_list = list(frame_index.axes[1])
    techs_list = list(frame_index.axes[2])
    frame_index = frame_index.to_frame()
    i = 0
    for each_link in info_attrdict['config_model']['links'].keys():
        for each_tech in \
                info_attrdict['config_model']['links'][each_link].keys():
            i += 1
    info_attrdict['labels'] = {'frame': {'index': frame_index.axes,
                                         'shape': frame_index.shape},
                               'techs': techs_list,
                               'locations': location_list,
                               'connections': i}
    return info_attrdict


def run_model(path):
    """Runs the model.  Returns the calliope.solution."""
    model = calliope.Model(config_run=path)
    model.run()
    solution = model.solution
    return solution


def get_original_caps(modelinfo):
    """
    Reforms the capacity pd.Dataframe to be used consistently.
    Returns pd.Series that is reformed.
    """
    e_cap = modelinfo['e_cap'].transpose()
    original_cap_frame = e_cap.unstack()
    return original_cap_frame


def copy_models_frame_based(modelinfos, initial_constraints,
                            create_parallel_script=False):
    """
    Prepare copy run files with all needed overrides.
    Returns dict with pd.Frame capacities and the folder name.
    """
    with open('example_model/model_config/locations.yaml') as loc_reader:
        locations_dict = yaml.load(loc_reader)

    with open('example_model/model_config/techs.yaml') as tec_reader:
        techs_dict = yaml.load(tec_reader)

    # Force settings per override
    cost_frame = modelinfos.costs.to_frame()
    caps_frame = pd.DataFrame({
        'max_caps': np.full(cost_frame.shape[0], np.nan),
        'min_caps(original_model)': np.zeros(cost_frame.shape[0])
    }, cost_frame.axes[0])

    override_list = list()

    for x in list(modelinfos.e_cap.index):        # Locations and links
        for y in list(pd.DataFrame(modelinfos. e_cap,
                                   index=[x])):   # Techs per location and link
            if modelinfos['config_model']['locations'][x]['level'] != 0:
                caps_frame.loc[x, y]['min_caps(original_model)'] = \
                    modelinfos.e_cap.ix[x][y]
            else:
                if ':' in y:
                    if int(str(x).split('r')[1]) < int(str(y).split(':r')[1]):
                        # Link overrides
                        split_link = str(x) + ',' + str(y).split(':')[1]
                        # Recreate link by location & destination

                        # ATTENTION:
                        # override structure in nested dict must already exist!

                        locations_dict['links'][split_link][
                            str(y).split(':')[0]
                        ]['constraints']['e_cap.min'] = \
                            float(modelinfos.e_cap.loc[x][y])
                        caps_frame.loc[x, y]['min_caps(original_model)'] = \
                            modelinfos.e_cap.loc[x][y]
                        caps_frame.loc[x, y]['max_caps'] = np.inf
                        new_override_dict = {
                            'override.links.' + str(split_link) + '.'
                            + str(y).split(':')[0]
                            + '.constraints.e_cap.max': 0,
                            'override.links.' + str(split_link) + '.'
                            + str(y).split(':')[0]
                            + '.constraints.e_cap.min': 0
                        }
                        total_override_dict = {**new_override_dict,
                                               **initial_constraints}
                        override_list.append(total_override_dict)
                    elif int(str(x).split('r')[1]) != \
                            int(str(y).split(':r')[1]):
                        # Necessary? see loop above
                        caps_frame.loc[x, y]['min_caps(original_model)'] = \
                            modelinfos.e_cap.loc[x][y]

                else:
                    if techs_dict['techs'][y]['parent'] == 'supply' \
                            and y in locations_dict['locations'][x]['techs']:
                            locations_dict['locations'][
                                x
                            ]['override'][y]['constraints']['e_cap.min'] = \
                                float(modelinfos.e_cap.loc[x][y])
                            caps_frame.loc[x, y][
                                'min_caps(original_model)'
                            ] = modelinfos.e_cap.loc[x][y]
                            caps_frame.loc[x, y]['max_caps'] = \
                                techs_dict['techs'][
                                    y
                                ]['constraints']['e_cap.max']
                    else:
                        caps_frame.loc[x, y]['min_caps(original_model)'] = \
                            modelinfos.e_cap.loc[x][y]

    # At this point all settings for the operate setting
    # should be ready to create the exact copy
    with open('example_model/model_config/copy_locations.yaml', 'w') \
            as loc_override:
        loc_override.write(yaml.dump(locations_dict))

    # Set up copy_model to run copy settings
    with open('example_model/model_config/model.yaml') as model_reader:
        model_dict = yaml.load(model_reader)
    model_dict['import'] = [item.replace('locations', 'copy_locations') for
                            item in model_dict['import']]
    model_dict['name'] = 'copy '+model_dict['name']
    with open('example_model/model_config/copy_model.yaml', 'w') \
            as model_writer:
        model_writer.write(yaml.dump(model_dict))

    with open('example_model/run.yaml') as run_reader:
        run_dict = yaml.load(run_reader)
    cut_number = run_dict['model'].find('model.yaml')
    run_dict['model'] = run_dict['model'][:cut_number] + 'copy_' \
                        + run_dict['model'][cut_number:]
    run_dict['output']['path'] += '/copy_runs'
    run_dict['parallel']['iterations'] = override_list
    folder_name = run_dict['parallel']['name']
    with open('example_model/copy_run.yaml', 'w') as run_writer:
        run_writer.write(yaml.dump(run_dict))
    if create_parallel_script is True:
        path_extension = os.path.abspath('')
        parallel_sh = '#!/bin/bash\ncalliope generate '\
                      + path_extension\
                      + '/example_model/copy_run.yaml '\
                      + path_extension \
                      + '/example_model\n'       # local paths!!
        with open('example_model/parallel.sh', 'w') as create_shell:
            create_shell.write(parallel_sh)
    return_dict = {'caps': caps_frame, 'name': folder_name}
    return return_dict


def read_results(model_info, run):
    """
    Reads the costs and capacities from one run.
    Returns a dict with pd.Dataframe of costs and capacities.
    """
    if not type(run) == str:
        run = str(run).zfill(4)
    # else:
    #    run = run.zfill(4)

    frame_index = model_info['labels']['frame']['index'][0]

    cost_dict = {run+'_cost': list()}
    for item in list(model_info['costs'].items):
        for major in list(model_info['costs'].major_axis):
            for minor in list(model_info['costs'].minor_axis):
                cost_dict[run + '_cost'] += \
                    [model_info['costs'].loc[item, major, minor]]
    cost_frame = pd.DataFrame(cost_dict, frame_index)

    caps_dict = {run + '_caps': list()}
    for x in model_info['labels']['locations']:
        for y in model_info['labels']['techs']:
            caps_dict[run + '_caps'] += [model_info['e_cap'].loc[x, y]]

    caps_frame = pd.DataFrame(caps_dict, frame_index)

    return_dict = {'costs': cost_frame, 'caps': caps_frame}
    return return_dict


def create_parallel_sh(path_to_run, model_info, folder_name, folder_name_2):
    """
    Creates, prepares and runs a bash script with the according
    folder structure for the parallel run with calliope.
    """

    # I need to add an intermediate step to give parallel script output
    # and run read_results(see above) with consistent input
    os.system('bash ' + path_to_run + 'parallel.sh')
    os.mkdir(path_to_run + folder_name + '/Runs/model_config')
    shutil.copytree(path_to_run + '/model_config/data', path_to_run + '/'
                    + folder_name + '/Runs/model_config/data')

    for i in range(model_info['labels']['connections']):
        # number of all connections
        cmd_base = 'cd '+os.path.abspath('') + '/' + folder_name_2 + '/' \
                   + folder_name + ' && '
        cmd_special = './run.sh ' + str(i+1)
        cmd = cmd_base + cmd_special
        os.system(cmd)


def create_parallel_sequentially(path_to_run, model_info):
    """Runs all calliope.Models sequencially instead of parallel."""

    # Some problems and not necessary/usefull in most cases
    with open(path_to_run + 'copy_run.yaml') as run_reader:
        run_dict = yaml.load(run_reader)

    folder_name = run_dict['parallel']['name']

    if not os.path.exists(path_to_run + '/' + folder_name + '/Output'):
        os.makedirs(path_to_run + '/' + folder_name + '/Output')

    initial_override = run_dict['override']
    initial_path = run_dict['output']['path']

    for i in range(model_info['labels']['connections']):
        if initial_override is None:
            temp_dict = run_dict['parallel']['iterations'][i]
            new_dict = dict()
            for key in temp_dict:
                new_dict[key] = temp_dict[key]
            run_dict['override'] = new_dict
            del new_dict
            del temp_dict
        else:
            for keys in run_dict['parallel']['iterations'][i].keys():
                run_dict['override'][keys] = \
                    run_dict['parallel']['iterations'][i][keys]
        run_dict['output']['path'] = \
            path_to_run + folder_name + '/Output/'+str(i+1).zfill(4)
        if not os.path.exists(
            path_to_run + '/' + folder_name + '/Output/'+str(i+1).zfill(4)
        ):
            os.makedirs(
                path_to_run + '/' + folder_name + '/Output/'+str(i+1).zfill(4)
            )
        with open(path_to_run + 'copy_run.yaml', 'w') as run_writer:
            run_writer.write(yaml.dump(run_dict))
        temp_model = calliope.Model(config_run=path_to_run + 'copy_run.yaml')
        temp_model.run()
        temp_model.save_solution('hdf')
    run_dict['output']['path'] = initial_path
    run_dict['override'] = initial_override
    with open(path_to_run + 'copy_run.yaml', 'w') as run_writer:
        run_writer.write(yaml.dump(run_dict))


def concat_frames(model_info, folder_name, path_name):
    """
    Takes informations of all runs.
    Returns dict with pd.Dataframes of costs and capacities.
    """

    cap_dict = dict()
    cost_dict = dict()
    tot_list = list()

    for i in range(1, model_info['labels']['connections']+1):
        # i runs from first to last Model (starting with 1)
        temp_copy_info = get_stuff_from_hdf(path_name + '/' + folder_name
                                            + '/Output/' + str(i).zfill(4)
                                            + '/solution.hdf')
        answer_dict = read_results(temp_copy_info, i)
        tot_list = tot_list + [answer_dict]
        cap_dict[str(i).zfill(4)] = answer_dict['caps']
        cost_dict[str(i).zfill(4)] = answer_dict['costs']

    cap_frame = pd.concat(cap_dict, axis=1)
    cost_frame = pd.concat(cost_dict, axis=1)

    max_cap_list = list()
    for i in range(int(cap_frame.shape[0])):
        max_cap_list = max_cap_list + [max(cap_frame.iloc[i][:])]

    cap_frame['max'] = max_cap_list

    max_cost_list = list()
    for i in range(cost_frame.shape[0]):
        max_cost_list = max_cost_list + [max(cost_frame.iloc[i][:])]
    cost_frame['max'] = max_cost_list

    frame_dict = {'costs': cost_frame, 'caps': cap_frame}
    return frame_dict


def analyse_results(cap_cost_dict, folder_name):
    """
    Adds a column with the maximal values of each row.
    Saves the tables of costs and capacities as csv.
    Returns pd.Dataframe with capacities.
    """
    path_extension = os.path.abspath('')
    cost = cap_cost_dict['costs']
    caps = cap_cost_dict['caps']
    max_list = list()
    for i in range(len(cost)):
        max_list += [max(cost.iloc[i][:])]
    cost['max_cost'] = pd.Series(max_list, index=cost.index)
    cost.to_csv(path_extension + '/example_model/'
                + folder_name + '/Overview/cost.csv')
    max_list = list()
    for i in range(len(caps)):
        max_list += [max(caps.iloc[i][:])]
    caps['max_caps'] = pd.Series(max_list, index=caps.index)
    caps.to_csv(path_extension + '/example_model/'
                + folder_name + '/Overview/caps.csv')
    return caps


def plot_bars(frame, content='caps', fixed_scale=False):
    """Creates bar plots and saves them in respective directories."""

    max_list = list()
    for i in range(frame.shape[0]):
        temp_frame = frame.iloc[i][:]
        if not temp_frame[-1] == 0 and not temp_frame.eq(temp_frame[0]).all():
            max_list.append(max(temp_frame))
    if content == 'caps':
        y_pos = list(range(frame.shape[1]-2))
        x_label = [str(i) for i in range(len(y_pos))]
        x_label[0] = 'original'
        x_label[-1] = 'safe'
        for i in range(frame.shape[0]):
            if not frame.iloc[i][-1] < abs(10**-10) \
                    and not frame.iloc[i][1:].eq(frame.iloc[i][1]).all():
                temp_fig = plt.figure()
                plt.bar(y_pos[0], frame.iloc[i][1], align='center',
                        alpha=0.75, color='red')
                plt.bar(y_pos[1:-1], frame.iloc[i][2:-2], align='center',
                        alpha=0.75, color='black')
                plt.bar(y_pos[-1], frame.iloc[i][-1], align='center',
                        alpha=0.75, color='blue')
                plt.xticks(y_pos, x_label)
                plt.xticks(rotation=30)
                plt.xlim(-2, 23)
                plt.ylabel('installed capacity')
                plt.title('Capacity comparison for '
                          + str(frame.axes[0][i][1])
                          + ' at ' + str(frame.axes[0][i][0]))
                if os.path.exists('temp/figures/caps') is False:
                    os.makedirs('temp/figures/caps/')
                if os.path.exists('temp/figures/cost') is False:
                    os.makedirs('temp/figures/cost/')
                if os.path.exists('temp/figures/load') is False:
                    os.makedirs('temp/figures/load/')
                if fixed_scale is True:
                    plt.ylim((0, max(max_list)))
                    temp_fig.savefig('temp/figures/caps/cap_fig_'
                                     + str(frame.axes[0][i][0]) + '_'
                                     + str(frame.axes[0][i][1])
                                     + '_fixed_scale.png')
                else:
                    temp_fig.savefig('temp/figures/caps/cap_fig_'
                                     + str(frame.axes[0][i][0]) + '_'
                                     + str(frame.axes[0][i][1])+'.png')
    elif content == 'costs':
        y_pos = list(range(frame.shape[1]-1))
        x_label = [str(i) for i in range(len(y_pos))]
        x_label[0] = 'original'
        x_label[-1] = 'safe'
        for i in range(frame.shape[0]):
            if not frame.iloc[i][-1] == 0 \
                    and not frame.iloc[i][:].eq(frame.iloc[i][0]).all():
                temp_fig = plt.figure()
                plt.bar(y_pos[0], frame.iloc[i][0], align='center',
                        alpha=0.75, color='red')
                plt.bar(y_pos[1:-1], frame.iloc[i][1:-2], align='center',
                        alpha=0.75, color='black')
                plt.bar(y_pos[-1], frame.iloc[i][-1], align='center',
                        alpha=0.75, color='blue')
                plt.xticks(y_pos, x_label)
                plt.xticks(rotation=30)
                plt.ylabel('costs')
                plt.title('Costs comparison for ' + str(frame.axes[0][i][1])
                          + ' at ' + str(frame.axes[0][i][0]))
                if fixed_scale is True:
                    plt.ylim((0, max(max_list)))
                    temp_fig.savefig('temp/figures/cost/cost_fig_'
                                     + str(frame.axes[0][i][0]) + '_'
                                     + str(frame.axes[0][i][1])
                                     + '_fixed_scale.png')
                else:
                    temp_fig.savefig('temp/figures/cost/cost_fig_'
                                     + str(frame.axes[0][i][0]) + '_'
                                     + str(frame.axes[0][i][1]) + '.png')


def plot_lines(model_info, frame, cost_frame,
               safe_info=False, bars=False, fixed_scale=False):
    """
    Creates line plots and pie plots and saves them in
    the respective directories.
    """
    loc_dict = dict()
    for i in range(model_info['node_power'].shape[2]):
        if model_info['config_model']['locations'][
            model_info['node_power'].axes[2][i]
        ]['level'] == 0:
            loc_dict[model_info['node_power'].axes[2][i]] = dict()
            for j in range(model_info['node_power'].shape[0]):
                if model_info['config_model']['techs'][
                    model_info['node_power'].axes[0][j].split(':')[0]
                ]['parent'] == 'transmission' \
                        and not model_info[
                            'node_power'
                        ].axes[0][j].split(':')[1] == \
                                model_info['node_power'].axes[2][i]:
                    if model_info['node_power'].axes[0][j].split(':')[-1] \
                            not in loc_dict[
                                model_info['node_power'].axes[2][i]
                            ].keys():
                        loc_dict[model_info['node_power'].axes[2][i]][
                            model_info['node_power'].axes[0][j].split(':')[-1]
                        ] = list()
                    loc_dict[model_info['node_power'].axes[2][i]][
                        model_info['node_power'].axes[0][j].split(':')[-1]
                    ].append(model_info['node_power'].axes[0][j])

    font = {'weight': 'light', 'size': 12}
    matplotlib.rc('font', **font)

    if fixed_scale is True:
        max_list = list()
        for l in range(frame.shape[0]):
            temp_frame = frame.iloc[l][:]
            if not temp_frame[-1] == 0 \
                    and not temp_frame.eq(temp_frame[0]).all():
                max_list.append(max(temp_frame))

    key_list = list(loc_dict.keys())
    for i in range(len(key_list)-1):
        for j in range(i+1, len(key_list)):
            connections = len(loc_dict[key_list[i]][key_list[j]])
            sub_plot = plt.figure(figsize=(12, 10), dpi=300)
            for k in range(connections):
                if bars is True:
                    plt.subplot(int(str(connections) + '2' + str(2*k+1)))
                else:
                    plt.subplot(int(str(connections) + '1' + str(k+1)))
                plt.plot(abs(model_info['node_power'].ix[
                             loc_dict[key_list[i]][key_list[j]][k],
                             :,
                             key_list[i]
                             ]), 'r')
                if safe_info:
                    plt.plot(abs(safe_info['node_power'].ix[
                                 loc_dict[key_list[i]][key_list[j]][k],
                                 :,
                                 key_list[i]
                                 ]), 'b')
                    tmp = safe_info['node_power'].to_frame()
                    tmp.to_csv('safe_nodepower.csv')
                    tmp2 = model_info['node_power'].to_frame()
                    tmp2.to_csv('orig_nodepower.csv')
                caps_origin = frame.ix[(key_list[i],
                                        loc_dict[key_list[i]][key_list[j]][k]),
                                       1]
                caps_safe = frame.ix[(key_list[i],
                                      loc_dict[key_list[i]][key_list[j]][k]),
                                     -1]
                origin_series = \
                    pd.Series(caps_origin, model_info['node_power'].axes[1])
                safe_series = \
                    pd.Series(caps_safe, model_info['node_power'].axes[1])
                plt.plot(origin_series, 'r')
                plt.plot(safe_series, 'b')
                plt.title('via ' + loc_dict[key_list[i]][key_list[j]][k].split(
                    ':'
                )[0],
                          **font)
                plt.xticks(rotation=20)
                if fixed_scale is True:
                    plt.ylim((-5, 1.05*max(max_list)))
                else:
                    plt.ylim([-0.05*caps_safe-1, caps_safe*1.05+1])
                plt.ylabel('e_cap')
                if bars is True:
                    y_pos = list(range(2, frame.shape[1]-2))
                    x_label = [str(m) for m in range(frame.shape[1])]
                    x_label[0] = 'original'
                    x_label[-1] = 'max value'

                    plt.subplot(int(str(connections) + '2' + str(2*k+2)))
                    plt.bar(0, frame.ix[
                        (key_list[i], loc_dict[key_list[i]][key_list[j]][k])
                    ][1], align='center', alpha=0.75, color='red')
                    plt.bar(y_pos, frame.ix[
                                       (key_list[i],
                                        loc_dict[key_list[i]][key_list[j]][k])
                                   ][2:-2],
                            align='center', alpha=0.75, color='black')
                    plt.bar(1, frame.ix[(
                        key_list[i], loc_dict[key_list[i]][key_list[j]][k]
                    )][-1], align='center', alpha=0.75, color='blue')
                    plt.title('original cap: ' + str(round(caps_origin, 1))
                              + ', max cap: ' + str(round(caps_safe, 1)))
                    if fixed_scale is True:
                        plt.ylim((-5, 1.05*max(max_list)))
                    else:
                        plt.ylim([-0.05 * caps_safe - 1, caps_safe * 1.05 + 1])
                    plt.xlim([-1, len(y_pos)+2])

            plt.suptitle('Connections from ' + key_list[i]
                         + ' to ' + key_list[j] + '\n')
            sub_plot.tight_layout()
            save_name = 'temp/figures/load/load_connections_' \
                        + key_list[i] + '_' + key_list[j]
            if fixed_scale is True:
                save_name += '_fixed_scale'
            if bars is True:
                save_name += '_with_bars'
            sub_plot.savefig(save_name + '_' + str(sub_plot.dpi) + '.png')
    plt.close('all')

    # Cost-plot in pie charts
    print(cost_frame.columns)
    if 'safe_costs' in cost_frame.columns:
        labels = 'production, fuel & \nsupply infrastructure', 'grid'
        transmission_cost_original = 0
        transmission_cost_safe = 0
        for frame_index in cost_frame.index:
            if ':' in frame_index[1]:
                transmission_cost_original += float(cost_frame.ix[frame_index,
                                                                  'monetary'])
                transmission_cost_safe += float(cost_frame.ix[frame_index,
                                                              ('max', '')])
        print(cost_frame)
        print(cost_frame.axes)
        costs_original = [float(cost_frame.ix['total', 'monetary'])
                          - transmission_cost_original,
                          transmission_cost_original]
        safe_col_name = 'safe_costs'
        costs_safe = [float(cost_frame.ix['total', safe_col_name])
                      - transmission_cost_safe, transmission_cost_safe]
        colors = ['red', 'blue']
        new_radius = np.sqrt(cost_frame.ix['total', safe_col_name]
                             / cost_frame.ix['total', 'monetary'])
        factor = cost_frame.ix['total', safe_col_name] \
                 / cost_frame.ix['total', 'monetary']
        explode = (0.1, 0)

        fig = plt.figure()
        plt.subplot(121)
        plt.pie(costs_original, explode=explode, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90, pctdistance=1.1, labeldistance=1.2)
        plt.axis('equal')
        plt.title('original model cost: '+str(cost_frame.ix['total', 'monetary']))
        plt.subplot(122)
        plt.pie(costs_safe, explode=explode, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90, radius=new_radius,
                pctdistance=1.1*new_radius, labeldistance=1.2*new_radius)
        plt.axis('equal')
        plt.title('N-1 safe model cost: '
                  + str(cost_frame.ix['total', safe_col_name]))
        plt.tight_layout()
        plt.suptitle('Total change of ' + str((factor-1)*100)
                     + ' percent', fontsize=15)
        plt.axis('off')
        plt.subplots_adjust(top=0.85)
        plt.legend(labels, loc=(-0.73, 0))
        fig.savefig('temp/figures/cost/Change_in_cost_composition.png')


def outage_safe_system(path_to_run_directory, max_caps):
    """
    Creates and runs N-1 safe system.
    Returns pd.Series with N-1 safe capacities.
    """
    with open(path_to_run_directory + 'run.yaml') \
            as run_reader:
        run_dict = yaml.load(run_reader)
    with open(path_to_run_directory + 'model_config/model.yaml') \
            as model_reader:
        model_dict = yaml.load(model_reader)
    with open(path_to_run_directory + 'model_config/locations.yaml') \
            as loc_reader:
        loc_dict = yaml.load(loc_reader)
    run_dict['output']['path'] = path_to_run_directory \
                                 + run_dict['parallel']['name'] \
                                 + '/Output/N-1_safe_model'

    for row in max_caps.axes[0]:
        if max_caps[row] != 0 or True:
            # loc_level to exclude locations within other locations
            if 'level' in loc_dict['locations'][row[0]].keys() \
                    and loc_dict['locations'][row[0]]['level'] != 0:
                loc_level = 1
            else:
                loc_level = 0
            if ':' in row[1] and loc_level == 0:
                if int(row[0].split('r')[-1]) < int(row[1].split(':r')[-1]):
                    loc_dict['links'][
                        row[0] + ',' + row[1].split(':')[-1]
                    ][row[1].split(':')[0]]['constraints']['e_cap.min'] = \
                        int(np.ceil(max_caps[row]))
            else:
                if row[1] in loc_dict['locations'][row[0]]['techs'] \
                        and row[1] != 'unmet_demand_power' \
                        and row[1] != 'demand_power':
                    loc_dict['locations'][
                        row[0]
                    ]['override'][row[1]]['constraints']['e_cap.min'] = \
                        int(np.ceil(max_caps[row]))

    run_dict['mode'] = 'plan'       # plan or operate
    run_dict['model'] = 'model_config/N-1_safe_model.yaml'
    model_dict['import'] = [item.replace('locations', 'N-1_safe_locations')
                            for item in model_dict['import']]
    with open(
                    path_to_run_directory + 'N-1_safe_run.yaml', 'w'
    ) as override_writer:
        override_writer.write(yaml.dump(run_dict))
    with open(
                    path_to_run_directory
                    + 'model_config/N-1_safe_model.yaml', 'w'
    ) as model_writer:
        model_writer.write(yaml.dump(model_dict))
    with open(
                    path_to_run_directory
                    + 'model_config/N-1_safe_locations.yaml', 'w'
    ) as loc_writer:
        loc_writer.write(yaml.dump(loc_dict))
    safe_model = calliope.Model(
        config_run=path_to_run_directory + 'N-1_safe_run.yaml'
    )
    safe_model.run()
    safe_model.save_solution('hdf')
    safe_sol = get_stuff(safe_model.solution)
    safe_caps = get_original_caps(safe_sol)
    return {'safe_caps': safe_caps, 'model_info': safe_sol}


def complete_analysis(
        path_to_run_directory, mode='design', new_run=True,
        new_copies='parallel',  # New_copies options: False, sequence, parallel
        new_plots=True, get_csv=False,
        constraint_dict=dict()):
    """
    Main function to run everything
    and manages the respective options via input flags.
    Returns pd.Series with N-1 safe caps.
    """
    # Could be done with a GUI if ever a lot of people might use it

    path_name = path_to_run_directory.split('/')[-2]

    if constraint_dict:
        with open(path_to_run_directory + 'run.yaml') as run_reader:
            run_dict = yaml.load(run_reader)
        run_dict['override'] = constraint_dict
        with open(path_to_run_directory + 'run.yaml', 'w') as run_writer:
            run_writer.write(yaml.dump(run_dict))

    if new_run is True:
        original_solution = run_model(path_to_run_directory + 'run.yaml')
        model_info = get_stuff(original_solution)
    else:
        with open(path_to_run_directory + 'run.yaml') as run_reader:
            run_dict = yaml.load(run_reader)
        model_info = get_stuff_from_hdf(run_dict['output']['path']
                                        + '/solution.hdf')

    # Create copy base
    if new_copies == 'parallel':
        temp_dict = copy_models_frame_based(model_info, constraint_dict,
                                            create_parallel_script=True)
        original_caps = temp_dict['caps']
        folder_name = temp_dict['name']
        create_parallel_sh(path_to_run_directory, model_info,
                           folder_name, path_name)
    elif new_copies == 'sequence':
        temp_dict = copy_models_frame_based(model_info, constraint_dict,
                                            create_parallel_script=False)
        original_caps = temp_dict['caps']
        folder_name = temp_dict['name']
        create_parallel_sequentially(path_to_run_directory, model_info)
    else:
        original_caps = get_original_caps(model_info)
        with open(path_to_run_directory + 'run.yaml') as run_reader:
            run_dict = yaml.load(run_reader)
        folder_name = run_dict['parallel']['name']

    # Create compare frames
    copy_frames = concat_frames(model_info, folder_name, path_name)
    total_caps_frame = pd.concat([original_caps, copy_frames['caps']], axis=1)
    total_cost_frame = pd.concat([model_info['costs'].to_frame(),
                                  copy_frames['costs']], axis=1)

    # Run N-1 safe model
    if mode == 'design':
        safe_solution = outage_safe_system(path_to_run_directory,
                                       total_caps_frame[
                                           total_caps_frame.axes[1][-1]
                                       ])
        safe_caps = safe_solution['safe_caps']
        total_caps_frame = pd.concat([total_caps_frame, safe_caps], axis=1)
        safe_costs = safe_solution['model_info']['costs'].to_frame()
        safe_costs.columns = ['safe_costs']
        total_cost_frame = pd.concat([total_cost_frame, safe_costs], axis=1)

    tot_dict = dict()
    for col in list(total_cost_frame.axes[1]):
        tot_dict[col] = sum(total_cost_frame[col])
    total = pd.DataFrame(tot_dict, index=['total'])
    total_cost_frame_final = pd.concat([total_cost_frame, total], axis=0)

    # Plot N-1 Barriers
    if new_plots is True:
        plot_bars(total_caps_frame, 'caps')
        if mode =='design':
            plot_bars(total_cost_frame_final, 'costs')
            plot_lines(model_info, total_caps_frame, total_cost_frame_final,
                       safe_info=safe_solution['model_info'], bars=True,
                       fixed_scale=False)
        elif mode == 'check':
            plot_bars(total_cost_frame, 'costs')
            plot_lines(model_info, total_caps_frame, total_cost_frame_final,
                       safe_info=False, bars=True,
                       fixed_scale=False)

    if mode == 'check':
        # Check all capacities in total_caps_frame if they are the largest
        lines_counter = 0
        lines_failure = list()
        for i in range(1, total_caps_frame.shape[1]-1):
            for j in range(total_caps_frame.shape[0]):
                if total_caps_frame.ix[j, 1] >= total_caps_frame.ix[j, i]:
                    pass
                else:
                    lines_counter += 1
                    lines_failure.append(total_caps_frame.axes[1][i])
                    break
            print(total_caps_frame.axes[1][i])
        if lines_counter == 0:
            print('The system is already N-1 safe')
        else:
            print('Number of lines, '
                  'which outage can not be compensated by your system:')
            print(lines_counter)

    if get_csv is True:
        # Save compare_frames
        total_cost_frame.to_csv('cost_frame.csv')
        total_caps_frame.to_csv('caps_frame.csv')

    return total_caps_frame[total_caps_frame.axes[1][-1]]
