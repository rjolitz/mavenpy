import os

import numpy as np

from .mag import read as read_mag
from .swia import read as read_swi
from .euv import read as read_euv
from .swea import read as read_swe
from .sep import read as read_sep

# Loads and merges multiple MAVEN data files
# of the same instrument type into the same dict
# structure, appending along the time axis
# when the time axis is present and preserving
# invariant columns such as energy.

# This routine is *ONLY* to be used for data
# that is binned by time. That is,
# EUV, LPW, MAG, SEP, STATIC, SWIA, SWEA
# Will NOT work for NGIMS, IUVS, and ROSE.


read_functions =\
    {"euv": read_euv,
     "mag": read_mag,
     "swi": read_swi,
     "swe": read_swe,
     "sep": read_sep}


def load_data(maven_data_files, data_read_function=None,
              include_unit=True, **kwargs):

    '''Loads a MAVEN dataset and a data read function.
    data_directory: string, path
    target_date: datetime object
    n_days: integer, >1
    data_read_function: function object'''

    n_files = len(maven_data_files)
    # print(include_unit)

    # Figure out read function from data name
    if not data_read_function:
        file_0 = os.path.split(maven_data_files[0])[1]
        # print(file_0)
        # input()
        tla = file_0.split("_")[1]

        data_read_function = read_functions[tla]

    # Initialize the final data structure as the first *successfully*
    # read data output. Attempt to read the files in order until
    # a success is found, otherwise abort since no data.
    data_struct = {}
    for i in range(0, n_files - 1):
        file_i = maven_data_files[i]
        print(file_i)

        # Skip file if empty string.
        if not file_i:
            continue

        # Try reading the data:
        maven_data_day_i = data_read_function(
            file_i, include_unit=include_unit, **kwargs)
        data_struct = maven_data_day_i

        # Retrieve the names of all contained datasets,
        # particulary the time axis.
        data_names = [i for i in data_struct.keys()]

        # Get the length of the time axis, which'll be used to identify timevarying
        # data names.
        if include_unit:
            units = {}
            for d in data_names:
                data_day_i, unit_i = maven_data_day_i[d]
                units[d] = unit_i
                data_struct[d] = data_day_i

        # print(data_names)
        # Get the time varying axis:
        time_axis = [i for i in data_names if "epoch" in i]
        print(time_axis)
        if len(time_axis) == 0:
            time_axis = [d for d in data_names if "time" in d]
            if len(time_axis) == 0:
                print(
                    "No time axis identified in {}, moving to next...".format(file_i))
                continue
                # raise IOError("No time axis identified to concatenate, aborting.")

        # For all time-varying axes, get the lengths:
        n_time_0 = [len(data_struct[time_axis]) for time_axis in time_axis]
        # print(n_time_0)
        n_time_0, time_ax_index = np.unique(n_time_0, return_index=True)
        time_axis = [time_axis[i] for i in time_ax_index]
        # print(n_time_0, time_axis)
        # input()

        # Find all data with time varying axes, as these will be concatenated
        # for multiple days
        iterate_names = []
        for d in data_names:
            data_day_i = data_struct[d]
            # print(d)
            # print(data_day_i)
            # print(data_day_i[:10])
            # print(type(data_day_i))
            # print(data_day_i.shape)
            # input()
            # try:
            data_i_d_dim = data_day_i.shape
            # print(data_i_d_dim)
            for n_time_i in n_time_0:
                if n_time_i in data_i_d_dim:
                    iterate_names.append(d)
            # except AttributeError:
            #     continue
        break

    # Exit routine and raise error if absolutely no data in the files:
    if len(data_struct) == 0:
        raise IOError(
            "No time axis identified in any files, so cannot be "
            "concatenated. Aborting.")

    # Iterate through remaining files:
    start_remaining_files = i + 1
    for i in range(start_remaining_files, n_files):

        # Skip file if empty string.
        if not maven_data_files[i]:
            continue

        maven_data_day_i = data_read_function(
            maven_data_files[i], include_unit=False, **kwargs)

        # Check if there are no time axes to append. This
        # can happen if the files are new and telemetry isn't down
        # for some APIDs.
        # If so, skip to next:
        no_time_axis_in_file = all(
            [(t not in maven_data_day_i) for t in time_axis])
        if no_time_axis_in_file:
            continue
        # if any()

        # Count the elements of the time axis for time-axis identification
        n_time_i = [len(maven_data_day_i[t]) for t in time_axis]

        # Iterate over all the time-varying data for concatenation
        for d in iterate_names:
            # print(d)
            # print(n_time_i)
            # print(maven_data_day_i[d][:10])
            # print(type(maven_data_day_i[d]))

            # Retrieve the data for the day
            data_day_i = maven_data_day_i[d]
            # print(d, data_struct[d].shape, data_day_i.shape)

            # Select the time-varying axis of the data array
            # (usually 0, but always good to verify)
            data_i_d_dim = data_day_i.shape
            # print(data_i_d_dim)

            # t_index = np.where(np.array(data_i_d_dim) == n_time_i)[0][0]
            for n_i in n_time_i:
                # print(data_i_d_dim)
                if n_i in data_i_d_dim:
                    t_index = data_i_d_dim.index(n_i)
            # print(t_index)

            # Append the days data to the full structure along the timevarying
            # axis and return with the unit.
            new_data_i_d = np.append(data_struct[d], data_day_i, axis=t_index)
            data_struct[d] = new_data_i_d


    # print("Load compelte:")
    # for n in data_struct:
    #     print(i, n, data_struct[n].shape)
    # input()

    # Reapply unit
    # print(include_unit)
    if include_unit:
        for d in data_names:
            data_struct[d] = (data_struct[d], units[d])

    return data_struct
