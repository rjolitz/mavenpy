import numpy as np
import datetime as dt
from collections.abc import Iterable

from dateutil.parser import parse
# import calendar

#########################################
#           Helper routines             #
#########################################

utc_19700101_dt = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)

# UTC_to_UNX and UNX_to_UTC can
# convert utc <-> unx to seconds level accuracy (2 msec
# discrepancy for Feb 27 2017 SEP L2data).


def process_data_dict(dataset_dict, conditional_array=None,
                      units=None, alias=None, scrub_NaN_times=None):
    '''Routine to iterate through datasets in
    a dictionary (usually from read_cdf and read_sav
    in read.py) and modify according to supplied kwargs,
    delete the key-value and make a new data dictionary.
    '''
    # Data associated with the appropriate unit

    # Determine the indices of conditional array
    # (and non-NAN times) to select data:
    if conditional_array is not None or scrub_NaN_times:
        time_unix_key =\
            [i for i in dataset_dict if 'unix' in i or 'unx' in i][0]
        time_unix = dataset_dict[time_unix_key]
        n_t = len(time_unix)

        if scrub_NaN_times:
            # Select the non-NaN indices
            # and only retain those in each column
            not_nan_indices = ~np.isnan(time_unix)

            # Update the conditional array to where not NAN
            # and matching condition
            conditional_array = (conditional_array & not_nan_indices)

        condition_index = np.where(conditional_array)[0]
        N = condition_index.size

    # Make new data dict
    final_data_dict = {}
    var_names = [i for i in dataset_dict.keys()]
    for dataset_name in var_names:

        # Get dataset
        data_i = dataset_dict[dataset_name]

        # Check dims to see if data is a
        # numpy array with a matching conditional
        # dimension:
        if isinstance(data_i, np.ndarray) and conditional_array is not None:
            data_i_shape = data_i.shape
            data_i_dim = len(data_i_shape)
            # input()

            if n_t in data_i_shape:
                matching_data_i = data_i[condition_index, ...]
            else:
                matching_data_i = data_i

            data_i = matching_data_i

        # Assign unit, if provided
        if units is not None:
            unit_i = units[dataset_name]
            data_i = (data_i, unit_i)

        # Rename dataset
        if alias is not None:
            if dataset_name in alias:
                alias_i = alias[dataset_name]
                new_dataset_name = alias_i
            else:
                new_dataset_name = dataset_name
        else:
            new_dataset_name = dataset_name

        final_data_dict[new_dataset_name] = data_i
        # print(unit_i)

        # Empty out old dict
        del dataset_dict[dataset_name]

    return final_data_dict


def invert_energy_axis(data_dict, energy_names=None,
                       energy_axis=None,
                       energy_dependent_var_names=None,
                       energy_dependent_var_axis=None,
                       n_energy=None):

    ''' Electrostatic analyzers like SWIA, SWEA,
    and STATIC generally sweep in voltage downwards,
    going from high energies to low energies. Accordingly
    it is useful to invert the energy axis
    so it is monotonically increasing for
    analysis.

    This routine **REVERSES** the direction of the energy axis so
    energies is an increasing instead of decreasing array.

    energy_names: iterable of strings, identifier of energy field
    energy_axis: integer specifying the axis that varies over energy
        for the energy axes (necessary for multidim energy axis,
        e.g. for STATIC)
    energy_dependent_var_names: iterable of strings,
        which identify the energy-dependent variables
    energy_dependent_var_axis: integer specifying the axis
        that varies over energy
        for the energy axes (necessary for multidim energy axis,
        e.g. for STATIC)

    '''

    if energy_names is None:
        # Retrieve the field corresponding to energy
        energy_names = [name for name in data_dict if "energy" in name]

    if energy_dependent_var_names is None:
        # Retrieve fields corresponding to energy-dependent variables
        # such as count or diff_en_flux
        energy_dependent_var_names =\
            [name for name in data_dict if "diff_en_fluxes"
             in name or "count" in name]

    for e_name in energy_names:
        energy = data_dict[e_name]
        e_dim = energy.shape
        if len(e_dim) == 1:
            energy = energy[::-1]
        elif len(e_dim) > 1:
            energy = np.flip(energy, axis=energy_axis)

        data_dict[e_name] = energy

    for e_var_name in energy_dependent_var_names:
        energy = data_dict[e_name]
        e_variable = data_dict[e_var_name]
        # e_dim = energy.shape
        if energy_dependent_var_axis is None:
            energy_dependent_var_axis = e_variable.shape.index(n_energy)

        e_variable = np.flip(e_variable, axis=energy_dependent_var_axis)
        data_dict[e_var_name] = e_variable

    return data_dict


def UNX_to_UTC(POSIX_time):
    """Convert a numpy array of POSIX times into a
    numpy array of datetime objects.
    Warning: doesnt account for leap seconds

    Q: Why don't I use numpy datetime64?
    A: Even less testing and support, plus still
    no leap seconds

    Equivalent to time_string() in IDL
    2/10: Verified time_string('2015 1 1') = UNX_to_UTC('2015 1 1')
    """

    if isinstance(POSIX_time, Iterable):
        return np.array([UNX_to_UTC(i) for i in POSIX_time])

    if np.isnan(POSIX_time):
        print("NaN Posix times detected, should not occur"
              ", please check input.")
        return dt.datetime(1900, 1, 1)
    else:
        return dt.datetime.fromtimestamp(POSIX_time, tz=dt.timezone.utc)


def UTC_to_UNX(UTC_time):
    """Convert a (array of) UTC times into an
    (array of) POSIX times.
    Warning: doesnt account for leap seconds"""

    # If it is an iterable and NOT a string, return array
    # variant
    # print(UTC_time)
    if isinstance(UTC_time, Iterable) and not isinstance(UTC_time, str):
        return np.array([UTC_to_UNX(i) for i in UTC_time])

    # If supplied as string, convert to datetime obj.
    if isinstance(UTC_time, str):
        UTC_time = parse(UTC_time)

    # If supplied as np.datetime64, convert to datetime obj:
    if isinstance(UTC_time, np.datetime64):
        # UTC_time = UTC_time.tolist()
        UTC_time = UTC_time.astype(dt.datetime)
    # print(UTC_time)

    if UTC_time.tzinfo is None:
        UTC_time = UTC_time.replace(tzinfo=dt.timezone.utc)

    # WARNING: using timestamp() references the SYSTEM timezone, not
    # UTC! get around by subtracting 1970/1/1 to get total seconds.
    # return UTC_time.timestamp()
    return (UTC_time - utc_19700101_dt).total_seconds()


def daterange(start_date=None, end_date=None, orbnum=None,
              n_days=None, orb_to_t_func=None):

    if not start_date and not orbnum:
        raise NameError("Need to provide either a date and # days,"
                        " list of days, or orbit number and "
                        "function that maps orbit number to a time.")

    # print(start_date, end_date, orbnum, n_days)

    # Convert start/end/n_days to appropriate datetype
    start_date, n_days, end_date = sanitize_date_inputs(
        start_date=start_date, n_days=n_days, end_date=end_date,
        default_start_date=None, default_end_date=None)
    # print(start_date, n_days, end_date)

    # if end_date:
    #     # Inclusive of initial day requested
    #     # (add one day)
    #     n_days = (end_date - start_date).days + 1

    if n_days:
        # return date info for requested days
        # Raise error if no start_date provided
        dt_list = [start_date + dt.timedelta(days=i) for i
                   in range(n_days)]

    elif orbnum:
        # return date info for requested orbits
        unx_list = orb_to_t_func(orbnum)
        if isinstance(unx_list, list):
            dt_list = UNX_to_UTC(unx_list)

    # print(dt_list)
    # print(n_days)
    # input()

    return dt_list


def dt_fstring(day_dt):
    # Convert the loaded day to strings:
    yyyy = day_dt.strftime("%Y")
    mm = day_dt.strftime("%m")
    dd = day_dt.strftime("%d")
    doy = day_dt.strftime("%j")

    HHMMSS = day_dt.strftime("%H%M%S")

    # Since we sometimes dont know the
    # HH:MM:SS referred to in the filename,
    # if it isn't provided (returned 00:00:00)
    # then just substitute the regex for search.
    if HHMMSS == "000000":
        HHMMSS = "(.*)"

    dt_dict = {"yyyy": yyyy, "mm": mm, "dd": dd,
               "doy": doy, "hhMMSS": HHMMSS}

    return dt_dict


def date_to_dt(date, default=None):
    '''Convert ambiguous date object (str or numpy datetime64) into
    datetime.'''

    if isinstance(date, dt.datetime):
        # Do nothing:
        date_dt = date
    elif isinstance(date, str):
        # The 'default' keyword in parse means that any string
        # missing the year-month-day will default
        # to 2015-01-01. So if '2018 2' if entered,
        # will become dt.datetime(2018, 2, 1)
        date_dt = parse(date, default=dt.datetime(2015, 1, 1))
    elif isinstance(date, np.datetime64) or\
            np.issubdtype(date, np.datetime64):
        date_dt = date.astype(dt.datetime)
    elif date is None and default is not None:
        default_dt = date_to_dt(default)
        date_dt = default_dt
    else:
        raise TypeError(
            "Date can only be str, datetime64,"
            " or datetime object. If None, "
            "must set a default date.")
    return date_dt


def sanitize_date_inputs(start_date=None, n_days=None, end_date=None,
                         default_start_date=None, default_end_date=None):

    # Set start date to default if provided, or raise error.
    start_date = date_to_dt(start_date, default=default_start_date)

    # If even after that, no end_date and no n_days, raise error.
    if n_days is None and end_date is None:
        raise ValueError("Need either n_days or an end_date.")

    if not end_date:
        end_date = start_date + dt.timedelta(days=(n_days - 1))

    end_date = date_to_dt(end_date, default=default_end_date)

    if end_date < start_date:
        raise ValueError(
            "Error: End date is earlier than start date.")

    # Get # of days
    # inclusive of initial day
    if not n_days:
        n_days = 1 + (end_date - start_date).days

    # print(start_date, end_date)
    return start_date, n_days, end_date


def make_dt_range(start_date, n_points_per_day=2,
                  n_days=None, end_date=None):

    # Convert start/end/n_days to appropriate datetype
    start_date_dt, n_days, end_date_dt = sanitize_date_inputs(
        start_date=start_date, n_days=n_days, end_date=end_date,
        default_start_date=None, default_end_date=None)

    N = n_points_per_day * n_days
    dt_range =\
        [n_days * dt.timedelta(days=i) / N + start_date_dt
         for i in range(N + 1)]

    return dt_range


def find_closest_index(array, value):
    return (np.abs(array - value)).argmin()


def find_closest_index_dt(time_dt, posix_time_array):

    target_time_unx = UTC_to_UNX(time_dt)
    closest_time_index = find_closest_index(posix_time_array, target_time_unx)

    return closest_time_index


def format_energy_as_string(energy):
    """Helper function to convert energies into a human-readable format
    with correct unit."""

    if energy < 1e2:
        round_e = int(energy)
        unit = "eV"
    elif energy < 1e5:
        round_e = np.around(int(energy) / 1e3, decimals=1)
        unit = "keV"
    elif energy < 1e8:
        round_e = np.around(int(energy) / 1e6, decimals=1)
        unit = "MeV"
    else:
        round_e = np.around(int(energy) / 1e9, decimals=1)
        unit = "GeV"
    return "{E} {unit}".format(E=round_e, unit=unit)


def select_data_index_below_altitude(sc_posix_time, instrument_epoch_time,
                                     sc_altitude, altitude_cutoff):

    # Convert instrument epoch time into POSIX
    instrument_posix_time = UTC_to_UNX(instrument_epoch_time)
    # Find time indices when s/c altitude below a limit
    index_below_alt_sc = np.where(sc_altitude < altitude_cutoff)[0]
    # Note this time selection will be a series of disconnected intervals.
    # To get the endpoints of each interval, we can use the fact that consecutive indices
    # will only increase by one in the region of interest, and more than that at the edges.
    alt_index_edges = np.where(np.abs(np.ediff1d(index_below_alt_sc)) > 1)[0]
    alt_index_edges = np.append(alt_index_edges, len(index_below_alt_sc) - 1)
    # Select the altitudes for that time
    alt_below_alt_sc = sc_altitude[index_below_alt_sc]
    index_below_alt_instrument = [
        find_closest_index(instrument_posix_time, t) for t in sc_posix_time[index_below_alt_sc]
    ]

    return alt_index_edges, alt_below_alt_sc, index_below_alt_instrument


def rolling_sum(time_s, data, dt, skip=None):

    dt_i = np.ediff1d(time_s, to_end=(time_s[-1] - time_s[-2]))

    if skip is not None:
        dt_i = np.where(skip == 1, np.nan, dt_i)

    i = 0
    f = 1

    new_dt = []
    new_f = []
    new_t = []

    N = time_s.size

    while i < (N - 1):
        dt_if = np.nansum(dt_i[i:f])
        # print(dt_i[i:f], i, f)
        # input()

        if dt_if < dt and f != (N - 1):
            f += 1
            continue

        f_i = np.nansum(data[i:f])/(f - i)
        t_i = time_s[i] + dt_if/2

        new_dt.append(dt_if)
        new_f.append(f_i)
        new_t.append(t_i)

        i = f
        f = i + 1
        # print(i, f, time_i.size - 1)

    # print('loop done')

    new_epoch = UNX_to_UTC(new_t)
    # print("conv to epoch")

    # input()
    new_t = np.array(new_t)
    new_f = np.array(new_f)

    return new_epoch, new_t, new_f


def add_colorbar_outside(im, fig, ax, colorbar_width=0.01,
                         margin=0.01, **kwargs):

    '''Makes a colorbar that is located just outside of an existing axis
    (useful for multipanel comparison)'''

    # Based on StackOverflow post here:
    # https://stackoverflow.com/questions/71500468/positioning-multiple-colorbars-outside-of-subplots-matplotlib

    # Get bounding box that marks the boundaries of the axis:
    # [x0 (left), y0 (bottom), x1 (right), y1 (top)] of the axis.
    bbox = ax.get_position()

    # [left most position, bottom position, width, height] of color bar.
    cax = fig.add_axes(
        [bbox.x1 + margin, bbox.y0, colorbar_width, bbox.height])

    # Add colorbar using the new axis:
    cbar = fig.colorbar(im, cax=cax, **kwargs)

    return cbar


def continuous_index_interval(indices):
    '''Given a list of indices to access an array,
    identifies gaps, sorts the index by gaps,
    and returns the start and end index for each
    range of continuous increment.'''

    # Select indices in the middle of the apoapse pass
    N = indices.size

    # Get the difference between consecutive indices
    d_index = np.ediff1d(indices).astype("int")

    # Get where in the difference array, there's a gap:
    index_gap = np.where(d_index != 1)[0]

    # Make the starting index of each continuous
    # series of indices: (0, ...)
    start = np.insert(index_gap + 1, 0, 0).astype("int")

    # And use this to index the original index array
    # to get the actual start:
    start_index = indices[start]

    # Get the length of each continuous series
    # (starts with -1)
    n = np.ediff1d(np.append(np.insert(index_gap, 0, -1), N - 1))
    n = n.astype("int")

    end_index = start_index + n
    # midpoints = indices[(indexes + lengths / 2).astype("int")]

    return start_index, end_index


def broadcast_index(array_nd, array_1d, matching_axis_index=Ellipsis,
                    other_axis_index=np.newaxis):

    '''For a given 1D array (array_1d) that describes a single axis of
    a multidimensional array (array_nd), determine which axis index that
    has the same number of elements as the 1D array (axis_index)
    and the tuple slice that will allow the 1D array to be directly
    operated on with the multidimensional array (selection_tuple)

    NOTE: This will not work if there are multiple axes with the same length,
    use cautiously.

    matching_axis_index: the index access along the matching axis of
        the ND array. Ellipsis will access all, otherwise can select
        along the index.
    other_axis_index: the index access along all other indices.

    '''

    # Check if array ND actually an ND array:
    if not isinstance(array_nd, np.ndarray):
        raise ValueError(
            "Must provide multidimensional numpy array to"
            " determine broadcast index over.")

    # Check if array 1D is a flat iterable:
    # First, convert into a numpy array if not provided:
    if isinstance(array_1d, Iterable) and not isinstance(array_1d, np.ndarray):
        array_1d = np.array(array_1d)

    # Next, flatten and get shape:
    array_1d = array_1d.flatten()
    array_1d_shape = array_1d.shape

    # If array 1D has more than one dimension, raise error:
    if len(array_1d_shape) > 1:
        raise ValueError(
            "array_1d must be 1-dimensional, "
            "received array_1d with more that 1D.")

    n_array_1d = array_1d.size
    matching_index_found = False

    # print(array_nd.shape, array_1d.shape)

    selection_tuple = []
    for ax_index, ax_length in enumerate(array_nd.shape):
        if ax_length == n_array_1d:
            if matching_index_found:
                raise ValueError(
                    "Multiple matching array lengths found"
                    " in ND array, exiting...")
            selection_tuple.append(matching_axis_index)
            axis_index = ax_index
            matching_index_found = True
        else:
            selection_tuple.append(other_axis_index)
    selection_tuple = tuple(selection_tuple)

    return axis_index, selection_tuple
