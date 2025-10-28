import os
import glob
import datetime as dt
from collections.abc import Iterable

from dateutil.relativedelta import relativedelta
from dateutil.parser import parse
import numpy as np
import matplotlib.dates

from .helper import find_closest_index_dt, daterange,\
    UNX_to_UTC, UTC_to_UNX, sanitize_date_inputs
from .read import read_sav
from .retrieve import download_file, Spinner
from .specification import in_datagap
from . import spice

# Routines for accessing ancillary MAVEN data,
# including:
# - reading the monthly IDL sav ephemeris (from 'maven_spice_eph.pro')
#   available at http://sprg.ssl.berkeley.edu/data/maven/anc/spice/sav/
# - reading the SPK orbit ephemeris (the NAIF '.orb' files)
# - get the orbit number for a given time OR the time
#   for a given orbit number, provided the orbit ephemeris
# - add a secondary axis to an existing Matplotlib Axis object
#   that describes the orbit number.


# Column information for the orbit-ephemeris files
# "maven_orb_rec_YYYYMMDD_YYYYMMDD_v?.orb"
dtype_orb_eph =\
    [("orbnum", "u4"),
     ("periapse_utc", "U20"),
     ("periapse_met", "U20"),
     ("apoapse_utc", "U20"),
     ("solar_longitude_deg", "f8"),
     ("solar_latitude_deg", "f8"),
     ("spacecraft_longitude_deg", "f8"),
     ("spacecraft_latitude_deg", "f8"),
     ("spacecraft_altitude_km", "f8"),
     ("spacecraft_sun_distance_km", "f8")]
delimiter_orb_eph =\
    [5 + 2, 20 + 2, 20 + 2, 20 + 2, 7 + 2, 7 + 2,
     7 + 2, 7 + 2, 10 + 2, 12 + 2]


# Monthly ephemeris in IDL sav files
# Contain x, y, z (km) in MSO and GEO
# as well as vx, vy, vz (km/s), but not orb number
monthly_ephemeris_path = os.path.join("maven", "anc", "spice", "sav")
monthly_column_names =\
    {"x": "spacecraft_{}_x_km", "y": "spacecraft_{}_y_km",
     "z": "spacecraft_{}_z_km", "vx": "spacecraft_{}_vx_kms",
     "vy": "spacecraft_{}_vy_kms", "vz": "spacecraft_{}_vz_kms",
     "t": "time_unix"}
monthly_ephemeris_url =\
    "http://sprg.ssl.berkeley.edu/data/maven/anc/spice/sav/"


def read_spacecraft_ephemeris(data_directory, coordinate_system,
                              start_date=None, end_date=None, n_days=None,
                              restrict_to_timerange=False, verbose=None,
                              download_if_not_available=True):

    """Return a dictionary containing the MAVEN spacecraft position
    (x, y, z) and velocity (vx, vy, vz) in a coordinate system (mso or geo)
    for a given time range. Read from monthly IDL sav ephemeris
    available at http://sprg.ssl.berkeley.edu/data/maven/anc/spice/sav/
    (produced by 'maven_spice_eph.pro').

    data_directory: containing directory
    coordinate_system: string, ephemeris coordinate ('mso' or 'geo')
    start_date, end_date: strings or datetimes indicating start/end time
    n_days: number of days, as opposed to end_date
    restrict_to_timerange: Boolean, set true if want to cut the returned
        ephemeris to the seeked time. default False."""

    # Make datetime range
    dt_range = daterange(
        start_date=start_date, end_date=end_date, n_days=n_days)
    start_dt, end_dt = dt_range[0], dt_range[-1]
    # print(start_dt, end_dt)
    # input()

    # Get the file name
    # coordinate_system can be 'mso' or 'geo'
    data_name = "maven_spacecraft_{coord}_{{yyyymm_i}}.sav".format(
        coord=coordinate_system)

    n_months = relativedelta(
        dt.datetime(end_dt.year, end_dt.month, 1),
        dt.datetime(start_dt.year, start_dt.month, 1)).months
    n_months += 1

    dt_i = [dt.datetime(start_dt.year, start_dt.month, 1) +
            relativedelta(months=m) for m in range(n_months)]
    files_re =\
        [data_name.format(yyyymm_i=d_i.strftime("%Y%m")) for d_i in dt_i]

    if verbose:
        print("Files to be retrieved:")
        print(files_re)

    # Check if data exists, else, download:
    files = []
    paths = []
    local_dir = os.path.join(data_directory, monthly_ephemeris_path)
    for file_i in files_re:
        path_i = os.path.join(local_dir, file_i)
        matching_path_i = glob.glob(path_i)
        matching_file_i = [os.path.split(i)[1] for i in matching_path_i]

        if verbose:
            print('Searched path:', path_i)
            print('Matching path:', matching_path_i)
            print('Matching file:', matching_file_i)

        if len(matching_file_i) == 0 and download_if_not_available:
            # download
            download_file(monthly_ephemeris_url + file_i, path_i, verbose=True)
            matching_file_i = file_i
            # pass
        else:
            matching_file_i = matching_file_i[0]
        files.append(matching_file_i)
        paths.append(os.path.join(local_dir, matching_file_i))

    # Load the data
    if verbose:
        print("Loading ephemeris...")
        spinner_i = Spinner(len(paths), 'files')
        n_loaded = 0

    for i, file_i in enumerate(paths):
        # print(file_i)
        anc_dataset = read_sav(file_i)
        if i == 0:
            all_data = anc_dataset
        else:
            # Append subsequent datasets
            for n in anc_dataset:
                dat_i = np.append(all_data[n], anc_dataset[n])
                all_data[n] = dat_i
        if verbose:
            n_loaded += 1
            spinner_i.increment(n_loaded)

    # Add UTC time to data structure
    t = all_data['t']
    utc_i = UNX_to_UTC(t)
    all_data["time_utc"] = utc_i

    # Cut the data to the appropriate time
    if restrict_to_timerange:
        i = find_closest_index_dt(start_dt, t)
        f = find_closest_index_dt(end_dt, t)

        for name_i in all_data:
            dat_i = all_data[name_i]
            dat_i = dat_i[i:f]
            all_data[name_i] = dat_i

    return all_data


def read_orbit_ephemeris(data_directory,
                         start_date=None, end_date=None, n_days=None,
                         only_over_timerange=None,
                         mirror_spedas_dir_tree=True,
                         fields=None,
                         download_if_not_available=True,
                         prompt_for_download=True):

    """Return the MAVEN orbit numbers and periapse/apoapse times
    for a given duration in the mission

    data_directory: containing directory
    only_over_time_range
    """

    # Make datetime range
    dt_range = daterange(
        start_date=start_date, end_date=end_date, n_days=n_days)
    start_dt, end_dt = dt_range[0], dt_range[-1]

    # Retrieve the .orb SPK spice kernels:
    paths = spice.retrieve_kernels(
        data_directory, 'maven', 'spk',
        start_date=start_dt, end_date=end_dt,
        download_if_not_available=download_if_not_available,
        spk_ext='orb',
        use_most_recent=None,
        mirror_spedas_dir_tree=mirror_spedas_dir_tree,
        prompt_for_download=prompt_for_download)

    if isinstance(paths, str):
        paths = [paths]

    # Load the data
    for i, file_i in enumerate(paths):
        # print(file_i)
        anc_dataset = read_orb(file_i, utc_to_dt=True, fields=fields)
        unx_i = UTC_to_UNX(anc_dataset['periapse_utc'])
        anc_dataset["periapse_time_unix"] = unx_i

        if i == 0:
            all_data = anc_dataset
        else:
            # Append subsequent datasets
            for n in anc_dataset:
                dat_i = np.append(all_data[n], anc_dataset[n])
                all_data[n] = dat_i

    # Cut the data to the appropriate time
    if only_over_timerange:
        t = all_data["periapse_time_unix"]
        i = find_closest_index_dt(start_dt, t)
        f = find_closest_index_dt(end_dt, t)

        for name_i in all_data:
            dat_i = all_data[name_i]
            dat_i = dat_i[i:f]
            all_data[name_i] = dat_i

    return all_data


def read_orb(filename, utc_to_dt=None, fields=None):
    '''Read function for .orb files in spk,
    which are plain text with two header rows
    followed by orbit number and multiple spacecraft
    parameters associated with orbit number.'''

    if not fields:
        fields = [i[0] for i in dtype_orb_eph]

    data = np.genfromtxt(
        filename, delimiter=delimiter_orb_eph,
        skip_header=2, dtype=dtype_orb_eph)

    # Convert to dictionary
    keep = [("Unable" not in i) and ("Unable" not in j) for
            i, j in zip(data["apoapse_utc"], data["periapse_utc"])]

    data_dict = {}
    for name in fields:
        data_i = data[name][keep]

        # if converting utc to datetime object
        if utc_to_dt and "utc" in name:
            data_i = [dt.datetime.strptime(i, "%Y %b %d %H:%M:%S")
                      for i in data_i]

        data_dict[name] = data_i

    return data_dict


def orbit_num(time_utc=None, time_unix=None, orbit_num=None,
              ephemeris=None, data_directory=None,
              start_date=None, end_date=None):
    '''For a given UTC time and/or POSIX time(s),
    return orbit number. Will load the ephemeris if not
    provided. Adapted from MVN_ORBIT_NUM.pro.'''

    # Raise error if data directory AND ephemeris not provided
    # Need data sets to retrieve orbit number.
    if data_directory is None and ephemeris is None:
        raise FileNotFoundError(
            "Need either data_directory (which contains NAIF"
            " files) or ephemeris (loaded by read_orbit_ephemeris)!")

    # Assuming data directory provided, check if exists/plugged in:
    if ephemeris is None:
        if not os.path.exists(data_directory):
            raise FileNotFoundError(
                "MAVEN directory not found, check "
                "if exists / drive connected.")

    # Check if time has been provided, either as a orbit number
    # or a time_utc/time_unx
    if (time_utc is None and time_unix is None) and orbit_num is None:
        raise ValueError(
            "Need a time (UTC or Unix) or orbit number"
            " to retrieve matching orbit number or time for.")

    # Convert time_utc to datetime if string
    if isinstance(time_utc, str):
        time_utc = parse(time_utc)

    if orbit_num is None:
        if time_utc is None:
            time_utc = UNX_to_UTC(time_unix)

        if time_unix is None:
            time_unix = UTC_to_UNX(time_utc)

    # Load ephemeris if not provided:
    if not ephemeris:
        if time_utc is None and time_unix is None:
            eph_start = spice.start_ephemeris_dt
            eph_end = dt.datetime.now()

        if orbit_num is None:
            if isinstance(time_unix, Iterable):
                eph_start, eph_end = time_utc[0], time_utc[-1]
            else:
                eph_start = time_utc - dt.timedelta(days=1)
                eph_end = time_utc + dt.timedelta(days=1)

            ephemeris = read_orbit_ephemeris(
                data_directory,
                start_date=eph_start, end_date=eph_end,
                fields=['orbnum', 'periapse_utc'])

    orbnum = ephemeris["orbnum"]
    eph_unx = ephemeris["periapse_time_unix"]

    if time_utc is None and time_unix is None:
        return UNX_to_UTC(np.interp(orbit_num, orbnum, eph_unx))
    else:
        return np.interp(time_unix, eph_unx, orbnum)


def add_orbit_axis(ax, data_directory=None, ephemeris=None,
                   label=True):

    '''Add secondary axis label with
    orbit number to existing matplotlib axis 'ax' '''

    # Raise error if data directory AND ephemeris not provided
    # Need data sets to retrieve orbit number.
    if data_directory is None and ephemeris is None:
        raise FileNotFoundError(
            "Need either data_directory (which contains NAIF"
            " files) or ephemeris (loaded by read_orbit_ephemeris)!")

    # Check if ephemeris loaded, and if not, see if directory exists.
    # Then make it with the limits of the provided axis.
    # Note! requires a time axis in UTC
    if ephemeris is None:
        if not os.path.exists(data_directory):
            raise FileNotFoundError(
                "MAVEN directory not found, check "
                "if exists / drive connected.")
        numminmax = ax.get_xlim()
        epochmin, epochmax = matplotlib.dates.num2date(numminmax)

        ephemeris = read_orbit_ephemeris(
            data_directory,
            start_date=epochmin, end_date=epochmax,
            fields=['orbnum', 'periapse_utc'])

    # Functions from UTC time to orbit and back
    def t_to_orb(t):
        t = matplotlib.dates.num2date(t)
        # print('t is ', t)
        # print(type(t[0]))
        return orbit_num(time_utc=t, ephemeris=ephemeris)

    def orb_to_t(x):
        # print('x is ', x)
        t = orbit_num(orbit_num=x, ephemeris=ephemeris)
        return matplotlib.dates.date2num(t)

    # Add the secondary axis
    secax = ax.secondary_xaxis('top', functions=(t_to_orb, orb_to_t))
    if label is not None:
        secax.set_xlabel(label)

    return secax


# Routines to retrieve / sort sections of an orbit.
# These were developed to do smart loading of parts of a
# dataset.

# 'segment_orbit' will load the orbit ephemeris and return
# a dictionary containing the Unix time, UTC time, and orbit number
# for the center (+ start and end) of an orbit segment
# given the string description of the center (apoapse or periapse)
# and width of the orbit segment (e.g. 0.15).

# 'sort_segments_by_day' will identify periods (# of days)
# where orbit segments are completely enclosed.
# e.g. if getting 15% of orbits within periapse and
# an orbit segment is continuing onto the next day,
# add a day and check again if orbit segment stretching to next day.
# Repeat until the orbit segment doesn't cross the day boundary.

# This is particularly useful for a mission-long analysis that
# is focused on specific orbit segments, since it means you can
# load a few days at a time instead of an entire dataset into
# memory (at which point, just do a day by day analysis).


def segment_orbit(data_directory, start_time, end_time, dorb,
                  center_on='apoapse',
                  download_if_not_available=None,
                  t_format_string='%Y-%m-%d/%H:%M',
                  verbose=False):

    '''Retrieve segments of time between a start UTC
    and end UTC centered on a given orbit number
    and with a +/- of dorb.

    If dorb is None or 0, then can be used to retrieve
    time info of a point along the orbit segment, like
    the apoapse or periapse.'''

    if center_on == 'apoapse':
        orbit_offset = 0.5
    elif center_on == 'periapse':
        orbit_offset = 0
    else:
        raise ValueError(
            "Unrecognized argument for 'center_on' ({})."
            " Try 'apopase' or 'periapse'.".format(center_on))

    # if not using a dorb width, don't need start/end
    # info:
    if dorb == 0:
        orbit_segment_name = ("center",)
    else:
        orbit_segment_name = ("start", "center", "end")

    # Retrieve orbit ephemeris for this time period
    eph = read_orbit_ephemeris(
        data_directory,
        start_date=start_time, end_date=end_time,
        download_if_not_available=download_if_not_available)
    print("Loaded ephemeris.")

    # Get orbit info
    orbnum = eph["orbnum"]
    utc = eph["periapse_utc"]
    unx = UTC_to_UNX(utc)
    # print(utc)
    # input()

    # Start/end datetime and UNX to get orbits:
    start_utc, n_days, end_utc = sanitize_date_inputs(
        start_date=start_time, end_date=end_time)
    end_utc = end_utc + dt.timedelta(days=1, seconds=-1)

    # Get the unx at the boundaries
    start_unx = UTC_to_UNX(start_utc)
    end_unx = UTC_to_UNX(end_utc)

    # Get orbit num at the start / end
    start_orbnum = np.interp(start_unx, unx, orbnum)
    end_orbnum = np.interp(end_unx, unx, orbnum)

    # Get the orbits if the center of the apoapse, +/- dorb,
    # is within the start/end
    start_orb = np.floor(start_orbnum).astype('int')
    if start_orbnum > (start_orb + (orbit_offset - dorb)):
        start_orb += 1
    end_orb = np.floor(end_orbnum).astype('int')
    if end_orbnum < (end_orb + (orbit_offset + dorb)):
        end_orb -= 1

    n_orbits = (end_orb - start_orb) + 1

    if verbose:
        print("For UTC time range: {} - {}".format(
            start_utc.strftime(t_format_string),
            end_utc.strftime(t_format_string)))
        print("Range of orbits: {} - {}".format(start_orbnum, end_orbnum))
        print("Loaded orbits: {} - {}".format(start_orb, end_orb))

        print("Retrieved UTC: {} - {}".format(start_orb, end_orb))
        # print(orbnum)
        print("# orbits: ", n_orbits)
    orbits = np.linspace(start_orb, end_orb, n_orbits)
    # print(orbits)

    # Get the center of the desired orbit,
    # and then get the time info before and after segment:
    ctr_orbnum = orbits + orbit_offset
    orbit_dict = {}
    for t_at in orbit_segment_name:

        if t_at == "start":
            orbnum_i = (ctr_orbnum - dorb)
        elif t_at == "end":
            orbnum_i = (ctr_orbnum + dorb)
        elif t_at == "center":
            orbnum_i = ctr_orbnum
        else:
            raise IOError(
                "t_at unrecognized, currently only "
                "'start'/'end'/'center'.")

        orb_unx_i = np.interp(orbnum_i, orbnum, unx)
        orb_utc_i = UNX_to_UTC(orb_unx_i)
        orb_text_i = [i.strftime(t_format_string) for i in orb_utc_i]

        # Add to structure:
        orbit_t_i =\
            {"unx": orb_unx_i, "utc": orb_utc_i,
             "num": orbnum_i, "str": orb_text_i}
        orbit_dict[t_at] = orbit_t_i
        # print(t_at)
        # print(orbnum_i)
        # print(orb_utc_i)
        # input()

    return orbit_dict


def sort_segments_by_day(orbit_dicts,
                         start_date=None, end_date=None, n_days=None,
                         verbose=None, data_gaps=None):

    '''Reorganize orbit segments into a data structure
    that is ordered in increments of days. To handle orbit segments
    that cross over a day boundary, a day is added.

    This approach allows selective downloading of periods of days
    based on orbit fragments, to reduce memory impact.

    start, center, end: dicts containing num, utc, and unx.

    '''
    start_date, n_days, end_date = sanitize_date_inputs(
        start_date=start_date, end_date=end_date)

    # Pull the used parts of the orbit segment info:
    start_orb_unx = orbit_dicts["start"]["unx"]
    end_orb_unx = orbit_dicts["end"]["unx"]

    # Identify and parse datagaps if supplied:
    if data_gaps:
        # Convert into datetime objects:
        if isinstance(data_gaps[0], str):
            data_gaps_dt = [[parse(j) for j in i.split("-")] for i in data_gaps]
        elif isinstance(data_gaps[0], dt.datetime):
            data_gaps_dt = data_gaps
        else:
            raise IOError(
                "Unfamiliar datatype, only recognize data_gaps"
                "as series of strings with a hyphen (-) to separate"
                " the time range (e.g. '2021/10/9-2021/10/10') "
                "or pairs of datetime objects.")

    # Identify orbit segments that start on one day
    # and end on the next:
    day_index = 0
    n_load_days = 1

    # Arrays that contain the day indexing
    day_index_arr = []
    n_load_day_arr = []

    # Initialize dictionary to dump sorted arrays into:
    sorted_orbit_dicts = {}

    while day_index < n_days:
        # Get the datetime & UNX for the start of the day:
        day_start_utc_i = start_date + dt.timedelta(days=day_index)
        day_start_unx = UTC_to_UNX(day_start_utc_i)

        # Check if the start is in the datagap. If so, skip to next.
        day_end_utc = day_start_utc_i + dt.timedelta(
            days=n_load_days, seconds=-1)
        # if in_datagap(day_start_utc_i, data_gaps_dt):
        # Need better way to handle last data on a datagap or first data
        # will completely skip days containing a datagap
        if in_datagap(day_start_utc_i, data_gaps_dt) or in_datagap(day_end_utc, data_gaps_dt):
            day_index += 1
            n_load_days = 1
            continue

        # if in_datagap(day_start_utc_i, data_gaps_dt):
        #     day_index += 1
        #     n_load_days = 1
        #     continue

        # if in_datagap(day_end_utc, data_gaps_dt):
        #     n_load_days += 1
        #     continue

        # Get the index of the orbit segment after the day start:
        i = np.searchsorted(start_orb_unx, day_start_unx)

        # Datetime & UNX of last second of the day(s)
        day_end_utc = day_start_utc_i + dt.timedelta(
            days=n_load_days, seconds=-1)
        day_end_unx = UTC_to_UNX(day_end_utc)

        # Check if the subsequent day is in the datagap,

        # Get the index of the first orbit segment start and end after
        # the end of day
        last_start_f = np.searchsorted(start_orb_unx, day_end_unx)
        last_end_f = np.searchsorted(end_orb_unx, day_end_unx)

        # If these indices are not the same, its because the first orbit
        # segment that ends after the day ends started on the first day.
        # Thus we want to load more days:
        # But don't want to add a day if that day is in the data gap.
        next_day = day_end_utc + dt.timedelta(seconds=1)
        if last_start_f != last_end_f and not in_datagap(
                next_day, data_gaps_dt):
            n_load_days += 1
            # print("added a day")
            continue

        f = last_start_f

        if verbose:
            # Conv to UTC times for intelligibility:
            start_utc_first_orb_i = UNX_to_UTC(start_orb_unx[i])
            end_utc_first_orb_i = UNX_to_UTC(end_orb_unx[i])
            start_utc_last_orb_i = UNX_to_UTC(start_orb_unx[f - 1])
            end_utc_last_orb_i = UNX_to_UTC(end_orb_unx[f - 1])
            if f < len(start_orb_unx):
                start_utc_first_excl_orb_i = UNX_to_UTC(start_orb_unx[f])
                end_utc_first_excl_orb_i = UNX_to_UTC(end_orb_unx[f])

            # print(start_orb_utc)
            t_fmt = "%Y/%m/%d-%H:%M"
            print(i, f, n_load_days)
            print(last_start_f, last_end_f)
            print("Start UTC of day: ",
                  day_start_utc_i.strftime(t_fmt))
            print("Start UTC of first orbit of period: ",
                  start_utc_first_orb_i.strftime(t_fmt))
            print("End UTC of first orbit of period: ",
                  end_utc_first_orb_i.strftime(t_fmt))
            print("Start UTC of last orbit:",
                  start_utc_last_orb_i.strftime(t_fmt))
            print("End UTC of last orbit:",
                  end_utc_last_orb_i.strftime(t_fmt))
            if f < len(start_orb_unx):
                print("Start UTC of first excluded orbit:",
                      start_utc_first_excl_orb_i.strftime(t_fmt))
                print("End UTC of first excluded orbit:",
                      end_utc_first_excl_orb_i.strftime(t_fmt))
            else:
                print("No excluded orbits, end of loaded ephemeris.")
            # input()

        for obs_type in orbit_dicts:
            # Get the orbit information for the
            # obs_type (e.g. start, end, center)
            obs_data_i = orbit_dicts[obs_type]

            # Get the equiv for the sorted, or
            # create if not found:
            if obs_type not in sorted_orbit_dicts:
                sorted_obs_dict_i = {}
            else:
                sorted_obs_dict_i = sorted_orbit_dicts[obs_type]

            # for each of the variables, select the
            # subset and then append that as an array:
            for var in obs_data_i:
                # get the variable obs info (e.g. utc)
                obs_var_i = obs_data_i[var]
                # print(var)
                # get the old sorted orb info,
                # create the equivalent array if not in
                # the sorted equivalent:
                if var not in sorted_obs_dict_i:
                    sorted_obs_var_i = []
                else:
                    sorted_obs_var_i = sorted_obs_dict_i[var]
                # subset:
                subset_obs_var_i = obs_var_i[i:f]
                # append:
                sorted_obs_var_i.append(subset_obs_var_i)
                # reassign:
                sorted_obs_dict_i[var] = sorted_obs_var_i

            sorted_orbit_dicts[obs_type] = sorted_obs_dict_i

        # orbsegment_ctr_time_utc_arr.append(center_utc[i:f])
        # orbsegment_ctr_orbnum_arr.append(center_orb[i:f])
        # orbsegment_start_unx_arr.append(start_orb_unx[i:f])
        # orbsegment_end_unx_arr.append(end_orb_unx[i:f])

        day_index_arr.append(day_index)
        n_load_day_arr.append(n_load_days)

        day_index += n_load_days
        n_load_days = 1

    if verbose:
        print(day_index_arr)
        print(n_load_day_arr)
        print(sorted_orbit_dicts["center"]["num"])
        # print([[i.strftime("%Y-%m-%d/%H:%M") for i in j]
        #        for j in sorted_orbit_dicts["center"]["utc"]])
        print(sorted_orbit_dicts["center"]["str"])

    return day_index_arr, n_load_day_arr, sorted_orbit_dicts
