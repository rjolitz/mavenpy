import os
import re
import glob
import datetime as dt

from . import helper
from . import specification

# Routines to look up where MAVEN data is located.
# Includes:
# - get_IDL_data_dir: Use IDL but forgot where your data
#   is saved? This function will locate your environment
#   variable or IDL startup file and pull the root_data dir
#   from it.
# - local_file_names: Given a data directory, an instrument
#   and/or instrument data type & level, and a time range,
#   returns the local paths to the dataset on your machine
#   if present.
# - regroup: sorts a list of file names by distinct version
#   numbers, to be used for multiple orbit files.
# - most_recent_version: retrieves the highest version #
#   and r # data file for a given list of files.


def get_IDL_data_dir():
    '''This routine will search the environment variables
    and in the IDL startup file for ROOT_DATA_DIR,
    which is prepended to every save path in SPEDAS.'''

    env = os.environ

    if "ROOT_DATA_DIR" in os.environ:
        # Return the root_data_dir if in the environment variable list:
        return env["ROOT_DATA_DIR"]

    elif "IDL_STARTUP" in env:
        idl_startup_file_path = env["IDL_STARTUP"]
        with open(idl_startup_file_path, 'r') as fh:
            for line_i in fh:
                line_lc_i = line_i.lower().strip()
                if line_lc_i.startswith("setenv"):
                    env_i = line_lc_i.split(',')[1]

                    if "root_data_dir" in env_i:
                        dir_i = env_i.split("=")[1].replace(
                            '\'', '').replace('\"', '').replace("\n", "")
                        return dir_i
        print("IDL startup file found, but no variable"
              " named 'root_data_dir' found.")
    else:
        print("No environment variable named 'IDL_STARTUP',"
              " can't find where IDL software saves the data.")

    return None


def regroup(paths, instrument_tla='v'):

    '''Split a list of filenames into lists of distinct
    orbit names with different versions, e.g. NGIMS'''

    groupings = {}
    key_order = []

    for p in paths:
        filename = os.path.split(p)[1]
        if instrument_tla == 'ngi':
            distinct_filename = (filename.split("_"))[3]
        else:
            distinct_filename = re.split("v[0-9]", filename)[0]

        if distinct_filename in groupings:
            groupings[distinct_filename].append(p)
        else:
            key_order.append(distinct_filename)
            groupings[distinct_filename] = [p]

    return key_order, groupings


def local_file_names(local_dir, instrument,
                     start_date=None, end_date=None, orbnum=None, n_days=None,
                     orb_to_t_func=None,
                     level="l2", dataset_name="", ext="", coord="", res="",
                     orbit_segment="", imaging_mode="",
                     mirror_remote_tree=True, source="ssl_sprg",
                     skip_missing_data=True):

    '''
    skip_only_safemode_days: only useful for PFP data, since
        NGIMS/IUVS does not always collect scans every day
    '''

    # First, see if maven dir exists:
    if not os.path.exists(local_dir):
        raise FileNotFoundError(
            "MAVEN directory not found, check if"
            " exists / drive connected.")

    # print(ext)
    # See if the requested dataset exists:
    specification.check_if_dataset_exist(
        instrument, level=level, ext=ext,
        dataset=dataset_name, coord=coord, res=res,
        orbit_segment=orbit_segment, imaging_mode=imaging_mode,
        orbit_num=orbnum)

    # Get the TLA for the instrument
    tla = instrument[:3].lower()

    # If we are mirroring the remote directory tree
    # (usually done if have existing data downloaded via SPEDAS),
    # construct the path to a given file.
    # Else, assume files in local directory.
    if mirror_remote_tree:

        # Filepath tuple
        filepath_tuple = specification.path(
            tla, level, ext=ext, res=res, dataset_name=dataset_name)
        # print(filepath_tuple)

        # Include data root if source provided
        if source:
            root = specification.remote_base_directory(source)

            # Check if the data root exists: halt if not
            local_root_dir = os.path.join(local_dir, *root)
            # print(local_root_dir)

            if not os.path.exists(local_root_dir):
                raise IOError(
                    "No dir available at {}, check if source"
                    " is correct.".format(local_root_dir))

            filepath_tuple = (*root, *filepath_tuple)
        else:
            local_root_dir = local_dir

        # Select the dirs up to fstring and see if that directory exists
        format_index =\
            [filepath_tuple.index(i) for i in filepath_tuple if '{' in i][0]
        local_path_wo_fstring = os.path.join(
            local_dir, *filepath_tuple[:format_index])

        if not os.path.exists(local_path_wo_fstring):
            raise IOError("No dir available at {}, check if source"
                          " is included".format(local_path_wo_fstring))

        local_path_fstring = os.path.join(local_dir, *filepath_tuple)

    else:
        local_path_fstring = local_dir
    # print(local_path_fstring)
    # input()
    filename_regex = specification.filename(
        tla, level=level, dataset_name=dataset_name, ext=ext,
        coord=coord, res=res,
        orbit_segment=orbit_segment, imaging_mode=imaging_mode,
        orbit_num=orbnum)
    # print(filename_regex)
    filepath_regex = os.path.join(local_path_fstring, filename_regex)
    # print(filepath_regex)
    # input()

    # Make datetime range
    dt_range = helper.daterange(
        start_date=start_date, end_date=end_date,
        orbnum=orbnum, n_days=n_days, orb_to_t_func=orb_to_t_func)
    dt_fstrings = [helper.dt_fstring(i) for i in dt_range]

    # Format the filepath with dt_info
    local_path_regex =\
        [filepath_regex.format(**dt_f_i) for dt_f_i in dt_fstrings]
    # print(local_path_regex)
    # input()

    # Search for files that match the regex:
    local_paths = []
    for path_regex_i, dt_i in zip(local_path_regex, dt_range):

        # print(path_regex_i)
        if "(.*)" in path_regex_i:
            dir_i, file_i = os.path.split(path_regex_i)
            if os.path.exists(dir_i):
                local_files_i = os.listdir(dir_i)
                matching_files =\
                    [i for i in local_files_i if re.match(re.compile(file_i), i)]

                matching_files = [os.path.join(dir_i, i) for i in matching_files]
            else:
                matching_files = []
        else:
            matching_files = glob.glob(path_regex_i)
        # print(matching_files)

        # input()

        if not matching_files:
            # print("Checking for missing")
            # If within last seven days and missing,
            # skip since might not be downlinked yet.
            if abs(dt_i - dt.datetime.now()).days < 7:
                local_paths.append("")
                continue

            if not skip_missing_data:
                # If deliberately not skipping missing data times,
                # first check if during safe mode
                if not specification.during_safemode(dt_i):
                    raise IOError(
                        "Missing data for {name} {level} {ext} {ds_i} on"
                        " date {date}'".format(
                            name=instrument, level=level,
                            ext=ext, ds_i=dataset_name,
                            date=dt_i.strftime("%Y-%m-%d")))

            local_paths.append("")
        else:

            # Split into groups of samefile name up to the version #
            # for multiple orbits in a day
            keyorder, distinct_filegroup_dict = regroup(
                matching_files, instrument_tla=tla)

            for filegroup in keyorder:
                matching_files_i = distinct_filegroup_dict[filegroup]
                local_path_v = most_recent_version(matching_files_i)
                # print(local_path_v)
                local_paths.append(local_path_v)

    return local_paths


def most_recent_version(files):
    ''''Retrieve the highest version/revision
    file from a list of files ending in
    "v##_r##.*"

    The v number is supposed to reflect significant
    changes in the dataset construction (e.g.
    new fields), while the r number is supposed
    to indicate the dataset version (e.g. reprocessing
    for telemetry gaps might have a higher r #).

    In practice, it is applied inconsistently across insruments.

    It is optimal to pull the newest,
    which is generally the highest v #
    then the highest r #.'''

    v_versions_match = [re.search("v[0-9][0-9]", f) for f in files]
    v_versions_int = [int(i.group()[1:]) for i in v_versions_match if i]

    alt_v_versions_match = [re.search("v[0-9][0-9][0-9]", f) for f in files]
    alt_v_versions_int = [int(i.group()[1:]) for i in alt_v_versions_match if i]

    r_versions_match = [re.search("r[0-9][0-9]", f) for f in files]
    r_versions_int = [int(i.group()[1:]) for i in r_versions_match if i]

    if r_versions_int and v_versions_int:
        r_v_version =\
            [(10*i + j) for (i, j) in zip(v_versions_int, r_versions_int)]
    elif v_versions_int:
        r_v_version = v_versions_int
    elif alt_v_versions_int:
        r_v_version = v_versions_int

    if v_versions_int or alt_v_versions_int:
        max_r_v = max(r_v_version)
        index_max_r_v = r_v_version.index(max_r_v)
        return files[index_max_r_v]
    else:
        return files[0]
