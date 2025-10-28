import os
import datetime as dt
import re
from collections.abc import Iterable

import numpy as np
import spiceypy
from dateutil.parser import parse as parsedt
from dateutil.relativedelta import relativedelta
from dateutil.rrule import rrule, WEEKLY, DAILY

from . import helper
from . import retrieve
import requests


naif_url = "https://naif.jpl.nasa.gov/pub/naif"
kernel_ids = ['ck', 'ek', 'fk', 'ik', 'lsk', 'pck', 'sclk', 'spk']

# CK formats
# The platforms available:
ck_platform_options = ['sc', 'swea', 'app']

# the finalized weekly files have "rel" in them:
# ex: 'mvn_sc_rel_140901_140907_v01.bc'
# ex: 'mvn_app_rel_141013_141019_v01.bc'
# exceptions:
# mvn_sc_rel_131118_131122_v01.bc,
# 'mvn_sc_rel_150415_150416_v01.bc',
# 'mvn_sc_rel_160105_160112_v01.bc',
# 'mvn_sc_rel_210830_210831_v01.bc
ck_weekly_file_format = "mvn_{platform}_rel_{yymmdd}_" +\
    "[0-9]"*6 + "_v[0-9][0-9].bc"

# daily files have "red"
# ex: 'mvn_sc_red_240101_v01.bc'
yymmdd = "[0-9]"*6

ck_daily_file_format = "mvn_{platform}_red_" + yymmdd + "_v[0-9][0-9].bc"

# recent files every 5 days
# ex: 'mvn_sc_rec_240311_240315_v03.bc'
ck_5day_recent_file_format =\
    "mvn_{platform}_rec_" + yymmdd + "_" + yymmdd + "_v[0-9][0-9].bc"

# Mappings of kernel name to file format.
# Generic kernels:
# - pck (planetary constant kernel) pck?????.tpc:
#   Contains spin axis and size of most solar system bodies.
# - lsk (Leap second kernel) (naif????.tls)
#   Contains times of leap seconds.
#   ???? contains the version number and increments by 1
#   with every new leap second.
# - spk (Relative solar system barycenters): (de430.bsp)
# - spk_(planet) e.g. Mars
# - Mars: mar097.bsp
# - Saturn: sat428.bsp
# - Jupiter:jup3[0-9][0-9].bsp
generic_kernel_file_fmt = {
    "pck": 'pck0001[0-9].tpc',
    "lsk": 'naif001[0-9].tls',
    "spk_planets": 'de4[0-9][0-9].bsp',
    "spk_satellites_mars": 'mar[0-9][0-9][0-9].bsp',
    "spk_satellites_jupiter": 'jup3[0-9][0-9].bsp',
    "spk_satellites_saturn": 'sat4[0-9][0-9].bsp'
}

# There is one FK MAVEN file ONLY in the tplot software (maven_misc.tf)
# This is the one that contains info for the "MSO" frame
# (as opposed to maven_v??.tf, which contains info for "MAVEN_MSO")
# Can cause pxform failures bc no frame for MAVEN_MSO
# findable if not loaded.
# 2023/04/04: Reviewed with Dave Mitchell what maven_misc.tf includes.
# - MSO done correctly (same as MAVEN_MSO)
# - defines sun z by flipping MSO
# - alternate spacecraft frame that flips (MAVEN_SCL)
# - frame MAVEN_MG_PY and _MY
#   (Changing the names to MAVEN_MAG1 and M2): Same, different names
# - STATIC flips X axes
# - adds two codes for Comet siding spring, two SIDING_SPRING or CSS
# mvn_fk_misc = "/Users/rjolitz/Software/spd/projects/maven/general"\
#     "/spice/kernels/fk/maven_misc.tf"
# naif_kernels.append(mvn_fk_misc)

# Note: some of the instrument kernels are searched by mvn_spice_kernels
# but don't/have never existed on NAIF, to wit:
# - "maven_lpw_v[0-9][0-9].ti",
# - "maven_mag_v[0-9][0-9].ti"

maven_file_fmt =\
    {"fk": 'maven_v[0-9][0-9].tf',
     'sclk': 'MVN_SCLKSCET.001[0-9][0-9].tsc',
     "spk": {"predict": "maven_orb.{ext}$",
             "quarterly": "maven_orb_rec_{yymmdd_i}_{yymmdd_f}_v[0-9].{ext}",
             "recent": "maven_orb_rec.{ext}$"},
     "ck": {"weekly": ck_weekly_file_format,
            "daily": ck_daily_file_format,
            "recent": ck_5day_recent_file_format},
     "ik": ["maven_ant_v[0-9][0-9].ti", "maven_euv_v[0-9][0-9].ti",
            "maven_iuvs_v[0-9][0-9].ti", "maven_ngims_v[0-9][0-9].ti",
            "maven_sep_v[0-9][0-9].ti", "maven_static_v[0-9][0-9].ti",
            "maven_swea_v[0-9][0-9].ti", "maven_swia_v[0-9][0-9].ti"]
     }

all_file_fmts =\
    {"maven": maven_file_fmt, "generic_kernels": generic_kernel_file_fmt}

expanded_kernel_name =\
    {"fk": "Frame kernel",
     "lsk": "Leap seconds kernel",
     "spk_planets": "Relative planetary barycenter kernel",
     "spk_satellites_mars": "Mars info kernel",
     "sclk": "Spacecraft clock kernel",
     "spk": "Spacecraft and planet ephemeris",
     "ck": "Orientation/Attitude kernel",
     "ik": "Instrument kernel",
     "pck": "Planetary constants kernel ()"}

# Missions located in the root directory (pub/naif/)
# Note: some of the missions are listed in both
#  active and the tobearchived.
# Also some are co-listed with officially archived ones,
# but those dirs are typically empty (e.g. MGS)
active_missions =\
    ["mex", "bepicolumbo", "exomars2016", "vex", "rosetta",
     "dart", "jwst", "europaclipper", "psyche", "cassini",
     "maven", "spp"]

# Tobearchived missions are stored in (pub/naif/pds/pds4):
# - bc (bepi), dart, em16 (TGO), insight, ladee, mars2020
notyet_officially_archived =\
    ['bc', "dart", "em16", "insight", "ladee",
     "mars2020", "maven", "orex", "vco"]

# Archived missions are stored in (pub/naif/pds/data):
officially_archived_missions =\
    {"clementine": "clem1-l",
     "cassini": "co-s_j_e_v",
     "dawn": "dawn-m_a",
     "deep impact": "di-c",
     "epoxi": "dif-c_e_x",
     "deep space 1": "ds1-a_c",
     "grail": "grail-l", "hayabusa": "hay-a",
     "juno": "jno-j_e_ss",
     "lro": "lro-l", "mer1": "mer1-m",
     "mer2": "mer2-m", "messenger": "mess-e_v_h",
     "mex": "mex-e_m", "mgs": "mgs-m", "mro": "mro-m",
     "msl": "msl-m", "near": "near-a",
     "new horizons": "nh-j_p_ss", "m01": "ody-m",
     "rosetta": "ro_rl-e_m_a_c",
     "stardust": "sdu-c", "akatsuki": "vco-v", "vex": "vex-e_v"}


def get_kernel_subfolder(group):
    '''
    group: string, the group of kernels that are
      searched, e.g. "generic_kernels" or "MAVEN".
    '''

    group = group.lower()

    # Now add the kernel category you're searching for:
    if group in ["generic_kernels", "deprecated_kernels"]:
        kernel_path = (group,)
    elif group in active_missions:
        kernel_path = (group.upper(), "kernels")
    elif group in notyet_officially_archived:
        kernel_path =\
            ("pds", "pds4", group,
             "{}_spice".format(group), "spice_kernels")
    elif group in officially_archived_missions:
        group_shortname = officially_archived_missions[group]
        group_tla = group_shortname[:3]
        group_dir_name = "{}-spice-6-v1.0".format(group_shortname)
        group_subdir = "{}sp_1000".format(group_tla)

        kernel_path =\
            ("pds", "data", group_dir_name,
             group_subdir, "data")

    else:
        raise IOError(
            "Don't know where {} kernels are, not in"
            " list of active flight, to-be-achived, or "
            "currently archived data.".format(group))

    return kernel_path


def find_local_files(local_dir, kernel_group, kernel_name,
                     start_dt=None, end_dt=None, spk_ext='bsp',
                     ck_platform=None,
                     use_most_recent=None,
                     mirror_spedas_dir_tree=True,
                     verbose=None):

    ''' Returns local files matching a given format.

    folder: string, path to location to search.
    kernel_group: string, the group of kernels that are
      searched, e.g. "generic_kernels" or "MAVEN".
    file_name_format: a string or list of strings that describes
        the searched for folder, such as
        'mvn_sc_red_240213_v[0-9][0-9].bc'
    mirror_spedas_file_tree: boolean, True/False if searching/creating
        spice kernels in the same subdirectory where SPEDAS saves them
        (root_dir + "/misc/spice/naif/" based on spice_file_source.pro)
    '''
    # print(kernel_group, kernel_name)

    # Look for or make a file tree consistent with
    # where the MAVEN spedas software saves the spice kernels:
    if mirror_spedas_dir_tree:

        # First, see if local dir exists:
        if not os.path.exists(local_dir):
            raise IOError(
                "No folder '{}' exists, check if "
                "path correct or drive connected.".format(local_dir))

        # Modify the local dir with the spice location per SPEDAS
        local_dir = os.path.join(local_dir, "misc", "spice", "naif")

    # Location of kernels
    # Determine local directory where spice kernels are.
    # This is based on where the SPEDAS distribution puts them
    # (/misc/spice/naif/MAVEN/kernels).
    subpath = get_kernel_subfolder(kernel_group)

    # I use "_" to indicate subfolders for the
    # generic kernels. This gets me into trouble
    # for discerning between Mars/Jupiter/etc.
    # So I exempt the last element of those.
    # There *has* to be a better way of doing this,
    # but I am running out of time.
    kernel_name_split = kernel_name.split("_")
    if "satellites" in kernel_name_split:
        kernel_name_split = kernel_name_split[:-1]
    subpath = (*subpath, *kernel_name_split)

    if mirror_spedas_dir_tree:
        local_dir_i = os.path.join(local_dir, *subpath)
    else:
        # 2/11/25: If not mirroring SPEDAS (which keeps
        # generic_kernels separated from the MAVEN distribution)
        # and instead pulling it into a specific folder containing
        # all MAVEN-related kernels only identified by kernel type
        # (e.g. spk, lsk), then only make a directory for that
        # subtype and save everything there.
        local_dir_i = os.path.join(local_dir, kernel_name_split[0])
    if verbose:
        print("Searching in:", local_dir_i)

    # Get list of local files present in the directory:

    # Get the file format
    files_fmt = all_file_fmts[kernel_group][kernel_name]
    # print(files_fmt)

    if kernel_name == "spk":
        # print(files_fmt)
        if isinstance(spk_ext, str):
            files_fmt_i = spk_file_names(start_dt, end_dt, files_fmt, spk_ext)
        elif isinstance(spk_ext, tuple):

            files_fmt_i_all = []
            for spk_ext_i in spk_ext:
                files_fmt_i = spk_file_names(
                    start_dt, end_dt, files_fmt, spk_ext_i)
                files_fmt_i_all += files_fmt_i
            files_fmt_i = files_fmt_i_all

        # print(files_fmt_i)
        # input()
        # if spk_ext == "bsp":
        #     version_format = 'v[0-9]'
        # else:
        #     version_format = None
    elif kernel_name == "ck":
        # print('ck')
        files_fmt_i = ck_file_names(ck_platform, start_dt, end_dt, files_fmt)
        # print(files_fmt_i)
        version_format = 'v[0-9][0-9]'
    elif isinstance(files_fmt, str):
        files_fmt_i = (files_fmt,)
        version_format = None
    else:
        files_fmt_i = files_fmt
        version_format = None

    # If doesn't exist, exit! There are no local files to find!
    # However, do need to provide the searched for file format,
    # subpath, and local dir saved to in order to download spice
    # kernels.
    if not os.path.exists(local_dir_i):
        print(local_dir_i, "does not exist")
        local_info_i =\
            {"fmt": files_fmt_i, "path": ['', ],
             "filenames": ['', ], "remote_subdir": subpath,
             "local_dir": local_dir_i}
        # print(files_fmt_i)
        # return []
        return local_info_i

    # print(files_fmt_i)
    # input()

    # Find local files:

    # Get a list of all the files to search:
    local_files_i = os.listdir(local_dir_i)
    # print(local_files_i)
    # input()

    local_files = []
    local_paths = []

    for i, filename_ij in enumerate(files_fmt_i):

        if verbose:
            print("Folder to be searched: ", local_dir_i)
            print("File searched for: ", filename_ij)

        # Find any local copy that matches:
        # local_file_ij = [i for i in local_files_i if
        #                  re.match(re.compile("{}$".format(filename_ij)), i)]
        local_file_ij = [i for i in local_files_i if
                         re.match(re.compile(filename_ij), i)]

        ext_i = filename_ij.split(".")[-1]
        # print(ext_i)
        # input()
        if ext_i == "bsp":
            version_format = 'v[0-9]'
        elif ext_i == "orb":
            version_format = None

        if verbose:
            print("Local matching copy: ", local_file_ij)

        if len(local_file_ij) == 0:
            local_files.append('')
        elif len(local_file_ij) == 1:
            matching_file_i = local_file_ij[0]
            if verbose:
                print("File found: ", matching_file_i)
            local_files.append(matching_file_i)
        else:
            # Get the file names that match
            # (useful for debugging)

            # select higher version number:
            if version_format is None and use_most_recent is None:
                raise ValueError(
                    "Multiple files ({}) found matching file"
                    " format, either set use_most_recent to True"
                    " or set version_format.".format(local_file_ij))

            matching_path_i =\
                [os.path.join(local_dir_i, i) for i in local_file_ij]

            # print(local_file_ij)
            # input()

            # if more than one, select newer one.
            if use_most_recent:
                # Pull the most recently modified
                # file matching the file_name_format

                local_modtime_posix_i =\
                    [os.path.getmtime(i) for i in matching_path_i]
                most_recent_index = local_modtime_posix_i.index(
                    max(local_modtime_posix_i))

                local_files.append(local_file_ij[most_recent_index])
                if verbose:
                    print("Multiple files found, use "
                          "most recent: ", matching_path_i[most_recent_index])

            elif version_format is not None:
                v_index_i = local_file_ij[0].index("_v") + 2
                version_str = [j[v_index_i:(v_index_i + 2)] for j in local_file_ij]
                version_num = [int(j[1:]) for j in version_str]
                highest_version = max(version_num)
                highest_v_index = version_num.index(highest_version)
                highest_v_file = local_file_ij[highest_v_index]

                local_files.append(highest_v_file)

                if verbose:
                    print("Multiple files found, use "
                          "highest version #: ", highest_v_file)

    # After the search is completed for weekly CK files,
    # want to grab daily files after the last CK:

    # print(local_files)
    local_files = [i for i in local_files if i]

    if kernel_name == "ck":
        # local_files_non_none = [i for i in local_files if i]
        if verbose:
            print("CK additional search:")

        if len(local_files) != 0:
            last_ck_name = local_files[-1]
            last_yymmdd = dt.datetime.strptime(
                last_ck_name.split("_")[4], "%y%m%d")
        else:
            last_yymmdd = start_dt.replace(tzinfo=None)
        end_dt = end_dt.replace(tzinfo=None)

        # get the time range for the dailys:
        daily_dt_range = [last_yymmdd + dt.timedelta(days=i) for i
                          in range((end_dt - last_yymmdd).days + 1)]

        # If the last recovered yy/mm/dd for the final
        # ck files is before the requested end date, need
        # to search for the alternative files: daily and
        # "recent"/"quick" files.
        # print(last_yymmdd, end_dt)
        # print("Checking...")
        if last_yymmdd < end_dt:

            # Generate the daily file names after the last rel file ends:
            daily_fmt = files_fmt["daily"].format(platform=ck_platform)
            daily_fmt_after = [daily_fmt.replace(yymmdd, i.strftime("%y%m%d"))
                               for i in daily_dt_range]
            # print(daily_fmt_after)

            # Search for the dailys that match this format:
            daily_after_rel = []
            for daily_fmt_i in daily_fmt_after:
                match_day_i =\
                    [i for i in local_files_i if
                     re.match(re.compile(daily_fmt_i), i)]
                daily_after_rel += match_day_i
            # print(daily_after_rel)
            # input()

            # Same for the recents:
            recent_fmt = files_fmt["recent"].format(platform=ck_platform)
            recent_fmt_after =\
                [recent_fmt.replace(yymmdd, i.strftime("%y%m%d"), 1)
                 for i in daily_dt_range]

            # print(recent_fmt_after)
            # input()

            # Search for the recents that match this format:
            recent_after_rel = []

            for recent_fmt_i in recent_fmt_after:
                recent_match_i =\
                    [i for i in local_files_i if
                     re.match(re.compile(recent_fmt_i), i)]
                recent_after_rel += recent_match_i

            local_files = local_files + daily_after_rel + recent_after_rel
            # print(daily_after_rel, recent_after_rel)
            # print(recent_fmt)
            # input()

            # Reset the files format (useful for lookup)
            files_fmt_i = files_fmt_i + [*daily_fmt_after, *recent_fmt_after]

    if verbose:
        # print(local_files)
        print("Local search done.")
        # input()

    # if kernel_name == "ck":
    #     print(local_files)
    #     print(files_fmt_i)
    # input()

    local_paths = [os.path.join(local_dir_i, i) if i else '' for i in local_files]

    local_info =\
        {"fmt": files_fmt_i, "path": local_paths,
         "filenames": local_files, "remote_subdir": subpath,
         "local_dir": local_dir_i}

    return local_info


def retrieve_kernels(data_directory, kernel_group, kernel_name,
                     start_date=None, end_date=None, n_days=None,
                     download_if_not_available=True,
                     ck_platform=None, spk_ext='bsp',
                     use_most_recent=None, mirror_spedas_dir_tree=True,
                     session=None,
                     verbose=None,
                     prompt_for_download=True):

    '''Returns a local filename of the generic kernel.
    If it doesn't exist or is created before an update
    on the remote (e.g. new file, such as leap second),
    will retrieve.

    kernel_group: string, 'generic_kernels', 'MAVEN', etc.
    kernel_type: string, e.g. 'lsk' or 'spk'
    file_regex: string, e.g. 'pck0001[0-9].tpc'
    '''

    if kernel_name in ('ck', 'spk'):
        start_date, n_days, end_date = helper.sanitize_date_inputs(
            start_date=start_date, n_days=n_days, end_date=end_date,
            default_start_date=start_ephemeris_dt,
            default_end_date=dt.datetime.now())

    # Return a list of matching local files
    # with the highest version number
    # (will be empty if no file found)
    local_info = find_local_files(
        data_directory, kernel_group, kernel_name,
        start_dt=start_date, end_dt=end_date, spk_ext=spk_ext,
        ck_platform=ck_platform,
        use_most_recent=use_most_recent,
        mirror_spedas_dir_tree=mirror_spedas_dir_tree,
        verbose=verbose)

    local_savedir = local_info["local_dir"]
    local_paths = local_info["path"]
    local_filenames = local_info["filenames"]
    files_fmt = local_info['fmt']

    if verbose:
        print("Searched: ", local_savedir)
        print("All files found: ", local_filenames)
    # input()

    # Retrieve remote:
    if download_if_not_available:

        remote_url = "/".join(
            (naif_url, *local_info['remote_subdir']))

        if session is None:
            # Open session:
            session = requests.session()

        # Download html to search for the most
        # recently generated kernel:
        html_i = retrieve.html_retrieve(remote_url, session=session)

        # Get local modtimes
        local_modtime_utc =\
            [dt.datetime.fromtimestamp(
                os.path.getmtime(path_i), tz=dt.timezone.utc)
             for path_i in local_paths if path_i]
        local_filenames = [i for i in local_filenames if i]

        # Retrieve most recently modified file(s)
        remote_filename_ij, remote_filename_modtime_dt =\
            retrieve.newest_file_from_html_index(
                html_i, files_fmt, verbose=verbose,
                ck_check=(kernel_name == 'ck'),
                ck_end_dt=end_date,
                multiple_files_per_day=(len(files_fmt) > 1))

        # Get all newest files on remote that are not present locally:
        only_on_remote = [i for i in remote_filename_ij
                          if i not in local_filenames]

        # ... and all that are present on both
        on_both = [i for i in remote_filename_ij if i in local_filenames]
        # print(on_both)
        # print(remote_filename_ij)
        # print(remote_filename_modtime_dt)
        # print(local_modtime_utc)
        # print(local_filenames)
        updated_after_local =\
            [i for i in on_both if
             remote_filename_modtime_dt[remote_filename_ij.index(i)] >
             local_modtime_utc[local_filenames.index(i)]]

        # print(on_both)
        # print(updated_after_local)
        # input()

        files = [i for (i, j) in zip(local_paths, local_filenames)
                 if j in local_filenames]

        if verbose:
            print("Searching on remote: ", remote_url)
            print('remote names: ', remote_filename_ij)
            print('remote dt: ', remote_filename_modtime_dt)
            print('local names: ', local_filenames)
            print('local dt: ', local_modtime_utc)
            print("only on remote: ", on_both)
            print("avail local, but updated on remote: ", updated_after_local)

        only_on_remote += updated_after_local

        # Now scan to find any filenames that are on remote that are not
        # on local, and if there is a newer version on the remote:
        for remote_filename_i in only_on_remote:
            # If it is the same, compare to remote and see if updated.
            # If it hasn't been, proceed to next.
            # if verbose:

            # Provide a prompt:
            if prompt_for_download:
                print('Download: ', remote_filename_i)
                input("Hit enter to allow download:")

            # If the remote file has been modified since
            # the local file has been present (or never
            # downloaded):
            # Since has been modified or further versions,
            # download:
            remote_file_url_i = "{}/{}".format(remote_url, remote_filename_i)
            new_local_path_i = os.path.join(local_savedir, remote_filename_i)
            retrieve.download_file(
                remote_file_url_i, new_local_path_i, session=session)

            files.append(new_local_path_i)
    else:
        files = local_paths

    return files


################
# SPK ROUTINES
# The ephemeris files are written for 3-month long
# periods starting 1/1/2015, with the exception
# of orbits in 2014 which are in the file starting
# 9/22/2014
###############
start_ephemeris_dt = dt.datetime(2014, 9, 22)
ephemeris_months = [1, 4, 7, 10]


def closest_spk_dt_range(time_dt, incl_day_before=False):
    '''Since the ephemeris data files (.orb/bsp in spk) contain data
    for three month periods (with the exception of
    the beginning of the mission, which starts
    9/11/2014), need to identify the right file containing
    ephemeris info for a given data time.'''

    # First, rule out all times before Mars orbit insertion:
    time_dt = time_dt.replace(tzinfo=None)

    if time_dt < start_ephemeris_dt:
        raise ValueError(
            "No orbits before MOI on 2014/09/22, must enter a time after.")

    # Only one file for 2014:
    if time_dt.year == 2014:
        return [start_ephemeris_dt, dt.datetime(2015, 1, 1)]

    # if starting at the exact same day as the start day
    # for the spk file, it is important to get the preceding file
    # since there can be some overlap (although this confused postdoc
    # does not know why.)
    if time_dt.day == 1 and incl_day_before:
        time_dt = time_dt - relativedelta(days=1)

    # The first time in the ephemeris file label should
    # immediately precede the searched for time:
    year = time_dt.year
    month = time_dt.month
    index = np.searchsorted(ephemeris_months, month, side='right') - 1
    closest_m = ephemeris_months[index]
    closest_dt = dt.datetime(year, closest_m, 1)
    end_dt = closest_dt + relativedelta(months=3)

    return [closest_dt, end_dt]


def spk_file_names(start_time_dt, end_time_dt, file_fmt, ext):
    '''For a given period of time, retrieve the list of SPK file
    names.'''

    # Generate the file name(s) that contain subtended
    # requested time:
    # (These are strings amenable to glob.glob search)
    if not start_time_dt:
        start_time_dt = start_ephemeris_dt
    if not end_time_dt:
        end_time_dt = dt.datetime.now()

    # Get month for the file containing the data,
    # which will include the start and end time.
    dt_range_i = closest_spk_dt_range(start_time_dt, incl_day_before=True)
    dt_range_f = closest_spk_dt_range(end_time_dt)

    # Determine the number of months between start and end.
    # if under three, only need to load one data file.
    # if over three, need to load multiple data files.
    n_months = (dt_range_f[-1].year - dt_range_i[0].year)*12 +\
        (dt_range_f[-1].month - dt_range_i[0].month)

    # Number of files is the  number of months divided by 3
    n_files = int(n_months/3)

    dt_i, dt_f = dt_range_i
    # print(dt_f)
    # print(dt_i)
    # print(n_files)
    # input()

    present_date_dt = dt.datetime.now()

    files = []
    for i in range(n_files):
        if i > 0:
            dt_i = dt_f
            dt_f = dt_f + relativedelta(months=3)

        # print(dt_i, dt_f)
        # Orbit ephemeris (saved in the Spice kernels)
        # ALERT: Currently only "v1" files present,
        # but will need to be modified if v[1-9]
        # files are created!
        file_i = file_fmt["quarterly"].format(
                yymmdd_i=dt_i.strftime("%y%m%d"),
                yymmdd_f=dt_f.strftime("%y%m%d"),
                ext=ext)
        # print(file_i)
        files.append(file_i)

    if dt_f > (present_date_dt - dt.timedelta(days=40)):
        # if dt_i >= present_date_dt:
        # Predictions for MAVEN orbit starting
        # from latest ephemeris window.
        file_i = file_fmt["predict"].format(ext=ext)
        files.append(file_i)

        # If orbits are retrieved within the last few months,
        # need to use maven_orb_rec.orb
        # It can take over 40 days for the new quarterly file
        # to be written:
        # file_i = file_fmt["recent"].format(ext=ext)
        files.append(file_fmt["recent"].format(ext=ext))

    # print('SPK filenames:', files)
    # input()

    return files


def ck_file_names(platform, start_time_dt, end_time_dt, file_fmt):

    # CK files contain the pointing information
    # for a platform ('app', 'sc', 'swea').
    # They are saved for each week of science operation
    # with names:
    # (weekly): mvn_sc_rel_231204_231210_v01.bc,
    # mvn_app_rel_231218_231224_v01.bc
    # print(platform, start_dt, end_dt)

    if not start_time_dt:
        start_time_dt = start_ephemeris_dt
    if not end_time_dt:
        end_time_dt = dt.datetime.now()

    start_time_dt = start_time_dt.replace(tzinfo=None)
    end_time_dt = end_time_dt.replace(tzinfo=None)

    if platform not in ck_platform_options:
        raise ValueError(
            "Platform must be defined to download pointing "
            "information (ck kernels), options: {}".format(
                ck_platform_options))
    if platform == "swea":
        files_search = ["mvn_swea_nom_131118_300101_v[0-9][0-9].bc",]
    else:
        if platform == 'sc':
            ck_start_dt = dt.datetime(2014, 9, 1)
        elif platform == 'app':
            ck_start_dt = dt.datetime(2014, 10, 13)

        # mvn_(sc/app)_rel are contain 7-days of data starting
        # 2014/09/01 (sc) and 2014/10/13 (app). Start dates
        # are a week apart. Return datetimes matching.
        weekly_rrule = rrule(
            freq=WEEKLY, dtstart=ck_start_dt, until=dt.datetime.now())
        weekly_file_dt = weekly_rrule.between(
            start_time_dt, end_time_dt, inc=True)

        # Sometimes will miss the prev file
        if not weekly_file_dt:
            weekly_file_dt = [weekly_rrule.before(start_time_dt)]

        # print(weekly_file_dt)
        # input()

        if start_time_dt < weekly_file_dt[0]:
            weekly_file_dt.insert(0, weekly_rrule.before(start_time_dt))

        files_search =\
            [file_fmt["weekly"].format(
                platform=platform, yymmdd=dt_i.strftime("%y%m%d")) for
             dt_i in weekly_file_dt]

    # print(files_search, ck_weekly_file_format)
    # input()

    return files_search


def MAVEN_kernels(data_directory, kernels, kernel_groups,
                  start_dt=None, end_dt=None,
                  download_if_not_available=True,
                  session=None, verbose=None,
                  mirror_spedas_dir_tree=True,
                  spk_ext='bsp',
                  prompt_for_download=True):

    """Routine to list/download MAVEN-relevant SPICE kernels into memory.
    This will need to be reduced for the specific relevant Spice kernels."""
    # Spacecraft clock file (only most recent used)
    # SPK files contain info on MAVEN's location
    # relative to the spacecraft clock.

    if session is None and download_if_not_available:
        session = requests.session()

    kernel_filepath = []

    # Iterate through requested kernels
    for k_i, kg_i in zip(kernels, kernel_groups):

        # Lower case the kernel for access:
        k_i = k_i.lower()

        # If ck, split into ck and platform:
        if k_i.startswith('ck'):
            # print(k_i)
            k_i, ck_platform = k_i.split("_")
        else:
            ck_platform = None

        # Name of kernel
        if verbose:
            print("Getting {} files ({})...".format(
                k_i, expanded_kernel_name[k_i]))

        # Disabling this definition of use_most_recent,
        # since these kernels can and do get updated:
        # use_most_recent = (k_i not in ('spk_planets', "spk_satellites_mars"))

        kernel_filepath_i = retrieve_kernels(
            data_directory, kg_i, k_i,
            download_if_not_available=download_if_not_available,
            use_most_recent=True,
            session=session,
            start_date=start_dt, end_date=end_dt,
            verbose=verbose,
            spk_ext=spk_ext, ck_platform=ck_platform,
            mirror_spedas_dir_tree=mirror_spedas_dir_tree,
            prompt_for_download=prompt_for_download)

        kernel_filepath += kernel_filepath_i

        if verbose:
            print("Done, saved here: ", kernel_filepath_i)

    return kernel_filepath


def load_kernels(data_directory,
                 start_date=None, end_date=None, n_days=None,
                 kernels=None,
                 download_if_not_available=True,
                 mirror_spedas_dir_tree=True,
                 verbose=None,
                 load_spacecraft=True,
                 load_spacecraft_pointing=True,
                 load_APP=False,
                 spk_ext=("bsp", "orb"),
                 prompt_for_download=True):

    '''Downloads or retrieves MAVEN-relevant Spice kernels via Spiceypy
    and furnsh.
    for using spice commands.'''

    # If kernels not specicified, load the following:
    if not kernels:

        # Mars-related generic kernels:
        kernels = ['pck', 'lsk', 'spk_planets', "spk_satellites_mars"]
        kernel_groups = ["generic_kernels"]*4

        # Add the MAVEN-specific kernels
        if load_spacecraft:
            kernels += ['fk', 'sclk', 'spk', 'ik']
            kernel_groups += ["maven"]*4

            # Add spacecraft pointing info:
            if load_spacecraft_pointing:
                kernels += ['ck_sc']
                kernel_groups += ['maven']

            # Add the APP pointing info, if desired.
            if load_APP:
                kernels += ['ck_app']
                kernel_groups += ['maven']

    # start_dt = helper.date_to_dt(start_date, default=start_ephemeris_dt)
    # end_dt = helper.date_to_dt(end_date, default=dt.datetime.now())
    # Convert start/end/n_days to appropriate datetype
    start_dt, n_days, end_dt = helper.sanitize_date_inputs(
        start_date=start_date, n_days=n_days, end_date=end_date,
        default_start_date=start_ephemeris_dt,
        default_end_date=dt.datetime.now())

    k = MAVEN_kernels(
        data_directory, kernels, kernel_groups,
        spk_ext=spk_ext,
        start_dt=start_dt, end_dt=end_dt,
        download_if_not_available=download_if_not_available,
        mirror_spedas_dir_tree=mirror_spedas_dir_tree,
        verbose=verbose,
        prompt_for_download=prompt_for_download)

    # Remove any null strings:
    k = [i for i in k if i]

    if verbose:
        print("All kernels:", k)
    # input()

    for k_i in k:
        # print(k_i)
        spiceypy.furnsh(k_i)

    return k


def currently_loaded_kernels():
    n_kernels = spiceypy.ktotal('ALL')
    k_file = []
    for i in range(n_kernels):
        k_info_i = spiceypy.kdata(i, 'ALL')
        file_path_i = k_info_i[0]
        # kernel_type_i = k_info_i[1]
        # source_file_i = k_info_i[2]
        # handle_i = k_info_i[3]
        k_file.append(file_path_i)
    return k_file


def dt_to_et(i):
    '''Convert any time string or datetime
    into an ephemeris time'''

    # First, convert the str into datetime,
    # so it won't trip the Iterable check:
    if isinstance(i, str):
        i = parsedt(i)

    # Now, detect if the provided item is an Iterable
    # and return a list iterating over items if in list/array:
    if isinstance(i, Iterable):
        return [dt_to_et(i_i) for i_i in i]

    # print(i, type(i))
    # If this is a numpy datetime64, need to convert into
    # a datetime.

    # HOWEVER: just doing "i.astype(dt.datetime)"
    # does not always work and will sometimes instead
    # return the # of nanoseconds since 1970
    # (https://github.com/numpy/numpy/issues/20351).

    # So what we do instead is convert into a string
    # via datetime_as_string, which returns in the format
    # '%Y-%m-%dT%H:%M:%S', and then convert into a datetime via strptime.
    if isinstance(i, np.datetime64):
        # print('conv to dt')
        i = dt.datetime.strptime(
            str(np.datetime_as_string(i, unit='s')),
            '%Y-%m-%dT%H:%M:%S')

    return spiceypy.str2et(i.strftime("%b %d, %Y %H:%M:%S"))


def mars_sun_distance(t):
    """Retrieve the Mars-sun distance as a function of time.
    t: datetime object or list of datetime objects"""

    # get ephemeris time:
    et = dt_to_et(t)

    if isinstance(t, Iterable):
        pos = np.array(spiceypy.spkpos("Mars", et, "J2000", "None", "Sun")[0])
        dist = np.sqrt(np.sum(pos ** 2, axis=1))
    else:
        pos = spiceypy.spkpos("Mars", et, "J2000", "None", "Sun")[0]
        dist = np.sqrt(sum(pos ** 2))

    return dist


def MAVEN_position(t_i, frame="MAVEN_MSO"):
    """Retrieve MAVENs position relative to Mars in MSO as a function of time.
    t_i: datetime object or list of datetime objects"""

    # get ephemeris time:
    et = dt_to_et(t_i)

    if isinstance(t_i, Iterable):
        position = spiceypy.spkpos(
            "MAVEN", et, frame, "None", "Mars")[0]
        x = np.array([i[0] for i in position])
        y = np.array([i[1] for i in position])
        z = np.array([i[2] for i in position])

    else:
        x, y, z = spiceypy.spkpos(
            "MAVEN", et, frame, "None", "Mars")[0]

    return x, y, z


def bpl_to_bmso(time_UTC, bx_pl, by_pl, bz_pl):
    """Rotate the measured B from MAVEN MAG from payload coordinates
    into MSO coordinates."""

    b_mso = pxform(
        time_UTC, bx_pl, by_pl, bz_pl,
        "MAVEN_SPACECRAFT", "MAVEN_MSO")

    return b_mso


def pxform(time_UTC, v_x, v_y, v_z, initial_frame, final_frame):
    """Rotate the measured vector from initial frame to final frame.
    Wrapper for pxform."""

    # get ephemeris time:
    ephemeris_time = dt_to_et(time_UTC)
    N = len(ephemeris_time)

    # Check if any CK kernels are loaded, and if so, if they
    # are the correct ones
    n_ck = spiceypy.ktotal('CK')
    if n_ck == 0:
        raise IOError("No CK files loaded, please load.")
    else:
        for i in range(n_ck):
            k_info_i = spiceypy.kdata(i, 'CK')
            file_path_i = k_info_i[0]
            ck_code_i = spiceypy.ckobj(file_path_i)[0]

    # initial vector array to fill in:
    v_f = np.zeros(shape=(N, 3))

    for i, et_i in enumerate(ephemeris_time):

        try:
            m = spiceypy.pxform(initial_frame, final_frame, et_i)
        except spiceypy.utils.exceptions.SpiceNOFRAMECONNECT:
            # When no frame available, which can happen if
            # the frame wasn't loaded (load_kernels must be
            # run before this routine to get a sensible result).
            # This happens also for Spice gaps, e.g. 2015 Mar -4 00:00

            v_f[i, :] = np.nan, np.nan, np.nan
            continue

        q = spiceypy.m2q(m)
        # print(q)
        # print(bx_pl[idx], by_pl[idx], bz_pl[idx])

        if isinstance(v_x, Iterable):
            v_i = (v_x[i], v_y[i], v_z[i])
        else:
            v_i = (v_x, v_y, v_z)

        v_x_f, v_y_f, v_z_f = quaternion_rotation(q, v_i)
        # print(bx_mso, by_mso, bz_mso)
        v_f[i, :] = v_x_f, v_y_f, v_z_f

    return v_f



def quaternion_rotation(q, v):
    """Rotation of a vector v into a new coordinate system
    provided the quaternion."""

    v1 = v[0]
    v2 = v[1]
    v3 = v[2]

    a = q[0]
    b = q[1]
    c = q[2]
    d = q[3]

    t2 = a * b
    t3 = a * c
    t4 = a * d
    t5 = -b * b
    t6 = b * c
    t7 = b * d
    t8 = -c * c
    t9 = c * d
    t10 = -d * d

    v1new = 2 * ((t8 + t10) * v1 + (t6 - t4) * v2 + (t3 + t7) * v3) + v1
    v2new = 2 * ((t4 + t6) * v1 + (t5 + t10) * v2 + (t9 - t2) * v3) + v2
    v3new = 2 * ((t7 - t3) * v1 + (t2 + t9) * v2 + (t5 + t8) * v3) + v3

    return v1new, v2new, v3new


def load_MAVEN_position(start_date, n_days=None, end_date=None,
                        n_sample_points=400, frame="MAVEN_MSO"):

    """Returns a spline function that has been fitted to MAVEN
    spacecraft position information as retrieved from a given datasource
    data_directory: containing MAVEN data
    date: datetime, time to begin retrieval, e.g. dt.datetime(2015, 3, 1)
    n: int,  number of days retrieved for, e.g. 1
    n_sample_points: number of points
        sampled per day, to direct SPICE retrieval
    """

    # Get start date, n_days, end_date
    # if end_date is None:
    start_date, n_days, end_date = helper.sanitize_date_inputs(
        start_date=start_date, n_days=n_days, end_date=end_date)

    sc_time_utc = helper.dt_range(
        start_date, end_date=end_date, n_points_per_day=n_sample_points)

    x, y, z = MAVEN_position(sc_time_utc, frame=frame)
    # Get the unx time to build the cartesian spline for the s/c position
    sc_time_unx = helper.UTC_to_UNX(sc_time_utc)

    return sc_time_utc, sc_time_unx, x, y, z


def local_solar_time(start_date, n_days=None, end_date=None,
                     n_sample_points=400, frame="MAVEN_MSO"):

    '''Adaptation of mvn_mars_localtime.pro, requires time.

    Returns local solar time where 0=midnight'''

    # Get start date, n_days, end_date
    if end_date is None:
        start_date, n_days, end_date = helper.sanitize_date_inputs(
            start_date=start_date, n_days=n_days, end_date=end_date)

    sc_time_utc = helper.dt_range(
        start_date, end_date=end_date, n_points_per_day=n_sample_points)
    et = dt_to_et(sc_time_utc)

    maven_iau = spiceypy.spkpos(
        'MAVEN', et, 'IAU_Mars', 'None', 'Mars')[0]

    maven_lon = np.degrees(np.arctan2(maven_iau[:, 1], maven_iau[:, 0]))
    maven_lon = np.where(maven_lon < 0, 360 + maven_lon, maven_lon)
    maven_lat = np.degrees(np.arcsin(maven_iau[:, 2]))

    sun_mso = [1, 0, 0]
    q = [spiceypy.pxform('MAVEN_MSO', 'IAU_Mars', i) for i in et]

    sun = [quaternion_rotation(spiceypy.m2q(q_i), sun_mso) for q_i in q]
    sun = np.array(sun)

    subsolar_lon = np.degrees(np.arctan2(sun[:, 1], sun[:, 0]))
    subsolar_lon = np.where(subsolar_lon < 0, 360 + subsolar_lon, subsolar_lon)

    subsolar_lat = np.degrees(np.arcsin(sun[:, 2]))

    # ; 0 = midnight, 12 = noon
    lst = (maven_lon - subsolar_lon)*(12./180.) - 12
    lst -= 24*np.floor(lst/24)  #; wrap to 0-24 range

    # from matplotlib import pyplot as plt
    # fig, ax = plt.subplots(nrows=3)
    # ax[0].plot(sc_time_utc, maven_lon)
    # ax[0].plot(sc_time_utc, subsolar_lon)
    # ax[1].plot(sc_time_utc, maven_lat)
    # ax[1].plot(sc_time_utc, subsolar_lat)
    # ax[2].plot(sc_time_utc, lst)
    # plt.show()

    return lst



