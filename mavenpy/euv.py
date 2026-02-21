import os

import numpy as np
import spiceypy

from .read import read_cdf, read_tplot
from .helper import process_data_dict, UTC_to_UNX, UNX_to_UTC
from .spice import retrieve_kernels


# EUV read routines

# Version / revision information for EUV:
# - v## refers to the file version, which changes if CDF fields are modified.
#    As of 6/17/23, it is locked at v15.
# - r## refers to the version of data available for that version.
# want the newest version & revision.

# Units for EUV based on the SIS:
units = {"epoch": "utc", "time": "unx",
         "flag": "usability flag",
         "maven_sun_distance": "km",
         "spectral_irradiance": "W/m2/nm",
         "wavelength": "nm",
         "uncertainty": "%",
         "irradiance": "W/m2 (17-22 nm, 0-7 nm, 121-122 nm)",
         "precision": "%",
         "accuracy": "%",
         "counts": "#/s"}

# Mapping from common column names to the fields in a given
# CDF:
column_to_field_name =\
    {"l2": {"irradiance": "data", "precision": "ddata", "accuracy": "dfreq",
            "time": "time_unix"},
     "l3": {"time": "x", "wavelength": "v", "spectral_irradiance": "y"},
     "l0": {}}

# Inversion of the field name, update the units for field name mapping
field_to_column_name = {}
units_by_field_name = {}
for level in column_to_field_name:
    column_to_field_i = column_to_field_name[level]
    field_to_column_name[level] = {v: k for k, v in column_to_field_i.items()}
    units.update({v: units[k] for k, v in column_to_field_i.items()})


def read(dataset_filename, lib='cdflib', level='', field_names="",
         column_names="", include_unit=True, relabel_columns=True,
         NaN_errant_data=True,
         lsk_file_path="", sclk_file_path="", spice_kernel_dir="",
         clock_drift_correction=True):

    """Loads daily files of EUV Level 3 and 2 data.

    lib: string, 'cdflib' or 'spacepy'
    field_names: CDF variable names, e.g. 'v'
    column_names: variable names, e.g. 'irradiance'
    relabel_columns: Boolean to enable relabeling the fields into
        more common column names:
    include_unit: returns a tuple for each dataset with the unit

    This routine is based on the EUV SIS document and the MAVEN SPEDAS
    routine 'mvn_euv_l3_load.pro'. We thank and credit the authors of both.

    L2 data consists of solar irradiances in three bandpasses calibrated
    for background signal, out-of-band signal, normalized gain, etc.
    Uncertainties for each irradiance also provided, with a flag indicating if
    EUV has favorable pointing. Also includes corrected count rates pre-
    conversion to irradiance units.

    L3 data from EUV consists of outputs of the FISM-M model based on
    EUV calibrated band irradiance. This includes an averaged spectra
    (day or minute) as a function of time, a quality flag,
    and the Mars-sun distance at time of observation.

    flag: integer representing data useability
         (0 - good, 1 - occultation, 2 - no pointing, 3 - sun partial in FOV,
          4 - sun not in FOV, 5 - windowed,  6 - eclipse, 7 - spare), (n_time)

    """

    # If not provided the level, recover that info from the filename:
    if not level:
        filename = os.path.split(dataset_filename)[1]
        level = filename.split("_")[2]

    # If not provided the fields to retrieve, generate from the column names
    # requested. Else, use the defaults:
    if not field_names:
        if column_names:
            column_to_field_name_i = column_to_field_name[level]
            field_names = [column_to_field_name_i[i] if i in
                           column_to_field_name_i else i for i in column_names]
        else:
            field_to_column_name_i = field_to_column_name[level]
            if level == 'l3':
                field_names = ("x", "epoch", "v", "y", "flag")
            elif level == 'l2':
                field_names = ('data', 'ddata', 'dfreq', 'flag',
                               'time_unix', 'epoch', 'maven_sun_distance',
                               'counts')
            elif level == 'l0':
                field_names = ("mvn_lpw_euv", "mvn_lpw_euv_temp_C")
            column_names = [field_to_column_name_i[i] if i in
                            field_to_column_name_i else i for i in field_names]

    # Read CDF:
    if level == 'l0':
        data = read_tplot(dataset_filename, return_plot_parameters=False)[0]
        # Since tplot var, same time_unix for both fields in the
        # file ('mvn_lpw_euv', 'mvn_lpw_euv_temp_C').

        # WARNING: time_unix is NOT CORRECTED for clock drift,
        # Need to convert back to MET_uncorrected and then use Spice
        # to get correct unix time (~400 sec off)
        time_unix_uncorr = data['mvn_lpw_euv']['time_unix']

        if clock_drift_correction:

            # Check if any Spice kernels loaded:
            if spiceypy.ktotal('ALL') < 2:

                if not lsk_file_path and not sclk_file_path:
                    if not spice_kernel_dir:
                        raise IOError(
                            "Require a directory with Spice kernels "
                            "to download/access for ET correction.")
                    lsk_file_path = retrieve_kernels(
                        spice_kernel_dir, 'generic_kernels',
                        'lsk', use_most_recent=True)
                    sclk_file_path = retrieve_kernels(
                        spice_kernel_dir, 'maven',
                        'sclk', use_most_recent=True)
                # Furnsh the kernels (will be cached
                # for subsequent days, if further loaded:)
                for k in [lsk_file_path, sclk_file_path]:
                    spiceypy.furnsh(k)

            # UNIX_uncorrected = MET_uncorrected + 2000-01-01/12:00:00
            # (sec since 1970)
            met_uncorr = time_unix_uncorr - (946771200 - 12*3600)

            # The onboard clock is formatted as strings
            # with format "SSSSSSSSSS.FFFFF", where SSSSSSSSSS is
            # onboard seconds and FFFFF is the fractions of a tick
            # s.t. 1 fraction = 1/65536 of a second
            abs_sec = np.floor(met_uncorr)
            frac_ticks = np.round((met_uncorr % 1)*65536)

            sclk_uncorr =\
                ["{onboard_sec}:{ticks}".format(
                    onboard_sec=int(i), ticks=int(j))
                 for (i, j) in zip(abs_sec, frac_ticks)]
            # print(sclk_uncorr[:3])
            et_corr = [spiceypy.scs2e(-202, i) for i in sclk_uncorr]
            time_utc = np.array([spiceypy.et2datetime(i) for i in et_corr])
            time_unix = UTC_to_UNX(time_utc)
        else:
            time_unix = time_unix_uncorr
            time_utc = UNX_to_UTC(time_unix)

        diode_currents = data["mvn_lpw_euv"]['y'] + 4.6e5
        diode_temperature = data["mvn_lpw_euv_temp_C"]['y']

        if include_unit:
            time_unix = (time_unix, "Unix time (seconds since 1970)")
            time_utc = (time_utc, "UTC time (datetime)")
            diode_currents = (diode_currents, "DN")
            diode_temperature = (diode_temperature, "C")

        data_dict = {"time_unix": time_unix, "epoch": time_utc,
                     "temperature": diode_temperature,
                     "diode_current": diode_currents}

        return data_dict

    else:
        data = read_cdf(dataset_filename, field_names, lib=lib)

    # Errant EUV data filled with -1 E 31
    if NaN_errant_data:
        for i in data:
            if "epoch" in i or "flag" in i:
                continue
            # print(i)
            data_i = data[i]
            errant = (data_i < -1e30)
            data_i[errant] = np.nan

    # Data relabeled according to new column names
    # Include unit if requested:

    data = process_data_dict(
        data, units=(units if include_unit else None),
        alias=(field_to_column_name_i if relabel_columns else None))

    return data
