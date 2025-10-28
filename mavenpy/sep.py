import itertools
import os

import numpy as np

# from .file_path import instrument_data_file
from .helper import UNX_to_UTC, process_data_dict
from .read import read_cdf, read_sav
from .sep_calib import raw_to_fto, fto_to_calibrated


# SEP read routines

atten_unit = "1 (open) or 2 (closed)"
count_unit = "#"

units = {"time_unix": "unx", "time": "unx",
         "epoch": "utc",
         "time_met": "Mission-elapsed time",
         "time_ephemeris": "Ephemeris time",
         "mapid": "# of the map used",
         "seq_cntr": "PDFPU sequence counter for readout",
         "att":  atten_unit,
         "atten":  atten_unit,
         "attenuator_state": atten_unit,
         "raw_counts": count_unit,
         "counts": count_unit,
         "accum_time": "# 1-s accumulations per sample",
         "delta_time": "s duration of observation",

         "energy": "keV",
         "flux": "#/cm2/s/sr/keV",
         "eflux": "keV/cm2/s/sr/keV",

         "mvn_pos_mso": "km, MSO (x, y, z)",
         "mvn_pos_geo": "km, IAU (x, y, z)",
         "mvn_sza": "deg.",
         "mvn_lat_geo": "deg., GEO", "mvn_lon_geo": "deg., GEO",
         "mvn_slt": "HR",
         "fov_sun_angle": "deg.", "fov_ram_angle": "deg.",
         "fov_mars": "fraction of field",
         "fov_pointing_mso": "MSO unit vector",
         "bmso": "nT, MSO (Bx, By, Bz)",
         "pitch_angle": "deg.",
         "ion_energy": "keV", "electron_energy": "keV",
         "gyroradius_electron": "km", "gyroradius_proton": "km",
         "gyroradius_oxygen": "km",
         "ion_efluxpa": "keV/cm2/s/sr/keV",
         "electron_efluxpa": "keV/cm2/s/sr/keV",
         "ion_norm_efluxpa": "F/avg(F(PA))",
         "electron_norm_efluxpa": "F/avg(F(PA))"}


def assign_unit_sep_cal(data, energy_unit):
    '''For the SEP calibrated data, get the relevant
    unit.'''

    new_data = {}
    for field_i in data:
        data_i = data[field_i]
        if "energy" in field_i:
            unit_i = energy_unit
            # Scale to requested unit:
            if energy_unit == 'eV':
                data_i = data_i*1e3

        elif "flux" in field_i:
            unit_i = units["flux"]
            if energy_unit == 'eV':
                data_i = data_i*1e-3
                unit_i = "#/cm2/s/sr/eV"
        elif "time" in field_i:
            unit_i = "s"

        else:
            unit_i = units[field_i]
        new_data[field_i] = (data[field_i], unit_i)
        data[field_i] = None

    return new_data


#########################################
#            SEP routines               #
#########################################


def read(file_path, lib='cdflib', include_unit=True,
         level="None",
         label_by_detector=True,
         dataset='None', sensor='None', telemetry_mode='None',
         telescope=("A", "B"), detector=("F", "O", "FTO"),
         particles=("elec", "ion"), look_directions=("r", "f"),
         energy_unit='eV',
         output_calibration_level='cal',
         output_data_units=("counts", "rate", "flux"),
         fields=None):

    '''
    data_level: either 'cal', 'raw', or 'fto'.
        (cal: calibrated in units of particle flux,
         fto: partially calibrated in detector units,
         raw: uncalibrated, counts in 256 bins) '''

    # Retrieve the dataset information from the filename,
    # if not provided:
    if dataset == "None":
        filename = os.path.split(file_path)[1]
        dataset_tuple = filename.split("_")
        level = dataset_tuple[2]
        mask_atten_actuation = True

        cadence_options = ("32sec", "5min", "01hr")

        if level == "l1":
            # Level 1 files contain both detectors,
            # and both telemetry modes ('svy' and 'arc').
            dataset = 'raw'
            if sensor == "None":
                sensor = ('1', '2')
            for c in cadence_options:
                if c in filename:
                    mask_atten_actuation = False

        elif level == "l2":
            # Level 2 files contain both detectors,
            # and both telemetry modes ('svy' and 'arc').
            # e.g. s2-cal-svy-full
            sep_l2_file_type = dataset_tuple[3].split("-")
            sensor = (sep_l2_file_type[0])[1]
            dataset = sep_l2_file_type[1]
            telemetry_mode = sep_l2_file_type[2]

        elif level == "l3":
            # Only level 3 is PAD
            dataset = dataset_tuple[3]
        elif level == "anc":
            dataset = level

        if telemetry_mode == "None":
            telemetry_mode = 'svy'

    if isinstance(output_data_units, str):
        output_data_units = (output_data_units,)

    # print(filename, dataset_tuple, dataset)

    # Retrieve right function:
    if dataset == "cal":
        # Impossible to get fto or raw data from SEP cal data,
        # since the cal data is already pruned of other channels.
        if output_calibration_level != 'cal':
            raise IOError(
                "SEP cal data can only be returned"
                " as calibrated fluxes, set data_unit to 'cal'.")
        data = read_cal(
            file_path, include_unit=include_unit,
            label_by_detector=label_by_detector, fields=fields,
            output_data=output_data_units)
    elif dataset == "pad":
        data = read_pad(
            file_path, particle=particles, field_names=fields,
            include_unit=include_unit)
    elif dataset == "anc":
        data = read_anci(
            file_path,  ext='', detector=sensor,
            look_direction=look_directions,
            field_names=fields, lib=lib,
            include_unit=include_unit)
    elif dataset == "raw":

        data = read_raw(
            file_path, lib=lib, level=level, fields=fields,
            dataset=telemetry_mode,
            sensors=sensor,
            telescopes=telescope, detector=detector,
            particle=particles, look_directions=look_directions,
            data_unit=output_calibration_level,
            include_unit=include_unit,
            output_data=output_data_units,
            label_by_detector=label_by_detector,
            mask_atten_actuation=mask_atten_actuation
            )

    return data


def uncertainty(counts, background_count_rate=0.2):

    return np.sqrt(counts + background_count_rate)


def read_raw(filename, lib='cdflib', level='None',
             fields=None,
             dataset='svy', sensors=('1', '2'),
             telescopes=("A", "B"), detector=("F", "O"),
             particle=("elec", "ion"),
             look_directions=("F", "R"),
             data_unit=('raw', 'fto', 'cal'),
             output_data=("counts", "rate", "flux"),
             label_by_detector=True,
             include_unit=True,
             mask_atten_actuation=True):

    """Routine to read data from Level-2 or 1 SEP data.
    Counts are grouped into 256 bins: two sets of 128 bins per
    telescope (A or B), and for each telescope, an energy
    map that samples each event type (e.g. F, O, T, FT, OT, FTO)
    and the ADC pulseheight readout (which is directly proportional
    to energy deposited).

    The Level 1 data contains information for housekeeping,
    noise floor, and both the archive and survey telemetry. These
    are not in the Level 2 dataset and cannot be accessed from that
    data structure.

    Level 1 also contains both sensors '1' and '2', while Level 2
    has a different file for each.

    data_unit: 'raw', 'cal', or 'fto'

    t_ids: telescopes, ('a' and/or 'b)
    dp_ids: event as defined by which detector(s) where crossed,
      e.g. ('f', 'o', 'fto').
      'F': particles cross the outer 'F'oil-covered detector
      'T': particles cross the middle detector which is 2x as
        'T'hick as 'F'/'O' (Note: this is sensitive to X-rays!)
      'O': particles cross the outer 'O'pen-facing detector, provided
       they aren't deflected by the yoked magnet.

    dataset: VALID ONLY FOR L1
        string or tuple ('svy', 'arc', 'nse', 'hkp')
        One per SEP sensor ('1', '2'), addressed as
        s[1,2]_[svy,arc,nse,hkp] in the file.

    """

    if level == 'l2':
        # Supply fields if not provided:
        if not fields:
            fields = ('time_unix', 'epoch',
                      'attenuator_state',
                      'accum_time',
                      'raw_counts')
            # "MAP_FTO", "MAP_TID", "MAP_NRG_MEAS_AVG",
            # "MAP_NRG_MEAS_DELTA")

        # Read the CDF data into a dict:
        cdf_data = read_cdf(filename, fields, lib=lib)

        if isinstance(sensors, tuple) and len(sensors) > 1:
            raise IOError(
                "Only one sensor can be read from L2,"
                " please specify which one.")

        # Index the raw_data
        raw_data = {sensors: cdf_data}

        time_var_name = "time_unix"
        counts_var_name = "raw_counts"
        att_var_name = "attenuator_state"

    elif level == 'l1':
        # Supply fields if not provided:
        # if not fields:
        #     if "svy" in dataset or "arc" in dataset:
        #         # ['TIME', 'MET', 'ET', 'SUBSEC', 'F0', 'DELTA_TIME', 'TRANGE',
        #         # 'SEQ_CNTR', 'DSEQCNTR', 'SENSOR', 'CCODE', 'MAPID', 'ATT',
        #         # 'DURATION', 'COUNTS_TOTAL', 'RATE', 'CFACTOR', 'DATA',
        #         # 'VALID']
        #         fields = ["TIME", "DELTA_TIME", "DATA", "ATT"]
        #     if "nse" in dataset:
        #         fields = ["TIME", "DATA"]

        # If dataset_type/detectors/telescopes is a string,
        # convert into tuple
        if isinstance(dataset, str):
            dataset = (dataset,)
        if isinstance(sensors, str):
            sensors = (sensors,)

        # L1 contains 19 structures, to wit:
        # - s[1,2]_svy: data collected from survey telemetry
        # - s[1,2]_arc: data collected from archive telemetry
        # - s[1,2]_nse: measurement of floating ground for each
        # detector (A-F,T,O and B-F,T,O).
        # - s[1,2]_hkp: "housekeeping" data e.g. thermistor info
        # - m[1,2]_hkp: housekeeping data from MAG (no idea why here)
        # - ap[20,21,22,23,24]: housekeeping for PFP
        # - misc description columns (sw_version, prereq_info, spice_info,
        #   source_filename)
        struc_names =\
            ["s{}_{}".format(d_i, s) for (d_i, s) in
             itertools.product(sensors, dataset)]
        # print(struc_names, fields)

        raw_data = read_sav(
            filename, struct_name=struc_names, field_names=fields,
            flatten_struct=False)

        counts_var_name = "data"
        time_var_name = "time"
        att_var_name = "att"

    if isinstance(data_unit, str):
        data_unit = (data_unit,)

    # Scrub the NaNs for the time axis of each
    # dataset (as they have different commanding
    # times).
    # print(filename)
    for i in raw_data:
        raw_data_i = raw_data[i]

        if not isinstance(raw_data_i, dict):
            # If there's no telemetry,
            # the raw data will be an integer (0).
            # In this case, cannot preprocess anything:
            continue

        ctime_unx = raw_data_i[time_var_name]
        non_nan_index = (~np.isnan(ctime_unx))
        raw_data_i = process_data_dict(
            raw_data_i, time_var_name=time_var_name,
            conditional_array=non_nan_index)

        if level == "l1":
            raw_data_i["epoch"] = UNX_to_UTC(raw_data_i[time_var_name])

        if ("svy" in dataset or "arc" in dataset) and mask_atten_actuation:
            # N_time x 256 bins
            raw_counts = raw_data_i[counts_var_name]
            att = raw_data_i[att_var_name]

            # When the attenuator opens/closes, phony counts
            # can be generated. Usually the time step before and after
            # actuation is NaN'ed to remove this data.
            atten_actuate = np.where(np.ediff1d(att) != 0)[0]
            raw_counts[atten_actuate, :] = np.nan
            raw_counts[atten_actuate + 1, :] = np.nan

            raw_data_i[counts_var_name] = raw_counts

        # print(raw_data_i.keys())

        raw_data[i] = raw_data_i

    # print("Raw data proc complete.")
    # input()

    # If translating the data into 'fto' or 'cal' units,
    # need additional arguments:

    n_datasets = len(raw_data.keys())
    data_dict = {}

    # For each sensor:
    for i in raw_data:
        raw_data_i = raw_data[i]
        # print(i, raw_data_i.keys())
        # continue

        if not isinstance(raw_data_i, dict):
            # If there's no telemetry,
            # the raw data will be an integer (0).
            # In this case, want to return nothing:
            data_dict[i] = {}
            continue

        # Iterate through the named categories in each sensor file:
        data_dict_i = {}
        for n in raw_data_i:
            # print(i, n, raw_data_i[n].shape)
            # print(type(raw_data_i[n]))
            # input()

            # Make a lower case version:
            n_lower = n.lower()
            if n_lower == "data":
                n_lower = "raw_counts"
            elif n_lower == "time":
                n_lower = "time_unix"
            elif n_lower == "att":
                n_lower = "attenuator_state"

            # I do not understand this line:
            # if "raw" not in data_unit and \
            #         ("time" not in n_lower and "epoch" not in n_lower):
            #     continue

            # Add unit to the data entry:
            if include_unit:
                if n_lower in units:
                    unit_i = units[n_lower]
                else:
                    unit_i = ''
                raw_data_ij = (raw_data_i[n], unit_i)
            else:
                raw_data_ij = raw_data_i[n]

            data_dict_i[n_lower] = raw_data_ij

        if 'fto' in data_unit or "cal" in data_unit:
            # Returns counts by FTO pattern:
            if level == "l2":
                sensor_num = i
            elif level == "l1":
                sensor_num = i[1]

            fto_dict_i = raw_to_fto(
                raw_data_i, level=level,
                sensors=sensor_num,
                raw_data_unit='counts',
                telescope=telescopes, detector=detector,
                output_data=output_data,
                include_unit=include_unit)
            # print(fto_dict_i.keys())
            # input()

        if 'fto' in data_unit:
            data_dict_i.update(fto_dict_i)

        if "cal" in data_unit:
            calib_dict_i = fto_to_calibrated(
                fto_dict_i,
                look_directions=look_directions,
                particle=particle,
                output_data=output_data,
                include_unit=include_unit)
            data_dict_i.update(calib_dict_i)

        # If pulling more than one dataset or
        # detector labeling enabled, change the key
        # to refer to the detector.
        if label_by_detector or n_datasets > 1:
            for j in data_dict_i:
                if len(dataset) > 1:
                    precede_str_i = i
                else:
                    precede_str_i = i[1]
                # print(precede_str_i)
                new_label = "{}_{}".format(precede_str_i, j)
                data_dict[new_label] = data_dict_i[j]
        else:
            data_dict = data_dict_i


    # print("final:")
    # for n in data_dict:
    #     print(i, n, data_dict[n].shape)
    # input()

    return data_dict


def cal_FOV_field_names(particle_names, look_dirs, field_names):

    field_names_per_fov = []
    for (p, l, f) in itertools.product(particle_names, look_dirs, field_names):
        fov_name_i = "{}_{}_{}".format(l, p, f)
        field_names_per_fov.append(fov_name_i)

    return field_names_per_fov


def read_cal(filename, lib='cdflib', fields=None,
             energy_unit='eV',
             detector="", p=("elec", "ion"), ld=("r", "f"),
             output_data=('flux',),
             label_by_detector=True, include_unit=True):

    """Return dict of SEP (e)fluxes, energy bins, attenuator state for
    a SEP look direction and particle.

    dataset_filename: string, name of SEP L2 data file.
    spectra_type: string, which spectra type to read (eflux or flux)
    fields: a tuple containing all requested datasets.
    By default, time, energy, attenuator, and flux data is returned.
    But can also include uncertainties "unc", energy bin width "de".
    """
    if not detector:
        detector = os.path.split(filename)[1].split('-')[0][-1:]

    # Supply the field names
    if not fields:
        fields = ["time_unix", "epoch", "attenuator_state", "energy",
                  "denergy", "flux"]

    # Each dataset with "energy" or "flux" in the name should
    # be expanded with the particle name (elec or ion)
    # and look direction (f or r)
    en_flux_vars = [i for i in fields if ("energy" in i or "flux" in i)]
    en_flux_names = cal_FOV_field_names(p, ld, en_flux_vars)
    fields = [i for i in fields if i not in en_flux_vars]
    fields += en_flux_names

    # print(fields)

    # Read the CDF data into a dict:
    cdf_data = read_cdf(filename, fields, lib=lib)

    # Next, flatten the energy axis if it is in the fields.
    # (SEP has a time-varying energy & denergy variable,
    # which was intended to account for the slow change in
    # the dead layers over time
    # HOWEVER -- right now, it is just the same repeated
    # array because this dead-time correction change has not been implemented.
    # Thus, if you want the energy axis from a SEP data file,
    # you'll need to select the first non-nan index for the actual
    # energy bins.)
    energy_fields = [i for i in fields if "energy" in i]
    # input()
    for en_name in energy_fields:
        en_2d_i = cdf_data[en_name]
        first_non_nan_en_index = np.argwhere(~np.isnan(en_2d_i[:, 0]))[0]
        en_1d_i = en_2d_i[first_non_nan_en_index, :].flatten()
        cdf_data[en_name] = en_1d_i

    # Now identify / elimate NaNs in time axis.
    # Glitches are present in SEP data as NaNs in the time axis.
    # These will contaminate results if not extracted.
    # While NaNs are present all non-time and non-attenuator
    # axes as well, but these are deliberate and kept.
    # These are times when the attenuator activates,
    # which causes phony counts in the pre/post actuation
    # samples. Thus, they remain NaNs and don't require deletion.
    not_nan_indices = ~np.isnan(cdf_data["time_unix"])
    data = process_data_dict(
        cdf_data, conditional_array=not_nan_indices)

    # for name in data:
    #     print(name, data[name].shape)
    # input()

    # And assign units / rescale as requested
    if include_unit:
        data = assign_unit_sep_cal(data, energy_unit)

    # Finally, relabel by detector name
    if label_by_detector:
        new_data = {}
        for field_i in data:
            new_field_name_i = "{}_{}".format(detector, field_i)
            new_data[new_field_name_i] = data[field_i]
            data[field_i] = None
        data = new_data


    return data


anc_cdf_to_sav_name =\
    {"mvn": "spacecraft", "pos": "position", "sza": "solar_zenith_angle",
     "slt": "local_time", "lat": "latitude", "elon": "east_longitude",
     "2": "_to_", "frac": "fraction", "ill": "sunlit_mars"}

anc_sav_to_cdf_name = {}
for i in anc_cdf_to_sav_name:
    anc_sav_to_cdf_name[anc_cdf_to_sav_name[i]] = i

anc_conversion = {}
anc_conversion["sav"] = anc_cdf_to_sav_name
anc_conversion["cdf"] = anc_sav_to_cdf_name


def anc_field_names(field_names, sensors, look_dirs, ext):

    # Isolate time names and remove from field names
    time_names = [i for i in field_names if "time" in i or "epoch" in i]
    field_names = [i for i in field_names if "time" not in i and "epoch" not in i]

    print(time_names)
    print(field_names)

    if ext == "sav":
        # Remove any time name not time_ephemeris  or time_unix
        time_names = []
        if "time_unix" in time_names:
            time_names.append("time_unix")
        if "time_ephemeris" in time_names:
            time_names.append("time_ephemeris")

    # Get anc conversion
    anc_conversion_i = anc_conversion[ext]

    # Now spacecraft/positioning names
    # spacecraft varying names:

    non_fov_varying_names =\
        [i for i in field_names if ("mvn" in i or ("position" in i or "spacecraft" in i))]
    field_names =\
        [i for i in field_names if not ("mvn" in i or ("position" in i or "spacecraft" in i))]

    for idx, f_name in enumerate(non_fov_varying_names):
        f_i = f_name.split("_")
        f_i = [anc_conversion_i[i] if i in anc_conversion_i
               else i for i in f_i]
        non_fov_varying_names[idx] = "_".join(f_i)

    # Special handling for mars_frac_sky
    if ("mars_frac_sky" in field_names or "fraction_sky_filled_by_mars"
            in field_names) and ext == "sav":
        non_fov_varying_names.append(
            field_names.pop(field_names.index("fraction_sky_filled_by_mars")))

    if ("mars_frac_sky" in field_names or "fraction_sky_filled_by_mars"
            in field_names) and ext == "cdf":
        non_fov_varying_names.append(
            field_names.pop(field_names.index("mars_frac_sky")))

    print(non_fov_varying_names)
    print(field_names)

    # Now for the fov-varying parameters
    fov_varying_names = []
    for f in field_names:
        f_i = f.split("_")
        print(f_i)

        # fov_[theta,phi]_[centers,edges] ONLY in sav
        # Dont pull a field from cdf with that name

        if len(f_i) == 3:
            if (f_i[2] == "edges" or f_i[2] == "centers") and ext != "sav":
                continue

        # sep-[1,2][f,r]_fov_[theta,phi] ONLY in cdf
        # and is not documented and is just a nonsensical
        # variable. skip.
        if f == "fov_theta" or f == "fov_phi":
            continue

        # "fov_[mso/geo/sso]" <-> "look_direction_[mso/sso/geo]"
        if ext == "sav" and (f_i[0] == "fov" and "angle" not in f_i):
            f_i = ["look_direction", f_i[1]]
        if ext == "cdf" and (f_i[0] == "look" and "angle" not in f_i):
            f_i = ["fov", f_i[2]]

        # "fov_full_mso" <-> "full_fov"
        if f_i == "full_fov" and ext == "sav":
            f_i = ["fov", "full", "mso"]
        if f_i == "fov_full_mso" and ext == "cdf":
            f_i = ["full", "fov"]

        if "angle" in f:
            f_i = f_i[1:]

        # "qrot_to_mso" <-> "qrot2mso"
        # "qrot_sep{}_to_mso" (sav) <-> "sep_{}_qrot2mso" (cdf)

        # Filter/convert any remaining
        f_i = [anc_conversion_i[i] if i in anc_conversion_i else i for i in f_i]

        print(f_i)
        print(f)

        if ext == "sav":
            if "qrot" in f:
                f_i = ["qrot", "sep{sensor}", "to", "mso"]
            else:
                f_i += ["sep{sensor}", "{look_dir_alt}"]
        elif ext == 'cdf':
            if "qrot" in f:
                f_i = ["sep", "{sensor}"] + f_i
            else:
                f_i = ["sep", "{sensor}{look_dir}"] + f_i

        new_fname = "_".join(f_i)
        print(new_fname)

        for (l_i, s_i) in itertools.product(look_dirs, sensors):
            l_i_expand = ("reverse" if l_i == "r" else "forward")
            fov_varying_names.append(
                new_fname.format(
                    sensor=s_i, look_dir_alt=l_i_expand, look_dir=l_i))

    all_names = time_names + non_fov_varying_names + fov_varying_names

    return all_names


def read_anci(filename, ext='', detector="", look_direction="",
              field_names=None, lib='cdflib', include_unit=True):

    """Return ancillary information in SEP cadence
    for a SEP look direction.

    filename: string, name of SEP L2 data file.
    field_names:
        - time-variables: time(_unix, nssdc, met, ephemeris), epoch
        - position variables:
            mvn_pos_mso, mvn_pos_geo, mvn_sza,
            mvn_lon_geo, mvn_lat_geo, mvn_slt, mvn_pos_eclipj2000,
            earth_pos_eclipj2000, mars_pos_eclipj2000
        - SEP sensor specific variables:
            CDF: sep-[1,2][f,r]_fov_[geo, mso, sso]
            SAV: look_direction_[geo, mso, sso]_sep[1,2]_[forward,reverse]
                SEP FOV pointing unit vectors in [IAU, MSO, SSO]
            CDF: sep_[1,2][f,r]_[sun,ram]_angle
            SAV: [sun,ram,pitch]_angle_sep[1,2]_[forward,reverse]
                angle between SEP FOV and the sun or s/c RAM direction
            sep-[1,2]_qrot2[geo, mso, sso]:
                quaternions for rotating from SEP frame -> IAU, MSO, SSO
            sep-[1,2][f,r]_frac_fov_[mars, ill]:
                fraction of SEP FOV occupied by Mars or illuminated
                disk of Mars
            mars_frac_sky:
                fraction of sky relative to MAVEN that is taken up by Mars.

    """

    if not ext:
        ext = filename[-3:]

    # print(dataset_filename)
    if not field_names:
        field_names =\
            ["time_unix", "epoch", "mvn_pos_mso",
             "mvn_pos_geo", "mvn_sza", "mvn_slt"]

        if detector and look_direction:
            field_names.extend(
                ["fov_mso", "fov_sun_angle", "frac_fov_mars", "qrot2mso"])
            field_names = anc_field_names(
                field_names, detector, look_direction, ext)

    # Read the CDF data into a dict:
    if ext == "cdf":
        anc = read_cdf(filename, field_names, lib=lib)
    elif ext == "sav":
        anc = read_sav(filename, field_names)

    if include_unit:
        anc = process_data_dict(anc, units=units)

    return anc


def read_pad(filename, field_names=None, particle="", include_unit=True):

    """Return SEP pitch angle distributions

    filename: string, name of SEP L2 data file.
    particle: tuple or string, if empty will only load
        B info. otherwise, if includes ion or elec, will
        retrieve that info"""

    if not field_names:
        field_names = ["time", "bmso"]

        if "ion" in particle or "elec" in particle:
            field_names.append("pitch_angle")

        if "ion" in particle:
            field_names.extend(
                ['ion_energy', 'gyroradius_proton', 'gyroradius_oxygen',
                 'ion_efluxpa', 'ion_norm_efluxpa'])
        if "elec" in particle:
            field_names.extend(
                ['electron_energy', 'gyroradius_electron', 'electron_efluxpa',
                 'electron_norm_efluxpa'])

    # Open IDL sav (named "pad")
    pad = read_sav(filename, struct_name="pad", field_names=field_names)

    # Add epoch since not in sav
    pad["epoch"] = UNX_to_UTC(pad["time"])

    # Assign unit
    if include_unit:
        pad = assign_unit(pad, units)

    return pad
