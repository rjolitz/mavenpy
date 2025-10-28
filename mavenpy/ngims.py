import os
import re
import datetime as dt

import numpy as np


l2_csn_dtype = [('epoch_utc', 'U19'),
                ('time_unix', '<f8'),
                ('t_sclk', '<f8'),
                ("t_tid", "f8"),
                ("tid", "u4"),
                ("orbit", "u4"),
                ("focusmode", "U3"),
                ("alt", "<f8"),
                ("lst", "<f8"),
                ("long", "<f8"),
                ("lat", "<f8"),
                ("sza", "<f8"),
                ("mass", "u4"),
                ("species", "U3"),
                ("cps_dt_bkd", "f8"),
                ("abundance", "f8"),
                ("precision", "f8"),
                ("quality", "U2")]


l2_ion_dtype = [('epoch_utc', 'U19'),
                ('time_unix', '<f8'),
                ('t_sclk', '<f8'),
                ("t_tid", "f8"),
                ("tid", "u4"),
                ("orbit", "u4"),
                ("focusmode", "U3"),
                ("alt", "<f8"),
                ("lst", "<f8"),
                ("long", "<f8"),
                ("lat", "<f8"),
                ("sza", "<f8"),
                ("ion_mass", "u4"),
                ("cps_dt", "f8"),
                ("abundance", "f8"),
                ("sensitivity", "f8"),
                ("SC_potential", "f8"),
                ("precision", "f8"),
                ("quality", "U2")]


l3_rsn_dtype = [('epoch_utc', 'U19'),
                ('time_unix', '<f8'),
                ('t_sclk', '<f8'),
                ("t_tid", "f8"),
                ("tid", "u4"),
                ("orbit", "u4"),
                ("focusmode", "U3"),
                ("alt", "<f8"),
                ("mass", "u4"),
                ("species", "U3"),
                ("density_bins", "f8"),
                ("quality", "U2")]

l3_rsh_dtype = [('epoch_utc', 'U19'),
                ('time_unix', '<f8'),
                ('t_sclk', '<f8'),
                ("t_tid", "f8"),
                ("tid", "u4"),
                ("orbit", "u4"),
                ("exo_alt", "f8"),
                ("mass", "u4"),
                ("species", "U3"),
                ("scale_height", "f8"),
                ("scale_height_error", "f8"),
                ("temperature", "f8"),
                ("temperature_error", "f8"),
                ("fit_residual", "f8"),
                ("quality", "U2")]


# NGIMS time string format:
# Ex: 2023-12-09T14:00:56
l2_time_fmt = "%Y-%m-%dT%H:%M:%S"


def filename_to_datatype(name):
    split_str = name.split("_")
    level = split_str[2]
    datatype = "-".join(split_str[3].split("-")[:2])

    return level, datatype


def select(ngims_dat, orbit_segment='', verification=''):

    '''
    orbit_segment: string, 'inbound' or 'outbound'
    verification: string, 'verified' or 'unverified' '''

    # If the orbit segment and verification level are empty
    # or None, there is nothing to subset.
    if not orbit_segment and not verification:
        return ngims_dat

    # orbit segment: 'inbound' -> "I", 'outbound' -> "O"
    if orbit_segment:
        orbit_segment = orbit_segment[:1].upper()
    else:
        orbit_segment = "[I,O]"

    # orbit segment: 'verified' -> "V", 'unverified' -> "U"
    if verification:
        verification = verification[:1].upper()
    else:
        verification = "[V,U]"

    # Construct string to match:
    search_str = "{}{}".format(orbit_segment, verification)

    if isinstance(ngims_dat, (dict, np.ndarray)):

        # Get the qflag column:
        qflag = ngims_dat["quality"]

        match_bool = [bool(re.match(search_str, i)) for i in qflag]

        # print(search_str)
        # print(match_bool)
        # print(ngims_dat.shape)
        ngims_dat = ngims_dat[match_bool]
        # print(ngims_dat.shape)
        # input()

    elif isinstance(ngims_dat, list):

        return [select(i, orbit_segment=orbit_segment,
                       verification=verification) for i in ngims_dat]

    else:
        raise IOError(
            "Not a NGIMS dict or list of NGIMS dicts, "
            "check first argument.")

    return ngims_dat


def tstring_to_dt(epoch_utc):
    '''Routine that converts the epoch column into
    a datetime object'''

    # Convert into datetimes:
    # Time format: 2023-12-09T14:00:56
    if isinstance(epoch_utc, str):

        return dt.datetime.strptime(epoch_utc, l2_time_fmt)
    elif isinstance(epoch_utc, list):
        return [tstring_to_dt(i) for i in epoch_utc]
    elif isinstance(epoch_utc, np.ndarray):
        return np.array([tstring_to_dt(i) for i in epoch_utc])
    else:
        raise IOError(
            "tstring can only convert lists and np arrays.")


def read(filename, dataset_type=None, fields=None,
         include_unit=None, remove_errant_data=True,
         orbit_segment='', verification=''):

    '''Routine to read Level 2 and 3 CSV files
    produced by the NGIMS team

    Note: Level 3 has significant errors and doesn't make any sense.
        Disrecommend use.

    dataset_type: string, can be 'csn-abund',
        'ion-abund', 'res-den', 'res-sht'.
        NOTE: res-den is not currently in use and DISRECOMMENDED
        for use.
    remove_errant_data: a keyword that when True, removes all rows containing
        -999 as abundance
      '''

    if not dataset_type:
        level, dataset_type = filename_to_datatype(os.path.split(filename)[1])

    # print(filename, level, dataset_type)
    # input()

    if dataset_type == 'csn-abund':
        dtype_i = l2_csn_dtype
    elif dataset_type == 'ion-abund':
        dtype_i = l2_ion_dtype
    elif dataset_type == 'res-den':
        dtype_i = l3_rsn_dtype
    elif dataset_type == 'res-sht':
        dtype_i = l3_rsh_dtype

    # Retrieve all fields
    if not fields:
        fields = [i[0] for i in dtype_i]

    # genfromtxt is faster (0.02 s for 1000x iterations,
    # v 3.7 s for 1000x iterations for loadtxt)
    # data = np.genfromtxt(filename, delimiter=',', skip_header=1, dtype=dtype_i)
    data = np.loadtxt(filename, delimiter=',', skiprows=1, dtype=dtype_i)

    # Inspect columns:
    # data = np.genfromtxt(filename, delimiter=',', names=True)
    # with open(filename, 'r') as f:
    #     print(f.readline())
    # print(dtype_i)
    # print(orbit_segment, verification, data.shape)
    # print(data.dtype.names)
    # print(data['epoch_utc'].shape)
    # input()

    if dataset_type == 'csn-abund':
        # Selects only the specific orbit segment:
        data = select(
            data, orbit_segment=orbit_segment,
            verification=verification)

    if remove_errant_data:
        not_errant = np.where(data['abundance'] != -999.0)[0]
        # print(data.shape, not_errant.shape)
        data = data[not_errant]
        # input()

    # Convert to dict:
    data_dict = {}
    for n in data.dtype.names:
        data_dict[n] = data[n]

    data_dict["time_utc"] = tstring_to_dt(data_dict["epoch_utc"])
    data = data_dict

    return data


def resample(ngims_csn_dat, orbit_nums=None, altitude_bins=None,
             species=('CO2', "Ar")):
    '''Resample the NGIMS CSN data for a given set of orbits and species
    into a provided altitude bin'''

    # Get the available orbits:
    orbnum_dat = [int(dat["orbit"][0]) for dat in ngims_csn_dat]

    # if no orbit #s requested, use the orbnum as orbit_nums:
    if orbit_nums is None:
        orbit_nums = orbnum_dat

    # If no orbit numbers to label by, label by the sorted orbits
    # of the available data:
    N_orbits = len(orbnum_dat)

    # Set up the altitude bins if not provided:
    if altitude_bins is None:
        dalt_km = 5
        altitude_bins = np.arange(100, 500, dalt_km)

    N_altitude_bins = len(altitude_bins) - 1

    # Make arrays to fill in with resampled data:
    shape = (len(species), N_orbits, N_altitude_bins)
    avg_abund = np.empty(shape=shape) + np.nan
    avg_alt = np.empty(shape=shape) + np.nan
    std_alt = np.empty(shape=shape) + np.nan
    std_abund = np.empty(shape=shape) + np.nan

    # Iterate through the orbits:
    for i, orbnum_i in enumerate(orbit_nums):

        # Get the data:
        dat_index = orbnum_dat.index(orbnum_i)
        dat_csn = ngims_csn_dat[dat_index]
        alt_i = dat_csn["alt"]
        sp_i = dat_csn["species"]
        qual = dat_csn["quality"]
        abundance_i = dat_csn["abundance"]

        # get the inbound (& verified) index:
        IV = np.where(qual == "IV")[0]
        I = np.where((qual == "IV") | (qual == "IU"))[0]
        # print(len(I), len(IV), qual.shape)
        # input()

        for sp_index, sp in enumerate(species):

            # Get the matching species index:
            sp_dat_index_i = np.where(sp_i == sp)[0]

            # Subset the altitude and abundance:
            # if sp == "Ar":
            #     # Since the abundances are all set to IU below 0.75-0.8 cm-3
            #     # use all IU and IV for argon:
            #     sp_I_i = np.intersect1d(sp_dat_index_i, I)
            #     alt_sp_i = alt_i[sp_I_i]
            #     abund_sp_i = abundance_i[sp_I_i]
            # else:
            #     sp_IV_i = np.intersect1d(sp_dat_index_i, IV)
            #     alt_sp_i = alt_i[sp_IV_i]
            #     abund_sp_i = abundance_i[sp_IV_i]

            sp_I_i = np.intersect1d(sp_dat_index_i, I)
            alt_sp_i = alt_i[sp_I_i]
            abund_sp_i = abundance_i[sp_I_i]

            # Get the indices of the altitudes in the altitude
            # bins array:
            index = np.digitize(alt_sp_i, altitude_bins)
            uniq_index = np.unique(index)

            # Iterate through the unique indices and calculate
            # the average + standard deviation for each matching
            # altitude bin:
            for index_i in uniq_index:
                if index_i >= N_altitude_bins:
                    continue
                index_j = np.where(index == index_i)[0]

                alts_subset = alt_sp_i[index_j]
                abund_subset = abund_sp_i[index_j]

                avg_alt_i = np.nanmean(alts_subset)
                avg_abund_i = np.nanmean(abund_subset)

                std_alt_i = np.nanstd(alts_subset)
                std_abund_i = np.nanstd(abund_subset)

                avg_abund[sp_index, i, index_i] = avg_abund_i
                avg_alt[sp_index, i, index_i] = avg_alt_i

                std_alt[sp_index, i, index_i] = std_alt_i
                std_abund[sp_index, i, index_i] = std_abund_i

            # Histogram approach:
            # weighted_alt = np.histogram(
            #     alt_sp_IV, bins=resample_alt_km, weights=alt_sp_IV)[0]
            # weighted_abund = np.histogram(
            #     alt_sp_IV, bins=resample_alt_km, weights=abund_sp_IV)[0]
            # N_obs = np.histogram(alt_sp_IV, bins=resample_alt_km)[0]
            # resampled_alt = weighted_alt/N_obs
            # resampled_abund = weighted_abund/N_obs
            # final_abund[sp_index, i, :] = resampled_abund
            # variance_alt = np.histogram(
            #     alt_sp_IV, bins=resample_alt_km,
            #     weights=(alt_sp_IV - resampled_alt)**2)[0]
            # stdev_alt = np.sqrt(variance_alt/N_obs)
            # variance_abund = np.histogram(
            #     alt_sp_IV, bins=resample_alt_km,
            #     weights=(abund_sp_IV - resampled_abund)**2)[0]
            # stdev_abund = np.sqrt(variance_abund/N_obs)

    return {"abund_avg": avg_abund, "altitude_avg": avg_alt,
            "abund_std": std_abund, "altitude_std": std_alt,
            "orbit_nums": orbit_nums, "species": species,
            "altitude_bins": altitude_bins}


def orbit_average(resampled_ngims, orbit_range):

    '''Get the average/stdev over multiple orbits
    given a resampled NGIMS structure:'''

    # Get the avg & stdev info from struct:
    abund_avg = resampled_ngims["abund_avg"]
    alt_avg = resampled_ngims["altitude_avg"]
    abund_std = resampled_ngims["abund_std"]
    alt_std = resampled_ngims["altitude_std"]
    orbs = resampled_ngims["orbit_nums"]

    start, end = orbit_range

    index = [i for i, orb_i in enumerate(orbs) if (start <= orb_i <= end)]
    print(index)

    orb_abun_std = np.sqrt(np.nansum(abund_std[:, index, :]**2, axis=1))
    orb_alt_std = np.sqrt(np.nansum(alt_std[:, index, :]**2, axis=1))

    orb_abund_avg = np.nanmean(abund_avg[:, index, :], axis=1)
    orb_alt_avg = np.nanmean(alt_avg[:, index, :], axis=1)
    lower_quartile = np.nanquantile(
        abund_avg[:, index, :], 0.25, axis=1)
    upper_quartile = np.nanquantile(
        abund_avg[:, index, :], 0.75, axis=1)

    orb = {}
    orb["species"] = resampled_ngims["species"]
    orb["altitude_bins"] = resampled_ngims["altitude_bins"]
    orb["abund_avg"] = orb_abund_avg
    orb["altitude_avg"] = orb_alt_avg
    orb["abund_std"] = orb_abun_std
    orb["altitude_std"] = orb_alt_std

    return orb

