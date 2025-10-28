import argparse
import datetime as dt

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from dateutil.relativedelta import relativedelta

from mavenpy import file_path, spice, load, anc, helper, mars_shape_conics, retrieve, plot_tools


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_directory",
        help="Directory containing MAVEN data.",
        required=True
    )
    parser.add_argument(
        "--start_date",
        help="Start date of retrieved data (YYYY-MM-DD).",
        type=lambda s: dt.datetime.strptime(s, "%Y-%m-%d"),
        required=True,
    )
    parser.add_argument(
        "--end_date",
        help="Stop date of retrieved data (YYYY-MM-DD).",
        type=lambda s: dt.datetime.strptime(s, "%Y-%m-%d"),
    )
    parser.add_argument(
        "--n_days",
        help="Number of days of data retrieved.",
        type=int,
    )
    parser.add_argument(
        "--download",
        help="Download files to data_directory if not found.",
        action="store_true",
        default=False,
    )

    parser.add_argument("--username", help="Username for PFP download.")
    args = parser.parse_args()

    # Clean the date inputs:
    start_date, n_days, end_date = helper.sanitize_date_inputs(
        start_date=args.start_date, n_days=args.n_days, end_date=args.end_date)

    # If download, set up the username / password:
    if args.download:
        if not args.username:
            six_months_ago = dt.datetime.now() - relativedelta(months=6)
            if start_date > six_months_ago:
                raise NameError(
                    "Need username to download newer than 6 months.")
            username = None
            password = None
        else:
            username = args.username
            password = "{}_pfp".format(username)

    # Get the data directory:
    data_directory = args.data_directory

    # Load the spice kernels:
    k = spice.load_kernels(
        data_directory,
        start_date=start_date, end_date=end_date,
        download_if_not_available=args.download,
        verbose=True)
    print("Loaded kernels.")

    # Get SWIA moment file names / load the SWIA onboard moments
    if args.download:
        retrieve.sdc_retrieve(
            'swia', destination_dir=data_directory,
            username=username, password=password,
            dataset_name='onboardsvymom', ext='cdf', level='l2',
            start_date=start_date, end_date=end_date,
            verbose=True)
        print("SWIA files updated.")

    swim_file_names = file_path.local_file_names(
        data_directory, 'swia', start_date=start_date,
        end_date=end_date, dataset_name='onboardsvymom',
        ext='cdf', source='ssl_sprg')
    swia_moments = load.load_data(swim_file_names)

    # Get MAG file names / load the MAG data
    mag_coord = 'pl'

    # mag_coord, mag_ext, mag_res, mag_level = 'ss', 'sts', '1sec', 'l2'
    for mag_ext, mag_res, mag_level in zip(('sav', 'sav'), ('30sec', '1sec'), ('l2', 'l1')):
        if args.download:
            retrieve.sdc_retrieve(
                'mag', destination_dir=data_directory,
                username=username, password=password,
                ext=mag_ext, res=mag_res, level=mag_level, coord=mag_coord,
                start_date=start_date, end_date=end_date)
            print("MAG files updated.")

        try:
            mag_file_names = file_path.local_file_names(
                data_directory, 'mag', start_date=start_date, end_date=end_date,
                ext=mag_ext, res=mag_res, level=mag_level, coord=mag_coord,
                source='ssl_sprg',)
        except IOError:
            continue

    actual_mag_files = [i for i in mag_file_names if i]

    # if len(actual_mag_files) != 0:
    #     break
    # print(mag_ext, mag_res, mag_level)
    # input()

    mag = load.load_data(
        mag_file_names, ext=mag_ext, res=mag_res,
        level=mag_level, coord=mag_coord)

    mag_epoch = mag["epoch"][0]
    bx = mag["Bx"][0]
    by = mag["By"][0]
    bz = mag["Bz"][0]
    bmag = np.sqrt(bx**2 + by**2 + bz**2)

    # Rotate from payload to MSO coordinates:
    b_mso = spice.bpl_to_bmso(mag_epoch, bx, by, bz)
    bx, by, bz = b_mso[:, 0], b_mso[:, 1], b_mso[:, 2]

    # Get SWEA file names / load the SWEA data
    if args.download:
        retrieve.sdc_retrieve(
            'swea', destination_dir=data_directory,
            username=username, password=password,
            start_date=start_date, end_date=end_date,
            dataset_name='svyspec', ext='cdf', level='l2')
        print("SWEA files updated.")
    swespec_files = file_path.local_file_names(
        data_directory, 'swea',
        start_date=start_date, end_date=end_date,
        dataset_name='svyspec', ext='cdf', level='l2')
    swe_spec = load.load_data(swespec_files, include_unit=False)
    print("SWE Spec loaded.")

    # Get SEP file names / load SEP data
    if args.download:
        retrieve.sdc_retrieve(
            'sep', destination_dir=data_directory,
            username=username, password=password,
            dataset_name='s1-cal-svy-full', ext='cdf',
            start_date=start_date, end_date=end_date)
        print("SEP files updated.")
    sep_1_file_names = file_path.local_file_names(
        data_directory, 'sep', start_date=start_date,
        end_date=end_date, dataset_name='s1-cal-svy-full',
        ext='cdf', source='ssl_sprg')
    print(sep_1_file_names)
    sep_1_calib = load.load_data(
        sep_1_file_names, include_unit=True, label_by_detector=False)

    # Retrieve orbit ephemeris for this time period
    eph = anc.read_orbit_ephemeris(
        data_directory,
        start_date=start_date, end_date=end_date,
        download_if_not_available=False)

    # Get MAVEN position
    sc_time_utc, sc_time_unx, x, y, z = spice.load_MAVEN_position(
        start_date, end_date=end_date, n_sample_points=400)
    alt = np.sqrt(x**2 + y**2 + z**2) - 3390

    # Make plot
    fig, ax = plt.subplots(nrows=6, sharex=True, figsize=(7, 8))

    swia_epoch = swia_moments["epoch"][0]
    n, n_unit = swia_moments["density"]
    v, v_unit = swia_moments["velocity_mso"]
    v_m = np.sqrt(np.sum(v**2, axis=1))

    # SWIA moment plot
    ax[0].plot(swia_epoch, n, color='b')
    ax[0].set_ylabel("n, {}".format(n_unit), color='b')
    ax[0].set_yscale('log')
    ax_v = ax[0].twinx()
    ax_v.plot(swia_epoch, v_m, color='r')
    ax_v.set_ylabel("Velocity, km/s", color='r')

    # MAG plot
    ax[1].plot(mag_epoch, bx, color='b', label='Bx')
    ax[1].plot(mag_epoch, by, color='g', label='By')
    ax[1].plot(mag_epoch, bz, color='r', label='Bz')
    ax[1].plot(mag_epoch, bmag, color='k', label='|B|')
    ax[1].axhline(0, color='gray', linestyle='--')
    ax[1].set_ylabel("B, nT")
    ax[1].set_ylim([-50, 50])
    ax[1].legend()

    # SWEA plot
    swe_epoch = swe_spec["epoch"]
    p = ax[2].pcolormesh(
        swe_epoch, swe_spec['energy'],
        swe_spec['diff_en_fluxes'].T,
        norm=LogNorm(vmin=1e5, vmax=1e10), cmap='viridis')
    ax[2].set_yscale('log')
    ax[2].set_ylabel("SWEA\nEnergy, eV")
    # plt.colorbar(p, ax=ax_all[1])
    plot_tools.add_colorbar_outside(
        p, fig, ax[2], orientation='vertical',
        label='Eflux\neV/eV/cm2/ster/s')

    # SEP plot
    sep_epoch = sep_1_calib["epoch"][0]
    elec_flux, flux_unit = sep_1_calib["f_elec_flux"]
    elec_energy, en_unit = sep_1_calib["f_elec_energy"]
    ion_flux = sep_1_calib["f_ion_flux"][0]
    ion_energy = sep_1_calib["f_ion_energy"][0]
    p = ax[-3].pcolormesh(
        sep_epoch, elec_energy, elec_flux.T, norm=LogNorm(1, 1e6))
    ax[-3].set_yscale('log')
    ax[-3].set_ylabel('1F elec')
    plot_tools.add_colorbar_outside(p, fig, ax[-3])
    p = ax[-2].pcolormesh(
        sep_epoch, ion_energy, ion_flux.T, norm=LogNorm(1, 1e6))
    ax[-2].set_yscale('log')
    ax[-2].set_ylabel('1F ion')
    plot_tools.add_colorbar_outside(p, fig, ax[-2])

    alt2 = mars_shape_conics.region_separation(x, y, z, alt)
    c = mars_shape_conics.region_colors

    for region in alt2:
        ax[-1].plot(sc_time_utc, alt2[region], label=region, color=c[region])
    ax[-1].legend()
    ax[-1].set_ylabel('Alt, km')
    ax[-1].set_yscale('log')

    anc.add_orbit_axis(ax[0], ephemeris=eph, label='Orb. #')

    plt.show()
