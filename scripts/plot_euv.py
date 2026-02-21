import datetime as dt
import argparse

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize, ListedColormap

from mavenpy import file_path, load, retrieve, plot_tools, helper


# Properties of the Level 2 data product:
l2_bands = ('17-22nm', '0-7nm', '121-122nm')
l2_flag_names = ['Good', 'Occultation', "No pointing",
                 "Partial sun in FOV", "No sun in FOV",
                 "Windowed", "Eclipse", ""]
l2_band_range = [[int(i) for i in (j.split("nm")[0]).split("-")] for j in l2_bands]

l2_flag_colors = ['b', 'salmon', 'gray',
                  'gold', 'r', 'g', 'k', 'pink']


l3_flag_names = ("Good MAVEN proxy",
                 "Poor MAVEN proxy",
                 "(No MAVEN) Good Earth proxy",
                 "(No MAVEN) Poor Earth proxy")
l3_flag_colors = ['b', 'cornflowerblue', 'green', 'r']
irradiance_unit = "$mW/m^{2}$"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_directory", help="Directory containing MAVEN EUV data.",
        required=True
    )
    parser.add_argument(
        "--start",
        help="Start day of MAVEN EUV data (YYYY-MM-DD).",
        type=lambda s: dt.datetime.strptime(s, "%Y-%m-%d"),
        required=True,
    )
    parser.add_argument(
        "--end",
        help="End day of MAVEN EUV data (YYYY-MM-DD).",
        type=lambda s: dt.datetime.strptime(s, "%Y-%m-%d")
    )
    parser.add_argument(
        "--n_days",
        help="# of days to load, alternative to --end.",
        type=int,
        default=1
    )

    # Level of data to access:
    parser.add_argument(
        "--level", help="Level of MAVEN EUV data (l0, l2, or l3)",
        type=str, nargs="+",
        default=('l2',)
    )
    parser.add_argument(
        "--cadence",
        help="Cadence of Level 3 EUV data (minute or daily)."
             " Defaults to minute.",
        type=str,
        default='minute'
    )
    parser.add_argument(
        "--no_clock_correction",
        help="Deactivates the spacecraft clock correction.",
        default=False,
        action='store_true',
    )

    # Remote/download args:
    parser.add_argument(
        "--remote",
        help="Remote to download from if not available (default ssl_sprg).",
        default="ssl_sprg")
    parser.add_argument(
        "--download",
        help="Argument to download files if not present, default False.",
        default=False,
        action='store_true',
    )
    parser.add_argument(
        "--username", help="Username to download from remote.",
        default='')
    parser.add_argument(
        "--password", help="Password for PFP download.",
        default='')
    parser.add_argument(
        "--verbose", help="Enable debugging messages for retrieve.",
        action="store_true")

    # Plot types:
    parser.add_argument(
        "--all_l2", help="Make plots of all Level 2 variables.",
        action="store_true")
    parser.add_argument(
        "--compare_l2_l3", help="Make plot comparing level 2 and 3.",
        action="store_true")
    parser.add_argument(
        "--compare_l0_l2", help="Make plot comparing level 0 and 2.",
        action="store_true")

    args = parser.parse_args()

    # Get local data dir from IDL env if not provided:
    if not args.data_directory:
        data_directory = file_path.get_IDL_data_dir()
    else:
        data_directory = args.data_directory

    # Time range:
    start = args.start
    if not args.end:
        end = start + dt.timedelta(days=args.n_days - 1)
    else:
        end = args.end

    # Check remote access info:
    # Do the download:
    if args.download:
        remote = args.remote

        # Username
        username = args.username
        password = args.password
        if remote == "ssl_sprg":
            password = (password if password else "{}_pfp".format(username))

    # File check:
    if args.download:
        print("Downloading files...")
        for level in args.level:
            if level == 'l2':
                dataset_name = 'bands'
                ext = 'cdf'
            elif level == 'l0':
                dataset_name = 'raw'
                ext = 'tplot'
            else:
                dataset_name = args.cadence
                ext = 'cdf'

            retrieve.sdc_retrieve(
                'euv', destination_dir=data_directory,
                username=username, password=password,
                source=remote,
                dataset_name=dataset_name, ext=ext, level=level,
                start_date=start, end_date=end, verbose=args.verbose)
        print("EUV files updated.")

    # Now find the files on local machine:
    euv_data = {}
    for level in args.level:
        # only one dataset type for EUV L2
        if level == 'l2':
            dataset_name = 'bands'
            ext = 'cdf'
        elif level == 'l0':
            dataset_name = 'raw'
            ext = 'tplot'
        else:
            dataset_name = args.cadence
            ext = 'cdf'

        # Get L2
        euv_li_files = file_path.local_file_names(
            data_directory, 'euv',
            start_date=start, end_date=end,
            level=level, dataset_name=dataset_name, ext=ext)
        if args.verbose:
            print("For Level ", level)
            print("Files to be loaded: ")
            print(euv_li_files)
        euv_li = load.load_data(
            euv_li_files, include_unit=True,
            spice_kernel_dir=data_directory,
            clock_drift_correction=(not args.no_clock_correction))
        euv_data[level] = euv_li

    # Show all Level 2 data:
    if args.all_l2:
        euv_l2 = euv_data["l2"]
        euv_l2_keys = [i for i in euv_l2 if "time" not in
                       i and "epoch" not in i]
        n_plots = len(euv_l2_keys)
        fig, ax = plt.subplots(nrows=n_plots, sharex=True)

        for i, key in enumerate(euv_l2_keys):
            if "epoch" not in key and "time" not in key:
                ax[i].plot(euv_l2["epoch"][0], euv_l2[key][0])
                ax[i].set_ylabel(key)

    # Make Level 0 plot
    if "l0" in args.level:
        euv_l0 = euv_data["l0"]

        # Make plot of band irradiances with quality flag:
        fig, ax = plt.subplots(
            nrows=2, height_ratios=[0.3, 1], sharex=True, figsize=(7.5, 7))

        l0_epoch = euv_l0["epoch"][0]
        l0_T = euv_l0["temperature"][0]
        l0_I = euv_l0["diode_current"][0]
        l0_color = ['g', 'b', 'r', 'k']
        diode_name = ('A', 'B', 'C', 'D')

        ax[0].set_title("EUV Level 0")
        ax[0].plot(l0_epoch, l0_T)
        ax[0].set_ylabel("Cryodiode\ntemperature, C")
        for i in range(4):
            ax[1].plot(l0_epoch, l0_I[i, :], color=l0_color[i],
                       label=diode_name[i])
        ax[1].legend()
        ax[1].set_ylabel("Diode current, DN")

        fig.autofmt_xdate(rotation=30)

        if args.compare_l0_l2:
            comp_fig, comp_ax = plt.subplots(
                nrows=4, sharex=True, figsize=(7.5, 7))
            for i in range(4):
                comp_ax[0].plot(
                    l0_epoch, l0_I[i, :], color=l0_color[i],
                    label=diode_name[i])
            comp_ax[0].legend()
            comp_ax[0].set_ylabel("Diode current, DN")
            comp_ax[0].set_yscale('log')

    # Make Level 2 plot:
    if "l2" in args.level:

        # Get the data:
        euv_l2 = euv_data["l2"]

        # Get properties:
        l2_epoch = euv_l2["epoch"][0]
        l2_irradiance = euv_l2["irradiance"][0] * 1e3
        l2_flag = euv_l2["flag"][0]

        l2_good = np.where(l2_flag == 0)[0]

        # Skip if channel all NaN, which is
        # the case for 0-7nm after a point:
        n_bands = len(l2_bands)
        nonnan_band = [l2_bands[i] for i in range(n_bands) if
                       not np.all(np.isnan(l2_irradiance[:, i]))]

        # Make plot of band irradiances with quality flag:
        fig, ax = plt.subplots(
            nrows=len(nonnan_band) + 1, sharex=True, figsize=(7.5, 7))
        fig.autofmt_xdate(rotation=30)

        ax[0].set_title("EUV Level 2")

        for flag_i in range(len(l2_flag_names) - 1, -1, -1):
            index_i = np.where(l2_flag == flag_i)[0]
            ax[-1].scatter(
                l2_epoch[index_i], l2_flag[index_i],
                marker='.', color=l2_flag_colors[flag_i],
                label=l2_flag_names[flag_i])
        # ax[-1].set_ylabel("Flag")
        ax[-1].set_yticks([i for i in range(8)])
        ax[-1].set_yticklabels(l2_flag_names, rotation=30)
        # ax[-1].yaxis.set_label_position("right")
        # ax[-1].yaxis.tick_right()

        for plot_index, band_i in enumerate(nonnan_band):
            i = l2_bands.index(band_i)

            for flag_i in range(len(l2_flag_names) - 1, -1, -1):

                index_i = np.where(l2_flag == flag_i)[0]

                ax[plot_index].scatter(
                    l2_epoch[index_i], l2_irradiance[index_i, i],
                    marker='.', color=l2_flag_colors[flag_i])

            ax[plot_index].set_ylabel(
                "Irradiance\n{}\n{}".format(band_i, irradiance_unit))


        if args.compare_l0_l2:

            for flag_i in range(len(l2_flag_names) - 1, -1, -1):
                index_i = np.where(l2_flag == flag_i)[0]
                comp_ax[-1].scatter(
                    l2_epoch[index_i], l2_flag[index_i],
                    marker='.', color=l2_flag_colors[flag_i],
                    label=l2_flag_names[flag_i])
            # ax[-1].set_ylabel("Flag")
            comp_ax[-1].set_yticks([i for i in range(8)])
            comp_ax[-1].set_yticklabels(l2_flag_names, rotation=30)
            # ax[-1].yaxis.set_label_position("right")
            # ax[-1].yaxis.tick_right()

            for plot_index, band_i in enumerate(nonnan_band):
                plot_index = plot_index + 1
                i = l2_bands.index(band_i)

                for flag_i in range(len(l2_flag_names) - 1, -1, -1):

                    index_i = np.where(l2_flag == flag_i)[0]

                    comp_ax[plot_index].scatter(
                        l2_epoch[index_i], l2_irradiance[index_i, i],
                        marker='.', color=l2_flag_colors[flag_i])

                comp_ax[plot_index].set_ylabel(
                    "Irradiance\n{}\n{}".format(band_i, irradiance_unit))



    # Get L3
    if "l3" in args.level:
        euv_l3 = euv_data["l3"]
        print(euv_l3.keys())
        l3_epoch = euv_l3["epoch"][0]
        l3_flag = euv_l3["flag"][0]
        l3_spec = euv_l3["spectral_irradiance"][0] * 1e3
        l3_wavelength = euv_l3["wavelength"][0]

        l3_good = np.where(l3_flag == 0)[0]
        l3_wavelength = l3_wavelength[0, :]
        good_l3_epoch = l3_epoch[l3_good]
        good_l3_spec = l3_spec[l3_good, :]

        fig, ax = plt.subplots(figsize=(7.5, 7), nrows=3, sharex=True)
        fig.autofmt_xdate(rotation=30)
        ax[0].set_title("EUV Level 3")
        p = ax[0].pcolormesh(l3_epoch, l3_wavelength, l3_spec.T)
        ax[0].set_ylabel('Wavelength, nm')
        plot_tools.add_colorbar_outside(p, fig, ax[0], label="$mW/m^{2}/nm$")

        # Add plot for integrated fluxes:
        xuv_index = np.where(l3_wavelength <= 10)[0]
        euv_index = np.where((l3_wavelength >= 10) & (l3_wavelength <= 120))[0]
        xuv_irradiance = np.trapz(
            l3_spec[:, xuv_index], l3_wavelength[xuv_index], axis=1)
        euv_irradiance = np.trapz(
            l3_spec[:, euv_index], l3_wavelength[euv_index], axis=1)

        for flag_i in range(len(l3_flag_names) - 1, -1, -1):
            l3_good_i = np.where(l3_flag == flag_i)[0]
            ax[1].scatter(
                l3_epoch[l3_good_i], xuv_irradiance[l3_good_i],
                marker='.', label=l3_flag_names[flag_i],
                color=l3_flag_colors[flag_i])
            ax[2].scatter(
                l3_epoch[l3_good_i], euv_irradiance[l3_good_i],
                marker='.', label=l3_flag_names[flag_i],
                color=l3_flag_colors[flag_i])
        ax[1].set_ylabel("XUV (0-10 nm)\n$mW/m^{2}$")
        ax[2].set_ylabel("EUV (10-120 nm)\n$mW/m^{2}$")
        ax[2].legend()


    if args.compare_l2_l3:

        # Make plot of band irradiances with quality flag:
        fig, ax = plt.subplots(
            nrows=len(l2_bands) + 1, sharex=True, figsize=(7, 5))

        ax[-1].scatter(l2_epoch, l2_flag, marker='.')
        ax[-1].set_ylabel("Flag")
        ax[-1].set_yticks([i for i in range(8)])
        ax[-1].set_yticklabels(l2_flag_names)
        # ax[-1].yaxis.set_label_position("right")
        ax[-1].yaxis.tick_right()

        for i, band_i in enumerate(l2_bands):

            ax[i].scatter(
                l2_epoch[l2_good], l2_irradiance[l2_good, i],
                label='L2', marker='.')

            l_i, l_f = l2_band_range[i]
            l_index = np.where(
                (l3_wavelength >= l_i) & (l3_wavelength <= l_f))[0]
            print(l_i, l_f)
            print(l3_wavelength[l_index])
            l3_irradiance_i = np.trapz(
                good_l3_spec[:, l_index], l3_wavelength[l_index], axis=1)
            ax[i].scatter(
                good_l3_epoch, l3_irradiance_i, marker='.', label='L3')

            ax[i].set_ylabel(
                "Irradiance\n{}\n{}".format(band_i, irradiance_unit))

        ax[0].legend()

    plt.show()
