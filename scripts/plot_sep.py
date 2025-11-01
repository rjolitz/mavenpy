import datetime as dt
import argparse
import sys

import numpy as np
from dateutil.parser import parse
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize, ListedColormap

from mavenpy import file_path, load, retrieve, plot_tools, helper


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_directory",
        help="Directory containing MAVEN SEP data.",
        required=True
    )
    parser.add_argument(
        "--start",
        help="Start day of MAVEN SEP data (YYYY-MM-DD HH:MM:SS).",
        type=lambda s: dt.datetime.strptime(s, "%Y-%m-%d"),
        required=True,
    )
    parser.add_argument(
        "--end",
        help="End day of MAVEN SEP data (YYYY-MM-DD HH:MM:SS).",
        type=lambda s: dt.datetime.strptime(s, "%Y-%m-%d")
    )
    parser.add_argument(
        "--n_days",
        help="# of days to load, alternative to --end.",
        type=int
    )

    # Plotting controls:
    parser.add_argument(
        "--zlim",
        help="Interval of flux/counts to map colors onto,"
             "defaults to (1, 10^4) for counts/rate and (1, 10^5) for flux.",
        type=float,
        nargs="+"
    )
    parser.add_argument(
        "--xlim",
        help="Interval to show data between (YYYY-MM-DDTHH:MM_1,"
             " YYYY-MM-DDTHH:MM_2), defaults to (start, end + 1 day).",
        type=str,
        nargs="+"
    )
    parser.add_argument(
        "--ylim",
        help="Interval to show data between energies (E_1,"
             " E_2), defaults to max range.",
        type=float,
        nargs="+"
    )
    parser.add_argument(
        "--vline",
        help="Time to draw a black dashed vertical line at"
             "(YYYY-MM-DD HH:MM:SS).",
        type=lambda s: parse(s)
    )
    parser.add_argument(
        "--cmap",
        help="Color map referenced (e.g. rainbow, jet, viridis, etc).",
        type=str,
        default='jet'
    )
    parser.add_argument(
        "--exclude_atten",
        help="Argument to exclude attenuator if activated.",
        action="store_true"
    )

    # Data type to plot as:
    parser.add_argument(
        "--show_as",
        help="Return plots in given units ('counts', 'rate' or 'flux')."
             " Defaults to flux.",
        type=str,
        default='flux'
    )

    parser.add_argument(
        "--panel",
        help="Split plots so each panel is a detector element e.g. fto "
             " or a particle e.g. elec. Defaults to particle.",
        type=str,
        default='particle'
    )

    parser.add_argument(
        "--plot",
        help="'line' or 'spectra' plot, default is 'spectra'.",
        type=str,
        default='spectra'
    )

    # Subtypes of SEP
    parser.add_argument(
        "--level", help="Level of MAVEN SEP data (l1 or l2)", type=str,
        default='l2'
    )
    parser.add_argument(
        "--sensor",
        help="MAVEN SEP sensor number ('1' and '2'), defaults to both.",
        type=str, nargs="+",
        default=('1', '2')
    )

    # Level 1 argument
    parser.add_argument(
        "--cadence",
        help="Resolution of MAVEN SEP data "
             "('01hr', 'full', '32sec', '5min'). "
             "Defaults to 01hr.",
        type=str,
        default='01hr'
    )

    parser.add_argument(
        "--detector",
        help="Detector pattern in SEP instrument, "
             "defaults to 'f', 't', 'o', 'fto', 'ft', 'ot').",
        type=str, nargs="+",
        default=('f', 't', 'o', 'fto', 'ft', 'ot')
    )

    parser.add_argument(
        "--telemetry_mode",
        help="Telemetry mode of SEP data, "
             "defaults to 'svy'.",
        type=str, nargs="+",
        default='svy'
    )

    # Level 2 arguments:
    parser.add_argument(
        "--type",
        help="Type of L2 MAVEN SEP data ('raw' or 'cal')."
             " Defaults to cal.",
        type=str,
        default='cal'
    )
    parser.add_argument(
        "--particle",
        help="MAVEN SEP particle type ('elec' and 'ion'), defaults to both.",
        type=str, nargs="+",
        default=('ion', 'elec')
    )

    parser.add_argument(
        "--look_dir",
        help="MAVEN SEP look direction ('f' and 'r'), defaults to both.",
        type=str, nargs="+",
        default=('f', 'r')
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
        "--password", help="Password for PFP download. "
        "If not provided, defaults to username_pfp.")

    parser.add_argument(
        "--verbose", help="Enable debugging messages for retrieve.",
        action="store_true")

    args = parser.parse_args()

    # Get local data dir from IDL env if not provided:
    if not args.data_directory:
        data_directory = file_path.get_IDL_data_dir()
    else:
        data_directory = args.data_directory

    remote_source = args.remote

    # Parse args on details:
    level = args.level
    sensors = args.sensor
    particles = args.particle
    look_directions = args.look_dir
    detector = [i.upper() for i in args.detector]
    telescope = ('A', 'B')

    if level == 'l2':
        # SEP parameters for calib dataset:
        dtype = args.type

        ext = 'cdf'
        print(dtype, particles, sensors, look_directions)

        # Search info:
        dataset_name = "s{{sensor_name}}-{dtype}-svy-full".format(dtype=dtype)
        search_dataset_name = dataset_name.format(
            sensor_name="(.*)" if len(sensors) > 1 else sensors[0])

    elif level == 'l1':
        # SEP parameters for level 1 dataset:
        dtype = 'raw'
        cadence = args.cadence
        telemetry_mode = args.telemetry_mode
        ext = 'sav'
        search_dataset_name = cadence

    # Whether or not data is returned as rates, counts, or flux:
    output_data_units = args.show_as
    output_data_cmap = args.cmap
    if args.zlim:
        vmax, vmin = args.zlim
    else:
        if "flux" in output_data_units:
            vmin, vmax = 1, 1e5
        else:
            vmin, vmax = 1, 1e4

    if output_data_units == "flux":
        plot_unit = "#/cm2\n/s/sr/keV"
    elif output_data_units == "eflux":
        plot_unit = "keV/cm2\n/s/sr/keV"
    elif output_data_units == "rate":
        plot_unit = "#/s"
    elif output_data_units == "counts":
        plot_unit = "#"

    if args.plot == 'line':
        plot_unit = plot_unit.replace("\n", "")

    output_data_norm = LogNorm(vmin=vmin, vmax=vmax)

    if dtype == "raw":
        read_detector = ('f', 't', 'o', 'fto', 'ft', 'ot')
        output_calibration_level = ('raw', 'fto', 'cal')
        # output_data_units = ("counts", "rate", "flux")
    else:
        output_calibration_level = 'cal'
        read_detector = None
        # output_data_units = "flux"

    # Time range:
    start = args.start

    if not args.end:
        if args.n_days:
            end = start + dt.timedelta(days=args.n_days - 1)
        else:
            end = start

    else:
        end = args.end

    # Do the download:
    if args.download:

        # Username
        username = args.username
        # password = ''
        if args.password:
            password = args.password

        else:
            if args.username and remote_source == "ssl_sprg":
                password = "{}_pfp".format(username)

        retrieve.sdc_retrieve(
            'sep', destination_dir=data_directory,
            level=level, ext=ext,
            username=username, password=password,
            dataset_name=search_dataset_name,
            start_date=start, end_date=end, verbose=args.verbose)

    # Get the filenames:
    if level == 'l1':
        file_names = file_path.local_file_names(
            data_directory, 'sep', start_date=start,
            end_date=end, level=level,
            dataset_name=cadence,
            ext=ext, source='ssl_sprg')
    elif level == 'l2':
        file_names = {}
        all_files = []
        for sensor_i in sensors:
            # Retrieve file names of calibrated SEP data:
            dataset_name_i = dataset_name.format(sensor_name=sensor_i)
            # print(dataset_name_i)
            file_names_i = file_path.local_file_names(
                data_directory, 'sep', start_date=start,
                end_date=end, level=level,
                dataset_name=dataset_name_i,
                ext=ext, source='ssl_sprg')
            file_names[sensor_i] = file_names_i
            all_files += file_names_i
            # input()

        actual_available_files = [i for i in all_files if i]
        # print(file_names_i)
        # print(actual_available_files)
        # input()
        if not actual_available_files:
            print(
                "No SEP data available, exiting...".format(
                    sensor_i))
            sys.exit()

    # Load the data:
    if level == 'l1':
        sep_all = load.load_data(
            file_names, include_unit=False,
            sensor=sensors,
            telemetry_mode=telemetry_mode,
            detector=read_detector, telescope=telescope,
            output_calibration_level=output_calibration_level,
            output_data_units=('flux', 'rate', 'counts'))

        sep_keys = sep_all.keys()
        sep_dict = {}
        for sensor_i in sensors:
            # Get all keys starting with that sensor name
            preceding_str = "{}_".format(sensor_i)
            matching_keys_i = [i for i in sep_keys if preceding_str in i]
            shortened_key_i = [i.split(preceding_str)[1] for i in matching_keys_i]
            sep_dict[sensor_i] = {}
            for old_key, new_key in zip(matching_keys_i, shortened_key_i):
                sep_dict[sensor_i][new_key] = sep_all[old_key]
    else:
        sep_dict = {}
        for sensor_i in sensors:
            sep_i = load.load_data(
                file_names[sensor_i],
                detector=read_detector, telescope=telescope,
                include_unit=False, label_by_detector=False,
                output_calibration_level=output_calibration_level
                )
            sep_dict[sensor_i] = sep_i

    # print(sep_dict['1'].keys())

    # Get # of sensors / particles / fovs
    n_sensors = len(sensors)

    # Get the # of subplots from # of particles + lookdirs
    # OR detector elements + telescope:

    if args.panel == 'particle':
        iterate_x = particles
        iterate_y = look_directions

    else:
        iterate_x = detector
        iterate_y = telescope

    n_x_plot = len(iterate_x)
    n_y_plot = len(iterate_y)

    # Set up the plot:
    atten_plot_height = 0.2
    n_plots = n_sensors * (n_x_plot * n_y_plot)
    height_ratios = n_sensors * [1] * n_x_plot * n_y_plot
    if not args.exclude_atten:
        height_ratios = n_sensors * ([1] * n_x_plot * n_y_plot + [atten_plot_height])
        n_plots += n_sensors

    # fig_height = max(min(
    #     n_sensors*(1*n_x_plot * n_y_plot + atten_plot_height) * 1, 8), 4)
    fig_height = max(min(sum(height_ratios), 8), 4)
    fig, ax = plt.subplots(
        nrows=n_plots, height_ratios=height_ratios, sharex=True,
        figsize=(10, fig_height))

    if args.plot == 'spectra':
        # Adjust for fitting colorbar:
        # plt.subplots_adjust(
        #     left=0.135, bottom=0.05, right=0.88, top=0.95, wspace=0, hspace=0)
        plt.subplots_adjust(
            left=0.135, right=0.88, top=0.95, wspace=0, hspace=0)
    elif args.plot == 'line':
        plt.subplots_adjust(top=0.95, right=0.95, wspace=0, hspace=0)

    for sensor_index, sensor_i in enumerate(sensors):
        sep_i = sep_dict[sensor_i]

        # Retrieve time:
        time_i = sep_i["time_unix"]
        epoch_i = sep_i["epoch"]
        att_i = sep_i["attenuator_state"]

        for index_x, name_x in enumerate(iterate_x):
            for index_y, name_y in enumerate(iterate_y):

                if args.panel == 'particle':
                    data_label_i = "{}_{}"
                    data_label_i = data_label_i.format(name_y, name_x)
                    name_x_i = name_x.capitalize()[:4]
                    name_y_i = name_y.upper()
                elif args.panel == 'detector':
                    data_label_i = "{}-{}"
                    name_x_i = name_x.upper()
                    name_y_i = name_y.upper()
                    data_label_i = data_label_i.format(name_y_i, name_x_i)

                # print(data_label_i)
                energy = sep_i["{}_energy".format(data_label_i)]

                if output_data_units == "eflux":
                    nflux = sep_i["{}_flux".format(data_label_i)]
                    flux = energy[np.newaxis, :] * nflux
                else:
                    flux = sep_i["{}_{}".format(data_label_i, output_data_units)]

                print(flux.shape, epoch_i.shape, energy.shape)

                ylabel_i = "SEP {}{} {}\nEnergy [keV]".format(
                    sensor_i, name_y_i, name_x_i)

                plot_index =\
                    index_x*n_y_plot + index_y +\
                    sensor_index*(n_x_plot*n_y_plot + 1)
                ax_i = ax[plot_index]

                if args.plot == "spectra":
                    p = ax_i.pcolormesh(
                        epoch_i, energy, flux.T, norm=output_data_norm,
                        cmap=output_data_cmap)
                    cbar = plot_tools.add_colorbar_outside(
                        p, fig, ax_i, label=plot_unit)
                    ax_i.set_yscale('log')
                    ax_i.set_ylabel(ylabel_i, rotation='horizontal',
                                    va='center', labelpad=35)
                    # cbar.ax.tick_params(labelsize=10)
                    cbar.ax.yaxis.label.set(
                        rotation='horizontal', ha='left')
                elif args.plot == 'line':
                    cmap = plt.get_cmap(output_data_cmap)
                    n_energy_plot = 4

                    above_e = (energy > 25)
                    flux = flux[:, above_e]
                    energy = energy[above_e]
                    n_energy_act = len(energy)

                    N_spacing = int(n_energy_act/n_energy_plot)
                    index_i = [i for i in range(n_energy_act) if i % N_spacing == 0]

                    en_color = cmap(np.linspace(0, 1, n_energy_act))

                    for en_i in index_i:
                        ax_i.plot(
                            epoch_i, flux[:, en_i],
                            label=helper.format_energy_as_string(energy[en_i]*1e3),
                            color=en_color[en_i])
                    ax_i.set_yscale('log')
                    ax_i.set_ylabel("SEP {}{}, {}\n{}".format(sensor_i, name_y_i, name_x_i, plot_unit))
                    ax_i.legend()

                if args.ylim is not None:
                    ax_i.set_ylim(args.ylim)

                if args.vline is not None:
                    ax_i.axvline(args.vline, color='k', linestyle='--')

        if not args.exclude_atten:
            ax_atten = ax[plot_index + 1]
            ax_atten.pcolormesh(
                epoch_i, [0, 1], [att_i, att_i],
                cmap=ListedColormap(['b', 'r']), norm=Normalize(vmin=1, vmax=2))
            # ax_atten.yaxis.set_visible(False)
            # ax_atten.set_yticks([], minor=True)
            ax_atten.tick_params(axis='y', which='both', left=False, labelleft=False)
            ax_atten.set_ylabel(
                "Atten.", rotation='horizontal',
                va='center', labelpad=35)
    # fig.autofmt_xdate()
    # ax[0].xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[30,]))
    import matplotlib.dates as mdates
    from mavenpy import helper

    # ax[0].xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12,]))
    # ax[0].xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 4, 8, 12, 16, 20]))
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M:%S'))

    # if start_date == end_date:
    if not args.xlim:
        start_x, n_days, end_date = helper.sanitize_date_inputs(
            start_date=start, end_date=end)
        end_x = end_date + dt.timedelta(days=1)
    else:
        start_x, end_x = args.xlim
        start_x = parse(start_x)
        end_x = parse(end_x)

    ax[0].set_xlim([start_x, end_x])

    plt.show()
