import sys
import os
import argparse
import datetime as dt
import re

import requests
from dateutil.parser import parse
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize, ListedColormap

from mavenpy import read, helper, anc, spice, mars_shape_conics, retrieve

sprg_url = "http://sprg.ssl.berkeley.edu/data"
tplot_file_name = 'mvn_ql_pfp_{yyyy}{mm}{dd}.tplot'
tplot_dir = ("maven", "anc", "tohban", "{yyyy}", "{mm}")


spec_colors = {"mvn_mag_bamp_1sec": "k", "mvn_mod_bcrust_amp": "b"}

formatted_ylabels =\
    {"mvn_sep1f_ion_eflux": "SEP 1F Ion\nEnergy [keV]",
     "mvn_sep1r_ion_eflux": "SEP 1R Ion\nEnergy [keV]",
     "mvn_sep1f_elec_eflux": "SEP 1F Elec\nEnergy [keV]",
     "mvn_sep1r_elec_eflux": "SEP 1R Elec\nEnergy [keV]",
     "mvn_sta_c0_e": "STA C0\n Energy [eV]",
     "mvn_sta_c6_m": "STA C6\nMass [amu]",
     "mvn_swis_en_eflux": "SWI Spec\nEnergy[eV]",
     "mvn_swe_etspec": "SWE\nEnergy[eV]",
     "mvn_lpw_iv": "LPW (IV)\n[V]",
     "mvn_mag_bamp_1sec": "|B| [nT]",
     "alt2": "Alt [km]",
     "mvn_mag_bang_1sec": "MAG (MSO)\nBphi [deg]",
     "burst_flag": "BST"}

formatted_zlabels =\
    {"mvn_swis_en_eflux": "EFLUX",
     "mvn_swe_etspec": "EFLUX",
     "mvn_lpw_iv": "log|IV|\n[nA]"}


def load_tohban(local_data_directory="", data_files=[],
                start_date='', end_date="", n_days=None,
                mirror_remote_tree=True,
                download=True,
                sprg_username="", sprg_password="",
                verbose=None):

    # Retrieve data files if not provided
    if not data_files:

        data_files = []

        # Get the time range
        dt_range = helper.daterange(
            start_date=start_date, end_date=end_date, n_days=n_days)

        # First, see if maven dir exists:
        if not os.path.exists(local_data_directory):
            raise FileNotFoundError(
                "MAVEN directory not found, check if"
                " exists / drive connected.")

        # Make the remote URL
        remote_url_fstring = "/".join((sprg_url, *tplot_dir))

        # If we are mirroring the remote directory tree
        # construct the path to a given file.
        # Else, assume files in local directory.
        if mirror_remote_tree:
            destination_dir = os.path.join(
                local_data_directory, *tplot_dir)
        else:
            destination_dir = local_data_directory

        # Organize the requested data files by unique
        # URL to retrieve from:
        unique_remote_url_dict = {}
        for dt_i in dt_range:
            dt_fstring_i = helper.dt_fstring(dt_i)

            # Get URL and destination folder for year/month.
            # Will index by this, since this is the root downloaded
            # from and will account for same root files (e.g. pfp l0 svy)
            # remote_url_i = remote_url_fstring.format(
            #     yyyy=yyyy_i, mm=mm_i)
            remote_url_i = remote_url_fstring.format(
                **dt_fstring_i)
            # print(remote_url_i)
            filename_i = tplot_file_name.format(**dt_fstring_i)

            # Add dictionary entry for the remote URL if not created,
            # otherwise add the day to existing entry.
            if remote_url_i in unique_remote_url_dict:
                # Append to filenames:
                unique_remote_url_dict[remote_url_i]['filename'].append(
                    filename_i)
            else:
                destination_dir_i = destination_dir.format(
                    **dt_fstring_i)

                unique_remote_url_dict[remote_url_i] =\
                    {'filename': [filename_i],
                     'destination': destination_dir_i
                     }
        print(unique_remote_url_dict)

        # input()

        # Look for files locally and then download from remote
        # if not found:

        if download:
            # Iterate through each webpage:
            session = requests.session()
            # print(username, password)
            if not sprg_username:
                raise Exception(
                    "tohban files only available from SSL SPRG"
                    ", need username & password.")

            session.auth = (sprg_username, sprg_password)
            # if verbose:
            print("Session opened, iterating through yyyy/mm...")

        for url_i in unique_remote_url_dict:
            if verbose:
                print("URL: {}".format(url_i))
                # print('{}/{}:'.format(*yyyy_mm_i))
            info_i = unique_remote_url_dict[url_i]
            filenames_i = info_i['filename']
            destination_i = info_i['destination']

            # See if destination dir_exist:
            local_files_i = []
            if os.path.exists(destination_i):
                # Get list of local files present in the directory:
                local_files_i = os.listdir(destination_i)

            html_soup_i = []
            if download:
                # Get the HTML of the URL, formatted
                # as a "BeautifulSoup object":
                # SLOWEST STEP
                try:
                    html_soup_i = retrieve.html_retrieve(
                        url_i, session=session)
                except requests.exceptions.HTTPError:
                    # if verbose:
                    print("CAUTION: Files in {} don't exist online. Note, tohban files "
                          "are deleted after 3 months, use the actual data if "
                          "you are looking back that far.".format(url_i))

            for file_i in filenames_i:
                # print(file_i)
                # Make path:
                path_i = os.path.join(destination_i, file_i)
                # print(path_i)

                # Get the remote files corresponding to the requested files
                if html_soup_i:
                    a_tag_i = html_soup_i.find('a', href=file_i)
                else:
                    a_tag_i = None

                # If not present locally or remotely, do nothing:
                # If only present locally and not remotely, add local file:
                if not a_tag_i and file_i in local_files_i:
                    # print("Present only locally:")
                    data_files.append(path_i)
                    continue
                # If exists remotely and locally, see if updated on remote:
                elif a_tag_i:
                    # print("Present remotely")
                    if file_i in local_files_i:
                        # print("Present locally")
                        local_posix_time_i = os.path.getmtime(
                            os.path.join(destination_i, file_i))
                        local_modtime_utc = dt.datetime.fromtimestamp(
                            local_posix_time_i)
                        remote_time_size_str = a_tag_i.next_sibling.strip()
                        remote_modtime_utc = parse(
                            remote_time_size_str.split("  ")[0])

                        #  If not updated on remote, no need to DL again.
                        if (local_modtime_utc >= remote_modtime_utc):
                            data_files.append(path_i)
                            continue

                    if not os.path.exists(destination_i):
                        os.makedirs(destination_i)

                    retrieve.download_file(
                        "/".join((url_i, file_i)),
                        path_i,
                        session=session, chunk_size=1048576,
                        verbose=verbose)
                    data_files.append(path_i)

    if not isinstance(data_files, list):
        data_files = [data_files]

    print(data_files)

    # Now iterate through files:
    retrieve_plot_params = True

    final_data = {}
    final_plot_info = {}

    for file_i in data_files:
        print(file_i)

        data, plot_info = read.read_tplot(
            file_i, return_plot_parameters=retrieve_plot_params)

        if retrieve_plot_params:
            if "tplot_order" in plot_info:
                final_plot_info = plot_info
                retrieve_plot_params = False

        if not final_data:
            final_data = data
            continue

        # Iterate through contained data to append:
        for n in data:
            # print(n)
            # If the key not entered yet, just add:
            if n not in final_data:
                final_data[n] = data[n]
            else:
                data_i = data[n]
                final_data_i = final_data[n]

                if not isinstance(final_data_i, dict):
                    continue

                # print(final_data_i.keys())
                # print(data_i.keys())

                final_time = np.append(
                    final_data_i["time_unix"], data_i["time_unix"])
                final_data[n]["time_unix"] = final_time

                if "z" in data_i:
                    final_data[n]["z"] = np.append(
                        final_data_i["z"], data_i["z"], axis=-1)

                    # Only update the y axis if it varies with time
                    y_nd_i = len(data_i["y"].shape)
                    if y_nd_i == 2:
                        final_data[n]["y"] = np.append(
                            final_data_i["y"], data_i["y"], axis=1)
                else:
                    final_data[n]["y"] = np.append(
                        final_data_i["y"], data_i["y"], axis=-1)

                #     print()

                #     print(j, data_i[j].shape, final_data[n][j].shape)
                # input()

    return final_data, final_plot_info


def add_colorbar_outside(im, fig, ax, **kwargs):
    # from stack overflow:
    fig = ax.get_figure()
    # bbox contains the
    # [x0 (left), y0 (bottom), x1 (right), y1 (top)] of the axis.
    bbox = ax.get_position()
    width = 0.01
    eps = 0.01  # margin between plot and colorbar
    # [left most position, bottom position, width, height] of color bar.
    cax = fig.add_axes([bbox.x1 + eps, bbox.y0, width, bbox.height])
    cbar = fig.colorbar(im, cax=cax, **kwargs)

    return cbar


if __name__ == "__main__":

    # Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--data_file", help="Tohban file.", nargs='+')
    parser.add_argument(
        "-d", "--data_directory", help="Directory containing MAVEN data.")
    parser.add_argument(
        "--start_date",
        help="Start date of download range (YYYY-MM-DD).",
        type=lambda s: dt.datetime.strptime(s, "%Y-%m-%d"))
    parser.add_argument(
        "--end_date",
        help="End date of download range (YYYY-MM-DD).",
        type=lambda s: dt.datetime.strptime(s, "%Y-%m-%d"))
    parser.add_argument(
        "--n_days", help="Number of days since start date to recover date.",
        type=int)

    # Argument for providing the PFP username for download:
    parser.add_argument(
        "--username",
        help="Username for PFP download, only required if download enabled.")

    # Argument for controling whether or not to initiate download:
    parser.add_argument(
        "--download",
        help="Keyword to download data if not already available or supplied as -f.",
        action='store_true')

    # Argument to show (or not) the altitude colored by magnetic region
    # This can be disabled if no wish to download spice:
    parser.add_argument(
        "--skip_alt_by_region",
        help="Keyword to skip plotted altitude colored by magnetic region"
             " e.g. solar wind, sheath, pileup.",
        action='store_true')

    # Keyword to exclude certain datasets (useful for ignoring panels
    # if nothing interesting for further analysis, such as SEPs during a quiet
    # period).
    parser.add_argument(
        "--exclude",
        help="Instrument datasets to ignore, e.g. 'sep' or 'sep sta'.",
        type=str, nargs="+", default=())

    # Keyword to include print statements
    parser.add_argument(
        "--verbose", help="Enable print statements to help debugging.",
        action='store_true')

    args = parser.parse_args()

    # Keyword that includes the altitude profile plotted by region
    # (requires spice kernels)
    plot_alt_by_region = (not args.skip_alt_by_region)

    start = args.start_date
    n_days = args.n_days
    end = args.end_date
    dl_kernels = args.download
    dl_tohban = args.download


    end = end - dt.timedelta(seconds=1)

    if dl_tohban:
        username = args.username
        password = "{}_pfp".format(username)

        data, plot_info = load_tohban(
            local_data_directory=args.data_directory,
            start_date=start, n_days=n_days, end_date=args.end_date,
            data_files=args.data_file, download=dl_tohban,
            sprg_username=username, sprg_password=password)
    else:
        data, plot_info = load_tohban(
            local_data_directory=args.data_directory,
            start_date=start, n_days=n_days, end_date=args.end_date,
            data_files=args.data_file, download=dl_tohban)
    # input()

    plot_info_keys = list(plot_info.keys())
    data_keys = list(data.keys())

    if len(data_keys) == 0:
        print("Exiting, no datafiles / data to show.")
        sys.exit()

    # print(plot_info_keys)
    # print(data_keys)

    if "tplot_names" in plot_info_keys:
        tplot_names = plot_info["tplot_order"]

    else:
        tplot_names = ['mvn_sep1f_ion_eflux', 'mvn_sep1r_ion_eflux', 'mvn_sep1f_elec_eflux', 'mvn_sep1r_elec_eflux', 'mvn_sta_c0_e', 'mvn_sta_c6_m', 'mvn_swis_en_eflux', 'mvn_swe_etspec', 'mvn_lpw_iv', 'mvn_mag_bamp', 'mvn_mag_bang_1sec', 'alt2', 'burst_flag']
    print(tplot_names)

    for excl in args.exclude:
        tplot_names = [i for i in tplot_names if i.find(excl) == -1]

    # LPW IV data is always prelogged:
    plot_info["mvn_lpw_iv"]["zscale"] = 'linear'
    plot_info["mvn_lpw_iv"]["yscale"] = 'linear'

    # input()

    if n_days and not end:
        end = start + dt.timedelta(days=n_days)

    if not start and not end:
        data_i = data[tplot_names[0]]
        t_unx_i = data_i['time_unix']
        t_unx_nonnan = t_unx_i[~np.isnan(t_unx_i)]
        start_unx = t_unx_nonnan[0]
        end_unx = t_unx_nonnan[-1]
        start_utc = helper.UNX_to_UTC(start_unx)
        end_utc = helper.UNX_to_UTC(end_unx)

        print(start_utc, end_utc)

        trunc_start_utc = dt.datetime(start_utc.year, start_utc.month, start_utc.day)
        # trunc_end_utc = dt.datetime(end_utc.year, end_utc.month, end_utc.day)
        # if trunc_start_utc == trunc_end_utc:
        #     trunc_end_utc += dt.timedelta(days=1)

        # print(trunc_start_utc, trunc_end_utc)

        start = trunc_start_utc
        # end = trunc_end_utc
        n_days = len(args.data_file)

        # input()

    plot_axis = []
    plot_names = []
    secondary_labels = {}
    secondary_colors = {}

    for i, plot_name_i in enumerate(tplot_names):

        print(plot_name_i)
        if plot_name_i == 'alt2' and plot_alt_by_region:
            plot_names.append(plot_name_i)
            plot_axis.append(i)

            # Plot the altitude:
            # Retrieve orbit ephemeris for this time period
            k = spice.load_kernels(
                args.data_directory,
                start_date=start, end_date=end, n_days=n_days,
                download_if_not_available=dl_kernels,
                verbose=False,
                prompt_for_download=False)
            input("Kernels loaded, hit return to continue:")

            eph = anc.read_orbit_ephemeris(
                args.data_directory,
                start_date=start, end_date=end, n_days=n_days,
                download_if_not_available=dl_kernels)
            input("Loaded ephemeris, hit return to continue:")

            sc_time_utc = helper.dt_range(
                start, end_date=end, n_days=n_days,
                n_points_per_day=1000)

            sc_x, sc_y, sc_z = spice.MAVEN_position(sc_time_utc)
            sc_time_utc = np.array(sc_time_utc)
            sc_r = np.sqrt(sc_x**2 + sc_y**2 + sc_z**2)
            sc_alt = sc_r - 3390

            print(sc_alt)

            alt2 = mars_shape_conics.region_separation(
                sc_x, sc_y, sc_z, sc_alt)
            c = mars_shape_conics.region_colors

            continue


        if plot_name_i not in data:
            continue

        data_i = data[plot_name_i]

        if plot_name_i == "mvn_sta_c0_e":
            # For some reason the time axis for the MAVEN STATIC
            # energy spectra will reset to the start of the day
            # when a data gap starts, leading to peculiar overplotting.
            # To address this, best to identify where the reset occurs
            # and exclude all data after that point.

            t = data_i['time_unix']
            delta_t = np.ediff1d(t, to_end=t[-2] - t[-1])
            any_neg_time = np.where(delta_t < 0)[0]

            if len(any_neg_time) > 0:
                cutoff = any_neg_time[0]
                data_i['time_unix'] = (data_i['time_unix'])[:cutoff]
                data_i['y'] = (data_i['y'])[:, :cutoff]
                data_i['z'] = (data_i['z'])[:, :cutoff]

        if isinstance(data_i, list):

            for plot_name_j in data_i:
                if plot_name_j not in data:
                    continue
                plot_names.append(plot_name_j)
                plot_axis.append(i)

        else:
            plot_names.append(plot_name_i)
            plot_axis.append(i)

            # if "second_axis" in plot_info[plot_name_i]:
            #     secondary_labels[i] =\
            #         [plot_info[plot_name_i]["ylabel"].split("\n")[1],
            #          plot_info[plot_name_i]["second_axis"]["ylabel"]]
            #     # print(secondary_labels[i])
            #     # input()

    # secondary_labels = {"mvn_mag_bang_1sec": }

    # ['mvn_sep1f_ion_eflux', 'mvn_sep1r_ion_eflux', 'mvn_sep1f_elec_eflux',
    #  'mvn_sep1r_elec_eflux', 'mvn_sta_c0_e', 'mvn_sta_c6_m', 'mvn_swis_en_eflux',
    #  'mvn_swe_etspec', 'mvn_lpw_iv', 'mvn_mag_bamp', 'mvn_mag_bang_1sec', 'alt2', 'burst_flag']
    # input()

    burst_cmap = ListedColormap(['w', 'r'])

    height_ratios = [1 if i != "burst_flag" else 0.3 for i in tplot_names]
    fig, ax = plt.subplots(
        nrows=len(tplot_names), sharex=True,
        figsize=(10, 8), height_ratios=height_ratios)
    plt.subplots_adjust(
        left=0.135, bottom=0.05, right=0.88, top=0.95, wspace=0, hspace=0)

    if plot_alt_by_region:
        orb_ax = anc.add_orbit_axis(ax[0], ephemeris=eph, label=None)
        orb_ax.ticklabel_format(useOffset=False)

    for i, plot_name_i in zip(plot_axis, plot_names):

        if plot_name_i == "alt2" and plot_alt_by_region:

            # Get orbit info
            orbnum = eph["orbnum"]
            utc = eph["periapse_utc"]
            unx = helper.UTC_to_UNX(utc)
            # ax[i].plot(sc_time_utc, sc_alt, color='gray')
            for region in alt2:
                ax[i].plot(
                    sc_time_utc, alt2[region], label=region, color=c[region])

            # ax[i].plot(sc_time_utc, sheath_alt, color='lightgreen', label='sheath')
            # ax[i].plot(sc_time_utc, pileup_alt, color='orange', label='pileup')
            # ax[i].plot(sc_time_utc, sw_alt, color='k', label='SW')
            # ax[i].plot(sc_time_utc, shadow_alt, color='b', label='Shadow')
            ax[i].set_ylabel(
                "Alt., km", rotation='horizontal', va='center', labelpad=35)
            # ax[i].legend()

            continue

        # Pull info from struct for that plot_name
        data_i = data[plot_name_i]

        if plot_name_i == "mvn_mod_bcrust_amp":
            plot_i = {'ylabel': 'Morschhauser\n|B| [nT]', 'yscale': 'linear'}
        else:
            if plot_name_i not in plot_info:
                continue
            plot_i = plot_info[plot_name_i]

        # if isinstance(data_i, list):
        #     print(data_i)
        #     continue

        print(plot_name_i)
        # print(data_i.keys())
        print(plot_i)

        # Get time axis
        t = data_i["time_unix"]

        # Get y info:
        ylabel = plot_i["ylabel"]
        y = data_i["y"]

        # Get parameters of if theres more than 1 output:
        z_present = ("z" in data_i)
        print(z_present)
        # input()

        if z_present:
            print("Zlabel:")
            z_label = plot_i["zlabel"]
            z = data_i["z"]
            if "zlim" in plot_i:
                zlim = plot_i["zlim"]

        # NaN clean:
        non_nan = (~np.isnan(t))
        N_nan = np.argwhere(np.isnan(t)).size

        if N_nan > 0:
            if y.shape == t.shape:
                y = y[non_nan]
            elif t.size in y.shape:
                y = y[..., non_nan]
            if "zlabel" in plot_i and len(z.shape) > 1:
                z = z[:, non_nan]
            t = t[non_nan]
            print("NAN filtered: ", y.shape, z.shape, t.shape)

        # For a multidimensional y:
        if len(y.shape) > 1:
            # Get first nonnan
            non_nan_y = np.argwhere(~np.isnan(y[0, :]))[0]
            y_1d = y[:, non_nan_y]
            # print()
            # print(y_1d.T)
            check_same = y - (y_1d)
            print(np.nansum(check_same))
            if np.nansum(check_same) == 0:
                y = y_1d
            print("Flattened y: ", y.shape, z.shape, t.shape)

        # Get UTC time
        t_utc_i = helper.UNX_to_UTC(t)

        # Operate on the axis:
        ax_i = ax[i]
        if z_present:
            print(plot_i)

            vmin, vmax = None, None
            if "zrange" in plot_i:
                vmin, vmax = plot_i["zrange"]

            if plot_i["zscale"] == "log":
                norm = LogNorm(vmin, vmax)
            else:
                norm = Normalize(vmin, vmax)

            if plot_name_i == "burst_flag":
                cmap_i = burst_cmap
            else:
                cmap_i = 'viridis'

            p = ax_i.pcolormesh(t_utc_i, y, z, norm=norm, cmap=cmap_i)
            if plot_name_i in formatted_zlabels:
                z_label = formatted_zlabels[plot_name_i]
            cb = add_colorbar_outside(p, fig, ax_i, label=z_label)

            if plot_name_i == "burst_flag":
                cb.remove()
                ax_i.yaxis.set_visible(False)
                ax_i.set_yticks([], minor=True)

        else:

            # Check if second y axis present:
            if "second_axis" in plot_i:
                print("Second axis")
                print(plot_i["second_axis"])
                ax_ij = ax_i.twinx()

                # z_label = plot_i["second_axis"]["ylabel"]
                z = y[0, :]
                y = y[1, :]

                if plot_name_i == "mvn_mag_bang_1sec":
                    z = (z - 180)/2
                # print(plot_i)
                # # input()

                if "mvn_mag" in plot_name_i or "bcrust" in plot_name_i:
                    ax_ij.scatter(t_utc_i, z, color='r', marker='.', s=1)

                else:
                    ax_ij.plot(t_utc_i, z, color='r')
                if "yrange" in plot_i["second_axis"]:
                    ax_ij.set_ylim(plot_i["second_axis"]["yrange"])

                if "ylabel" in plot_i["second_axis"]:
                    ax_ij.set_ylabel(
                        plot_i["second_axis"]["ylabel"],
                        color='r', rotation='horizontal',
                        va='center', labelpad=25)

            color_i = None
            if plot_name_i in spec_colors:
                color_i = spec_colors[plot_name_i]

            print(color_i)

            # Make main plot otherwise
            if "mvn_mag" in plot_name_i or "bcrust" in plot_name_i:
                ax_i.scatter(t_utc_i, y, marker='.', s=1, color=color_i)
            else:
                ax_i.plot(t_utc_i, y, color=color_i)

            if "yrange" in plot_i:
                ax_i.set_ylim(plot_i["yrange"])

        # if plot_i["yscale"] == "log" and not second_axis_present:
        #     ax_i.set_yscale('log')
        if plot_i["yscale"] == "log" and "bang" not in plot_name_i:
            ax_i.set_yscale('log')

        if plot_name_i in formatted_ylabels:
            new_ylabel = formatted_ylabels[plot_name_i]
            # ax_i.set_ylabel(ylabel, rotation='horizontal',
            #                 va='center', labelpad=35)
            ax_i.set_ylabel(new_ylabel, rotation='horizontal',
                            va='center', labelpad=35)

    ax[-1].set_xlim(start, end)

    # Relabel the Bamp
    tplot_index = tplot_names.index("mvn_mag_bamp")
    subplot_names = data["mvn_mag_bamp"]

    b_ax = ax[tplot_index]

    b_ax.set_yscale('log')

    # bbox contains the
    # [x0 (left), y0 (bottom), x1 (right), y1 (top)] of the axis.
    bbox = b_ax.get_position()
    dy = bbox.height
    dy_i = dy/len(subplot_names)

    for j, sec_i in enumerate(subplot_names):
        color_i = spec_colors[sec_i]

        if sec_i in plot_info:

            b_label_i = plot_info[sec_i]["ylabel"].split("\n")[0]
            fig.text(bbox.x1 + 0.01, bbox.y0 + dy_i*(j + 0.5),
                     b_label_i, va='center', color=color_i)

    plt.show()
