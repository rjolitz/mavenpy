import os
import argparse
import datetime as dt
from dateutil.parser import parse
from collections.abc import Iterable

from scipy.io import readsav
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.patches import Wedge
import matplotlib.colors as mcolors
import matplotlib.dates as mdates


from mavenpy import spice, mars_shape_conics, coordinates, helper, anc, plot_tools

Rm = mars_shape_conics.Mars_radius


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_directory",
        help="Directory containing MAVEN data.",
        required=True
    )

    # Keywords for start/end time, or number of days, or the orbit #.
    parser.add_argument(
        "--start",
        help="Start of MAVEN plot timeframe (YYYY-MM-DD HH:MM:SS).",
        type=str
    )
    parser.add_argument(
        "--end",
        help="End of MAVEN plot timeframe (YYYY-MM-DD HH:MM:SS).",
        type=str,
    )
    parser.add_argument(
        "--n_days",
        help="Number of days of data retrieved.",
        type=int,
    )
    parser.add_argument(
        "--orbit_number",
        help="MAVEN orbit number.",
        type=float
    )

    # Keyword to download NAIF files:
    parser.add_argument(
        "--download",
        help="Download NAIF files to data_directory if not found.",
        action="store_true",
    )

    # Keyword controlling # of points plotted:
    parser.add_argument(
        "--n_points",
        help="Number of points plotted. Defaults to 500.",
        type=int,
        default=500,
    )

    # Optional plot keywords:
    parser.add_argument(
        "--plot_b", help="Add contours for Br.", action="store_true")
    parser.add_argument(
        "--plot_path_color",
        help="Color the path by progressing time.",
        action="store_true",
    )
    parser.add_argument(
        "--overlay_lonalt",
        help="Overlays longitude v altitude on lon/lat plot.",
        action="store_true",
    )

    parser.add_argument(
        "--path_cmap",
        help="Color map of the path color, defaults to viridis.",
        default='viridis'
    )

    args = parser.parse_args()

    n_periapse_points = 10
    default_xyzlim = [-3, 3]
    N = args.n_points
    axis_color = '0.75'
    # axis_color = '0.925'

    # Clean the date inputs:
    data_directory = args.data_directory
    # start_date, n_days, end_date = helper.sanitize_date_inputs(
    #     start_date=start, end_date=end)

    if args.orbit_number:
        eph = anc.read_orbit_ephemeris(
            data_directory,
            start_date='2014 12 1', end_date=dt.datetime.now(),
            download_if_not_available=args.download)
        onum = [args.orbit_number - 0.5, args.orbit_number + 0.5]
        start, end = anc.orbit_num(ephemeris=eph, orbit_num=onum)
    else:
        start, n_days, end = helper.sanitize_date_inputs(
            start_date=args.start, n_days=args.n_days, end_date=args.end)

    # print(start, end)

    k = spice.load_kernels(
        data_directory,
        start_date=start, end_date=end,
        download_if_not_available=args.download,
        verbose=None, spk_ext='bsp')
    # print(spice.currently_loaded_kernels())

    sc_time_utc = helper.dt_range(start, end_date=end, N=N)
    x_mso, y_mso, z_mso = spice.MAVEN_position(sc_time_utc, frame="MAVEN_MSO")
    x_geo, y_geo, z_geo = spice.MAVEN_position(sc_time_utc, frame="IAU_MARS")

    if args.plot_path_color:
        path_cmap = plt.get_cmap(args.path_cmap)
        color_i = path_cmap(np.linspace(0, 1, N))
        if args.overlay_lonalt:
            lonalt_color_i = cm.Reds(np.linspace(0, 1, N))
            lonlat_color_i = cm.Blues(np.linspace(0, 1, N))
        else:
            lonlat_color_i = color_i
    else:
        color_i = 'g'
        lonalt_color_i = 'r'
        lonlat_color_i = 'b'

    # Spacecraft alttitude, latitude, longitude:
    alt, lat, lon = coordinates.cartesian_to_geographic(
        x_geo, y_geo, z_geo)

    # Get the periapse indices:
    below_alt = np.where(alt < 250)[0]
    print('Min alt: {} km, max alt: {} km'.format(np.min(alt), np.max(alt)))
    # input()
    # print(below_alt)
    alt_edges = np.where(np.abs(np.ediff1d(below_alt)) > 1)[0]
    # print(alt_edges)
    alt_edges = np.append(alt_edges, len(below_alt) - 1)
    alt_edges = np.insert(alt_edges + 1, 0, 0).astype("int")
    periapse_index = []

    for a_l, a_h in zip(alt_edges[:-1], alt_edges[1:]):
        alt_lh = alt[below_alt[a_l:a_h]]
        # print(below_alt[a_l:a_h])
        # print(alt_lh)
        periapse_i = below_alt[a_l] + np.argmin(alt_lh)
        # print(periapse_i)
        periapse_index.append(periapse_i)
        # print(alt[below_alt[alt_edges]])

    print('Periapse index: ', periapse_index)
    print('# periapse: ', len(periapse_index))
    # input()

    # mso_spline = mars_shape_conics.cartesian_spline(
    #     sc_time_unx, sc_x_mso, sc_y_mso, sc_z_mso)

    # sc_mso_short = sc_mso[start_time_index:end_time_index]
    # rho_mso_scan = np.sqrt(sc_mso_scan[1]**2 + sc_mso_scan[2]**2)/Rm
    x_Rm = x_mso / Rm
    y_Rm = y_mso / Rm
    z_Rm = z_mso / Rm

    # Conics:
    x_bs, y_bs, z_bs = mars_shape_conics.bow_shock()
    x_mpb, y_mpb, z_mpb = mars_shape_conics.MPB()

    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(10, 6.5))
    path = ((x_Rm, y_Rm, z_Rm), (x_Rm, z_Rm, y_Rm), (y_Rm, z_Rm, -x_Rm))
    bs_line = ((x_bs, y_bs), (x_bs, z_bs), None)
    mpb_line = ((x_mpb, y_mpb), (x_mpb, z_mpb), None)
    plot_x_label = ("MSO X, Rm", "MSO X, Rm", "MSO Y, Rm")
    plot_y_label = ("MSO Y, Rm", "MSO Z, Rm", "MSO Z, Rm")

    for i, ax_i in enumerate(ax[0, :]):
        if i == 2:
            mars_circle = Wedge((0, 0), 1, 0, 360, fc="w", ec="salmon")
            ax_i.add_patch(mars_circle)
        else:
            mars_circle = Wedge((0, 0), 1, -90, 90, fc="w", ec="salmon")
            shadow_hatch = Wedge(
                (0, 0), 1, 90, 90 + 180, fc="w", hatch="//////", ec='salmon')
            ax_i.add_patch(mars_circle)
            ax_i.add_patch(shadow_hatch)

        # Bow shock:
        bs_i = bs_line[i]
        if bs_i is not None:
            plot_x_bs, plot_y_bs = bs_i[0], bs_i[1]
            ax_i.plot(plot_x_bs, plot_y_bs, linestyle="--", color="k")

        # MPB:
        mpb_i = mpb_line[i]
        if mpb_i is not None:
            plot_x_mpb, plot_y_mpb = mpb_i[0], mpb_i[1]
            ax_i.plot(plot_x_mpb, plot_y_mpb, linestyle=":", color="k")

        # path
        plot_x, plot_y, plot_z = path[i]
        index = ((np.sqrt(plot_x**2 + plot_y**2) < 1) & (plot_z > 0))
        # print(index)

        if args.plot_path_color:
            ax_i.scatter(plot_x[~index], plot_y[~index], c=color_i[~index],
                         marker='.', s=4)
            ax_i.scatter(plot_x[index], plot_y[index], c=color_i[index],
                         marker='.', s=0.1)
        else:
            ax_i.scatter(plot_x[~index], plot_y[~index], c=color_i,
                         marker='.', s=4)
            ax_i.scatter(plot_x[index], plot_y[index], c=color_i,
                         marker='.', s=0.1)

        ax_i.set_aspect("equal")
        ax_i.set_xlim(default_xyzlim)
        ax_i.set_ylim(default_xyzlim)
        ax_i.set_xlabel(plot_x_label[i])
        ax_i.set_ylabel(plot_y_label[i])
        ax_i.set_facecolor(axis_color)

    # Split the lower row into two plots:
    gs = ax[1, 0].get_gridspec()
    gs01 = gs[1, :].subgridspec(1, 2)
    for ax_i in ax[1, :]:
        ax_i.remove()
    # ax_l = fig.add_subplot(gs[1:, 1:])
    ax_l = fig.add_subplot(gs01[:, 0])
    ax_r = fig.add_subplot(gs01[:, 1])
    if args.overlay_lonalt:
        ax_r2 = ax_r.twinx()

    # The rho plot:
    rho_bs = np.sqrt(z_bs ** 2 + y_bs ** 2)
    rho_mpb = np.sqrt(z_mpb ** 2 + y_mpb ** 2)
    rho_Rm = np.sqrt(z_Rm**2 + y_Rm**2)

    mars_circle = Wedge((0, 0), 1, -90, 90, fc="w", ec="salmon")
    shadow_hatch = Wedge(
        (0, 0), 1, 90, 90 + 180, fc="w", hatch="////", ec='salmon')
    ax_l.add_patch(mars_circle)
    ax_l.add_patch(shadow_hatch)
    ax_l.plot(x_bs, rho_bs, linestyle="--", color="k")
    ax_l.plot(x_mpb, rho_mpb, linestyle=":", color="k")
    ax_l.scatter(x_Rm, rho_Rm, c=color_i, marker='.', s=2)
    ax_l.set_aspect('equal')
    ax_l.set_xlim(default_xyzlim)
    ax_l.set_ylim([0, default_xyzlim[1]])
    ax_l.set_xlabel("MSO X, Rm")
    ax_l.set_ylabel("ρ [√($Y^2$ + $Z^2$)], Rm")
    ax_l.set_facecolor(axis_color)


    # Lon / Lat

    # Longitude wrapping from 0 to 360 appears as horizontal bars.
    # We can eliminate those by plotting segments where wrapping occurs.
    wrapped_indices = np.where(np.abs(np.ediff1d(lon)) > 180)[0]
    wrapped_indices = np.append(wrapped_indices, len(lon) - 1)

    if len(periapse_index) < n_periapse_points:
        if isinstance(lonlat_color_i, str):
            periapse_color = lonlat_color_i
        else:
            periapse_color = lonlat_color_i[periapse_index]

        ax_r.scatter(
            lon[periapse_index], lat[periapse_index],
            color=periapse_color, marker='x', zorder=10)
        if args.overlay_lonalt:
            ax_r2.scatter(
                lon[periapse_index], alt[periapse_index],
                color=periapse_color, marker='x', zorder=10)

    # plt.show()

    # input()
    if len(wrapped_indices) == 0:
        ax_r.scatter(lon, lat, color=lonlat_color_i, s=3, marker='.', zorder=2)
        if args.overlay_lonalt:
            ax_r2.scatter(
                lon, alt, color=lonalt_color_i, s=3, marker='.', zorder=2)
    else:
        init_index = 0
        for wrap_index in wrapped_indices:
            lon_i = lon[init_index:(wrap_index + 1)]
            lat_i = lat[init_index:(wrap_index + 1)]
            alt_i = alt[init_index:(wrap_index + 1)]

            if args.plot_path_color:
                lonlat_color_j = lonlat_color_i[init_index:(wrap_index + 1)]
                ax_r.scatter(
                    lon_i, lat_i, color=lonlat_color_j,
                    s=3, marker='.', zorder=2)
                if args.overlay_lonalt:
                    lonalt_color_j = lonalt_color_i[init_index:(wrap_index + 1)]
                    ax_r2.scatter(
                        lon_i, alt_i, color=lonalt_color_j,
                        s=3, marker='.', zorder=2)

            else:
                ax_r.scatter(
                    lon_i, lat_i, c=lonlat_color_i, s=3, marker='.', zorder=2)
                if args.overlay_lonalt:
                    ax_r2.scatter(
                        lon_i, alt_i, c=lonalt_color_i,
                        s=3, marker='.', zorder=2)
            init_index = wrap_index + 1

    ax_r.set_aspect('equal')
    ax_r.set_xlim([0, 360])
    ax_r.set_yticks([-90 + 30 * i for i in range(7)])
    ax_r.set_ylim([-90, 90])
    ax_r.set_xticks([60 * i for i in range(7)])
    ax_r.set_xlabel("East Longitude, deg.")
    if args.overlay_lonalt:
        ax_r2.set_ylabel("Altitude, km", color='r')
        ax_r.set_ylabel("Latitude, deg.", color='b')
    else:
        ax_r.set_ylabel("Latitude, deg.")
    ax_r.set_facecolor(axis_color)

    fig.tight_layout()

    if args.plot_b:
        plot_b_file = "/Users/rjolitz/DataFiles/from_others/Morschhauser_spc_dlat0.25_delon0.25_dalt5.sav"
        b_struct = readsav(plot_b_file)['morschhauser']

        r = b_struct['radius'][0]
        lon = b_struct['longitude'][0]
        lat = b_struct['latitude'][0]
        b = b_struct['b'][0]

        altitude_index = helper.find_closest_index(r, 400 + 3390)
        b_subset = b[:, :, altitude_index, :]
        bx = b_subset[:, :, 0]
        by = b_subset[:, :, 1]
        bz = b_subset[:, :, 2]
        # print(bx.shape, by.shape, bz.shape)

        theta = (np.radians(90.0 - lat))[:, np.newaxis]
        phi = (np.radians(lon))[np.newaxis, :]

        br, bt, bp = coordinates.cartesian_to_spherical_vector(
            bx, by, bz, theta, phi)

        ax_r.contourf(lon, lat, br, vmin=-40, vmax=40, cmap=cm.RdBu, zorder=1)
        # plt.colorbar(label='Br, nT')


    if args.plot_path_color:

        # Get bounding box that marks the boundaries of the axis:
        # [x0 (left), y0 (bottom), x1 (right), y1 (top)] of the axis.
        ax_l_pos = ax_l.get_position()
        ax_r_pos = ax_r.get_position()
        new_ax_l_pos =\
            [ax_l_pos.x0, ax_r_pos.y0, ax_l_pos.width, ax_l_pos.height]
        ax_l.set_position(new_ax_l_pos)

        norm_t = mcolors.Normalize(
            vmin=mdates.date2num(sc_time_utc[0]),
            vmax=mdates.date2num(sc_time_utc[-1]))
        im = cm.ScalarMappable(norm=norm_t, cmap='viridis')

        # [left most position, bottom position, width, height] of color bar.
        # cax = fig.add_axes(
        #     [bbox.x1 - 0.02, bbox.y0, 0.05, bbox.height])
        # cax = fig.add_axes(
        #     [bbox.x0, bbox.y0 - 0.2, bbox.width, 0.01])
        if args.overlay_lonalt:
            cax = fig.add_axes(
                [ax_l_pos.x0, ax_r_pos.y1 - 0.03, ax_l_pos.width, 0.03])
        else:
            cax = fig.add_axes(
                [ax_l_pos.x0, ax_l_pos.y1 + 0.04, ax_l_pos.width, 0.02])

        rho_cbar = fig.colorbar(im, cax=cax, orientation='horizontal')

        rho_loc = mdates.AutoDateLocator()
        rho_fmt = mdates.ConciseDateFormatter(rho_loc)
        rho_cbar.ax.xaxis.set_major_locator(rho_loc)
        rho_cbar.ax.xaxis.set_major_formatter(rho_fmt)
        # rho_cbar.ax.xaxis.set_offset_position('bottom')
        # rho_cbar.ax.yaxis.set_major_formatter(
        #     mdates.AutoDateFormatter(rho_loc))

    plt.show()
