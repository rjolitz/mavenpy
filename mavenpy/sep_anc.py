import time
from collections.abc import Iterable

import numpy as np
import spiceypy

from . import coordinates
from . import spice
from . import mars_shape_conics
from . import helper


# Adaptation of mvn_sep_anc_fov_mars_fraction.pro,
# which is used to calculate the SEP-??_FRAC_FOV_MARS
# variable saved in the ancillary SEP data files.


SEP_fov_NAIF_ID =\
    {"SEP1A_front": -202126, "SEP1B_front": -202127,
     "SEP1A_back": -202128, "SEP1B_back": -202129,
     "SEP2A_front": -202121, "SEP2B_front": -202122,
     "SEP2A_back": -202123, "SEP2B_back": -202124}

# No difference between 1A_front and 1B_front

SEP_fovs = {"1F": "SEP1A_front", "1R": "SEP1A_back",
            "2F": "SEP2A_front", "2R": "SEP2A_back"}


def sep_mso_look_dir(time_UTC, sensor_number, look_direction):
    """Rotate the look direction of a given MAVEN SEP sensor
    into MSO coordinates."""

    # # get ephemeris time:
    # ephemeris_time = spice.dt_to_et(time_UTC)
    # N = len(ephemeris_time)

    look_direction = look_direction.lower()[:1]

    if look_direction == "f":
        fov_x = 1
    elif look_direction == "r":
        fov_x = -1
    else:
        raise ValueError(
            "Look direction '{}' not recognized, "
            "use 'r' or 'f' instead.".format(look_direction))

    # Use wrapper from spice module (wraps spiceypy.pxform)
    sep_mso = spice.pxform(
        time_UTC, fov_x, 0, 0,
        "MAVEN_SEP{}".format(sensor_number),
        "MAVEN_MSO")

    return sep_mso


def get_fov_pixels(sep_sensor, sep_look_direction, pixel_width_deg=1.5):

    '''Returns a list of (x, y, z) coordinates that describe the boresight
    for a given SEP sensor-look direction.
    '''

    sep_s_ld = "{}{}".format(sep_sensor, sep_look_direction.upper())

    naif_code = SEP_fov_NAIF_ID[SEP_fovs[sep_s_ld]]

    # Ask Spice getfov for the corner vectors of the FOV
    # Returns: shape (e.g. rectangle), frame, boresight, # edges
    #     bounds (array of 4 3-D vectors for rectangle)
    shape, frame_i, boresight, n_edges, corners = spiceypy.getfov(
        naif_code, 4)

    # Make a range of pixels around the boresight stretching to the corners
    # r is 1
    r, theta_bounds, phi_bounds = coordinates.cartesian_to_spherical(
        corners[:, 0], corners[:, 1], corners[:, 2])
    # print(np.degrees(theta_bounds))
    # print(np.degrees(phi_bounds))

    # Get pixel width in radians:
    pixel_width_rad = np.radians(pixel_width_deg)

    # Discretize the cone angle into pixels with width of ~1.5 deg apart:
    # # Method from SEP IDL routine, not needed but kept for documentation:
    # theta_range = max(theta_bounds) - min(theta_bounds)
    # ntheta = np.ceil(theta_range/pixel_width_rad)
    # dtheta = theta_range/ntheta
    # theta_edges_array = theta_bounds[0] + dtheta*np.arange(ntheta + 1)
    # theta_centers_array = theta_edges_array[:-1] +\
    #     np.ediff1d(theta_edges_array)/2
    theta_centers = np.arange(
        min(theta_bounds), max(theta_bounds), pixel_width_rad)
    # print(np.degrees(theta_centers))

    # Do the same for the phi angle:
    if phi_bounds[3] > phi_bounds[0]:
        phi_centers = np.arange(
            min(phi_bounds), max(phi_bounds), pixel_width_rad)
    else:
        phi_centers = np.arange(
            max(phi_bounds), min(phi_bounds) + 2*np.pi, pixel_width_rad) % 360
    # print(np.degrees(phi_centers))

    theta_centers_2d, phi_centers_2d = np.meshgrid(theta_centers, phi_centers)

    # fov_x, fov_y, fov_z = spherical_to_cartesian(
    #     1, theta_centers_array[np.newaxis, :],
    #     phi_centers_array[:, np.newaxis])
    fov_x, fov_y, fov_z = coordinates.spherical_to_cartesian(
        1, theta_centers_2d, phi_centers_2d)

    return fov_x, fov_y, fov_z, theta_centers_2d, phi_centers_2d


def fov_target_angle(sensor, look_direction, time_utc, target,
                     time_unix=None,
                     MAVEN_position_spline=None, n_spline_points=300,
                     calculate_position=False,
                     verbose=False):
    """
    target: solar system object(s) to take angle wrt, e.g.
        'sun' or 'Mars'
    calculate_position: Boolean to calculate the MAVEN
        position at each point instead of using spline
        (slower, but more precise)
    """

    # Turn target into iterable, so can return angles for each:
    if isinstance(target, str):
        target = (target,)

    # Convert UTC time to Unix, if not supplied:
    if time_unix is None:
        time_unix = helper.UTC_to_UNX(time_utc)

    # Get the MSO coordinates of the center of the boresight:
    if verbose:
        print("Get FOV coordinates in MSO...")
        time_init = time.time()
    fov_mso_unit_vec = sep_mso_look_dir(time_utc, sensor, 'f')
    if verbose:
        print("Done: {} s".format(time.time() - time_init))

    fov_angle_dict = {}
    for target_i in target:
        target_i_lc = target_i.lower()
        if target_i_lc == 'mars':
            # Get the position vector in MSO, which is needed
            # for Mars-FOV angle:
            if calculate_position:
                # Calculate spacecraft position using spice for
                # each increment in time:
                if verbose:
                    print("No position supplied, calculating MAVEN "
                          "position in MSO "
                          "(will be slow, use spline if speed "
                          "preferred)...")
                    time_init = time.time()
                sc_x, sc_y, sc_z = spice.MAVEN_position(time_utc)
                if verbose:
                    print("Done: {} s".format(time.time() - time_init))
            else:
                # Use the spline to eval the spacecraft position:
                if not MAVEN_position_spline:
                    if verbose:
                        print(
                            "No position spline supplied, calculating"
                            " spline with {} points...".format(
                                n_spline_points))
                        time_init = time.time()
                    # If no spline provided and not calculating position,
                    # first get the position (N=300 points per day
                    # gets sub-km precision)
                    time_utc_m, time_unx_m, xm, ym, zm = spice.load_MAVEN_position(
                        time_utc[0], end_date=time_utc[-1],
                        n_sample_points=n_spline_points, frame="MAVEN_MSO")
                    MAVEN_position_spline = mars_shape_conics.cartesian_spline(
                        time_unx_m, xm, ym, zm)
                    if verbose:
                        print("Done: {} s".format(time.time() - time_init))

                # Use the spline to eval the spacecraft position
                # at the times the FOV target angle is eval'ed:
                sc_x = MAVEN_position_spline[0](time_unix)
                sc_y = MAVEN_position_spline[1](time_unix)
                sc_z = MAVEN_position_spline[2](time_unix)

            # Get r and calculate unit vector.
            sc_r = np.sqrt(sc_x**2 + sc_y**2 + sc_z**2)
            mars_unit = np.zeros(shape=(sc_x.size, 3))
            mars_unit[:, 0] = -sc_x/sc_r
            mars_unit[:, 1] = -sc_y/sc_r
            mars_unit[:, 2] = -sc_z/sc_r

            fov_dot_unit_vector = np.sum(fov_mso_unit_vec * mars_unit, axis=1)

        elif target_i_lc == 'sun':
            # MSO x axis is sun-pointing, so unit_vector_sun dot
            # fov_mso_unit_vector is just the x coordinate of the
            # fov_mso_unit_vector
            fov_dot_unit_vector = fov_mso_unit_vec[:, 0]

        fov_angle_i = np.degrees(np.arccos(fov_dot_unit_vector))
        fov_angle_dict[target_i_lc] = fov_angle_i

    return fov_angle_dict


def fraction_Mars_in_FOV(sensor, look_direction, time_utc,
                         pixel_width_deg=1.5,
                         use_pointing_info=False,
                         MAVEN_position_spline=None,
                         n_spline_points=300,
                         fov_angle=None,
                         fov_angle_threshold=105):
    '''Returns the fraction of the field of view
    of a given SEP sensor is taken up by (sunlit) Mars

    time_utc: str or datetime or list of datetimes
    '''

    # Convert to ephemeris time:
    ephemeris_time = spice.dt_to_et(time_utc)
    unix_time = helper.UTC_to_UNX(time_utc)

    # Spice frame that all SEP boresight vectors are in:
    frame = "MAVEN_SEP{}".format(sensor)

    # Get the pixel ranges that will be iterated across
    # to see if Mars intersects:
    fov_x, fov_y, fov_z, fov_theta_deg, fov_phi_deg = get_fov_pixels(
        sensor, look_direction, pixel_width_deg=pixel_width_deg)

    # Flatten the arrays for iterating:
    fov_x = fov_x.flatten()
    fov_y = fov_y.flatten()
    fov_z = fov_z.flatten()

    if not isinstance(ephemeris_time, Iterable):
        ephemeris_time = [ephemeris_time]

    n_time = len(ephemeris_time)
    n_fov_vec = len(fov_x)

    mars_in_fov = np.zeros(shape=(n_time, n_fov_vec))
    sunlit_mars_in_fov = np.zeros(shape=(n_time, n_fov_vec))

    # generate an index array marking where we eval FOV
    eval_ephemeris_index = np.where(unix_time == unix_time)[0]
    # print(eval_ephemeris_index)
    # input()
    if use_pointing_info:
        # Calculate the FOV-Mars angle if not supplied:
        if fov_angle is None:
            fov_angle_dict = fov_target_angle(
                sensor, look_direction, time_utc, ('mars',),
                MAVEN_position_spline=MAVEN_position_spline,
                n_spline_points=n_spline_points,
                calculate_position=False)
            fov_angle = fov_angle_dict['mars']
        potential_obscure = np.where(fov_angle < fov_angle_threshold)[0]
        eval_ephemeris_index = eval_ephemeris_index[potential_obscure]

    # Disables the error when no intercept:
    with spiceypy.no_found_check():
        # for j, et_j in enumerate(ephemeris_time):
        for j in eval_ephemeris_index:
            et_j = ephemeris_time[j]
            for i, fov_xyz_i in enumerate(zip(fov_x, fov_y, fov_z)):
                # print(i, fov_xyz_i)
                # Just pxform takes 6 seconds/100 time segments:
                # sincpt (Surface intercept):
                # Takes 6 seconds/100 time segments
                try:
                    spoint, trgepc, srfvec, intercept = spiceypy.sincpt(
                        'Ellipsoid', 'Mars', et_j,
                        'IAU_MARS', 'NONE', 'MAVEN', frame,
                        fov_xyz_i)
                except spiceypy.utils.exceptions.SpiceNOFRAMECONNECT:
                    print("Missing Spice info for {}, skipping.".format(time_utc[j]))
                    break

                mars_in_fov[j, i] = intercept

                if intercept:
                    # Adds 0.5 sec/100 time segments

                    trgepc, srfvec, phase_angle,\
                        solar_zenith_angle, emission_angle = spiceypy.ilumin(
                            'Ellipsoid', 'MARS', et_j, 'IAU_MARS',
                            'NONE', 'MAVEN', spoint)

                    sunlit_mars_in_fov[j, i] =\
                        (solar_zenith_angle < np.pi/2) *\
                        np.cos(solar_zenith_angle)

    weight = np.sin(fov_theta_deg.flatten())[np.newaxis, :]

    frac_fov = np.sum(mars_in_fov*weight, axis=1)/np.sum(weight)
    frac_illum_fov = np.sum(sunlit_mars_in_fov*weight, axis=1)/np.sum(weight)

    return frac_fov, frac_illum_fov
