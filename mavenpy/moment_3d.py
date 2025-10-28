from collections.abc import Iterable

import numpy as np

from . import helper, units, constants


from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm


# Adaptation of SPEDAS code for calculating
# plasma moments for a 3D ESA.
# Built and tested for SWIA moments, may or may not
# work for others.


def jacobian(phi_deg=None, dphi_deg=None, theta_deg=None, dtheta_deg=None,
             shape=None, v_moment=0, theta_ax=None):

    '''Returns the Jacobian that represents the solid angle(s) subtended by each
    detector element.

    theta_deg: angle range for integration, assumes instrument
        elevation angle coordinates where -90 < theta < 90
    phi_deg: angle range for integration, assumes instrument
        azimuthal angle coordinates where 0 < theta < 360
    shape: shape of the energy flux / distrib function to
        multiply by
    v_moment: n where n is the nth velocity moment.
        0: density
        1: number flux
        2: '''

    # Set to 4pi if no angles available:

    if phi_deg is None and theta_deg is None:
        return 4 * np.pi

    # Add indices to multiply with larger arrays, if needed:
    phi_index, phi_broadcast = helper.broadcast_index(
        shape, phi_deg.shape, matching_axis_index=slice(None),
        other_axis_index=np.newaxis, raise_if_over_1d=False)
    theta_index, theta_broadcast = helper.broadcast_index(
        shape, theta_deg.shape, matching_axis_index=slice(None),
        other_axis_index=np.newaxis, raise_if_over_1d=False)

    # If no dphi provided and phi is an Iterable, get dphi
    # as the spacing between phi bins
    if dphi_deg is None and isinstance(phi_deg, Iterable):
        dphi_deg = np.ediff1d(
            phi_deg, to_end=(phi_deg[-1] - phi_deg[-2]))
        # Set to constant if all same:
        if all(np.isclose(dphi_deg[0], dphi_deg)):
            dphi_deg = dphi_deg[0]

    # If no dtheta provided and theta is an Iterable,
    # get dtheta as the spacing between theta bins:
    if dtheta_deg is None and isinstance(theta_deg, Iterable):
        # Assuming theta primarily varies over
        # the 0th axis, deals with energy-dependet
        # theta
        theta_shape = theta_deg.shape
        # print(theta_shape)
        if not theta_ax:
            theta_ax = 0

        dtheta_deg =\
            (np.roll(theta_deg, -1, axis=theta_ax) -
             np.roll(theta_deg, 1, axis=theta_ax))/2
        # input()

        index_0 = tuple(
            [slice(None) if i != theta_ax else 0
             for i, _ in enumerate(theta_shape)])
        index_1 = tuple(
            [slice(None) if i != theta_ax else 1
             for i, _ in enumerate(theta_shape)])
        index_n1 = tuple(
            [slice(None) if i != theta_ax else -2
             for i, _ in enumerate(theta_shape)])
        index_n = tuple(
            [slice(None) if i != theta_ax else -1
             for i, _ in enumerate(theta_shape)])

        # print(index_0, index_1, index_n1, index_n)

        dtheta_0 = theta_deg[index_1] - theta_deg[index_0]

        dtheta_deg[index_0] = dtheta_0

        dtheta_n = theta_deg[index_n] - theta_deg[index_n1]

        dtheta_deg[index_n] = dtheta_n
        # print(dtheta_deg)
        # input()
        # print(theta_deg)
        # print(dtheta_deg)

    # Reindex the theta, phi, dtheta, and dphi arrays:
    # print(dtheta_deg.shape, theta_deg.shape)
    if isinstance(dphi_deg, Iterable):
        dphi_deg = dphi_deg[phi_broadcast]
    if isinstance(phi_deg, Iterable):
        phi_deg = phi_deg[phi_broadcast]
    if isinstance(dtheta_deg, Iterable) and len(dtheta_deg.shape) != len(shape):
        dtheta_deg = dtheta_deg[theta_broadcast]
    if isinstance(theta_deg, Iterable) and len(theta_deg.shape) != len(shape):
        theta_deg = theta_deg[theta_broadcast]

    # print(dtheta_deg.shape, theta_deg.shape)
    # input()

    # Get the angles of the bin centers in radians:
    phi_rad = np.radians(phi_deg)
    dphi_rad = np.radians(dphi_deg)
    theta_rad = np.radians(theta_deg)
    dtheta_rad = np.radians(dtheta_deg)

    # domega has dims of N_phi x N_theta x N_energy
    # domega=2.*(dphi/!radeg)*cos(theta/!radeg)*sin(.5*dtheta/!radeg)
    domega = np.abs(
        2*dphi_rad * np.cos(theta_rad) * np.sin(0.5*dtheta_rad))

    # Now get the Jacobian element(s) for each moment:
    if v_moment == 0:
        # Density
        # Expression from n_3d_new.pro:
        domega = 2.*np.cos(theta_rad)*np.sin(dtheta_rad/2.)*dphi_rad

    elif v_moment == 1:
        # Number flux
        # Expression from j_3d_new.pro:
        dtheta_term =\
            (dtheta_rad/2. + np.cos(2*theta_rad) * np.sin(dtheta_rad)/2.) *\
            (2.*np.sin(dphi_rad/2.))

        x_domega = dtheta_term*np.cos(phi_rad)
        y_domega = dtheta_term*np.sin(phi_rad)
        z_domega = 2. * np.sin(theta_rad) * np.cos(theta_rad) *\
            np.sin(dtheta_rad/2.)*np.cos(dtheta_rad/2.)*dphi_rad

        # # Expression from j_3d.pro (not much different):
        # domega = 2.*np.cos(theta_rad)*np.sin(dtheta_rad/2.)*dphi_rad
        # x_domega = np.cos(phi_rad)*domega*np.cos(theta_rad)
        # y_domega = np.sin(phi_rad)*domega*np.cos(theta_rad)
        # z_domega = domega*np.sin(theta_rad)

        # domega = np.abs(x_domega), np.abs(y_domega), np.abs(z_domega)
        domega = x_domega, y_domega, z_domega

    elif v_moment == 2:
        # Momentum tensor:
        # Expression from m_3d_new.pro:
        th1 = theta_rad - dtheta_rad/2.
        th2 = theta_rad + dtheta_rad/2.
        ph1 = phi_rad - dphi_rad/2.
        ph2 = phi_rad + dphi_rad/2.
        cth1 = np.cos(th1)
        cth2 = np.cos(th2)
        sth1 = np.sin(th1)
        sth2 = np.sin(th2)
        cph1 = np.cos(ph1)
        cph2 = np.cos(ph2)
        sph1 = np.sin(ph1)
        sph2 = np.sin(ph2)
        s_2ph1 = np.sin(2.*ph1)
        s_2ph2 = np.sin(2.*ph2)
        s2_ph1 = sph1**2
        s2_ph2 = sph2**2
        s3_th1 = sth1**3
        s3_th2 = sth2**3
        c3_th1 = cth1**3
        c3_th2 = cth2**3

        d_omega_xx =\
            ((ph2-ph1)/2.+(s_2ph2-s_2ph1)/4.)*(sth2-sth1-(s3_th2-s3_th1)/3.)
        d_omega_yy =\
            ((ph2-ph1)/2.-(s_2ph2-s_2ph1)/4.)*(sth2-sth1-(s3_th2-s3_th1)/3.)
        d_omega_zz = dphi_rad*(s3_th2-s3_th1)/3.
        d_omega_xy = ((s2_ph2-s2_ph1)/2.)*(sth2-sth1-(s3_th2-s3_th1)/3.)
        d_omega_xz = (sph2-sph1)*((c3_th1-c3_th2)/3.)
        d_omega_yz = (cph1-cph2)*((c3_th1-c3_th2)/3.)

        domega = d_omega_xx, d_omega_yy, d_omega_zz,\
            d_omega_xy, d_omega_xz, d_omega_yz

    return domega


def n(energy_eV, d_energy_eV, diff_en_flux_eVcm2eVsters, species,
      phi_deg=None, dphi_deg=None, theta_deg=None, dtheta_deg=None,
      theta_ax=None):
    '''Adaptation of n_3d_new.pro, although this assumes an energy
    flux rather than f(v) (conversion handled). Verified by comparison
    with n_3d_new.pro result.'''

    # Get the mass:
    mass_eVkm2s2 = constants.mass[species]

    # Converts from eflux (eV/(cm2secstereV)) to distribution function
    # (f(v) = eflux * (m^2/2E^2)) and integrates over velocity
    # space (d3v = v2 dv d_SA = 2E/m dv d_SA =
    #            = (m^2 F(E)/2E)  (2E/m)^0.5 dE/m )
    N_constant = 1e-5 * np.sqrt(mass_eVkm2s2/2) *\
        d_energy_eV * energy_eV ** -1.5

    # Add the time axis to the front:
    N_constant = N_constant[np.newaxis, ...]
    # print(diff_en_flux_eVcm2eVsters.shape)

    # Get the Jacobian:
    eflux_shape = diff_en_flux_eVcm2eVsters.shape
    domega = jacobian(
        phi_deg=phi_deg, dphi_deg=dphi_deg, theta_deg=theta_deg,
        dtheta_deg=dtheta_deg, shape=eflux_shape, v_moment=0,
        theta_ax=theta_ax)

    # Sum over all but the time axis:
    sum_axis = tuple([i for i, j in enumerate(eflux_shape)])[1:]
    N = np.sum(
        N_constant * diff_en_flux_eVcm2eVsters * domega,
        axis=sum_axis)

    return N


def nflux(energy_eV, d_energy_eV, diff_en_flux_eVcm2eVsters, species,
          phi_deg=None, dphi_deg=None, theta_deg=None, dtheta_deg=None,
          theta_ax=None):
    '''Adaptation of j_3d_new.pro. Minor numerical discrepancies
    in the Jacobian compared to j_3d_new.'''

    V_constant = d_energy_eV / energy_eV
    V_constant = V_constant[np.newaxis, ...]

    # Get the Jacobian:
    eflux_shape = diff_en_flux_eVcm2eVsters.shape
    domega_x, domega_y, domega_z = jacobian(
        phi_deg=phi_deg, dphi_deg=dphi_deg, theta_deg=theta_deg,
        dtheta_deg=dtheta_deg, shape=eflux_shape, v_moment=1,
        theta_ax=theta_ax)

    # #/cm2/s
    # Sum over all but the time axis:
    sum_axis = tuple([i for i, j in enumerate(eflux_shape)])[1:]
    flux_x = np.sum(
        V_constant * domega_x * diff_en_flux_eVcm2eVsters, axis=sum_axis)
    flux_y = np.sum(
        V_constant * domega_y * diff_en_flux_eVcm2eVsters, axis=sum_axis)
    flux_z = np.sum(
        V_constant * domega_z * diff_en_flux_eVcm2eVsters, axis=sum_axis)

    # flux_3d = np.stack((flux_x, flux_y, flux_z))

    return flux_x, flux_y, flux_z


def v(energy_eV, d_energy_eV, diff_en_flux_eVcm2eVsters, species,
      phi_deg=None, dphi_deg=None, theta_deg=None, dtheta_deg=None,
      theta_ax=None, N=None):
    '''Adaptation of v_3d_new.pro'''

    if N is None:
        N = n(
            energy_eV, d_energy_eV, diff_en_flux_eVcm2eVsters, species,
            phi_deg=phi_deg, dphi_deg=dphi_deg, theta_deg=theta_deg,
            dtheta_deg=dtheta_deg, theta_ax=theta_ax)

    flux_3d = nflux(
        energy_eV, d_energy_eV, diff_en_flux_eVcm2eVsters, species,
        phi_deg=phi_deg, dphi_deg=dphi_deg, theta_deg=theta_deg,
        dtheta_deg=dtheta_deg, theta_ax=theta_ax)

    # cm/s --> km/s
    flux_3d = np.stack(flux_3d)
    velocity_3d = flux_3d / N[np.newaxis, :] * 1e-5

    return velocity_3d


def M(energy_eV, d_energy_eV, diff_en_flux_eVcm2eVsters, species,
      phi_deg=None, dphi_deg=None, theta_deg=None,
      dtheta_deg=None, theta_ax=None, diagonal_only=False):

    '''Momentum tensor, adapted from m_3d_new.pro'''

    # Get the mass:
    mass_eVkm2s2 = constants.mass[species]

    # For f(v): (2E/m)^1.5 dE
    # for EF(E): (2E/m)^1.5 (m^2/2E^2) dE = (2m/E)^ 0.5
    C_E = np.sqrt(2 * mass_eVkm2s2 / energy_eV) * d_energy_eV * 1e-5
    # print(C_E)
    C_E = C_E[np.newaxis, ...]

    # Sum over all but the time axis:
    eflux_shape = diff_en_flux_eVcm2eVsters.shape
    sum_axis = tuple([i for i, j in enumerate(eflux_shape)])[1:]

    # df versus eflux
    domega = jacobian(
        phi_deg=phi_deg, dphi_deg=dphi_deg, theta_deg=theta_deg,
        dtheta_deg=dtheta_deg, shape=eflux_shape, v_moment=2,
        theta_ax=theta_ax)

    d_omega_xx, d_omega_yy, d_omega_zz, d_omega_xy, d_omega_xz, d_omega_yz = domega

    # Get the elements of the tensor, beginning with diagonal:
    mom_xx = np.sum(C_E*diff_en_flux_eVcm2eVsters*d_omega_xx, axis=sum_axis)
    mom_yy = np.sum(C_E*diff_en_flux_eVcm2eVsters*d_omega_yy, axis=sum_axis)
    mom_zz = np.sum(C_E*diff_en_flux_eVcm2eVsters*d_omega_zz, axis=sum_axis)

    if diagonal_only:
        return mom_xx, mom_yy, mom_zz

    mom_xy = np.sum(C_E*diff_en_flux_eVcm2eVsters*d_omega_xy, axis=sum_axis)
    mom_xz = np.sum(C_E*diff_en_flux_eVcm2eVsters*d_omega_xz, axis=sum_axis)
    mom_yz = np.sum(C_E*diff_en_flux_eVcm2eVsters*d_omega_yz, axis=sum_axis)

    return mom_xx, mom_yy, mom_zz, mom_xy, mom_xz, mom_yz


def p(energy_eV, d_energy_eV, diff_en_flux_eVcm2eVsters, species,
      phi_deg=None, dphi_deg=None, theta_deg=None, dtheta_deg=None,
      theta_ax=None, N=None, diagonal_only=False,
      diagonalize_axis=None):
    '''Pressure tensor, adaptation of p_3d_new.pro'''

    # Get the mass:
    mass_eVkm2s2 = constants.mass[species]

    # Get density if not supplied:
    if N is None:
        N = n(
            energy_eV, d_energy_eV, diff_en_flux_eVcm2eVsters, species,
            phi_deg=phi_deg, dphi_deg=dphi_deg, theta_deg=theta_deg,
            dtheta_deg=dtheta_deg, theta_ax=theta_ax)

    momentum = M(
        energy_eV, d_energy_eV, diff_en_flux_eVcm2eVsters, species,
        phi_deg=phi_deg, dphi_deg=dphi_deg, theta_deg=theta_deg,
        dtheta_deg=dtheta_deg, theta_ax=theta_ax, diagonal_only=False)

    flux = nflux(
        energy_eV, d_energy_eV, diff_en_flux_eVcm2eVsters, species,
        phi_deg=phi_deg, dphi_deg=dphi_deg, theta_deg=theta_deg,
        dtheta_deg=dtheta_deg, theta_ax=theta_ax)

    # print(momentum[0][0], momentum[1][0], momentum[2][0])
    # print(flux[0][0], flux[1][0], flux[2][0])
    # print(N[0])
    # # input()

    pxx = (momentum[0]-mass_eVkm2s2*flux[0]*flux[0]/N/1.e10)
    pyy = (momentum[1]-mass_eVkm2s2*flux[1]*flux[1]/N/1.e10)
    pzz = (momentum[2]-mass_eVkm2s2*flux[2]*flux[2]/N/1.e10)

    if diagonal_only and diagonalize_axis is None:
        return pxx, pyy, pzz

    # off diagonal elements:
    pxy = (momentum[3]-mass_eVkm2s2*flux[0]*flux[1]/N/1.e10)
    pxz = (momentum[4]-mass_eVkm2s2*flux[0]*flux[2]/N/1.e10)
    pyz = (momentum[5]-mass_eVkm2s2*flux[1]*flux[2]/N/1.e10)

    if diagonalize_axis is not None:
        # This whole zone is a damn mess:
        diagonalize_axis = (1, 0, 0)
        sx, sy, sz = diagonalize_axis

        p = np.array([[pxx, pxy, pxz],[pxy, pyy, pyz],[pxz, pyz, pzz]])
        print(p.shape)
        # input()

        # ; Rotate p about Z-axis by the angle between X and the projection of B on the XY plane
        N_spec = N.shape[0]
        ph = np.arctan2(sy, sx)
        rot_ph = np.array(
            [[np.cos(ph), -np.sin(ph), 0],
             [np.sin(ph), np.cos(ph), 0],
             [0, 0, 1]])
        p = np.array([rot_ph @ p[..., i] @ rot_ph.T for i in range(N_spec)])
        p = np.swapaxes(p, 0, 2)

        print(p.shape)

        # ; Then rotate p about Y-axis by the angle between Bz and B
        th = np.pi/2 - np.arctan2(sz, np.sqrt(sx**2 + sy**2))
        rot_th = np.array(
            [[np.cos(th), 0, np.sin(th)],
             [0, 1, 0],
             [-np.sin(th), 0, np.cos(th)]])
        p = np.array([rot_th @ p[..., i] @ rot_th.T for i in range(N_spec)])
        p = np.swapaxes(p, 0, 2)

        # ; Finally diagonalize Pxx and Pyy
        dP = np.sqrt(
            p[0, 0, ...]**2 + p[1, 1, ...]**2 - 2*p[0, 0, ...]*p[1, 1, ...] + 4*p[0, 1, ...]**2)
        l1 = p[0, 0, ...] + (p[1, 1, ...] + dP)/2
        l2 = p[0, 0, ...] + (p[1, 1, ...] - dP)/2
        print(l1.shape, l2.shape)
        #  ph is the rotation angle to diagonalize
        ph = np.arccos(np.sqrt((p[0, 0, ...]*l1-p[1, 1, ...]*l2)/(l1**2-l2**2)))
        rot_ph = np.zeros(shape=(3, 3, N_spec))
        rot_ph[0, 0, :] = np.cos(ph)
        rot_ph[1, 0, :] = -np.sin(ph)
        rot_ph[0, 1, :] = np.sin(ph)
        rot_ph[1, 1, :] = np.cos(ph)
        rot_ph[2, 2, :] = 1
        p = np.array([rot_ph[..., i] @ p[..., i] @ rot_ph[..., i].T for i in range(N_spec)])
        p = np.swapaxes(p, 0, 2)
        print(p.shape)

        pxx = p[0, 0, ...]
        pyy = p[1, 1, ...]
        pzz = p[2, 2, ...]
        pxy = p[0, 1, ...]
        pxz = p[0, 2, ...]
        pyz = p[1, 2, ...]

    return pxx, pyy, pzz, pxy, pxz, pyz


def T(energy_eV, d_energy_eV, diff_en_flux_eVcm2eVsters, species,
      phi_deg=None, dphi_deg=None, theta_deg=None, dtheta_deg=None,
      theta_ax=None, N=None, diagonalize_axis=None):
    '''Temperature vector, adaptation of T_3d_new.pro'''

    # Get the density (cm-3) if not provided
    if N is None:
        N = n(
            energy_eV, d_energy_eV, diff_en_flux_eVcm2eVsters, species,
            phi_deg=phi_deg, dphi_deg=dphi_deg, theta_deg=theta_deg,
            dtheta_deg=dtheta_deg, theta_ax=theta_ax)

    # Calculate the pressure tensor:
    pressure_tensor = p(
        energy_eV, d_energy_eV, diff_en_flux_eVcm2eVsters, species,
        phi_deg=phi_deg, dphi_deg=dphi_deg, theta_deg=theta_deg,
        dtheta_deg=dtheta_deg, theta_ax=theta_ax, N=N,
        diagonal_only=diagonalize_axis,
        diagonalize_axis=diagonalize_axis)

    Tavg = (pressure_tensor[0]+pressure_tensor[1]+pressure_tensor[2])/(3*N)
    Tx = pressure_tensor[0]/N
    Ty = pressure_tensor[1]/N
    Tz = pressure_tensor[2]/N

    return Tx, Ty, Tz, Tavg


def H_alpha_moments(energy_eV, d_energy_eV, diff_en_flux_eVcm2eVsters,
                    phi_deg=None, dphi_deg=None,
                    theta_deg=None, dtheta_deg=None, theta_ax=None,
                    N=None, V=None, debug_plots=False):

    '''Adaptation of mvn_swia_protonalphamoms_minf.pro'''

    # First, get the moments assuming everything is protons:
    if N is None or V is None:
        N = n(energy_eV, d_energy_eV, diff_en_flux_eVcm2eVsters,
              'H+', phi_deg=phi_deg, dphi_deg=dphi_deg, theta_deg=theta_deg)
        V = v(energy_eV, d_energy_eV, diff_en_flux_eVcm2eVsters,
              'H+', phi_deg=phi_deg, dphi_deg=dphi_deg, theta_deg=theta_deg,
              N=N)

    # Get the magnitude of the velocity and the proton energy that
    # corresponds to:
    v_mag = np.sqrt(np.sum(V**2, axis=0))
    E = units.energy(v_mag, name='H+')

    # Sum over all angle bins to get the differential
    # energy flux (d(EF)/dEdS -> d(EF)/dE):
    eflux_shape = diff_en_flux_eVcm2eVsters.shape
    sum_axis = tuple(
        [i for i, j in enumerate(eflux_shape) if
         j not in energy_eV.shape][1:])
    domega = jacobian(
        phi_deg=phi_deg, dphi_deg=dphi_deg, theta_deg=theta_deg,
        dtheta_deg=dtheta_deg, shape=eflux_shape, v_moment=0,
        theta_ax=theta_ax)
    espec = np.sum(diff_en_flux_eVcm2eVsters*domega, axis=sum_axis) /\
        np.sum(domega, axis=sum_axis)
    # print(espec.shape)

    # Mark indices where we can seek a different Ecut:
    E = energy_eV[np.argmax(espec, axis=1)]
    E_cut = 1.5 * E
    E_l = 1.15 * E
    E_h = 2 * E

    # inspect = np.where((energy_eV[-2] > E_l) & (energy_eV[2] < E_h))[0]
    inspect = np.where((energy_eV[-1] > E_l) & (energy_eV[0] < E_h))[0]
    # print(inspect.shape)

    for j in inspect:
        # This section working pretty well (subset)
        espec_j = espec[j, :]
        subset_j = np.where((energy_eV > E_l[j]) & (energy_eV < E_h[j]))[0]
        # print(espec_j[subset_j])

        # # Jasper's algo: identify dEF/de > 0 (inc) and sign of inc
        # dEF_dE = np.gradient(espec_j[subset_j], energy_eV[subset_j])
        # # cross = dEF_dE[:-1] * dEF_dE[1:]
        # cross = np.roll(dEF_dE, 1) * dEF_dE
        # # inflection_point_j = np.where((cross < 0) & (dEF_dE[1:] > 0))[0]
        # inflection_point_j = np.where((cross < 0) & (dEF_dE > 0))[0]
        # if len(inflection_point_j) > 0:
        #     # new_start_index = subset_j[0] + inflection_point_j[0] + 1
        #     new_start_index = subset_j[0] + inflection_point_j[0]
        #     new_E_cut = energy_eV[new_start_index]
        #     # print(energy_eV[new_start_index], E_cut[j])

        #     E_cut[j] = new_E_cut

        # Alt method: min?
        inflection_point_j = np.argmin(espec_j[subset_j])
        E_cut[j] = energy_eV[subset_j[0] + inflection_point_j]

        # fig, ax = plt.subplots()
        # ax.plot(energy_eV, espec_j, color='gray')
        # ax.plot(energy_eV[subset_j], espec_j[subset_j])
        # # ax.plot(energy_eV[subset_j], dEF_dE, marker='x')
        # # ax.plot(energy_eV[subset_j], cross, marker='x')
        # ax.axvline((energy_eV[subset_j])[inflection_point_j], color='k')
        # ax.axvline(E_l[j], color='b')
        # ax.axvline(E_h[j], color='r')
        # ax.set_yscale('log')
        # ax.set_xscale('log')
        # plt.show()

        # plt.figure()
        # plt.loglog(energy_eV, espec_j)
        # plt.xlabel("Energy, eV")
        # plt.ylabel('Eflux, eV/cm2/s/ster/eV')
        # plt.axvline(E[j], color='k')
        # plt.axvline(E_l[j], color='gray', linestyle='--')
        # plt.axvline(E_h[j], color='gray', linestyle='--')
        # plt.axvline(E_cut[j], color='salmon')
        # if len(inflection_point_j) > 0:
        #     plt.axvline(new_E_cut, color='b')
        # else:
        #     plt.axvline(E_cut[j], color='r')
        # plt.show()
        # print(subset_j)

        # input()

    if debug_plots:
        from matplotlib.colors import LogNorm
        fig, ax = plt.subplots(nrows=3, ncols=3)

        time_indices = [13870, 14010, 14250]
        time_indices = [7431, 7670, 7807]
        for i, time_index in enumerate(time_indices):
            E_index = np.where(energy_eV <= E_cut[time_index])[0][-1]

            for k, e_offset_index in enumerate([-1, 0, 1]):
                j = E_index + e_offset_index
                theta_deg_i = theta_deg[:, j]
                print(eflux_shape)
                Eflux_snapshot = diff_en_flux_eVcm2eVsters[time_index, :, :, j]
                print(Eflux_snapshot.shape, phi_deg.shape, theta_deg.shape)
                p = ax[i, k].pcolormesh(
                    phi_deg, theta_deg_i, Eflux_snapshot.T,
                    norm=LogNorm(vmin=1e5, vmax=1e8), cmap='plasma')
                # ax[i, k].set_title("{} eV".format(energy_eV[j]))
        # fig.colorbar(p, label='Eflux', ax=ax[-1])

        fig, ax = plt.subplots()
        time_utc = np.linspace(0, eflux_shape[0] - 1, eflux_shape[0])
        p = ax.pcolormesh(
            time_utc, energy_eV, espec.T,
            norm=LogNorm(vmin=1e5, vmax=1e10), cmap='plasma')

        plt.show()

    # Get the Helium flux by setting fluxes below the energy
    # cut to 0
    He_eflux = np.where(
        energy_eV[np.newaxis, np.newaxis, np.newaxis, :] >
        E_cut[:, np.newaxis, np.newaxis, np.newaxis],
        diff_en_flux_eVcm2eVsters, 0)

    He_energy_eV = 2*energy_eV
    He_d_energy_eV = 2*d_energy_eV
    He_name = 'He++'

    # Get the moments, assuming the rest is He++
    He_n = n(
        He_energy_eV, He_d_energy_eV, He_eflux, He_name,
        phi_deg=phi_deg, dphi_deg=dphi_deg,
        theta_deg=theta_deg, dtheta_deg=dtheta_deg)
    He_v = v(
        He_energy_eV, He_d_energy_eV, He_eflux, He_name,
        phi_deg=phi_deg, dphi_deg=dphi_deg,
        theta_deg=theta_deg, dtheta_deg=dtheta_deg, N=He_n)
    He_T = T(
        He_energy_eV, He_d_energy_eV, He_eflux, He_name,
        phi_deg=phi_deg, dphi_deg=dphi_deg,
        theta_deg=theta_deg, dtheta_deg=dtheta_deg, N=He_n)

    # Get the proton flux by setting fluxes below the energy
    # cut to 0
    p_eflux = np.where(
        energy_eV[np.newaxis, np.newaxis, np.newaxis, :] <
        E_cut[:, np.newaxis, np.newaxis, np.newaxis],
        diff_en_flux_eVcm2eVsters, 0)

    p_n = n(
        energy_eV, d_energy_eV, p_eflux, 'H+',
        phi_deg=phi_deg, dphi_deg=dphi_deg,
        theta_deg=theta_deg, dtheta_deg=dtheta_deg)
    p_v = v(
        energy_eV, d_energy_eV, p_eflux, 'H+',
        phi_deg=phi_deg, dphi_deg=dphi_deg,
        theta_deg=theta_deg, dtheta_deg=dtheta_deg, N=p_n)
    p_T = T(
        energy_eV, d_energy_eV, p_eflux, 'H+',
        phi_deg=phi_deg, dphi_deg=dphi_deg,
        theta_deg=theta_deg, dtheta_deg=dtheta_deg, N=p_n)

    if debug_plots:
        fig, ax = plt.subplots(nrows=3, sharey=True, sharex=True)
        time_utc = np.linspace(0, eflux_shape[0] - 1, eflux_shape[0])
        p = ax[0].pcolormesh(
            time_utc, energy_eV, espec.T,
            norm=LogNorm(vmin=1e5, vmax=1e10), cmap='plasma')
        He_espec = np.sum(He_eflux*domega, axis=sum_axis) /\
            np.sum(domega, axis=sum_axis)
        p = ax[1].pcolormesh(
            time_utc, energy_eV, He_espec.T,
            norm=LogNorm(vmin=1e5, vmax=1e10), cmap='plasma')
        p_espec = np.sum(p_eflux*domega, axis=sum_axis) /\
            np.sum(domega, axis=sum_axis)
        p = ax[2].pcolormesh(
            time_utc, energy_eV, p_espec.T,
            norm=LogNorm(vmin=1e5, vmax=1e10), cmap='plasma')
        ax[0].set_yscale('log')
        for ax_i in ax:
            ax_i.plot(time_utc, E_cut, color='k')


    return p_n, p_v, p_T, He_n, He_v, He_T


def moment(desired_moments, time_unix, energy_eV, d_energy_eV,
           diff_en_flux_eVcm2eVsters, species,
           phi_deg=None, dphi_deg=None, theta_deg=None, dtheta_deg=None,
           time_windows_dt=None):


    run_test = False
    if run_test:
        # Avg spectra
        # Sum over theta (1) and phi (2), get peak energy:
        avg_eflux = np.sum(diff_en_flux_eVcm2eVsters, axis=(1, 2))/120
        peak_index = np.argmax(avg_eflux, axis=1)
        peak_flux = np.max(avg_eflux, axis=1)
        peak_en = energy_eV[peak_index]
        diffflux_shape = diff_en_flux_eVcm2eVsters.shape
        N_time = peak_en.shape[0]
        N_phitheta = diffflux_shape[1] * diffflux_shape[2]

        # Now get the angular distributions at the peak energies
        peakflux_dist = diff_en_flux_eVcm2eVsters[np.arange(N_time), :, :, peak_index]
        # peakflux_dist = np.take(diff_en_flux_eVcm2eVsters, peak_index, axis=3)
        print(peakflux_dist.shape)

        # Sum over the theta axis: (n_time x n_phi)
        # sum_over_th = np.sum(peakflux_dist, axis=2)

        peakflux_allangle = peakflux_dist.reshape(N_time, N_phitheta)
        # peakflux_sum = np.sum(peakflux_allangle, axis=1)/120
        maxflux_peak_allangle = np.max(peakflux_allangle, axis=1)
        minflux_peak_allangle = np.min(peakflux_allangle, axis=1)

        stdflux_peak_allangle = np.std(peakflux_allangle, axis=1)
        # ratio = maxflux_peak_allangle/peak_flux
        # ratio = maxflux_peak_allangle/minflux_peak_allangle
        # ratio = stdflux_peak_allangle/maxflux_peak_allangle
        ratio = stdflux_peak_allangle/peak_flux

        # subset = (ratio > 2)

        fig, ax = plt.subplots(nrows=4, sharex=True)
        ax[0].pcolormesh(
            epoch, energy_eV, avg_eflux.T, norm=LogNorm())
        ax[0].set_yscale('log')
        ax[1].semilogy(epoch, peak_en, marker='.')
        ax[2].semilogy(epoch, peak_flux, marker='.')
        ax[3].semilogy(epoch, ratio)

        # ax[0].semilogy(new_time[subset], peak_en[subset], marker='.')
        # ax[1].semilogy(new_time[subset], peak_flux[subset], marker='.')
        # ax[2].semilogy(new_time[subset], ratio[subset])

        plt.show()

    # Calculate moments:
    if "density" in desired_moments:

        N = n(
            energy_eV, d_energy_eV, diff_en_flux_eVcm2eVsters, species,
            phi_deg=phi_deg, dphi_deg=dphi_deg, theta_deg=theta_deg,
            dtheta_deg=dtheta_deg)
        moments["density"] = N

    if "velocity" in desired_moments:

        # print(flux_x.shape)
        # input()

        V = v(
            energy_eV, d_energy_eV, diff_en_flux_eVcm2eVsters, species,
            phi_deg=phi_deg, dphi_deg=dphi_deg, theta_deg=theta_deg,
            dtheta_deg=dtheta_deg)

        # print(flux_3d.shape)
        moments["velocity"] = V

    # print(N.shape)
    # input()

    return moments
