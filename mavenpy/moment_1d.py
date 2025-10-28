import numpy as np
from scipy.optimize import curve_fit
from scipy.special import factorial, gamma

from . import constants


# Adaptation of SPEDAS code for calculating
# plasma moments for a 1D spectra.
# Built and tested for SWEA spectra moments, may or may not
# work for others.


def trapezoidal_dE(energy):
    # dE defined to calculate trapezoidal integral
    n_en = energy.size
    dE = np.zeros(shape=energy.shape)
    dE[0] = np.abs(energy[1] - energy[0])
    for i in range(1, n_en - 1):
        dE[i] = np.abs(energy[i + 1] - energy[i - 1]) / 2
    dE[-1] = np.abs(energy[-1] - energy[-2])
    return dE


def moment_N_T(
    energy_eV, d_energy_eV, diff_en_flux_eVcm2eVsters,
    mass_eVkm2s2, sc_potential_V=None
):

    """ Estimate density and temperature by calculating moments over the
    plasma distribution function as given by the differential energy
    flux at any given time.

    Recall distribution function f(v) = m^2 F(E)/2E for flux F(E)

    N = Int( f(v) d3v ) = Int ( f(v) 4pi v^2 dv )
      = 4pi Int ( (m^2 F(E)/2E)  (2E/m)^0.5 dE/m )
      = 4pi Int ( (m F(E)/2E)  (2E/m)^0.5 dE )
      = 4pi (m/2)^0.5 Int ( EF(E) / E / E^0.5 dE )
      = 4pi (m/2)^0.5 Int ( EF (E) / E^1.5 dE )
    P = Int ( mww f(v) d3v)
      = 2/3 Int ( mv^2/2 f(v) d3v )
      = 2/3 4pi (m/2)^0.5 Int ( EF (E)  E^-0.5 dE )
    T = P/N

    Note that if using an uncorrected spectra for
    spacecraft potential, need to correct the
    moment integration.
    Since f(v) is invariant regardless of s/c charging by Liouville,
    F_actual / E_actual = F_measured * E_measured
    so F(E) --> F(E_measured) * (1 - Vsc/Emeasured)
    and E --> Emeasured - Vsc

    for N:
    EF(E)/E^1.5
       -> (Emeas-Vsc) (Emeas - Vsc) / (Emeas - Vsc)^1.5 / Emeas
       -> (Emeas - Vsc)^0.5/Emeas

    for P:
    EF(E) E^-0.5
       -> (Emeas-Vsc) (Emeas - Vsc) / (Emeas - Vsc)^0.5 / Emeas
       -> (Emeas-Vsc)^1.5 / Emeas

    (Assuming that SW electrons are isotropic, so the
    velocity in the plasma rest frame
    w = v - u ~ v for velocity v in the lab frame
    and plasma bulk velocity u. This should
    be an overestimate.)

    Integrate the distribution function over all velocity space to get density
    and pressure, and subsequently temperature."""

    data_dim = len(diff_en_flux_eVcm2eVsters.shape)
    if data_dim == 2:
        sum_axis = 1
    elif data_dim == 1:
        sum_axis = 0

    if sc_potential_V is not None:
        if data_dim == 2:
            sc_pot_2d = sc_potential_V[:, np.newaxis]
            e_2d = energy_eV[np.newaxis, :]
            ratio = np.where(e_2d > sc_pot_2d, sc_pot_2d / e_2d, 1)
        elif data_dim == 1:
            ratio = np.where(
                energy_eV > sc_potential_V, sc_potential_V / energy_eV, 1)

    moment_prefactor = 4 * np.pi * 1e-5 * np.sqrt(mass_eVkm2s2 / 2)
    N_constant = moment_prefactor * d_energy_eV * energy_eV ** -1.5
    if data_dim == 2:
        N_constant = N_constant[np.newaxis, :]
    if sc_potential_V is not None:
        N_constant = N_constant * np.sqrt(1 - ratio)

    N = np.sum(N_constant * diff_en_flux_eVcm2eVsters, axis=sum_axis)

    # Now calculate pressure
    P_constant = 2 / 3 * moment_prefactor * d_energy_eV * energy_eV ** -0.5
    if data_dim == 2:
        P_constant = P_constant[np.newaxis, :]
    if sc_potential_V is not None:
        P_constant = P_constant * (1 - ratio) ** 1.5
    P = np.sum(P_constant * diff_en_flux_eVcm2eVsters, axis=sum_axis)

    # Now temperature
    T = P / N

    return N, T


def maxwell_boltzmann_coefficients(mass_eVkm2s2=None, particle=None):
    '''Returns the coefficients for a Maxwell-Boltzmann
    distribution.
    (Assume kT ~ T(eV))
    f(v) = (m/2piT)^1.5 exp(-E/T)
    -> eflux EF(E) = (m/2piT)^1.5 exp(-E/T) * E * 2E/m^2
             = (m/2pi)^1.5 * (2/m^2) exp(-E/T) * E^2 /T^1.5
             = c1 * c2 * E^2 exp(-E/T) / T^1.5
    '''

    if particle is None and mass_eVkm2s2 is None:
        raise ValueError(
            "Need either mass_eVkm2s2 or particle to "
            "normalize the calculated flux.")
    elif particle:
        mass_eVkm2s2 = constants.mass[particle]

    # eV^3/2 / (km/s)^3
    c1 = (mass_eVkm2s2 / (2 * np.pi)) ** 1.5
    # (km/s)^4/(eV)^2
    c2 = 2e5 / mass_eVkm2s2 ** 2

    return c1, c2


def check_dims(E, N, T, V=0):
    '''Verify the dimensions of the array:'''
    # Convert into np arrays with multiple axes:
    if isinstance(N, np.ndarray):
        N = N[:, np.newaxis]
        T = T[:, np.newaxis]
        E = E[np.newaxis, :]

        if isinstance(V, np.ndarray):
            V = V[:, np.newaxis]

    return E, N, T, V


def maxwell_boltzmann_nflux(E, N, T, V=0, c1=None, c2=None,
                            mass_eVkm2s2=None, particle=None):

    # c1c2 -> 10^5 eV^-0.5 km/s = eV^0.5 cm/s
    # NE c1c2/t^-1.5 -> cm-3 eV  eV^0.5 cm/s eV^-1.5
    # -> cm-2 s-1 eV-1

    # Get coefficients, if not supplied:
    if c1 is None and c2 is None:
        c1, c2 = maxwell_boltzmann_coefficients(
            mass_eVkm2s2=mass_eVkm2s2, particle=particle)

    E, N, T, V = check_dims(E, N, T, V=V)

    nflux = N * c1 * (E * c2) * np.exp(-(E - V) / T) / (T ** 1.5)

    return nflux


def maxwell_boltzmann_eflux(E, N, T, V=0, c1=None, c2=None,
                            mass_eVkm2s2=None, particle=None):

    """
    Energy flux for a Maxwell-Boltzmann distribution function
    (Assume kT ~ T(eV))
    f(v) = (m/2piT)^1.5 exp(-E/T)
    -> EF(E) = (m/2piT)^1.5 exp(-E/T) * E * 2E/m^2
             = (m/2pi)^1.5 * (2/m^2) exp(-E/T) * E^2 /T^1.5
             = c1 * c2 * E^2 exp(-E/T) / T^1.5
    """
    # Get coefficients, if not supplied:
    if c1 is None and c2 is None:
        c1, c2 = maxwell_boltzmann_coefficients(
            mass_eVkm2s2=mass_eVkm2s2, particle=particle)

    E, N, T, V = check_dims(E, N, T, V=V)

    eflux = N * c1 * (E * E * c2) * np.exp(-(E - V) / T) / (T ** 1.5)

    return eflux


def kappa_nflux(E, mass_eVkm2s2, N, T):
    """
    Energy flux for a kappa (nonthermal) distribution function
    with velocity v, density N, and coefficient k

    Note: If coefficient k is included in the fit,
    it ends up fitting at k ~ 46. Odd. Maybe this is why
    swe_maxbol.pro anchors the fit at k ~ 5. However,
    Stverak et al 2007 estimates the coefficient should be ~2-3
    at distance > 1 AU.

    """
    # (km/s)^4/(eV)^2
    c2 = 2e5 / mass_eVkm2s2 ** 2
    k = 5
    v_halo = np.sqrt(3 * T / mass_eVkm2s2)

    vtot2 = 2 * E / mass_eVkm2s2
    vh2 = (k - 1.5) * v_halo ** 2
    kc = (np.pi * vh2) ** (-1.5) * factorial(k) / gamma(k - 0.5)
    nflux = N * kc * (E * c2) * (1 + (vtot2 / vh2)) ** (-k - 1)

    return nflux


class maxwell_boltzmann_function:
    def __init__(self, c1=None, c2=None, m=None, particle=None):

        # Get coefficients, if not supplied:
        if c1 is None and c2 is None:
            c1, c2 = maxwell_boltzmann_coefficients(
                mass_eVkm2s2=m, particle=particle)
        self.c1 = c1
        self.c2 = c2
        self.m = m

    def maxwell_boltzmann_eflux(self, E, N, T):

        return maxwell_boltzmann_eflux(
            E, N, T, V=self.V, c1=self.c1, c2=self.c2)

    def kappa_eflux(self, E, v, N):
        """
        Energy flux for a kappa (nonthermal) distribution function
        with velocity v, density N, and coefficient k

        Note: If coefficient k is included in the fit,
        it ends up fitting at k ~ 46. Odd. Maybe this is why
        swe_maxbol.pro anchors the fit at k ~ 5. However,
        Stverak et al 2007 estimates the coefficient should be ~2-3
        at distance > 1 AU.

        """
        k = 5

        vtot2 = 2 * (E - self.V) / self.m
        vh2 = (k - 1.5) * v ** 2
        kc = (np.pi * vh2) ** (-1.5) * factorial(k) / gamma(k - 0.5)
        eflux = N * kc * (E * E * self.c2) * (1 + (vtot2 / vh2)) ** (-k - 1)

        return eflux


def curve_fitted_N_T(energy_eV, d_energy_eV, diff_en_flux_eVcm2eVsters,
                     mass_eVkm2s2, sc_potential_V=None, plot_fits=None):

    """Fit a Maxwell-Boltzmann distribution to spectra over time to
    estimate the temperature and density of a solar wind electron distribution
    energy: (N_energies)
    diff_en_flux: (N_time x N_energies)"""

    # Get constants:
    c1, c2 = maxwell_boltzmann_coefficients(mass_eVkm2s2=mass_eVkm2s2)

    # Initial f to run the fitting over:
    f = maxwell_boltzmann_function(c1=c1, c2=c2, m=mass_eVkm2s2)

    N = diff_en_flux_eVcm2eVsters.shape[0]
    m = mass_eVkm2s2
    F = diff_en_flux_eVcm2eVsters
    E = energy_eV
    dE = d_energy_eV
    if sc_potential_V is None:
        V = 0
    else:
        V = sc_potential_V

    # Calculate the moments w/o curve fitting:
    N_all, T_all = moment_N_T(E, dE, F, m, sc_potential_V=sc_potential_V)
    F_allmom = maxwell_boltzmann_eflux(
        E, N_all, T_all, V=sc_potential_V, c1=c1, c2=c2)


    # Get the maximum flux and the energy at that maximum
    index_max = np.nanargmax(F, axis=1)
    Fmax = np.nanmax(F, axis=1)
    Emax = E[index_max]

    # Initial estimates of the temperature and density
    # assuming a purely thermal distribution.
    guess_T = Emax / 2
    guess_N = Fmax / (4 * c1 * c2 * np.sqrt(guess_T) * np.exp(V / guess_T - 2))

    # from matplotlib import pyplot as plt
    # # plt.figure()
    # # plt.plot(np.linspace(1, N, N), Emax)
    # # plt.yscale('log')
    # # plt.show()

    # i = 0
    # V_i = V[i]
    # F_i = F[0, :]
    # f.V = V_i
    # Elo = min(0.8 * Emax[i], max(V_i, 0.5 * Emax[i]))
    # Ehi = 3 * Emax[i]
    # core_indices = np.where((E > Elo) & (E < Ehi))[0]
    # E_core = E[core_indices]
    # F_core = F_i[core_indices]
    # # (N_core_fit_i, T_core_fit_i), _ = curve_fit(
    # #     f.maxwell_boltzmann_eflux, E_core, F_core,
    # #     p0=(guess_N[i], guess_T[i]),
    # #     nan_policy='omit')
    # # print(N_core_fit_i, T_core_fit_i)

    # E_min = np.min([0.8 * Emax, np.max([V, 0.5 * Emax], axis=0)], axis=0)
    # core = (E[np.newaxis, :] > E_min[:, np.newaxis]) & (E[np.newaxis, :] < 3*Emax[:, np.newaxis])
    # # E_core = np.where(core, E, np.nan)
    # E_core = E[np.newaxis, :]
    # # E_core = E
    # F_core = np.where(core, F, np.nan)

    # plt.figure()
    # plt.pcolormesh(np.linspace(1, N, N), E, F_core.T)
    # plt.yscale('log')
    # plt.show()

    # print(E_core.shape, F_core.shape, guess_T.shape, guess_N.shape)

    # # Remove all-NaN columns:
    # not_allnan = ~np.all(np.isnan(F_core), axis=1)
    # print(not_allnan.shape)

    # F_core = F_core[not_allnan, :]
    # guess_N = guess_N[not_allnan]
    # guess_T = guess_T[not_allnan]

    # print(E_core.shape, F_core.shape, guess_T.shape, guess_N.shape)
    # print("Run curvefit")
    # (N_core_fit, T_core_fit), _ = curve_fit(
    #     f.maxwell_boltzmann_eflux, E_core, F_core,
    #     p0=(guess_N, guess_T),
    #     nan_policy='omit')
    # print(N_core_fit.shape, T_core_fit.shape)

    # input()


    # initialize empty arrays to store solutions
    N_core_curvefit = np.zeros(shape=(N,)) + np.nan
    T_core_curvefit = np.zeros(shape=(N,)) + np.nan
    N_core_moment = np.zeros(shape=(N,)) + np.nan
    T_core_moment = np.zeros(shape=(N,)) + np.nan
    N_halo_curvefit = np.zeros(shape=(N,)) + np.nan
    T_halo_curvefit = np.zeros(shape=(N,)) + np.nan
    N_halo_moment = np.zeros(shape=(N,)) + np.nan
    T_halo_moment = np.zeros(shape=(N,)) + np.nan

    for i in range(N):

        # Select Eflux:
        F_i = F[i, :]
        # print(F_i)

        # Set the spacecraft potential, if supplied.
        V_i = sc_potential_V[i] if sc_potential_V is not None else 0

        # Skip array if all NaNs:
        # if ~np.isnan(np.sum(F_i)):
        if np.isnan(F_i).all():
            continue

        # Convert nans into small/large numbers
        F_i = np.nan_to_num(F_i)

        f.V = V_i

        # # Check if spectra is flat -- this
        # if Emax > 300:
        #     print(Emax)
        #     continue

        # Select the range over which we will do the fit.
        Elo = min(0.8 * Emax[i], max(f.V, 0.5 * Emax[i]))
        Ehi = 3 * Emax[i]
        core_indices = np.where((E > Elo) & (E < Ehi))[0]
        E_core = E[core_indices]
        F_core = F_i[core_indices]

        # Attempt to fit a Maxwell-Boltzmann to the core distribution.
        try:
            (N_core_fit_i, T_core_fit_i), _ = curve_fit(
                f.maxwell_boltzmann_eflux, E_core, F_core,
                p0=(guess_N[i], guess_T[i])
            )
        except RuntimeError:
            N_core_fit_i = np.nan
            T_core_fit_i = np.nan
            continue

        # Estimate the density and temperature via estimating the
        # moment over "core" range energies.
        N_core_moment_i, T_core_moment_i = moment_N_T(
            E_core, d_energy_eV[core_indices], F_core, mass_eVkm2s2, sc_potential_V=f.V
        )

        # Now select the halo component of the spectra
        halo_indices = np.where(E > 2 * Emax[i])[0]
        E_halo = E[halo_indices]
        F_halo = F_i[halo_indices] - f.maxwell_boltzmann_eflux(
            E_halo, N_core_fit_i, T_core_fit_i
        )
        # If any negative resultants, set to zero.
        F_halo = np.where(F_halo > 0, F_halo, 0)
        dE_halo = d_energy_eV[halo_indices]

        # Initial estimates of the temperature and density
        # assuming a purely thermal distribution.

        try:
            guess_T_i = E_halo[np.argmax(F_halo)] / 2
        except ValueError:
            N_all, T_all = moment_N_T(E, d_energy_eV, F_i, mass_eVkm2s2, sc_potential_V=0)
            F_moment_i = f.maxwell_boltzmann_eflux(E, N_all, T_all)
            F_core_fit_moment_i = f.maxwell_boltzmann_eflux(E, N_core_moment_i, T_core_moment_i)
            F_core_fit_curvefit_i = f.maxwell_boltzmann_eflux(E, N_core_fit_i, T_core_fit_i)

            from matplotlib import pyplot as plt

            plt.figure()
            plt.loglog(E, F_i, label="sc pot corrected spectra")
            plt.loglog(E, F_moment_i, label="all - MWB\nmoment integration")
            plt.loglog(E, F_core_fit_curvefit_i, label="core - MWB\nfit to MWB")
            plt.loglog(E, F_core_fit_moment_i, label="core - MWB\nmoment integration")
            plt.ylim([1e3, 1e9])
            plt.xlabel("Energy, eV")
            plt.ylabel("Energy flux, eV/cm2/ster/eV")
            plt.legend()
            # plt.gca().axvline(Elo)
            # plt.gca().axvline(Ehi)
            # plt.gca().axvline(2*Emax)
            plt.show()

        guess_N_i = np.max(F_halo) / (
            4 * f.c1 * f.c2 * np.sqrt(guess_T_i) * np.exp(f.V / guess_T_i - 2)
        )
        guess_vh = np.sqrt(E_halo[np.argmax(F_halo)] / mass_eVkm2s2)

        # Old reference to how kappa used to be fitted as well
        # Had problems with overfitting.
        guess_k = 5

        try:
            # (vh_i, N_alt_halo, k_i), _ = curve_fit(
            #     f.kappa_eflux, E_halo, F_halo, p0=(guess_vh, guess_N_i, guess_k))
            (v_halo_fit_i, N_halo_fit_i), _ = curve_fit(
                f.kappa_eflux, E_halo, F_halo, p0=(guess_vh, guess_N_i)
            )
        except RuntimeError:
            N_halo_fit_i = np.nan
            v_halo_fit_i = np.nan

        # 1/2mv2 = 3/2kT -> T = mv^2/3
        T_halo_fit_i = mass_eVkm2s2 * v_halo_fit_i ** 2 / 3

        # Estimate the density and temperature from the moment:
        N_halo_moment_i, T_halo_moment_i = moment_N_T(
            E_halo, dE_halo, F_halo, mass_eVkm2s2, sc_potential_V=f.V
        )

        # Store all determined parameters
        N_core_curvefit[i] = N_core_fit_i
        T_core_curvefit[i] = T_core_fit_i
        N_core_moment[i] = N_core_moment_i
        T_core_moment[i] = T_core_moment_i
        N_halo_curvefit[i] = N_halo_fit_i
        T_halo_curvefit[i] = T_halo_fit_i
        N_halo_moment[i] = N_halo_moment_i
        T_halo_moment[i] = T_halo_moment_i

        # if N_core_fit_i > 1e4:
        # if i > 3e4:
        if plot_fits:
            N_all, T_all = moment_N_T(E, d_energy_eV, F_i, mass_eVkm2s2, sc_potential_V=0)
            print(
                "Moment - all | density: {} cm-3 & temperature: {} eV".format(
                    np.around(N_all, 2), np.around(T_all, 1)
                )
            )

            print(
                "Curvefit - Core | density: {} cm-3 & temperature: {} eV".format(
                    np.around(N_core_fit_i, 2), np.around(T_core_fit_i, 1)
                )
            )

            print(
                "Moment - Core | density: {} cm-3 & temperature: {} eV".format(
                    np.around(N_core_moment_i, 2), np.around(T_core_moment_i, 1)
                )
            )

            print(
                "Curvefit - Halo | density: {} cm-3 &"
                " temperature: {} eV ({} x 10^5 K)".format(
                    np.around(N_halo_fit_i, 2),
                    np.around(T_halo_fit_i, 1),
                    np.around(T_halo_fit_i / 8.617e-5 / 1e5, 1),
                )
            )

            print(
                "Moment - Halo | density: {} cm-3 & temperature: {} eV ({} x 10^5 K)".format(
                    np.around(N_halo_moment_i, 2),
                    np.around(T_halo_moment_i, 1),
                    np.around(T_halo_moment_i / 8.617e-5 / 1e5, 1),
                )
            )

            F_core_fit_moment_i = f.maxwell_boltzmann_eflux(E, N_core_moment_i, T_core_moment_i)
            F_core_fit_curvefit_i = f.maxwell_boltzmann_eflux(E, N_core_fit_i, T_core_fit_i)
            F_halo_fit_moment_i = f.kappa_eflux(
                E, np.sqrt(T_halo_moment_i * 3 / mass_eVkm2s2), N_halo_moment_i
            )
            F_halo_fit_curvefit_i = f.kappa_eflux(E, v_halo_fit_i, N_halo_fit_i)

            from matplotlib import pyplot as plt
            from matplotlib.colors import LogNorm

            fig, ax = plt.subplots()
            ax.plot(E, F_i, label="raw spectra")
            ax.plot(E, F_allmom[i, :],
                    label="all - MWB\nmoment integration")
            ax.plot(E, F_core_fit_curvefit_i,
                    label="core - MWB\nfit to MWB")
            ax.plot(E, F_core_fit_moment_i,
                    label="core - MWB\nmoment integration")
            ax.plot(E, F_halo_fit_moment_i,
                    label="halo - halo\nmoment integration")
            ax.plot(E, F_halo_fit_curvefit_i,
                    label="halo -  kappa\nfit to kappa")
            if f.V > 0:
                ax.axvline(f.V)
            ax.set_ylim([1e3, 1e9])
            ax.set_xlabel("Energy, eV")
            ax.set_ylabel("Energy flux, eV/cm2/ster/eV")
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.legend()
            # plt.gca().axvline(Elo)
            # plt.gca().axvline(Ehi)
            # plt.gca().axvline(2*Emax)

            fig, ax = plt.subplots()
            time_i = np.linspace(0, N - 1, N)
            ax.pcolormesh(time_i, E, diff_en_flux_eVcm2eVsters.T, norm=LogNorm())
            ax.set_yscale('log')
            ax.scatter(time_i, V, color='r')
            ax.axvline(time_i[i], color='k')


            plt.show()

    return (
        N_core_curvefit,
        T_core_curvefit,
        N_core_moment,
        T_core_moment,
        N_halo_curvefit,
        T_halo_curvefit,
        N_halo_moment,
        T_halo_moment,
    )
