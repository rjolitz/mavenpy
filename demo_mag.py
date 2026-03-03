import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

# Modules from mavenpy to be used:
from mavenpy import spice, retrieve, file_path,\
    load, mars_shape_conics

# Set a start and end time range to study:
start = '2023-09-28'
end = '2023-09-29'

# DOWNLOAD PARAMETERS:
# Directory to store data files:
data_directory = "/Users/rjolitz/DataFiles/data"
# Boolean to control whether or not to download
download = False
# Remote name to download data from
remote = "ssl_sprg"
# Boolean to print additional information
# during file download
verbose = False
# Boolean for asking user to confirm download
prompt_for_download = False

# INSTRUMENT PARAMETERS:
# We want MAG data, which is available on ssl_sprg
# in level l2 with a 30 second cadence in payload
# coordinates in IDL sav files
instrument_name = "mag"
mag_coord = 'pl'
mag_ext = 'sav'
mag_res = '30sec'
mag_level = 'l2'

# Need Spice kernels for spacecraft pointing
# (rotating magnetometer data from payload
#  to planetary based MSO coordinates)
# and position.
k = spice.load_kernels(
    data_directory,
    start_date=start, end_date=end,
    download_if_not_available=download,
    verbose=verbose,
    prompt_for_download=prompt_for_download)
print("Loaded kernels.")

# Get MAG file names / load the MAG data
if download:
    retrieve.sdc_retrieve(
        instrument_name,
        destination_dir=data_directory,
        source=remote,
        ext=mag_ext, res=mag_res,
        level=mag_level, coord=mag_coord,
        start_date=start, end_date=end)
    print("MAG files updated.")

# Get the file names:
mag_file_names = file_path.local_file_names(
    data_directory, instrument_name,
    start_date=start, end_date=end,
    ext=mag_ext, res=mag_res,
    level=mag_level, coord=mag_coord,
    source=remote)

# Load the MAG files into a dict:
mag = load.load_data(
    mag_file_names,
    ext=mag_ext, res=mag_res, level=mag_level, coord=mag_coord)

# Access keys, which are mapped to a
# tuple with the data and unit:
epoch, epoch_unit = mag["epoch"]
bx, bx_unit = mag["Bx"]
by, by_unit = mag["By"]
bz, bz_unit = mag["Bz"]

# Can calculate the magnetic field magnitude:
bmag = np.sqrt(bx**2 + by**2 + bz**2)

# Rotate from payload to MSO coordinates:
b_mso = spice.bpl_to_bmso(epoch, bx, by, bz)
bx, by, bz = b_mso[:, 0], b_mso[:, 1], b_mso[:, 2]

# Get the s/c position in MSO during MAG observations:
sc_x, sc_y, sc_z = spice.MAVEN_position(epoch)
sc_alt = np.sqrt(sc_x**2 + sc_y**2 + sc_z**2) - 3390

# Get indices when in solar wind assuming a Trotignon 2006
# model fit:
sw_index = mars_shape_conics.solar_wind_indices(
    sc_x, sc_y, sc_z, reference="Trotignonetal2006")[0]

# Make time v Bx and altitude plot:
fig, ax = plt.subplots(nrows=2, sharex=True)
ax[0].plot(epoch, bx)
ax[0].scatter(epoch[sw_index], bx[sw_index], color='k')
ax[0].set_ylabel("Bx, {unit}".format(unit=bx_unit))
ax[1].plot(epoch, sc_alt)
ax[1].scatter(epoch[sw_index], sc_alt[sw_index], color='k')
ax[1].set_ylabel("Altitude, km")
ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%b-%-d\n%H:%M'))

# Make bins for |B|
bmag_bins = np.linspace(0, 100, 101)
fig, ax = plt.subplots()
ax.hist(bmag, bins=bmag_bins)
ax.hist(bmag[sw_index], bins=bmag_bins)
ax.set_ylabel("# of observations")
ax.set_xlabel("|B|, {unit}".format(unit=bx_unit))

# plt.plot(mpb_x*3390, rho_mpb*3390, color='gray')
plt.show()
