from collections.abc import Iterable
import datetime as dt

import numpy as np
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
from dateutil.parser import parse

from . import helper


def get_ax(ax, index):

    '''Get an axis from a list or singleton of pyplot
    axes (e.g. from plt.subplots() or plt.subplots(nrows=2)).'''

    if isinstance(ax, Iterable):
        ax = ax[index]
    else:
        ax = ax
    return ax


def iter_ax(ax):

    '''Returns an iterable list of axes, given
    an axis that may be a singleton or tuple
    (e.g. from plt.subplots() or plt.subplots(nrows=2)).'''

    if isinstance(ax, Iterable):
        ax = ax
    else:
        ax = (ax,)
    return ax


def add_colorbar_outside(im, fig, ax, colorbar_width=0.01,
                         margin=0.01, **kwargs):

    '''Makes a colorbar that is located just outside of an existing axis
    (useful for multipanel comparison)'''

    # Based on StackOverflow post here:
    # https://stackoverflow.com/questions/71500468/positioning-multiple-colorbars-outside-of-subplots-matplotlib

    # Get bounding box that marks the boundaries of the axis:
    # [x0 (left), y0 (bottom), x1 (right), y1 (top)] of the axis.
    bbox = ax.get_position()

    # [left most position, bottom position, width, height] of color bar.
    cax = fig.add_axes(
        [bbox.x1 + margin, bbox.y0, colorbar_width, bbox.height])

    # Add colorbar using the new axis:
    cbar = fig.colorbar(im, cax=cax, **kwargs)

    return cbar


def format_xaxis(ax, start_date, end_date):

    start_dt = parse(start_date)
    end_dt = parse(end_date)
    n_days = (end_dt - start_dt).days

    if n_days > 2:
        minor_locator = mdates.DayLocator()
        major_locator = mdates.DayLocator(interval=1)
        major_formatter = mdates.DateFormatter("%b %-d")

    else:
        minor_locator = mdates.MinuteLocator(byminute=[0, 15, 30, 45])
        major_locator = mdates.HourLocator(interval=3)
        major_formatter = mdates.DateFormatter("%b %-d\n%H:%M")

    ax.xaxis.set_major_formatter(major_formatter)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.xaxis.set_major_locator(major_locator)

    # ax.set_xlabel("Time")
    # fig.autofmt_xdate()
    ax.set_xlim(start_dt, end_dt + dt.timedelta(days=1))
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')


def legend_sidetext(fig, ax, labels, colors=None, ax_pad=0.01,
                    ax_min=None, ax_max=None):
    '''Adds labels to the right of the axis,
    equally spaced between two
    '''

    # # labels:
    n = len(labels)

    # If no label colors supplied,
    if not colors:
        prop = plt.rcParams['axes.prop_cycle']()
        colors = [next(prop)['color'] for i in range(n)]

    # Invert the list since we're stacking from bottom-to-top
    labels = labels[::-1]
    colors = colors[::-1]

    # bbox contains the
    # [x0 (left), y0 (bottom), x1 (right), y1 (top)] of the axis.
    bbox = ax.get_position()

    # Start writing the labels from this height:
    if not ax_min:
        ax_min = bbox.y0
    if not ax_max:
        ax_max = bbox.y1
    dy = ax_max - ax_min
    # dy = bbox.height
    dy_i = dy/n

    for j, (l_i, c_i) in enumerate(zip(labels, colors)):
        fig.text(bbox.x1 + ax_pad, ax_min + dy_i*(j + 0.5),
                 l_i, va='center', color=c_i)


def patch_gap(data_nm, x_n, y_m, x_maxgap,
              sample_dx=None, fill_value=np.nan,
              dy=None, verbose=False, utc=None):

    '''A preprocessing function for 2D data with gaps
    before showing in pcolormesh. Addresses the problem
    that pcolormesh will draw big gaping polygons across datagaps,
    despite lacking data.

    For a n x m 2D array sampled over x_n and y_m,
    identifies indices where x_n+1 - x_n >= x_maxgap,
    and if that number is >0, creates a new array
    of dimension (n + (# gaps) x m) and fills the new
    columns with fill_value.
    Also will return an extended x and y axis,
    which will be n + 1 and m + 1, which will ensure
    pcolormesh will draw the polygons properly. '''

    # Check if x array is a datetime:
    if np.issubdtype(x_n.dtype, np.datetime64) or isinstance(
            x_n[0], dt.datetime):
        utc = True

    if utc:
        x_n = helper.UTC_to_UNX(x_n)

    n_x, n_y = data_nm.shape

    # Get the duration between observations:
    dx = np.ediff1d(x_n)

    # indices in the ediff1d array with a greater
    # gap (offset by one from the original array):
    gap_index = np.where(dx > x_maxgap)[0]

    if len(gap_index) == 0:
        # no data gaps, do nothing:
        return data_nm, x_n, y_m

    # if no duration of the sample provided, assume it's
    # dx less than the duration
    if not sample_dx:
        sample_dx = dx[gap_index[0] - 1]

    # Extend dy:
    d_y = np.ediff1d(y_m)
    if d_y[0] < 0:
        y_m = y_m[::-1]

    if not dy:
        dy = y_m[-1] - y_m[-2]
    new_y = np.append(y_m, y_m[-1] + dy)
    if d_y[0] < 0:
        new_y = new_y[::-1]

    # Make new arrays to fill:
    new_n_x = n_x + len(gap_index)

    new_data = np.zeros(shape=(new_n_x, n_y)) + fill_value
    new_x = np.zeros(shape=(new_n_x + 1))

    # The index corresponding to x_n where (x_n - x_n-1) >
    # x_gap is actually offset by 1:
    gap_index = gap_index + 1
    if verbose:
        print("Gap index: ", gap_index)
    # input()

    new_i_index = 0
    i_index = 0

    for i, gap_i in enumerate(gap_index):

        # in the final array, the gap will be placed
        # at i + gap_index
        new_gap_index = gap_i + i
        # gap_i = gap_i + 1

        # mid_unx = (x_n[gap_i] + x_n[gap_i - 1])/2
        # mid_unx = (x_n[gap_i] + x_n[gap_i - 1])/2
        # sample_dx = (unx[gap_i - 1] - x_n[gap_i - 2])
        mid_unx = (x_n[gap_i - 1] + sample_dx)
        # print(x_n[gap_i - 1] - x_n[0], mid_unx - x_n[0], x_n[gap_i] - x_n[0])

        # print(ext_i_index, new_gap_index)
        # print(i_index, gap_i)
        # print((unx[i_index:gap_i] - unx[i_index])/60)
        # print(unx_ext[ext_i_index:new_gap_index])
        # print()
        # print(unx_ext[0:new_gap_index + 2])

        new_x[new_i_index:new_gap_index] = x_n[i_index:gap_i]
        # print(new_x[0:new_gap_index + 2])

        new_x[new_gap_index] = mid_unx
        # print((new_x[0:new_gap_index + 2] - x_n[0])/60)
        # print((x_n[0:gap_i + 2] - x_n[0])/60)
        # print(helper.UNX_to_UTC(unx_ext[0:new_gap_index + 2]))

        # extemd the H and alpha fluxes:
        # print()
        # print(data_h_ext[0:final_gap_i + 2, -6])
        new_data[new_i_index:new_gap_index, :] = data_nm[i_index:gap_i, :]
        # print(new_data[0:new_gap_index + 2, -6])

        # Progress indices:
        i_index = gap_i
        new_i_index = new_gap_index + 1
        # input()

    # patch the last
    new_x[new_i_index:-1] = x_n[i_index:]
    new_x[-1] = x_n[-1] + sample_dx
    new_data[new_i_index:, :] = data_nm[i_index:, :]

    if utc:
        new_x = helper.UNX_to_UTC(new_x)

    return new_data, new_x, new_y
