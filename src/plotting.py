import numpy as np

def plot_time_domain(axis, time_axis_ns, data, channels) -> None:
    """
    Plots time-domain data on a given axis.

    This function clears the provided axis and plots the input data with respect
    to the specified time axis. Each channel in the data is plotted individually
    and labeled corresponding to the given channel names. The axis labels, legend,
    and grid are updated accordingly. The changes are then re-drawn on the canvas.

    Args:
        axis: The matplotlib axis object where the data will be plotted.
        time_axis_ns: Array containing the time values in nanoseconds.
        data:"""
    axis.clear()
    for idx, label in enumerate(channels):
        y = data[:, idx]
        axis.plot(time_axis_ns, y, label=label)

    axis.set_xlabel("Time (ns)")
    axis.set_ylabel("Amplitude (V)")
    axis.legend(loc="best")
    axis.grid(True)
    axis.figure.canvas.draw()


def plot_original_spectrum(axis, data, dt_s, channels, center_freq_mhz=0.0,
                           remove_neg=False, freq_min=None, freq_max=None) -> None:
    """
    Plots the original spectrum of provided data on the given axis.

    This function computes the Fourier Transform of the input data to derive the
    spectrum, optionally removes negative frequencies, shifts the frequency axis
    by a specified center frequency, and plots the magnitude of the spectrum for
    each channel on the given axis. It also optionally limits the displayed frequency
    range.

    Parameters:
        axis : matplotlib.axes._axes.Axes
            The matplotlib axis on which to plot the spectrum.
        data : numpy.ndarray
            The input dataset containing time-domain signals for multiple channels.
            Should be of shape (N, C), where N is the number of samples and C is
            the number of channels.
        dt_s : float
            The sampling interval in seconds for the time-domain signals.
        channels : list of str
            List of labels corresponding to each channel in the data.
        center_freq_mhz : float, optional
            The center frequency in MHz used to shift the frequency axis. Defaults
            to 0.0.
        remove_neg : bool, optional
            Whether to remove negative frequencies from the plot. Defaults to False.
        freq_min : float, optional
            Minimum frequency in MHz for limiting the x-axis. Defaults to None.
        freq_max : float, optional
            Maximum frequency in MHz for limiting the x-axis. Defaults to None.

    Raises:
        None
    """
    N = data.shape[0]
    axis.clear()

    # Расчёт оси частот
    freq_axis = np.fft.fftfreq(N, d=dt_s) / 1e6  # в МГц
    freq_axis_shifted = freq_axis - center_freq_mhz

    for idx, label in enumerate(channels):
        y = data[:, idx]
        Y = np.fft.fft(y)
        spectrum = np.abs(Y)

        if remove_neg:
            # Уберём отрицательные частоты
            mask = freq_axis_shifted >= 0
            axis.plot(freq_axis_shifted[mask], spectrum[mask], label=label)
        else:
            axis.plot(freq_axis_shifted, spectrum, label=label)

    axis.set_xlabel("Frequency (MHz) [Shifted by Fc]")
    axis.set_ylabel("Magnitude (r.u.)")
    axis.legend(loc="best")
    axis.grid(True)

    if freq_min is not None and freq_max is not None and freq_max > freq_min:
        axis.set_xlim([freq_min, freq_max])

    axis.figure.canvas.draw()


def plot_demod_spectrum(axis, demod_data, dt_s, channels,
                        remove_neg=False, freq_min=None, freq_max=None):
    """
        Plot the demodulated spectrum for multiple channels on a given axis.

        The function computes the Fast Fourier Transform (FFT) of the given
        demodulated data for each specified channel and plots its magnitude
        spectrum against the frequency axis. Optionally, it can limit the
        frequency range or remove the negative frequency components from the
        plot. The function customizes the plot with axis labels, a legend, and
        a grid, and redraws the canvas upon completion.

        Parameters
        ----------
        axis : matplotlib.axes.Axes
            The axis on which the spectrum is plotted.
        demod_data : numpy.ndarray
            A 2D array of shape (N, C), where N is the number of samples and
            C is the number of channels. Each column represents the time-domain
            demodulated data for a corresponding channel.
        dt_s : float
            The sampling interval in seconds.
        channels : list of str
            The list of labels corresponding to each channel in the data array.
        remove_neg : bool, optional
            If True, negative frequencies are excluded from the spectrum plot.
            Defaults to False.
        freq_min : float, optional
            The minimum frequency in MHz to be displayed on the x-axis. If None,
            no lower limit is applied. Defaults to None.
        freq_max : float, optional
            The maximum frequency in MHz to be displayed on the x-axis. If None,
            no upper limit is applied. Defaults to None.
    """
    N = demod_data.shape[0]
    axis.clear()

    freq_axis = np.fft.fftfreq(N, d=dt_s) / 1e6  # в МГц

    for idx, label in enumerate(channels):
        y = demod_data[:, idx]
        Y = np.fft.fft(y)
        spectrum = np.abs(Y)

        if remove_neg:
            mask = freq_axis >= 0
            axis.plot(freq_axis[mask], spectrum[mask], label=label)
        else:
            axis.plot(freq_axis, spectrum, label=label)

    axis.set_xlabel("Frequency (MHz) [Baseband]")
    axis.set_ylabel("Magnitude (r.u.)")
    axis.legend(loc="best")
    axis.grid(True)

    if freq_min is not None and freq_max is not None and freq_max > freq_min:
        axis.set_xlim([freq_min, freq_max])

    axis.figure.canvas.draw()


def plot_demodulated_time_domain(ax, time_ns, demod_data, selected_channels) -> None:
    """
    Plots the demodulated time-domain data onto the provided Axes object.

    This function is designed to visualize data in the time domain after demodulation, displaying the real part
    of the signal from multiple channels as individual lines on the same plot. Each line is labeled with its
    corresponding channel information. The plot includes labeled axes, gridlines, and a legend for interpretability.

    Parameters:
        ax (matplotlib.axes.Axes): The axes object on which the data is plotted. It must be cleared before plotting.
        time_ns (numpy.ndarray): 1D array representing the time values in nanoseconds, corresponding to the x-axis.
        demod_data (numpy.ndarray): 2D array with demodulated data containing complex values. Rows correspond
                                    to time points, and columns correspond to different channels.
        selected_channels (Sequence[str]): Sequence of strings representing the labels of the selected channels
                                           to include in the plot.

    Raises:
        None. This function does not handle exceptions and assumes input validity.

    Returns:
        None. The function directly modifies the provided Axes object.
    """
    ax.clear()
    for idx, ch_label in enumerate(selected_channels):
        y_i = np.real(demod_data[:, idx])
        ax.plot(time_ns, y_i, label=f"{ch_label} (I)")

    ax.set_xlabel("Time after decimation (ns)")
    ax.set_ylabel("Amplitude (V)")
    ax.legend(loc="best")
    ax.grid(True)
    ax.figure.canvas.draw()
