import sys
import numpy as np
import string
import h5py

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QDockWidget,
    QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog,
    QListWidget, QListWidgetItem, QGroupBox, QMessageBox,
    QCheckBox, QSizePolicy
)
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from scipy.io import savemat

from mpl_canvas import MplCanvas
from plotting import (
    plot_time_domain,
    plot_original_spectrum,
    plot_demod_spectrum,
    plot_demodulated_time_domain
)


class MainWindow(QMainWindow):
    """
    Main window for signal processing application.

    This class represents the main interface for a signal processing application.
    It allows users to interact with signals using various controls such as loading
    a CSV file, applying filters, demodulating, plotting, and exporting data.
    The graphical interface is divided into three main control groups: Time Domain,
    Demod Domain, and Frequency Domain. Each group includes functionalities specific
    to that domain. Additionally, the window includes sections for visualizing
    time-domain plots and frequency spectra.

    Attributes:
        csv_data: Data loaded from the CSV file.
        demod_data: Data after demodulation.
        demod_time_ns: Timestamp data associated with the demodulated data.
        sampling_rate_ns: Sampling rate for signal processing in nanoseconds.
        loaded_file_path: Path of the loaded CSV file.

    Raises:
        NotImplementedError: Raised for unimplemented functionalities if applicable.
    """
    def __init__(self):
        super().__init__()

        self.csv_data = None
        self.demod_data = None
        self.demod_time_ns = None

        self.sampling_rate_ns = 13
        self.loaded_file_path = None

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Signal Demo by V.Pugovkin (1920x1080)")

        self.setGeometry(50, 50, 1920, 1080)


        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        top_controls_layout = QHBoxLayout()
        main_layout.addLayout(top_controls_layout)

        time_domain_group = QGroupBox("Time Domain")
        time_domain_group_layout = QVBoxLayout(time_domain_group)

        self.load_btn = QPushButton("Load CSV")
        self.load_btn.clicked.connect(self.load_csv)
        time_domain_group_layout.addWidget(self.load_btn)

        bandpass_layout = QHBoxLayout()
        self.use_bandpass_checkbox = QCheckBox("Bandpass Filter")
        bandpass_layout.addWidget(self.use_bandpass_checkbox)

        bandpass_label_low = QLabel("Low (MHz):")
        self.bandpass_low_input = QLineEdit("10")
        self.bandpass_low_input.setFixedWidth(60)

        bandpass_label_high = QLabel("High (MHz):")
        self.bandpass_high_input = QLineEdit("50")
        self.bandpass_high_input.setFixedWidth(60)

        bandpass_layout.addWidget(bandpass_label_low)
        bandpass_layout.addWidget(self.bandpass_low_input)
        bandpass_layout.addWidget(bandpass_label_high)
        bandpass_layout.addWidget(self.bandpass_high_input)
        time_domain_group_layout.addLayout(bandpass_layout)

        self.plot_time_btn = QPushButton("Plot Time Domain")
        self.plot_time_btn.clicked.connect(self.plot_time_domain)
        time_domain_group_layout.addWidget(self.plot_time_btn)

        top_controls_layout.addWidget(time_domain_group)

        demod_group = QGroupBox("Demod Domain")
        demod_group_layout = QVBoxLayout(demod_group)

        demod_params_layout = QHBoxLayout()

        label_fc = QLabel("Fc (MHz):")
        self.fc_input = QLineEdit("0")
        self.fc_input.setFixedWidth(60)

        label_st = QLabel("Start (ns):")
        self.start_time_input = QLineEdit("0")
        self.start_time_input.setFixedWidth(70)

        label_end = QLabel("End (ns):")
        self.end_time_input = QLineEdit("1000")
        self.end_time_input.setFixedWidth(70)

        label_dec = QLabel("Dec factor:")
        self.decimation_factor_input = QLineEdit("")
        self.decimation_factor_input.setFixedWidth(60)

        demod_params_layout.addWidget(label_fc)
        demod_params_layout.addWidget(self.fc_input)
        demod_params_layout.addWidget(label_st)
        demod_params_layout.addWidget(self.start_time_input)
        demod_params_layout.addWidget(label_end)
        demod_params_layout.addWidget(self.end_time_input)
        demod_params_layout.addWidget(label_dec)
        demod_params_layout.addWidget(self.decimation_factor_input)

        demod_group_layout.addLayout(demod_params_layout)

        lowpass_layout = QHBoxLayout()
        self.lowpass_checkbox = QCheckBox("Lowpass after demod")
        lowpass_layout.addWidget(self.lowpass_checkbox)

        lowpass_label = QLabel("Cutoff (MHz):")
        self.lowpass_cutoff_input = QLineEdit("20")
        self.lowpass_cutoff_input.setFixedWidth(60)
        lowpass_layout.addWidget(lowpass_label)
        lowpass_layout.addWidget(self.lowpass_cutoff_input)

        demod_group_layout.addLayout(lowpass_layout)

        self.demod_dec_btn = QPushButton("Demod + Decim")
        self.demod_dec_btn.clicked.connect(self.demodulate_and_decimate)
        demod_group_layout.addWidget(self.demod_dec_btn)

        self.plot_demod_btn = QPushButton("Plot demod time")
        self.plot_demod_btn.clicked.connect(self.plot_demod_data)
        demod_group_layout.addWidget(self.plot_demod_btn)

        export_layout = QHBoxLayout()
        self.export_mat_btn = QPushButton("Export to .mat")
        self.export_mat_btn.clicked.connect(self.export_data_to_mat)
        export_layout.addWidget(self.export_mat_btn)

        self.export_h5_btn = QPushButton("Export to .h5")
        self.export_h5_btn.clicked.connect(self.export_data_to_h5)
        export_layout.addWidget(self.export_h5_btn)

        demod_group_layout.addLayout(export_layout)

        top_controls_layout.addWidget(demod_group)

        freq_domain_group = QGroupBox("Frequency Domain")
        freq_domain_group_layout = QVBoxLayout(freq_domain_group)

        freq_range_layout = QHBoxLayout()
        freq_range_label = QLabel("Spectrum range (MHz):")
        self.freq_min_input = QLineEdit("-100")
        self.freq_min_input.setFixedWidth(60)
        self.freq_max_input = QLineEdit("100")
        self.freq_max_input.setFixedWidth(60)

        freq_range_layout.addWidget(freq_range_label)
        freq_range_layout.addWidget(self.freq_min_input)
        freq_range_layout.addWidget(self.freq_max_input)
        freq_domain_group_layout.addLayout(freq_range_layout)

        self.remove_neg_cb = QCheckBox("Remove negative freq (one-sided)")
        freq_domain_group_layout.addWidget(self.remove_neg_cb)

        self.plot_orig_spectrum_btn = QPushButton("Plot Original Spectrum")
        self.plot_orig_spectrum_btn.clicked.connect(self.plot_original_spectrum)
        freq_domain_group_layout.addWidget(self.plot_orig_spectrum_btn)

        self.plot_demod_spectrum_btn = QPushButton("Plot Demod Spectrum")
        self.plot_demod_spectrum_btn.clicked.connect(self.plot_demod_spectrum)
        freq_domain_group_layout.addWidget(self.plot_demod_spectrum_btn)

        export_spectrum_layout = QHBoxLayout()
        self.export_orig_spectrum_btn = QPushButton("Export Original Spectrum")
        self.export_orig_spectrum_btn.clicked.connect(self.export_original_spectrum)
        export_spectrum_layout.addWidget(self.export_orig_spectrum_btn)

        self.export_demod_spectrum_btn = QPushButton("Export Demod Spectrum")
        self.export_demod_spectrum_btn.clicked.connect(self.export_demod_spectrum)
        export_spectrum_layout.addWidget(self.export_demod_spectrum_btn)

        freq_domain_group_layout.addLayout(export_spectrum_layout)

        top_controls_layout.addWidget(freq_domain_group)

        plot_row_1 = QHBoxLayout()

        self.time_group = QGroupBox("Time Domain Plot")
        time_layout = QVBoxLayout(self.time_group)
        self.time_canvas = MplCanvas(self, width=5, height=4)
        time_layout.addWidget(self.time_canvas)
        self.time_toolbar = NavigationToolbar(self.time_canvas, self.time_group)
        time_layout.addWidget(self.time_toolbar)

        self.freq_group = QGroupBox("Spectrum Plot")
        freq_layout = QVBoxLayout(self.freq_group)
        self.freq_canvas = MplCanvas(self, width=5, height=4)
        freq_layout.addWidget(self.freq_canvas)
        self.freq_toolbar = NavigationToolbar(self.freq_canvas, self.freq_group)
        freq_layout.addWidget(self.freq_toolbar)

        plot_row_1.addWidget(self.time_group, stretch=1)
        plot_row_1.addWidget(self.freq_group, stretch=1)
        main_layout.addLayout(plot_row_1)

        plot_row_2 = QHBoxLayout()
        self.demod_group_plot = QGroupBox("Demodulated Signal Plot")
        demod_layout = QVBoxLayout(self.demod_group_plot)
        self.demod_time_canvas = MplCanvas(self, width=5, height=3)
        demod_layout.addWidget(self.demod_time_canvas)
        self.demod_toolbar = NavigationToolbar(self.demod_time_canvas, self.demod_group_plot)
        demod_layout.addWidget(self.demod_toolbar)
        plot_row_2.addWidget(self.demod_group_plot, stretch=1)
        main_layout.addLayout(plot_row_2)


        dock = QDockWidget("Channels", self)
        dock.setFeatures(
            QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetVerticalTitleBar
        )

        self.channels_list = QListWidget()
        self.channels_list.setSelectionMode(QListWidget.MultiSelection)
        self.channels_list.setMaximumWidth(250)

        dock_widget = QWidget()
        vbox = QVBoxLayout(dock_widget)
        label_ch = QLabel("Select channels:")
        vbox.addWidget(label_ch)
        vbox.addWidget(self.channels_list)
        dock_widget.setLayout(vbox)

        dock.setWidget(dock_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        self.time_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.freq_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.demod_group_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.setStyleSheet("""
            QMainWindow {
                background-color: #FAFAFA;
            }
            QDockWidget {
                background-color: #F4F7F9;
            }
            QDockWidget::title {
                background-color: #34495E;
                color: white;
                padding: 8px;
                font-size: 16px;
            }
            QGroupBox {
                border: 2px solid #3498db;
                border-radius: 9px;
                margin-top: 20px;
                font-weight: bold;
                font-size: 18px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center; 
                padding: -5px 25px;
            }
            QPushButton {
                background-color: #3DAEE9;
                color: white;
                border-radius: 9px;
                padding: 6px 16px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #3592CC;
            }
            QPushButton:pressed {
                background-color: #2A7098;
            }
            QCheckBox {
                font-size: 14px;
            }
            QLineEdit {
                padding: 3px;
                font-size: 14px;
            }
            QLabel {
                font-size: 14px;          
            }
        """)

        self.show()

    def load_csv(self):
        """
        Loads a CSV file, processes its content, and sets up the application variables.
        The method uses a file dialog to select a CSV file, reads and processes its contents,
        populates an array with numerical data, and updates a user interface component with
        channel names based on the column count in the loaded CSV. If the file cannot be
        loaded, an error message is displayed, and data variables are reset.

        Returns:
            None

        Raises:
            Exception: If there is an error while opening, reading, or processing the CSV file.
        """
        file_dialog = QFileDialog(self, "Select .csv file", ".", "CSV Files (*.csv)")
        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
        else:
            return

        if not file_path:
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                cleaned_lines = []
                for line in f:
                    line = line.strip()

                    if line.endswith(','):
                        line = line[:-1]
                    cleaned_lines.append(line)

            loaded_array = np.loadtxt(cleaned_lines, delimiter=',')
            if loaded_array.ndim == 1:
                loaded_array = loaded_array.reshape(-1, 1)

            self.csv_data = loaded_array
            num_cols = self.csv_data.shape[1]

            self.channels_list.clear()
            for i in range(num_cols):
                channel_name = string.ascii_uppercase[i] if i < 26 else f"CH{i}"
                item = QListWidgetItem(f"Channel {channel_name}")
                if i == 0:
                    item.setCheckState(Qt.Checked)
                else:
                    item.setCheckState(Qt.Unchecked)
                self.channels_list.addItem(item)

            self.loaded_file_path = file_path
            QMessageBox.information(
                self, "File loaded", f"CSV file loaded:\n{file_path}"
            )
        except Exception as e:
            QMessageBox.warning(self, "Error loading", f"Could not load file:\n{e}")
            self.csv_data = None

    def get_time_slice_indices(self):
        """
        Determine the time slice indices based on user input and sampling rate.

        This method calculates the indices for a specified time slice by converting
        start and end times input by the user into indices relative to the sampling
        rate. If the input is invalid (e.g., end time is less than or equal to start
        time), or if the computed indices are invalid, the method returns None for
        both indices. Otherwise, it adjusts out-of-range indices to fit within the
        data bounds and returns the start and end indices.

        Returns
        -------
        tuple[int or None, int or None]
            A tuple containing the start and end indices for the specified time
            slice. Returns (None, None) if the inputs are invalid or if the computed
            slice is outside the valid range of indices.
        """
        if self.csv_data is None:
            return None, None

        try:
            start_ns = float(self.start_time_input.text())
            end_ns = float(self.end_time_input.text())
            if end_ns <= start_ns:
                raise ValueError("end <= start")
        except ValueError:
            return None, None

        dt_ns = self.sampling_rate_ns
        start_idx = int(round(start_ns / dt_ns))
        end_idx = int(round(end_ns / dt_ns))

        if start_idx < 0:
            start_idx = 0
        if end_idx > self.csv_data.shape[0]:
            end_idx = self.csv_data.shape[0]

        if start_idx >= end_idx:
            return None, None

        return start_idx, end_idx

    def get_selected_channels(self):
        """
        Gets the indices of selected channels from a list.

        This method iterates through the items in the `channels_list` and checks
        their state. If the state of an item is `Qt.Checked`, its index is added
        to a list of selected indices. The method provides an easy way to retrieve
        the indices of all channels that have been checked by the user.

        Returns:
            list[int]: A list containing the indices of the channels that are
            checked in the `channels_list`.
        """
        indices = []
        for i in range(self.channels_list.count()):
            item = self.channels_list.item(i)
            if item.checkState() == Qt.Checked:
                indices.append(i)
        return indices

    def bandpass_filter_fft(self, data, dt_s, f_low_hz, f_high_hz):
        """
        Applies a bandpass filter to multi-channel time-series data using FFT.

        This function takes a multi-channel time-series `data` and applies a bandpass
        filter to each channel using the Fast Fourier Transform (FFT). The filter
        retains frequency components within the range `[f_low_hz, f_high_hz]` and
        suppresses frequencies outside this range.

        Parameters:
        data (np.ndarray): The input time-series data as a 2D array, where the rows
          represent time samples and the columns represent different channels.
        dt_s (float): The time step in seconds between consecutive samples in the input data.
        f_low_hz (float): The lower cutoff frequency of the bandpass filter in Hz.
        f_high_hz (float): The upper cutoff frequency of the bandpass filter in Hz.

        Returns:
        np.ndarray: A 2D array of the filtered data with the same shape as the input
          `data`. Each channel is individually filtered.
        """
        N = data.shape[0]
        num_ch = data.shape[1]
        filtered = np.zeros_like(data, dtype=float)

        freqs = np.fft.fftfreq(N, d=dt_s)
        for ch in range(num_ch):
            y = data[:, ch]
            Y = np.fft.fft(y)

            mask = (np.abs(freqs) >= f_low_hz) & (np.abs(freqs) <= f_high_hz)
            Y[~mask] = 0.0
            y_filt = np.fft.ifft(Y)
            filtered[:, ch] = np.real(y_filt)

        return filtered

    def lowpass_filter_fft(self, data, dt_s, cutoff_hz):
        """
        Apply a low-pass filter to multi-channel data using FFT. The function performs a
        Fast Fourier Transform (FFT) to convert signal data to the frequency domain, applies
        a low-pass filter by masking out frequency components above the cutoff frequency, and
        then performs an inverse FFT to convert the filtered signal back to the time domain.

        Parameters
        ----------
        data : numpy.ndarray
            A 2D array of shape (N, num_ch), where N is the number of time samples and num_ch
            is the number of channels. Represents the input time-domain data to be filtered.
        dt_s : float
            The sampling interval in seconds.
        cutoff_hz : float
            The cutoff frequency of the low-pass filter in Hertz (Hz).

        Returns
        -------
        numpy.ndarray
            A 2D array of shape (N, num_ch), containing the filtered time-domain data. The
            array has the same dtype as the input `data`.

        Raises
        ------
        None
        """
        N = data.shape[0]
        num_ch = data.shape[1]
        filtered = np.zeros_like(data, dtype=data.dtype)

        freqs = np.fft.fftfreq(N, d=dt_s)
        for ch in range(num_ch):
            Y = np.fft.fft(data[:, ch])
            mask = (np.abs(freqs) <= cutoff_hz)
            Y[~mask] = 0.0
            filtered[:, ch] = np.fft.ifft(Y)
        return filtered

    def apply_bandpass_if_needed(self, data, dt_s):
        """
        Applies a bandpass filter to the given data if the bandpass checkbox is checked.

        Based on the user input and the state of the bandpass checkbox, this function either
        applies a bandpass filter to the provided data or returns a copy of the original data.
        The bandpass filter is applied using specified low and high frequency bounds. If the
        user-provided frequency bounds are invalid (e.g., high frequency is less than or equal
        to low frequency), a warning is displayed to the user.

        Parameters:
            data: numpy.ndarray
                The input data to be filtered.
            dt_s: float
                The time step in seconds between consecutive data samples.

        Returns:
            numpy.ndarray: A filtered copy of the data if the bandpass checkbox is checked
            and valid frequency bounds are provided. Otherwise, a direct copy of the input
            data.
        """
        out = data.copy()
        if self.use_bandpass_checkbox.isChecked():
            try:
                f_low_mhz = float(self.bandpass_low_input.text())
                f_high_mhz = float(self.bandpass_high_input.text())
                if f_high_mhz <= f_low_mhz:
                    raise ValueError
            except ValueError:
                QMessageBox.warning(self, "Bandpass Error", "Incorrect bandpass frequencies.")
            else:
                f_low_hz = f_low_mhz * 1e6
                f_high_hz = f_high_mhz * 1e6
                out = self.bandpass_filter_fft(out, dt_s, f_low_hz, f_high_hz)
        return out

    def plot_time_domain(self):
        """
        Plots the time-domain data for selected channels within the specified time range.

        The method verifies whether CSV data is loaded, checks the validity of the
        specified time interval, ensures that at least one channel is selected,
        and processes the data accordingly. The processed data is then visualized
        on the time canvas using the specified time axis.

        Parameters
        ----------
        None

        Raises
        ------
        None

        Returns
        -------
        None
        """
        if self.csv_data is None:
            QMessageBox.warning(self, "Error", "No CSV loaded.")
            return

        start_idx, end_idx = self.get_time_slice_indices()
        if start_idx is None or end_idx is None:
            QMessageBox.warning(self, "Error", "Invalid time interval.")
            return

        selected_channels = self.get_selected_channels()
        if not selected_channels:
            QMessageBox.warning(self, "Error", "No channels selected.")
            return

        raw_cropped = self.csv_data[start_idx:end_idx, :]

        raw_cropped_v = raw_cropped / 32767.0 * 5.0

        dt_s = self.sampling_rate_ns * 1e-9
        data_for_plot = self.apply_bandpass_if_needed(raw_cropped_v, dt_s)

        self.time_canvas.ax.clear()
        time_axis_ns = np.arange(start_idx, end_idx) * self.sampling_rate_ns

        channel_labels = [
            f"Channel {string.ascii_uppercase[ch] if ch < 26 else ch}"
            for ch in selected_channels
        ]

        plot_time_domain(
            axis=self.time_canvas.ax,
            time_axis_ns=time_axis_ns,
            data=data_for_plot[:, selected_channels],
            channels=channel_labels
        )
        self.time_canvas.draw()

    def plot_original_spectrum(self):
        """
        Plot the original spectrum using FFT (Fast Fourier Transform) for the selected time slice and
        channels from the loaded CSV data. The method performs checks on data validity, user input,
        and applies a band-pass filter if necessary before plotting on the specified axis.

        Raises:
            Warning: Displays a warning message box if:
                - No CSV data has been loaded.
                - The specified time interval is invalid.
                - No channels are selected.

        Parameters:
            None

        Returns:
            None
        """
        if self.csv_data is None:
            QMessageBox.warning(self, "Error", "No CSV loaded.")
            return

        start_idx, end_idx = self.get_time_slice_indices()
        if start_idx is None or end_idx is None:
            QMessageBox.warning(self, "Error", "Invalid time interval.")
            return

        selected_channels = self.get_selected_channels()
        if not selected_channels:
            QMessageBox.warning(self, "Error", "No channels selected.")
            return

        try:
            frequency_mhz = float(self.fc_input.text())
        except ValueError:
            frequency_mhz = 0.0

        try:
            freq_min = float(self.freq_min_input.text())
            freq_max = float(self.freq_max_input.text())
        except ValueError:
            freq_min, freq_max = None, None

        raw_cropped = self.csv_data[start_idx:end_idx, :]
        raw_cropped_v = raw_cropped / 32767.0 * 5.0

        dt_s = self.sampling_rate_ns * 1e-9
        data_for_fft = self.apply_bandpass_if_needed(raw_cropped_v, dt_s)


        self.freq_canvas.ax.clear()
        channel_labels = [
            f"Channel {string.ascii_uppercase[ch] if ch < 26 else ch}"
            for ch in selected_channels
        ]

        plot_original_spectrum(
            axis=self.freq_canvas.ax,
            data=data_for_fft[:, selected_channels],
            dt_s=dt_s,
            channels=channel_labels,
            center_freq_mhz=frequency_mhz,
            remove_neg=self.remove_neg_cb.isChecked(),
            freq_min=freq_min,
            freq_max=freq_max
        )

        self.freq_canvas.draw()

    def plot_demod_spectrum(self):
        """
        Plots the frequency spectrum of the demodulated data on the designated canvas.

        The method checks for the availability of demodulated data and time values,
        prepares plotting parameters (like frequency range, selected channels, and
        decimation factor), and plots the spectrum. It also handles invalid inputs
        gracefully by defaulting to predefined values.

        Attributes
        ----------
        demod_data : Any
            Demodulated data to be plotted, expected to be in a complex format.
        demod_time_ns : Any
            Demodulated time data in nanoseconds, used for time-frequency relationships.
        freq_canvas : Any
            The matplotlib canvas where the spectrum is plotted. Expected to provide
            an axis for plotting.
        sampling_rate_ns : int
            Sampling rate in nanoseconds used during signal acquisition.
        freq_min_input : Any
            Input widget providing the minimum frequency for the plot.
        freq_max_input : Any
            Input widget providing the maximum frequency for the plot.
        decimation_factor_input : Any
            Input widget providing the decimation factor to resample the data.
        remove_neg_cb : Any
            Checkbox widget determining whether to remove negative frequencies.

        Parameters
        ----------
        self : PlotSpectrumClass
            Represents the instance of the class that contains this method.
        """
        if self.demod_data is None or self.demod_time_ns is None:
            QMessageBox.warning(self, "Error", "No demodulated data. Please do demodulation first.")
            return

        try:
            freq_min = float(self.freq_min_input.text())
            freq_max = float(self.freq_max_input.text())
        except ValueError:
            freq_min, freq_max = None, None

        self.freq_canvas.ax.clear()

        selected_channels = self.get_selected_channels()
        channel_labels = [
            f"Channel {string.ascii_uppercase[ch] if ch < 26 else ch}"
            for ch in selected_channels
        ]

        dec_factor = 1
        try:
            dec_factor = int(self.decimation_factor_input.text())
            if dec_factor < 1:
                dec_factor = 1
        except:
            pass
        dt_s = self.sampling_rate_ns * 1e-9 * dec_factor

        plot_demod_spectrum(
            axis=self.freq_canvas.ax,
            demod_data=self.demod_data,
            dt_s=dt_s,
            channels=channel_labels,
            remove_neg=self.remove_neg_cb.isChecked(),
            freq_min=freq_min,
            freq_max=freq_max
        )

        self.freq_canvas.draw()

    def demodulate_and_decimate(self):
        """
        Processes raw waveform data by applying bandpass filtering, demodulation,
        low-pass filtering (if selected), and decimation. This method is used to extract
        useful information from waveform data collected over the specified time interval
        and selected channels. Users have the option to specify the carrier frequency for
        demodulation and customize the decimation factor. Automatic decimation
        factor optimization is performed if no input is provided.

        Errors are handled gracefully with appropriate message box warnings
        to notify users of any input or selection issues.

        Attributes
        ----------
        self.csv_data : numpy.ndarray
            CSV data loaded in memory, expected as a 2D array with channels as columns.
        self.sampling_rate_ns : float
            Sampling rate of the data in nanoseconds.
        self.decimation_factor_input : QLineEdit
            User input for specifying the decimation factor.
        self.fc_input : QLineEdit
            User input for the carrier frequency in MHz.
        self.lowpass_checkbox : QCheckBox
            Checkbox to enable or disable the low-pass filtering step.
        self.lowpass_cutoff_input : QLineEdit
            User input specifying the cutoff frequency for low-pass filtering in MHz.
        self.demod_data : numpy.ndarray
            Output complex demodulated and decimated data.
        self.demod_time_ns : numpy.ndarray
            Time vector for demodulated data in nanoseconds.

        Parameters
        ----------
        None

        Raises
        ------
        ValueError
            Raised when invalid input is provided for carrier frequency, decimation factor,
            or low-pass cutoff frequency.
        Warning
            If no CSV is loaded, time intervals are invalid, or no channels are selected.

        Returns
        -------
        None
        """
        if self.csv_data is None:
            QMessageBox.warning(self, "Error", "No CSV loaded.")
            return

        start_idx, end_idx = self.get_time_slice_indices()
        if start_idx is None or end_idx is None:
            QMessageBox.warning(self, "Error", "Invalid time interval.")
            return

        selected_channels = self.get_selected_channels()
        if not selected_channels:
            QMessageBox.warning(self, "Error", "Select at least one channel.")
            return

        try:
            frequency_mhz = float(self.fc_input.text())
        except ValueError:
            frequency_mhz = 0.0

        cropped_data = self.csv_data[start_idx:end_idx, :]
        cropped_data_v = cropped_data / 32767.0 * 5.0
        dt_s = self.sampling_rate_ns * 1e-9


        data_for_demod = self.apply_bandpass_if_needed(cropped_data_v, dt_s)


        num_points = end_idx - start_idx
        auto_dec_factor = max(1, num_points // 256)

        try:
            user_dec_factor = int(self.decimation_factor_input.text())
            if user_dec_factor < 1:
                raise ValueError
            decimation_factor = user_dec_factor
        except:
            decimation_factor = auto_dec_factor
            self.decimation_factor_input.setText(str(decimation_factor))

        t_vec = np.arange(data_for_demod.shape[0]) * dt_s
        fc_hz = frequency_mhz * 1e6


        y_complex = np.zeros((data_for_demod.shape[0], len(selected_channels)), dtype=np.complex128)
        for idx, ch in enumerate(selected_channels):
            y = data_for_demod[:, ch]
            mixer = np.exp(-1j * 2 * np.pi * fc_hz * t_vec)
            y_complex[:, idx] = y * mixer


        if self.lowpass_checkbox.isChecked():
            try:
                cutoff_mhz = float(self.lowpass_cutoff_input.text())
                cutoff_hz = cutoff_mhz * 1e6
                y_complex = self.lowpass_filter_fft(y_complex, dt_s, cutoff_hz)
            except ValueError:
                QMessageBox.warning(self, "Error", "Invalid cutoff frequency for lowpass.")
                return


        self.demod_data = y_complex[::decimation_factor, :]
        decimated_dt_s = dt_s * decimation_factor
        self.demod_time_ns = np.arange(self.demod_data.shape[0]) * decimated_dt_s * 1e9

        QMessageBox.information(
            self, "Demodulation",
            f"Demod on {frequency_mhz} MHz + decim x{decimation_factor} done!"
        )

    # ========== Отдельная кнопка "Plot demod time" ==========
    def plot_demod_data(self):
        """
        Plots the demodulated data in the time domain after verifying data availability
        and selected channels. Clears the current axis and plots the data for the
        selected channels.

        Raises
        ------
        TypeError
            Raised when `self.demod_data` or `self.demod_time_ns` is None, or when no
            channels have been selected for plotting.

        Returns
        -------
        None
        """
        if self.demod_data is None or self.demod_time_ns is None:
            QMessageBox.warning(self, "Error", "No demodulated data. Please do demodulation first.")
            return

        selected_channels = self.get_selected_channels()
        if not selected_channels:
            QMessageBox.warning(self, "Error", "No channels selected.")
            return

        self.demod_time_canvas.ax.clear()
        channel_labels = [
            f"Channel {string.ascii_uppercase[ch] if ch < 26 else ch}"
            for ch in selected_channels
        ]

        plot_demodulated_time_domain(
            ax=self.demod_time_canvas.ax,
            time_ns=self.demod_time_ns,
            demod_data=self.demod_data,  # комплекс
            selected_channels=channel_labels
        )
        self.demod_time_canvas.draw()

    def export_data_to_mat(self):
        """
        Exports demodulated data and corresponding timestamps to a .mat file format.
        Provides a dialog for the user to select the save location and file name.
        It validates the presence of data before allowing the export and handles errors
        during the process.

        Raises a warning if the necessary data is unavailable, or if there is an error
        in saving the file.

        Parameters
        ----------
        self : object
            Instance of the class that holds 'demod_data' and 'demod_time_ns' attributes.

        Raises
        ------
        Exception
            If any error occurs during saving the .mat file, an error message box is
            displayed containing details of the error.
        """
        if self.demod_data is None or self.demod_time_ns is None:
            QMessageBox.warning(self, "Error", "No demodulated data.")
            return

        file_dialog = QFileDialog(self, "Save as .mat", ".", "MAT files (*.mat)")
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setDefaultSuffix("mat")

        if file_dialog.exec():
            save_path = file_dialog.selectedFiles()[0]
        else:
            return

        if not save_path:
            return

        try:
            savemat(save_path, {
                'time_ns': self.demod_time_ns,
                'demod_data': self.demod_data
            })
            QMessageBox.information(self, "Export", f"Saved:\n{save_path}")
        except Exception as e:
            QMessageBox.warning(self, "Error saving", f"Could not save:\n{e}")

    def export_data_to_h5(self):
        """
        Exports demodulated data and time information into an HDF5 file (.h5) format. The function
        checks if the required data attributes are available before initiating the save process. It
        uses a file dialog to allow the user to select the save location. The demodulated data is
        stored separately in real and imaginary components within the HDF5 file. Upon successful
        save, a confirmation message is shown to the user, and in case of an error, an appropriate
        warning dialog is displayed.

        Raises:
            Exception: Raised if any error occurs during the saving process, with the error
            details shown in a warning dialog.
        """
        if self.demod_data is None or self.demod_time_ns is None:
            QMessageBox.warning(self, "Error", "No demodulated data.")
            return

        file_dialog = QFileDialog(self, "Save as .h5", ".", "HDF5 files (*.h5)")
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setDefaultSuffix("h5")

        if file_dialog.exec():
            save_path = file_dialog.selectedFiles()[0]
        else:
            return

        if not save_path:
            return

        try:
            with h5py.File(save_path, "w") as h5f:
                h5f.create_dataset("time_ns", data=self.demod_time_ns)
                h5f.create_dataset("demod_data_real", data=np.real(self.demod_data))
                h5f.create_dataset("demod_data_imag", data=np.imag(self.demod_data))

            QMessageBox.information(self, "Export", f"Saved:\n{save_path}")
        except Exception as e:
            QMessageBox.warning(self, "Error saving", f"Could not save:\n{e}")

    def export_original_spectrum(self):
        """
        Exports the original spectrum data derived from the loaded CSV file and saves it as a .mat file after processing.

        This function performs the following operations:
        1. Validates that a CSV file has been loaded, a valid time interval is specified, and channels are selected.
        2. Retrieves and processes the selected time slice of data by applying a bandpass filter if needed.
        3. Performs a Fourier transform on the data to compute the frequency spectrum for the selected channels.
        4. Allows the user to save the spectrum data, including frequency and transformed values, in a .mat file.

        Raises informative errors, using warning dialogs, in case prerequisites are not met or issues occur during the saving process.

        Attributes
        ----------
        csv_data : numpy.ndarray or None
            The CSV data loaded into the application.

        sampling_rate_ns : float
            The sampling rate in nanoseconds used for frequency calculations.

        fc_input : QLineEdit
            Widget for user input to specify a frequency in MHz for shifting the spectrum.

        remove_neg_cb : QCheckBox
            Checkbox to determine if negative frequencies should be removed from the spectrum.

        Methods
        -------
        get_time_slice_indices() -> Tuple[int, int] or Tuple[None, None]
            Retrieves the start and end indices that define the time slice of data to analyze.

        get_selected_channels() -> List[int]
            Returns a list of selected channel indices.

        apply_bandpass_if_needed(data: numpy.ndarray, dt_s: float) -> numpy.ndarray
            Applies a bandpass filter to the given data if required.

        Raises
        ------
        QMessageBox.warning
            Raised when one of the validation checks fails or when an error occurs while saving the .mat file.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.csv_data is None:
            QMessageBox.warning(self, "Error", "No CSV loaded.")
            return

        start_idx, end_idx = self.get_time_slice_indices()
        if start_idx is None or end_idx is None:
            QMessageBox.warning(self, "Error", "Invalid time interval.")
            return

        selected_channels = self.get_selected_channels()
        if not selected_channels:
            QMessageBox.warning(self, "Error", "No channels selected.")
            return

        try:
            frequency_mhz = float(self.fc_input.text())
        except ValueError:
            frequency_mhz = 0.0

        cropped_data = self.csv_data[start_idx:end_idx, :]
        cropped_data_v = cropped_data / 32767.0 * 5.0
        dt_s = self.sampling_rate_ns * 1e-9
        data_for_fft = self.apply_bandpass_if_needed(cropped_data_v, dt_s)

        N = data_for_fft.shape[0]
        freq_axis = np.fft.fftfreq(N, d=dt_s) / 1e6  # в МГц
        freq_axis_shifted = freq_axis - frequency_mhz

        if self.remove_neg_cb.isChecked():
            mask = freq_axis_shifted >= 0
            freq_axis_shifted = freq_axis_shifted[mask]

        spectrum = {}
        for idx, ch in enumerate(selected_channels):
            Y = np.fft.fft(data_for_fft[:, ch])
            if self.remove_neg_cb.isChecked():
                Y = Y[mask]
            channel_label = f"Channel_{string.ascii_uppercase[ch]}" if ch < 26 else f"Channel_CH{ch}"
            spectrum[channel_label] = np.abs(Y)

        file_dialog = QFileDialog(self, "Save Original Spectrum as .mat", ".", "MAT files (*.mat)")
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setDefaultSuffix("mat")

        if file_dialog.exec():
            save_path = file_dialog.selectedFiles()[0]
        else:
            return

        if not save_path:
            return

        try:
            savemat(save_path, {
                'frequency_mhz': freq_axis_shifted,
                **spectrum
            })
            QMessageBox.information(self, "Export", f"Original Spectrum Saved:\n{save_path}")
        except Exception as e:
            QMessageBox.warning(self, "Error saving", f"Could not save:\n{e}")

    def export_demod_spectrum(self):
        """
        Exports the demodulated spectrum data to a .mat file. This method retrieves
        demodulation data and processes it into a spectral form through FFT (Fast Fourier
        """
        if self.demod_data is None or self.demod_time_ns is None:
            QMessageBox.warning(self, "Error", "No demodulated data. Please do demodulation first.")
            return

        selected_channels = self.get_selected_channels()
        if not selected_channels:
            QMessageBox.warning(self, "Error", "No channels selected.")
            return

        try:
            freq_min = float(self.freq_min_input.text())
            freq_max = float(self.freq_max_input.text())
        except ValueError:
            freq_min, freq_max = None, None

        N = self.demod_data.shape[0]
        try:
            dec_factor = int(self.decimation_factor_input.text())
            if dec_factor < 1:
                dec_factor = 1
        except:
            dec_factor = 1
        dt_s = self.sampling_rate_ns * 1e-9 * dec_factor
        freq_axis = np.fft.fftfreq(N, d=dt_s) / 1e6  # в МГц

        if self.remove_neg_cb.isChecked():
            mask = freq_axis >= 0
            freq_axis = freq_axis[mask]

        spectrum = {}
        for idx, ch in enumerate(selected_channels):
            Y = np.fft.fft(self.demod_data[:, idx])
            if self.remove_neg_cb.isChecked():
                Y = Y[mask]
            channel_label = f"Channel_{string.ascii_uppercase[ch]}" if ch < 26 else f"Channel_CH{ch}"
            spectrum[channel_label] = np.abs(Y)

        file_dialog = QFileDialog(self, "Save Demod Spectrum as .mat", ".", "MAT files (*.mat)")
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setDefaultSuffix("mat")

        if file_dialog.exec():
            save_path = file_dialog.selectedFiles()[0]
        else:
            return

        if not save_path:
            return

        try:
            savemat(save_path, {
                'frequency_mhz': freq_axis,
                **spectrum
            })
            QMessageBox.information(self, "Export", f"Demod Spectrum Saved:\n{save_path}")
        except Exception as e:
            QMessageBox.warning(self, "Error saving", f"Could not save:\n{e}")


def main():
    """
    This function initializes a PyQt application. It creates an instance of the QApplication class,
    constructs the main window, displays it to the user, and manages the event loop termination.

    Raises:
        SystemExit: Raised when the PyQt application's event loop finishes and the program needs
        to exit.
    """
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
