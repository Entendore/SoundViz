import sys
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                QHBoxLayout, QPushButton, QLabel, QSlider, QComboBox,
                                QCheckBox, QGroupBox, QSplitter, QFileDialog, QMessageBox,
                                QStatusBar, QToolBar, QSizePolicy)
from PySide6.QtCore import Qt, QTimer, QElapsedTimer
from PySide6.QtGui import QAction, QIcon, QFont, QColor
import pyqtgraph as pg
from scipy import signal as scipy_signal

# ==================== Configuration ====================
pg.setConfigOptions(antialias=True, useOpenGL=False, leftButtonPan=True)

# Color themes
DARK_THEME = {
    'bg': '#1a1a1a', 'fg': '#e0e0e0', 'grid': '#404040',
    'curve': '#00aaff', 'fft': '#ff5500', 'spectrogram': 'inferno',
    'region': '#88ff88', 'crosshair': '#ffff00', 'peak': '#ff00ff'
}

LIGHT_THEME = {
    'bg': '#ffffff', 'fg': '#000000', 'grid': '#cccccc',
    'curve': '#0066cc', 'fft': '#cc3300', 'spectrogram': 'viridis',
    'region': '#22aa22', 'crosshair': '#000000', 'peak': '#cc00cc'
}

class SignalGenerator:
    """Handles continuous signal generation with phase coherence"""
    def __init__(self):
        self.fs = 2000  # Sampling frequency (Hz)
        self.duration = 2.0  # Signal duration (seconds)
        self.n_samples = int(self.fs * self.duration)
        self.time = np.linspace(0, self.duration, self.n_samples, endpoint=False)
        
        self.freq1 = 50
        self.freq2 = 120
        self.amp1 = 1.0
        self.amp2 = 0.5
        self.noise_level = 0.1
        self.signal_type = "Sine Wave"
        
        # Streaming state
        self.stream_phase = 0.0
        
    def generate(self):
        """Generate the selected signal for static display"""
        t = self.time
        sig = self._generate_wave(t)
        if self.noise_level > 0:
            sig += np.random.normal(0, self.noise_level, len(sig))
        return sig
    
    def generate_stream_chunk(self, n_new):
        """Generate a continuous chunk for streaming"""
        t = np.linspace(self.stream_phase, self.stream_phase + n_new/self.fs, n_new, endpoint=False)
        self.stream_phase += n_new / self.fs
        
        sig = self._generate_wave(t)
        if self.noise_level > 0:
            sig += np.random.normal(0, self.noise_level, len(sig))
        return sig
        
    def _generate_wave(self, t):
        """Internal wave generation logic"""
        if self.signal_type == "Sine Wave":
            return self.amp1 * np.sin(2 * np.pi * self.freq1 * t)
        elif self.signal_type == "Square Wave":
            return self.amp1 * scipy_signal.square(2 * np.pi * self.freq1 * t)
        elif self.signal_type == "Sawtooth":
            return self.amp1 * scipy_signal.sawtooth(2 * np.pi * self.freq1 * t)
        elif self.signal_type == "Mixed Frequencies":
            return (self.amp1 * np.sin(2 * np.pi * self.freq1 * t) + 
                   self.amp2 * np.sin(2 * np.pi * self.freq2 * t))
        elif self.signal_type == "Chirp":
            return scipy_signal.chirp(t, f0=self.freq1, f1=self.freq1*5, 
                                     t=len(t)/self.fs, method='linear')
        elif self.signal_type == "AM Modulated":
            carrier = np.sin(2 * np.pi * self.freq1 * t)
            modulator = 0.5 * (1 + 0.5 * np.sin(2 * np.pi * self.freq2 * t))
            return self.amp1 * modulator * carrier
        return np.zeros_like(t)

class MeasurementsCalculator:
    """Calculate signal measurements: THD, SNR, peak frequency, RMS"""
    
    @staticmethod
    def calculate_thd(signal_data, fs, fundamental_freq):
        n = len(signal_data)
        fft_result = np.fft.rfft(signal_data)
        fft_mag = np.abs(fft_result) / n
        freqs = np.fft.rfftfreq(n, 1/fs)
        
        fundamental_idx = np.argmin(np.abs(freqs - fundamental_freq))
        harmonics = []
        for h in range(2, 6):
            h_freq = fundamental_freq * h
            if h_freq < fs/2:
                h_idx = np.argmin(np.abs(freqs - h_freq))
                harmonics.append(fft_mag[h_idx])
        
        if harmonics and fft_mag[fundamental_idx] > 0:
            thd = np.sqrt(np.sum(np.array(harmonics)**2)) / fft_mag[fundamental_idx] * 100
            return thd
        return 0.0
    
    @staticmethod
    def calculate_snr(signal_data, fs, signal_freq):
        n = len(signal_data)
        fft_result = np.fft.rfft(signal_data)
        fft_mag = np.abs(fft_result) / n
        freqs = np.fft.rfftfreq(n, 1/fs)
        
        signal_idx = np.argmin(np.abs(freqs - signal_freq))
        signal_power = fft_mag[signal_idx]**2
        
        noise_mask = np.ones(len(fft_mag), dtype=bool)
        signal_region = 10
        noise_mask[max(0, signal_idx-signal_region):min(len(fft_mag), 
                                                          signal_idx+signal_region)] = False
        noise_power = np.mean(fft_mag[noise_mask]**2)
        
        if noise_power > 0:
            return 10 * np.log10(signal_power / noise_power)
        return float('inf')
    
    @staticmethod
    def find_peaks(fft_mag, frequencies, n_peaks=3, min_dist=10):
        """Find top N peaks in FFT"""
        # Copy magnitude to avoid modifying original
        mag = fft_mag.copy()
        peaks = []
        
        for _ in range(n_peaks):
            idx = np.argmax(mag)
            if mag[idx] <= 0:
                break
                
            peaks.append((frequencies[idx], mag[idx]))
            
            # Zero out surrounding area to find next peak
            start = max(0, idx - min_dist)
            end = min(len(mag), idx + min_dist)
            mag[start:end] = 0
            
        return peaks

class SignalAnalysisDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Signal Analysis Dashboard Pro")
        self.setGeometry(50, 50, 1400, 900)
        
        # Initialize components
        self.generator = SignalGenerator()
        self.signal_data = None
        self.theme = DARK_THEME
        self.is_streaming = False
        self.stream_buffer_size = 4000
        
        # DSP Filter settings
        self.filter_type = "None"
        self.filter_cutoff = 100
        self.filter_order = 4
        
        # Performance tracking
        self.fps_timer = QElapsedTimer()
        self.fps_timer.start()
        self.frame_count = 0
        
        # Create UI
        self.create_actions()
        self.create_toolbar()
        self.create_central_widget()
        self.create_statusbar()
        
        # Apply theme
        self.apply_theme()
        
        # Initialize plot items for performance
        self.init_plot_items()
        
        # Generate initial signal
        self.update_signal()
        
        # Real-time streaming timer
        self.stream_timer = QTimer()
        self.stream_timer.timeout.connect(self.update_streaming_data)
        
    def create_actions(self):
        """Create menu actions"""
        self.export_csv_action = QAction("Export CSV", self)
        self.export_csv_action.setShortcut("Ctrl+E")
        self.export_csv_action.triggered.connect(self.export_csv)
        
        self.export_png_action = QAction("Export PNG", self)
        self.export_png_action.setShortcut("Ctrl+P")
        self.export_png_action.triggered.connect(self.export_png)
        
        self.toggle_theme_action = QAction("Toggle Theme", self)
        self.toggle_theme_action.setShortcut("Ctrl+T")
        self.toggle_theme_action.triggered.connect(self.toggle_theme)
        
        self.start_stream_action = QAction("Start Stream", self)
        self.start_stream_action.setShortcut("Space")
        self.start_stream_action.triggered.connect(self.toggle_streaming)
        
        self.reset_action = QAction("Reset", self)
        self.reset_action.setShortcut("Ctrl+R")
        self.reset_action.triggered.connect(self.reset_view)
        
    def create_toolbar(self):
        """Create main toolbar"""
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        
        toolbar.addAction(self.start_stream_action)
        toolbar.addAction(self.reset_action)
        toolbar.addSeparator()
        toolbar.addAction(self.export_csv_action)
        toolbar.addAction(self.export_png_action)
        toolbar.addSeparator()
        toolbar.addAction(self.toggle_theme_action)
        
    def create_central_widget(self):
        """Create the central widget with splitter layout"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        splitter = QSplitter(Qt.Horizontal)
        
        control_panel = self.create_control_panel()
        splitter.addWidget(control_panel)
        
        plot_panel = self.create_plot_panel()
        splitter.addWidget(plot_panel)
        
        measurements_panel = self.create_measurements_panel()
        splitter.addWidget(measurements_panel)
        
        splitter.setSizes([250, 900, 250])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        
        main_layout.addWidget(splitter)
        
    def create_control_panel(self):
        """Create the control panel with all signal parameters"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # Signal Type Group
        signal_group = QGroupBox("Signal Configuration")
        signal_layout = QVBoxLayout(signal_group)
        
        signal_layout.addWidget(QLabel("Signal Type:"))
        self.signal_combo = QComboBox()
        self.signal_combo.addItems([
            "Sine Wave", "Square Wave", "Sawtooth", 
            "Mixed Frequencies", "Chirp", "AM Modulated"
        ])
        self.signal_combo.setToolTip("Select the type of waveform to generate")
        self.signal_combo.currentTextChanged.connect(self.on_signal_type_changed)
        signal_layout.addWidget(self.signal_combo)
        
        # Frequency 1
        self.freq1_label = QLabel(f"Frequency 1: {self.generator.freq1} Hz")
        signal_layout.addWidget(self.freq1_label)
        self.freq1_slider = QSlider(Qt.Horizontal)
        self.freq1_slider.setRange(1, 500)
        self.freq1_slider.setValue(self.generator.freq1)
        self.freq1_slider.setToolTip("Set the primary frequency")
        self.freq1_slider.valueChanged.connect(self.on_freq1_changed)
        signal_layout.addWidget(self.freq1_slider)
        
        # Frequency 2
        self.freq2_label = QLabel(f"Frequency 2: {self.generator.freq2} Hz")
        signal_layout.addWidget(self.freq2_label)
        self.freq2_slider = QSlider(Qt.Horizontal)
        self.freq2_slider.setRange(1, 500)
        self.freq2_slider.setValue(self.generator.freq2)
        self.freq2_slider.setToolTip("Set the secondary frequency (used in Mixed/AM)")
        self.freq2_slider.valueChanged.connect(self.on_freq2_changed)
        signal_layout.addWidget(self.freq2_slider)
        
        # Amplitude 1
        self.amp1_label = QLabel(f"Amplitude 1: {self.generator.amp1:.1f}")
        signal_layout.addWidget(self.amp1_label)
        self.amp1_slider = QSlider(Qt.Horizontal)
        self.amp1_slider.setRange(1, 50)
        self.amp1_slider.setValue(int(self.generator.amp1 * 10))
        self.amp1_slider.valueChanged.connect(self.on_amp1_changed)
        signal_layout.addWidget(self.amp1_slider)
        
        # Amplitude 2
        self.amp2_label = QLabel(f"Amplitude 2: {self.generator.amp2:.1f}")
        signal_layout.addWidget(self.amp2_label)
        self.amp2_slider = QSlider(Qt.Horizontal)
        self.amp2_slider.setRange(1, 50)
        self.amp2_slider.setValue(int(self.generator.amp2 * 10))
        self.amp2_slider.valueChanged.connect(self.on_amp2_changed)
        signal_layout.addWidget(self.amp2_slider)
        
        # Noise Level
        self.noise_label = QLabel(f"Noise Level: {self.generator.noise_level:.2f}")
        signal_layout.addWidget(self.noise_label)
        self.noise_slider = QSlider(Qt.Horizontal)
        self.noise_slider.setRange(0, 50)
        self.noise_slider.setValue(int(self.generator.noise_level * 10))
        self.noise_slider.valueChanged.connect(self.on_noise_changed)
        signal_layout.addWidget(self.noise_slider)
        
        layout.addWidget(signal_group)
        
        # DSP Filters Group (NEW)
        filter_group = QGroupBox("Digital Filters (DSP)")
        filter_layout = QVBoxLayout(filter_group)
        
        filter_layout.addWidget(QLabel("Filter Type:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["None", "Lowpass", "Highpass", "Bandpass"])
        self.filter_combo.setToolTip("Apply a Butterworth filter to the signal")
        self.filter_combo.currentTextChanged.connect(self.on_filter_changed)
        filter_layout.addWidget(self.filter_combo)
        
        self.filter_cutoff_label = QLabel(f"Cutoff: {self.filter_cutoff} Hz")
        filter_layout.addWidget(self.filter_cutoff_label)
        self.filter_slider = QSlider(Qt.Horizontal)
        self.filter_slider.setRange(1, 500)
        self.filter_slider.setValue(self.filter_cutoff)
        self.filter_slider.setToolTip("Set the filter cutoff frequency")
        self.filter_slider.valueChanged.connect(self.on_filter_cutoff_changed)
        filter_layout.addWidget(self.filter_slider)
        
        layout.addWidget(filter_group)
        
        # Analysis Settings Group
        analysis_group = QGroupBox("Analysis Settings")
        analysis_layout = QVBoxLayout(analysis_group)
        
        analysis_layout.addWidget(QLabel("FFT Window Size:"))
        self.fft_combo = QComboBox()
        self.fft_combo.addItems(["256", "512", "1024", "2048", "4096", "8192"])
        self.fft_combo.setCurrentText("1024")
        self.fft_combo.currentTextChanged.connect(self.update_frequency_plot)
        analysis_layout.addWidget(self.fft_combo)
        
        analysis_layout.addWidget(QLabel("Window Function:"))
        self.window_combo = QComboBox()
        self.window_combo.addItems(["Rectangular", "Hann", "Hamming", "Blackman", "Flattop"])
        self.window_combo.currentTextChanged.connect(self.update_frequency_plot)
        analysis_layout.addWidget(self.window_combo)
        
        self.log_fft_check = QCheckBox("Log Scale (dB)")
        self.log_fft_check.stateChanged.connect(self.update_frequency_plot)
        analysis_layout.addWidget(self.log_fft_check)
        
        self.peak_detect_check = QCheckBox("Auto Peak Detection")
        self.peak_detect_check.setChecked(True)
        self.peak_detect_check.stateChanged.connect(self.update_frequency_plot)
        analysis_layout.addWidget(self.peak_detect_check)
        
        self.spectrogram_check = QCheckBox("Show Spectrogram")
        self.spectrogram_check.setChecked(True)
        self.spectrogram_check.stateChanged.connect(self.toggle_spectrogram)
        analysis_layout.addWidget(self.spectrogram_check)
        
        layout.addWidget(analysis_group)
        
        layout.addStretch()
        
        self.update_button = QPushButton("Generate Signal")
        self.update_button.clicked.connect(self.update_signal)
        layout.addWidget(self.update_button)
        
        return panel
    
    def create_plot_panel(self):
        """Create the plot panel with time, frequency, and spectrogram"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.graphics_layout = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics_layout)
        
        # Time Domain Plot
        self.time_plot = self.graphics_layout.addPlot(row=0, col=0, title="Time Domain")
        self.time_plot.setLabel('left', 'Amplitude')
        self.time_plot.setLabel('bottom', 'Time', units='s')
        self.time_plot.showGrid(x=True, y=True, alpha=0.3)
        
        # Frequency Domain Plot
        self.freq_plot = self.graphics_layout.addPlot(row=1, col=0, title="Frequency Domain")
        self.freq_plot.setLabel('left', 'Magnitude')
        self.freq_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.freq_plot.showGrid(x=True, y=True, alpha=0.3)
        
        # Spectrogram Plot
        self.spectrogram_plot = self.graphics_layout.addPlot(row=2, col=0, title="Spectrogram")
        self.spectrogram_plot.setLabel('left', 'Frequency', units='Hz')
        self.spectrogram_plot.setLabel('bottom', 'Time', units='s')
        
        self.spectrogram_image = pg.ImageItem()
        self.spectrogram_plot.addItem(self.spectrogram_image)
        
        pos = np.linspace(0.0, 1.0, num=256)
        color = pg.colormap.get('inferno').getColors(mode='float')
        self.spectrogram_image.setColorMap(pg.ColorMap(pos, color))
        
        # Region of interest for time selection
        self.region = pg.LinearRegionItem()
        self.region.setZValue(10)
        self.time_plot.addItem(self.region)
        self.region.setRegion([0.3, 1.2])
        self.region.sigRegionChanged.connect(self.update_frequency_plot)
        
        # Crosshair lines
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.time_plot.addItem(self.vLine, ignoreBounds=True)
        self.time_plot.addItem(self.hLine, ignoreBounds=True)
        
        self.proxy = pg.SignalProxy(self.time_plot.scene().sigMouseMoved, 
                                    rateLimit=60, slot=self.mouse_moved)
        
        return panel
    
    def init_plot_items(self):
        """Initialize persistent plot items for high performance updates"""
        # Time domain curve
        self.time_curve = self.time_plot.plot(pen=pg.mkPen(color=self.theme['curve'], width=1))
        
        # Frequency domain curve
        self.freq_curve = self.freq_plot.plot(pen=pg.mkPen(color=self.theme['fft'], width=1))
        
        # Peak markers (scatter plot)
        self.peak_scatter = pg.ScatterPlotItem(
            size=10, pen=pg.mkPen('w'), brush=pg.mkBrush(self.theme['peak'])
        )
        self.freq_plot.addItem(self.peak_scatter)
        
        # Peak labels
        self.peak_labels = []
        
    def create_measurements_panel(self):
        """Create the measurements panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        
        meas_group = QGroupBox("Measurements")
        meas_layout = QVBoxLayout(meas_group)
        
        self.peak_freq_label = QLabel("Peak Freq: --- Hz")
        self.peak_freq_label.setFont(QFont("Consolas", 10))
        meas_layout.addWidget(self.peak_freq_label)
        
        self.peak_mag_label = QLabel("Peak Mag: ---")
        self.peak_mag_label.setFont(QFont("Consolas", 10))
        meas_layout.addWidget(self.peak_mag_label)
        
        self.rms_label = QLabel("RMS: ---")
        self.rms_label.setFont(QFont("Consolas", 10))
        meas_layout.addWidget(self.rms_label)
        
        self.thd_label = QLabel("THD: --- %")
        self.thd_label.setFont(QFont("Consolas", 10))
        meas_layout.addWidget(self.thd_label)
        
        self.snr_label = QLabel("SNR: --- dB")
        self.snr_label.setFont(QFont("Consolas", 10))
        meas_layout.addWidget(self.snr_label)
        
        meas_layout.addWidget(QLabel(""))
        meas_layout.addWidget(QLabel("Signal Info:"))
        self.samples_label = QLabel(f"Samples: {self.generator.n_samples}")
        self.samples_label.setFont(QFont("Consolas", 10))
        meas_layout.addWidget(self.samples_label)
        
        self.duration_label = QLabel(f"Duration: {self.generator.duration} s")
        self.duration_label.setFont(QFont("Consolas", 10))
        meas_layout.addWidget(self.duration_label)
        
        self.fs_label = QLabel(f"Sample Rate: {self.generator.fs} Hz")
        self.fs_label.setFont(QFont("Consolas", 10))
        meas_layout.addWidget(self.fs_label)
        
        layout.addWidget(meas_group)
        
        cursor_group = QGroupBox("Cursor Info")
        cursor_layout = QVBoxLayout(cursor_group)
        
        self.cursor_time_label = QLabel("Time: ---")
        self.cursor_time_label.setFont(QFont("Consolas", 10))
        cursor_layout.addWidget(self.cursor_time_label)
        
        self.cursor_value_label = QLabel("Value: ---")
        self.cursor_value_label.setFont(QFont("Consolas", 10))
        cursor_layout.addWidget(self.cursor_value_label)
        
        layout.addWidget(cursor_group)
        
        layout.addStretch()
        
        return panel
    
    def create_statusbar(self):
        """Create status bar with FPS counter"""
        self.fps_label = QLabel("FPS: 0")
        self.statusBar().addPermanentWidget(self.fps_label)
        self.statusBar().showMessage("Ready")
        
        # FPS update timer
        self.fps_update_timer = QTimer()
        self.fps_update_timer.timeout.connect(self.update_fps)
        self.fps_update_timer.start(1000)
        
    def update_fps(self):
        """Update FPS counter"""
        elapsed = self.fps_timer.elapsed()
        if elapsed > 0:
            fps = self.frame_count * 1000 / elapsed
            self.fps_label.setText(f"FPS: {fps:.1f}")
        self.fps_timer.restart()
        self.frame_count = 0
        
    def apply_theme(self):
        """Apply the current theme to the application"""
        if self.theme == DARK_THEME:
            style = """
                QMainWindow { background-color: #1a1a1a; }
                QWidget { background-color: #2a2a2a; color: #e0e0e0; }
                QGroupBox { border: 1px solid #404040; border-radius: 4px; 
                           margin-top: 1ex; padding-top: 1ex; font-weight: bold; }
                QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
                QPushButton { background-color: #3a3a3a; border: 1px solid #505050; 
                             border-radius: 3px; padding: 5px; min-width: 80px; }
                QPushButton:hover { background-color: #4a4a4a; }
                QPushButton:pressed { background-color: #2a2a2a; }
                QComboBox { background-color: #3a3a3a; border: 1px solid #505050; 
                           border-radius: 3px; padding: 3px; }
                QSlider::groove:horizontal { border: 1px solid #404040; height: 4px; 
                                            background: #2a2a2a; border-radius: 2px; }
                QSlider::handle:horizontal { background: #505050; border: 1px solid #606060; 
                                            width: 12px; margin: -5px 0; border-radius: 6px; }
                QCheckBox { spacing: 5px; }
                QToolBar { background-color: #2a2a2a; border: none; spacing: 3px; 
                          padding: 3px; }
                QStatusBar { background-color: #2a2a2a; }
                QLabel { background: transparent; }
            """
        else:
            style = """
                QMainWindow { background-color: #f0f0f0; }
                QWidget { background-color: #ffffff; color: #000000; }
                QGroupBox { border: 1px solid #cccccc; border-radius: 4px; 
                           margin-top: 1ex; padding-top: 1ex; font-weight: bold; }
                QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
                QPushButton { background-color: #e0e0e0; border: 1px solid #aaaaaa; 
                             border-radius: 3px; padding: 5px; min-width: 80px; }
                QPushButton:hover { background-color: #d0d0d0; }
                QPushButton:pressed { background-color: #c0c0c0; }
                QComboBox { background-color: #ffffff; border: 1px solid #aaaaaa; 
                           border-radius: 3px; padding: 3px; }
                QSlider::groove:horizontal { border: 1px solid #cccccc; height: 4px; 
                                            background: #f0f0f0; border-radius: 2px; }
                QSlider::handle:horizontal { background: #cccccc; border: 1px solid #999999; 
                                            width: 12px; margin: -5px 0; border-radius: 6px; }
                QCheckBox { spacing: 5px; }
                QToolBar { background-color: #e0e0e0; border: none; spacing: 3px; 
                          padding: 3px; }
                QStatusBar { background-color: #e0e0e0; }
                QLabel { background: transparent; }
            """
        
        self.setStyleSheet(style)
        
        for plot in [self.time_plot, self.freq_plot, self.spectrogram_plot]:
            plot.setBackground(self.theme['bg'])
        
        if self.theme == DARK_THEME:
            color = pg.colormap.get('inferno').getColors(mode='float')
        else:
            color = pg.colormap.get('viridis').getColors(mode='float')
        
        pos = np.linspace(0.0, 1.0, num=256)
        self.spectrogram_image.setColorMap(pg.ColorMap(pos, color))
        
        # Update curve colors
        if hasattr(self, 'time_curve'):
            self.time_curve.setPen(pg.mkPen(color=self.theme['curve'], width=1))
            self.freq_curve.setPen(pg.mkPen(color=self.theme['fft'], width=1))
            self.peak_scatter.setBrush(pg.mkBrush(self.theme['peak']))
    
    # ==================== Event Handlers ====================
    
    def on_signal_type_changed(self, text):
        self.generator.signal_type = text
        self.update_signal()
    
    def on_freq1_changed(self, value):
        self.generator.freq1 = value
        self.freq1_label.setText(f"Frequency 1: {value} Hz")
        self.update_signal()
    
    def on_freq2_changed(self, value):
        self.generator.freq2 = value
        self.freq2_label.setText(f"Frequency 2: {value} Hz")
        self.update_signal()
    
    def on_amp1_changed(self, value):
        self.generator.amp1 = value / 10.0
        self.amp1_label.setText(f"Amplitude 1: {self.generator.amp1:.1f}")
        self.update_signal()
    
    def on_amp2_changed(self, value):
        self.generator.amp2 = value / 10.0
        self.amp2_label.setText(f"Amplitude 2: {self.generator.amp2:.1f}")
        self.update_signal()
    
    def on_noise_changed(self, value):
        self.generator.noise_level = value / 10.0
        self.noise_label.setText(f"Noise Level: {self.generator.noise_level:.2f}")
        self.update_signal()
    
    def on_filter_changed(self, text):
        self.filter_type = text
        self.update_signal()
    
    def on_filter_cutoff_changed(self, value):
        self.filter_cutoff = value
        self.filter_cutoff_label.setText(f"Cutoff: {value} Hz")
        self.update_signal()
    
    def apply_filter(self, data):
        """Apply the selected digital filter to the data"""
        if self.filter_type == "None":
            return data
            
        nyq = 0.5 * self.generator.fs
        normal_cutoff = self.filter_cutoff / nyq
        
        try:
            if self.filter_type == "Lowpass":
                b, a = scipy_signal.butter(self.filter_order, normal_cutoff, btype='low', analog=False)
                filtered = scipy_signal.filtfilt(b, a, data)
            elif self.filter_type == "Highpass":
                b, a = scipy_signal.butter(self.filter_order, normal_cutoff, btype='high', analog=False)
                filtered = scipy_signal.filtfilt(b, a, data)
            elif self.filter_type == "Bandpass":
                # For bandpass, we need a range. Let's use cutoff to cutoff*2
                low = normal_cutoff
                high = min(normal_cutoff * 2, 0.99)
                b, a = scipy_signal.butter(self.filter_order, [low, high], btype='band', analog=False)
                filtered = scipy_signal.filtfilt(b, a, data)
            else:
                filtered = data
            return filtered
        except Exception as e:
            print(f"Filter error: {e}")
            return data
    
    def update_signal(self):
        """Generate and display new signal with high performance updates"""
        if self.is_streaming:
            return
            
        self.signal_data = self.generator.generate()
        self.signal_data = self.apply_filter(self.signal_data)
        
        # High performance update using setData
        self.time_curve.setData(self.generator.time, self.signal_data)
        
        self.update_frequency_plot()
        self.update_spectrogram()
        self.update_measurements()
        
        self.statusBar().showMessage("Signal generated", 2000)
        self.frame_count += 1
    
    def update_frequency_plot(self):
        """Update frequency domain plot with automatic peak detection"""
        if self.signal_data is None:
            return
            
        region = self.region.getRegion()
        t_start, t_end = region
        
        time = self.generator.time
        start_idx = np.argmin(np.abs(time - t_start))
        end_idx = np.argmin(np.abs(time - t_end))
        
        if end_idx <= start_idx:
            return
            
        segment = self.signal_data[start_idx:end_idx]
        
        fft_size_str = self.fft_combo.currentText()
        fft_size = min(int(fft_size_str), len(segment))
        
        window_name = self.window_combo.currentText()
        if window_name == "Rectangular":
            window = np.ones(fft_size)
        elif window_name == "Hann":
            window = scipy_signal.windows.hann(fft_size)
        elif window_name == "Hamming":
            window = scipy_signal.windows.hamming(fft_size)
        elif window_name == "Blackman":
            window = scipy_signal.windows.blackman(fft_size)
        elif window_name == "Flattop":
            window = scipy_signal.windows.flattop(fft_size)
        else:
            window = np.ones(fft_size)
        
        windowed_segment = segment[:fft_size] * window
        fft_result = np.fft.rfft(windowed_segment)
        fft_mag = np.abs(fft_result) / (fft_size / 2)
        
        if self.log_fft_check.isChecked():
            fft_mag = 20 * np.log10(fft_mag + 1e-10)
        
        frequencies = np.fft.rfftfreq(fft_size, 1/self.generator.fs)
        
        # High performance update
        self.freq_curve.setData(frequencies, fft_mag)
        
        # Automatic peak detection
        if self.peak_detect_check.isChecked():
            peaks = MeasurementsCalculator.find_peaks(fft_mag, frequencies, n_peaks=3)
            
            # Clear old labels
            for label in self.peak_labels:
                self.freq_plot.removeItem(label)
            self.peak_labels = []
            
            # Update scatter points
            peak_points = [{'pos': (f, m), 'data': 1} for f, m in peaks]
            self.peak_scatter.setData(peak_points)
            
            # Add frequency labels
            for i, (freq, mag) in enumerate(peaks):
                label = pg.TextItem(f"{freq:.1f} Hz", color=self.theme['peak'], anchor=(0, 1))
                label.setPos(freq, mag)
                self.freq_plot.addItem(label)
                self.peak_labels.append(label)
        else:
            self.peak_scatter.setData([])
            for label in self.peak_labels:
                self.freq_plot.removeItem(label)
            self.peak_labels = []
        
        # Update labels
        if self.log_fft_check.isChecked():
            self.freq_plot.setLabel('left', 'Magnitude (dB)')
        else:
            self.freq_plot.setLabel('left', 'Magnitude')
    
    def update_spectrogram(self):
        """Update spectrogram display"""
        if self.signal_data is None:
            return
            
        nperseg = min(256, len(self.signal_data))
        f, t, Sxx = scipy_signal.spectrogram(
            self.signal_data, 
            fs=self.generator.fs,
            nperseg=nperseg,
            noverlap=nperseg//2
        )
        
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        self.spectrogram_image.setImage(Sxx_db)
        self.spectrogram_image.setRect(0, 0, t[-1], f[-1])
    
    def update_measurements(self):
        """Update measurements display"""
        if self.signal_data is None:
            return
            
        peak_freq, peak_mag = MeasurementsCalculator.find_peak_frequency(
            self.signal_data, self.generator.fs
        )
        
        rms = MeasurementsCalculator.calculate_rms(self.signal_data)
        
        if self.generator.signal_type in ["Sine Wave", "Square Wave", "Sawtooth", "Chirp"]:
            fundamental = self.generator.freq1
        elif self.generator.signal_type == "Mixed Frequencies":
            fundamental = self.generator.freq1
        else:
            fundamental = peak_freq
        
        thd = MeasurementsCalculator.calculate_thd(
            self.signal_data, self.generator.fs, fundamental
        )
        
        snr = MeasurementsCalculator.calculate_snr(
            self.signal_data, self.generator.fs, fundamental
        )
        
        self.peak_freq_label.setText(f"Peak Freq: {peak_freq:.2f} Hz")
        self.peak_mag_label.setText(f"Peak Mag: {peak_mag:.4f}")
        self.rms_label.setText(f"RMS: {rms:.4f}")
        self.thd_label.setText(f"THD: {thd:.2f} %")
        self.snr_label.setText(f"SNR: {snr:.2f} dB")
    
    def toggle_streaming(self):
        """Toggle real-time streaming mode"""
        if self.is_streaming:
            self.stream_timer.stop()
            self.is_streaming = False
            self.start_stream_action.setText("Start Stream")
            self.statusBar().showMessage("Streaming stopped", 2000)
        else:
            # Initialize streaming buffer
            self.stream_buffer = np.zeros(self.stream_buffer_size)
            self.stream_time = np.linspace(0, self.stream_buffer_size/self.generator.fs, 
                                           self.stream_buffer_size)
            self.stream_ptr = 0
            self.generator.stream_phase = 0.0
            self.stream_timer.start(50)  # 20 FPS
            self.is_streaming = True
            self.start_stream_action.setText("Stop Stream")
            self.statusBar().showMessage("Streaming started", 2000)
    
    def update_streaming_data(self):
        """Update streaming data with continuous phase"""
        n_new = 100
        new_data = self.generator.generate_stream_chunk(n_new)
        new_data = self.apply_filter(new_data)
        
        # Update buffer
        self.stream_buffer = np.roll(self.stream_buffer, -n_new)
        self.stream_buffer[-n_new:] = new_data
        
        # High performance update
        self.time_curve.setData(self.stream_time, self.stream_buffer)
        
        # Update FFT
        fft_size = min(1024, len(self.stream_buffer))
        window = scipy_signal.windows.hann(fft_size)
        windowed = self.stream_buffer[:fft_size] * window
        fft_result = np.fft.rfft(windowed)
        fft_mag = np.abs(fft_result) / (fft_size / 2)
        frequencies = np.fft.rfftfreq(fft_size, 1/self.generator.fs)
        
        self.freq_curve.setData(frequencies, fft_mag)
        
        self.frame_count += 1
    
    def toggle_theme(self):
        """Toggle between dark and light themes"""
        if self.theme == DARK_THEME:
            self.theme = LIGHT_THEME
        else:
            self.theme = DARK_THEME
        self.apply_theme()
        self.update_signal()
        self.statusBar().showMessage("Theme changed", 2000)
    
    def toggle_spectrogram(self):
        """Toggle spectrogram visibility"""
        self.spectrogram_plot.setVisible(self.spectrogram_check.isChecked())
    
    def mouse_moved(self, evt):
        """Handle mouse movement in time plot"""
        if self.signal_data is None:
            return
            
        pos = evt[0]
        if self.time_plot.sceneBoundingRect().contains(pos):
            mouse_point = self.time_plot.getViewBox().mapSceneToView(pos)
            x = mouse_point.x()
            y = mouse_point.y()
            
            self.vLine.setPos(x)
            self.hLine.setPos(y)
            
            self.cursor_time_label.setText(f"Time: {x:.4f} s")
            self.cursor_value_label.setText(f"Value: {y:.4f}")
    
    def reset_view(self):
        """Reset plot views"""
        self.time_plot.autoRange()
        self.freq_plot.autoRange()
        self.spectrogram_plot.autoRange()
        self.region.setRegion([0.3, 1.2])
        self.statusBar().showMessage("View reset", 2000)
    
    def export_csv(self):
        """Export signal data to CSV"""
        if self.signal_data is None:
            QMessageBox.warning(self, "Export", "No data to export")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                data = np.column_stack((self.generator.time, self.signal_data))
                np.savetxt(file_path, data, delimiter=',', 
                          header='Time,Amplitude', comments='')
                self.statusBar().showMessage(f"Exported to {file_path}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")
    
    def export_png(self):
        """Export plots to PNG"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export PNG", "", "PNG Files (*.png);;All Files (*)"
        )
        
        if file_path:
            try:
                self.graphics_layout.grab().save(file_path)
                self.statusBar().showMessage(f"Exported to {file_path}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_F5:
            self.update_signal()
        elif event.key() == Qt.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 9))
    
    window = SignalAnalysisDashboard()
    window.show()
    
    sys.exit(app.exec())