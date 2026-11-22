import sys
import numpy as np
import librosa
import librosa.display
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QComboBox, QSpinBox, QProgressBar,
    QCheckBox, QSlider
)
from PySide6.QtCore import Qt, QThread, Signal, QUrl
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.lines as mlines


class AudioWorker(QThread):
    """
    Worker thread to process audio without freezing the GUI.
    Now accepts the audio array directly to avoid re-reading from disk.
    """
    finished = Signal(object, object)
    error = Signal(str)

    def __init__(self, y, sr, method='CQT', hop_length=512, normalize=False):
        super().__init__()
        self.y = y
        self.sr = sr
        self.method = method
        self.hop_length = hop_length
        self.normalize = normalize

    def run(self):
        try:
            # Isolate harmonic content to remove drums/percussion noise
            y_harmonic = librosa.effects.harmonic(self.y)
            
            # Calculate chromagram based on selected method
            if self.method == "CQT":
                chroma = librosa.feature.chroma_cqt(
                    y=y_harmonic, sr=self.sr, hop_length=self.hop_length
                )
            else: # STFT
                chroma = librosa.feature.chroma_stft(
                    y=y_harmonic, sr=self.sr, hop_length=self.hop_length
                )
            
            # Normalize if requested (per frame)
            if self.normalize:
                chroma = librosa.util.normalize(chroma, norm=2, axis=0)
            
            self.finished.emit(chroma, self.sr)
        except Exception as e:
            self.error.emit(str(e))


class ChromaCanvas(FigureCanvas):
    """
    Matplotlib canvas to embed in PySide6.
    """
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 5), facecolor='#1e1e1e', tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111, facecolor='#2b2b2b')
        self.playhead = None
        self._init_plot()

    def _init_plot(self):
        self.ax.set_title("Chromagram", color='white')
        self.ax.set_xlabel("Time", color='white')
        self.ax.set_ylabel("Pitch Class", color='white')
        self.ax.tick_params(colors='white')
        for spine in self.ax.spines.values():
            spine.set_edgecolor('white')
        self.draw_idle()

    def plot_chroma(self, chroma, sr):
        # Clear figure and recreate axes
        self.fig.clear()
        self.ax = self.fig.add_subplot(111, facecolor='#2b2b2b')
        
        # Display the chromagram
        img = librosa.display.specshow(
            chroma, y_axis='chroma', x_axis='time', 
            ax=self.ax, sr=sr, cmap='magma'
        )
        
        # Add colorbar
        cbar = self.fig.colorbar(img, ax=self.ax, format='%+2.0f dB')
        cbar.ax.yaxis.set_tick_params(color='white')
        for label in cbar.ax.get_yticklabels():
            label.set_color('white')
            
        self.ax.set_title("Chromagram", color='white')
        self.ax.set_xlabel("Time", color='white')
        self.ax.set_ylabel("Pitch Class", color='white')
        self.ax.tick_params(colors='white')
        for spine in self.ax.spines.values():
            spine.set_edgecolor('white')
            
        # Initialize playhead
        self.playhead = self.ax.axvline(x=0, color='cyan', linewidth=2, linestyle='--')
        
        self.draw_idle()

    def update_playhead(self, time_sec):
        if self.playhead:
            self.playhead.set_xdata([time_sec, time_sec])
            self.draw_idle()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Chromagram Visualizer with Playback")
        self.resize(1000, 750)
        self.apply_dark_theme()

        # State variables
        self.current_file = None
        self.audio_data = None
        self.sr = None
        self.worker = None

        # Media Player Setup
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.8)
        
        self.player.positionChanged.connect(self.update_playhead)
        self.player.durationChanged.connect(self.update_duration_label)
        self.player.playbackStateChanged.connect(self.on_playback_state_changed)

        # Main UI setup
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Top Controls Layout (Processing) ---
        proc_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("Load Audio")
        self.load_btn.clicked.connect(self.load_file)
        
        self.method_combo = QComboBox()
        self.method_combo.addItems(["CQT (Constant-Q)", "STFT (Fourier)"])
        
        self.hop_spin = QSpinBox()
        self.hop_spin.setRange(64, 8192)
        self.hop_spin.setSingleStep(64)
        self.hop_spin.setValue(512)
        
        self.norm_check = QCheckBox("Normalize")
        self.norm_check.setToolTip("Normalizes energy per frame. Good for analyzing harmony regardless of volume.")
        
        self.process_btn = QPushButton("Re-process")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self.reprocess_audio)
        
        proc_layout.addWidget(QLabel("Method:"))
        proc_layout.addWidget(self.method_combo)
        proc_layout.addWidget(QLabel("Hop Length:"))
        proc_layout.addWidget(self.hop_spin)
        proc_layout.addWidget(self.norm_check)
        proc_layout.addStretch()
        proc_layout.addWidget(self.load_btn)
        proc_layout.addWidget(self.process_btn)

        # --- Playback Controls ---
        play_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("Play")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.toggle_playback)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_playback)
        
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)
        self.volume_slider.setFixedWidth(100)
        self.volume_slider.valueChanged.connect(lambda v: self.audio_output.setVolume(v / 100.0))
        
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet("color: #aaaaaa; font-family: monospace;")
        
        play_layout.addWidget(self.play_btn)
        play_layout.addWidget(self.stop_btn)
        play_layout.addWidget(QLabel("Vol:"))
        play_layout.addWidget(self.volume_slider)
        play_layout.addStretch()
        play_layout.addWidget(self.time_label)

        # --- Status and Progress ---
        self.status_label = QLabel("No file loaded.")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #aaaaaa; font-style: italic;")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0) # Indeterminate mode

        # --- Matplotlib Canvas ---
        self.canvas = ChromaCanvas(self)

        # --- Add to main layout ---
        main_layout.addLayout(proc_layout)
        main_layout.addLayout(play_layout)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.canvas)

    def apply_dark_theme(self):
        dark_stylesheet = """
            QMainWindow { background-color: #1e1e1e; }
            QWidget { color: #e0e0e0; background-color: #1e1e1e; }
            QPushButton {
                background-color: #3a3a3a; border: 1px solid #5a5a5a; 
                padding: 6px 12px; border-radius: 4px;
            }
            QPushButton:hover { background-color: #4a4a4a; }
            QPushButton:pressed { background-color: #2a2a2a; }
            QPushButton:disabled { color: #666666; background-color: #2a2a2a; }
            QComboBox, QSpinBox {
                background-color: #2a2a2a; border: 1px solid #5a5a5a; 
                padding: 4px; border-radius: 4px;
            }
            QComboBox::drop-down { border: none; }
            QSpinBox::up-button, QSpinBox::down-button { background-color: #3a3a3a; }
            QCheckBox { spacing: 6px; }
            QSlider::groove:horizontal {
                border: 1px solid #5a5a5a; height: 4px; background: #2a2a2a; border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #3a3a3a; border: 1px solid #5a5a5a; width: 10px; margin: -4px 0; border-radius: 5px;
            }
        """
        self.setStyleSheet(dark_stylesheet)

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "", 
            "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a);;All Files (*)"
        )
        
        if file_path:
            self.current_file = file_path
            self.process_btn.setEnabled(True)
            
            # Load into memory for processing
            self.status_label.setText("Loading audio into memory...")
            self.progress_bar.setVisible(True)
            
            try:
                # Load audio data into RAM
                self.audio_data, self.sr = librosa.load(file_path, sr=None, mono=True)
                
                # Setup media player
                self.player.setSource(QUrl.fromLocalFile(file_path))
                
                self.start_processing()
            except Exception as e:
                self.on_processing_error(str(e))

    def reprocess_audio(self):
        if self.audio_data is not None:
            self.start_processing()

    def start_processing(self):
        if self.audio_data is None:
            return
            
        file_name = self.current_file.split('/')[-1]
        method = "CQT" if self.method_combo.currentText().startswith("CQT") else "STFT"
        hop_length = self.hop_spin.value()
        normalize = self.norm_check.isChecked()
        
        self.status_label.setText(f"Processing: {file_name} [{method}, hop={hop_length}, norm={normalize}]...")
        self.progress_bar.setVisible(True)
        self.load_btn.setEnabled(False)
        self.process_btn.setEnabled(False)
        self.play_btn.setEnabled(False)

        # Pass cached audio data to worker
        self.worker = AudioWorker(self.audio_data, self.sr, method, hop_length, normalize)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.error.connect(self.on_processing_error)
        self.worker.start()

    def on_processing_finished(self, chroma, sr):
        self.canvas.plot_chroma(chroma, sr)
        self.status_label.setText("Processing complete. Ready to play.")
        self.progress_bar.setVisible(False)
        self.load_btn.setEnabled(True)
        self.process_btn.setEnabled(True)
        self.play_btn.setEnabled(True)

    def on_processing_error(self, error_msg):
        self.status_label.setText(f"Error: {error_msg}")
        self.progress_bar.setVisible(False)
        self.load_btn.setEnabled(True)
        self.process_btn.setEnabled(True)

    # --- Playback Logic ---
    
    def toggle_playback(self):
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def stop_playback(self):
        self.player.stop()

    def on_playback_state_changed(self, state):
        if state == QMediaPlayer.PlayingState:
            self.play_btn.setText("Pause")
            self.stop_btn.setEnabled(True)
        else:
            self.play_btn.setText("Play")
            if state == QMediaPlayer.StoppedState:
                self.stop_btn.setEnabled(False)
                self.canvas.update_playhead(0)

    def update_playhead(self, position_ms):
        # Convert media player position (milliseconds) to seconds
        time_sec = position_ms / 1000.0
        self.canvas.update_playhead(time_sec)
        
        # Update time label
        dur_ms = self.player.duration()
        self.time_label.setText(f"{self.format_time(position_ms)} / {self.format_time(dur_ms)}")

    def update_duration_label(self, duration_ms):
        pos_ms = self.player.position()
        self.time_label.setText(f"{self.format_time(pos_ms)} / {self.format_time(duration_ms)}")

    def format_time(self, ms):
        s = int(ms / 1000)
        m, s = divmod(s, 60)
        return f"{m:02d}:{s:02d}"


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())