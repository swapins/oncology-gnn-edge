import os
import time
import psutil
import torch
import pandas as pd
from collections import deque
from datetime import datetime

from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QGridLayout, QPushButton, QFileDialog,
    QMessageBox, QProgressBar, QFrame,
    QScrollArea, QTextEdit
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtChart import QChart, QChartView, QLineSeries, QValueAxis


# -----------------------------
# Metric Card Component
# -----------------------------

class MetricCard(QFrame):

    def __init__(self, title, value="--"):
        super().__init__()

        self.setFixedHeight(90)

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 8, 12, 8)

        self.title = QLabel(title)
        self.title.setFont(QFont("Segoe UI", 9))
        self.title.setStyleSheet("color: #90A4AE;")

        self.value = QLabel(value)
        self.value.setFont(QFont("Segoe UI", 18, QFont.Bold))
        self.value.setStyleSheet("color: #00E676;")

        layout.addWidget(self.title)
        layout.addWidget(self.value)

        self.setLayout(layout)

        self.setStyleSheet("""
            QFrame {
                background-color: #1B263B;
                border: 1px solid #324A5F;
                border-radius: 6px;
            }
        """)

    def set_value(self, val):
        self.value.setText(str(val))


# -----------------------------
# Status Indicator Component
# -----------------------------

class StatusIndicator(QFrame):
    
    def __init__(self):
        super().__init__()
        
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)
        
        # Status label
        status_label = QLabel("Status")
        status_label.setFont(QFont("Segoe UI", 10))
        status_label.setStyleSheet("color: #90A4AE;")
        layout.addWidget(status_label)
        
        # Status text with indicator
        self.status_text = QLabel("● Ready")
        self.status_text.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.status_text.setStyleSheet("color: #4CAF50;")
        layout.addWidget(self.status_text)
        
        # Timestamp
        self.timestamp = QLabel(datetime.now().strftime("%H:%M:%S"))
        self.timestamp.setFont(QFont("Segoe UI", 9))
        self.timestamp.setStyleSheet("color: #607D8B;")
        layout.addWidget(self.timestamp)
        
        self.setLayout(layout)
        
        self.setStyleSheet("""
            QFrame {
                background-color: #1B263B;
                border: 1px solid #324A5F;
                border-radius: 6px;
            }
        """)
        
        self.status = "Ready"
    
    def set_status(self, status_text, status_type="info"):
        """
        status_type: 'ready', 'processing', 'success', 'error'
        """
        self.status = status_text
        self.status_text.setText(f"● {status_text}")
        
        color_map = {
            "ready": "#4CAF50",
            "processing": "#FFC107",
            "success": "#4CAF50",
            "error": "#FF5252"
        }
        
        color = color_map.get(status_type, "#4CAF50")
        self.status_text.setStyleSheet(f"color: {color};")
        self.timestamp.setText(datetime.now().strftime("%H:%M:%S"))


# ---- Section Separator
class SectionSeparator(QFrame):
    
    def __init__(self, title=""):
        super().__init__()
        
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 8, 0, 8)
        
        if title:
            label = QLabel(title)
            label.setFont(QFont("Segoe UI", 11, QFont.Bold))
            label.setStyleSheet("color: #00E676;")
            layout.addWidget(label)
        
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("background-color: #00E676;")
        layout.addWidget(separator)
        
        self.setLayout(layout)
        self.setStyleSheet("background-color: transparent; border: none;")


# ---- Results Panel Component
class ResultsPanel(QFrame):
    
    def __init__(self):
        super().__init__()
        
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        
        title = QLabel("📊 Analysis Results")
        title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        title.setStyleSheet("color: #00E676;")
        layout.addWidget(title)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(140)
        self.results_text.setFont(QFont("Courier New", 9))
        self.results_text.setStyleSheet("""
            QTextEdit {
                background-color: #0D1B2A;
                color: #E8EEF5;
                border: 1px solid #324A5F;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.results_text)
        
        self.setLayout(layout)
        
        self.setStyleSheet("""
            QFrame {
                background-color: #1B263B;
                border: 1px solid #324A5F;
                border-radius: 6px;
            }
        """)
        
        self.clear_results()
    
    def clear_results(self):
        self.results_text.setPlainText("No analysis run yet.\nWait for analysis results...")
    
    def set_results(self, results_dict):
        """Display analysis results in formatted text."""
        text = "✓ Analysis Complete\n\n"
        for key, value in results_dict.items():
            text += f"{key}: {value}\n"
        self.results_text.setPlainText(text)

    def set_value(self, val):
        self.value.setText(str(val))


# -----------------------------
# Main Dashboard
# -----------------------------

class GNNSystemDashboard(QWidget):

    def __init__(self, engine):
        super().__init__()

        self.engine = engine
        self.expression_path = None
        self.ppi_path = None
        self.inference_times = deque(maxlen=60)
        self.last_output = None

        self.setWindowTitle("Oncology-GNN-Edge Research Console")
        self.setGeometry(0, 0, 1024, 768)

        self.setStyleSheet("""
            QWidget {
                background-color: #0D1B2A;
                color: #E8EEF5;
                font-family: Segoe UI;
            }
            QPushButton {
                background-color: #00E676;
                color: black;
                font-weight: bold;
                padding: 8px;
                border-radius: 5px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #69F0AE;
            }
            QPushButton:pressed {
                background-color: #00C853;
            }
            QProgressBar {
                background-color: #263238;
                border: 1px solid #37474F;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #00E676;
            }
        """)

        # Use scroll area for touch-friendly navigation
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #0D1B2A;
            }
        """)
        
        scroll_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # Header
        header = QLabel("🧬 Oncology-GNN-Edge Research Workstation")
        header.setFont(QFont("Segoe UI", 18, QFont.Bold))
        header.setStyleSheet("color: #00E676;")
        main_layout.addWidget(header)

        # ====== RESEARCH CONTROLS SECTION ======
        main_layout.addWidget(SectionSeparator("📁 Research Controls"))

        # Data Upload Section
        upload_layout = QHBoxLayout()
        upload_layout.setSpacing(6)

        self.btn_expression = QPushButton("📂 Expression CSV")
        self.btn_expression.clicked.connect(self.load_expression)

        self.btn_ppi = QPushButton("📂 PPI Network CSV")
        self.btn_ppi.clicked.connect(self.load_ppi)

        self.btn_run = QPushButton("▶ Run Analysis")
        self.btn_run.clicked.connect(self.run_analysis)

        self.btn_export = QPushButton("💾 Export Results")
        self.btn_export.clicked.connect(self.export_results)

        upload_layout.addWidget(self.btn_expression)
        upload_layout.addWidget(self.btn_ppi)
        upload_layout.addWidget(self.btn_run)
        upload_layout.addWidget(self.btn_export)

        main_layout.addLayout(upload_layout)

        # Status Indicator
        self.status_indicator = StatusIndicator()
        main_layout.addWidget(self.status_indicator)

        # Results Panel
        self.results_panel = ResultsPanel()
        main_layout.addWidget(self.results_panel)

        # Analysis Metrics
        metrics_layout = QGridLayout()
        metrics_layout.setSpacing(8)

        self.nodes_card = MetricCard("Graph Nodes")
        self.edges_card = MetricCard("Graph Edges")
        self.embedding_card = MetricCard("Embedding Shape")
        self.time_card = MetricCard("Inference (ms)")
        self.drift_card = MetricCard("Network Drift")

        metrics_layout.addWidget(self.nodes_card, 0, 0)
        metrics_layout.addWidget(self.edges_card, 0, 1)
        metrics_layout.addWidget(self.embedding_card, 0, 2)
        metrics_layout.addWidget(self.time_card, 1, 0)
        metrics_layout.addWidget(self.drift_card, 1, 1)

        main_layout.addLayout(metrics_layout)

        # Performance Chart
        self.chart = QChart()
        self.chart.setBackgroundBrush(QColor("#1B263B"))
        self.chart.setTitleBrush(QColor("#00E676"))
        self.chart.setTitle("Inference Trend (Last 60 Samples)")
        self.chart.legend().hide()

        self.chart_view = QChartView(self.chart)
        self.chart_view.setMinimumHeight(180)

        main_layout.addWidget(self.chart_view)

        # ====== SYSTEM MONITORING SECTION ======
        main_layout.addWidget(SectionSeparator("⚙️ System Monitoring"))

        # System Status Cards
        status_layout = QGridLayout()
        status_layout.setSpacing(8)

        self.device_card = MetricCard("Device", str(self.engine.device))
        self.cuda_card = MetricCard("CUDA Available",
                                    "Yes" if torch.cuda.is_available() else "No")
        self.fp16_card = MetricCard("FP16 Enabled",
                                    "Yes" if self.engine.use_fp16 else "No")

        status_layout.addWidget(self.device_card, 0, 0)
        status_layout.addWidget(self.cuda_card, 0, 1)
        status_layout.addWidget(self.fp16_card, 0, 2)

        main_layout.addLayout(status_layout)

        # System Resource Section
        resource_layout = QHBoxLayout()
        resource_layout.setSpacing(8)

        cpu_label = QLabel("CPU Usage")
        cpu_label.setStyleSheet("color: #90A4AE; font-size: 10px;")
        self.cpu_progress = QProgressBar()
        self.cpu_progress.setMaximumWidth(150)

        ram_label = QLabel("RAM Usage")
        ram_label.setStyleSheet("color: #90A4AE; font-size: 10px;")
        self.ram_progress = QProgressBar()
        self.ram_progress.setMaximumWidth(150)

        resource_layout.addWidget(cpu_label)
        resource_layout.addWidget(self.cpu_progress)
        resource_layout.addWidget(ram_label)
        resource_layout.addWidget(self.ram_progress)
        resource_layout.addStretch()

        main_layout.addLayout(resource_layout)
        main_layout.addStretch()

        scroll_widget.setLayout(main_layout)
        scroll.setWidget(scroll_widget)

        # Main container layout
        container = QVBoxLayout()
        container.setContentsMargins(0, 0, 0, 0)
        container.addWidget(scroll)
        self.setLayout(container)

        # Timer for system updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_system_metrics)
        self.timer.start(1000)

    # -----------------------------
    # CSV Upload Handlers
    # -----------------------------

    def load_expression(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Expression CSV", "", "CSV Files (*.csv)"
        )
        if path:
            self.expression_path = path
            self.status_indicator.set_status(f"Loaded: {os.path.basename(path)}", "success")
            QMessageBox.information(self, "Loaded", os.path.basename(path))

    def load_ppi(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select PPI CSV", "", "CSV Files (*.csv)"
        )
        if path:
            self.ppi_path = path
            self.status_indicator.set_status(f"Loaded: {os.path.basename(path)}", "success")
            QMessageBox.information(self, "Loaded", os.path.basename(path))

    # -----------------------------
    # Run Analysis
    # -----------------------------

    def run_analysis(self):

        if not self.expression_path or not self.ppi_path:
            self.status_indicator.set_status("Missing files", "error")
            QMessageBox.warning(self, "Missing Files",
                                "Please upload both Expression and PPI CSV files.")
            return

        try:
            self.status_indicator.set_status("Processing...", "processing")
            self.btn_run.setEnabled(False)

            # Load CSVs without headers (headerless data)
            expr_df = pd.read_csv(self.expression_path, header=None)
            ppi_df = pd.read_csv(self.ppi_path, header=None)

            features = torch.tensor(expr_df.values, dtype=torch.float32)

            n = features.shape[0]
            input_dim = features.shape[1]  # Get actual input dimension from CSV
            
            adj = torch.zeros((n, n))

            for _, row in ppi_df.iterrows():
                i, j = int(row[0]), int(row[1])
                if i < n and j < n:
                    adj[i, j] = 1
                    adj[j, i] = 1

            adj = adj.to_sparse()

            # Create engine dynamically based on CSV dimensions
            from gnn_edge.config import GNNConfig
            from gnn_edge.inference import GNNInference
            
            config = GNNConfig(
                input_dim=input_dim,  # Use actual input dimension from CSV
                hidden_dim=32,
                use_fp16=False,
                force_device="cpu"
            )
            analysis_engine = GNNInference(config)

            graph = {
                "features": features.to(analysis_engine.device),
                "adjacency": adj.to(analysis_engine.device)
            }

            t0 = time.time()
            output = analysis_engine.forward(graph)
            t1 = time.time()

            inference_time = (t1 - t0) * 1000
            self.last_output = output

            # Update UI
            self.nodes_card.set_value(n)
            self.edges_card.set_value(adj._nnz())
            self.embedding_card.set_value(tuple(output.shape))
            self.time_card.set_value(f"{inference_time:.2f}")
            drift = float(torch.norm(output).item())
            self.drift_card.set_value(f"{drift:.3f}")

            self.inference_times.append(inference_time)
            self.update_chart()

            # Update results panel
            results_dict = {
                "Status": "✓ Success",
                "Input Features": input_dim,
                "Nodes": n,
                "Edges": adj._nnz(),
                "Inference Time": f"{inference_time:.2f} ms",
                "Output Shape": str(output.shape),
                "Network Drift": f"{drift:.3f}",
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Expression File": os.path.basename(self.expression_path),
                "PPI File": os.path.basename(self.ppi_path)
            }
            self.results_panel.set_results(results_dict)

            self.status_indicator.set_status("Analysis Complete", "success")
            self.btn_run.setEnabled(True)

            QMessageBox.information(self, "Success",
                                    f"Embedding complete.\nTime: {inference_time:.2f} ms\nExport results to save.")

        except Exception as e:
            self.status_indicator.set_status("Error occurred", "error")
            self.btn_run.setEnabled(True)
            QMessageBox.critical(self, "Error", str(e))

    # -----------------------------
    # Export Results
    # -----------------------------

    def export_results(self):

        if self.last_output is None:
            self.status_indicator.set_status("No results to export", "error")
            QMessageBox.warning(self, "No Results",
                                "Run analysis first to generate results for export.")
            return

        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Embeddings CSV", "embeddings.csv", "CSV Files (*.csv)"
            )

            if file_path:
                embeddings_df = pd.DataFrame(
                    self.last_output.detach().cpu().numpy()
                )
                embeddings_df.to_csv(file_path, index=False)

                self.status_indicator.set_status(f"Exported to {os.path.basename(file_path)}", "success")
                QMessageBox.information(self, "Export Success",
                                        f"Embeddings saved to:\n{file_path}")
        except Exception as e:
            self.status_indicator.set_status("Export failed", "error")
            QMessageBox.critical(self, "Export Error", str(e))

    # ---- Update Inference Info (from main.py periodic calls)
    def update_inference_info(self, graph, output, inference_time_ms):
        """Called from main.py to update dashboard during periodic inference."""
        adj = graph["adjacency"]
        nodes = adj.size(0)
        edges = adj._nnz() if adj.is_sparse else int(adj.sum().item())

        self.nodes_card.set_value(nodes)
        self.edges_card.set_value(edges)
        self.embedding_card.set_value(str(tuple(output.shape)))
        self.time_card.set_value(f"{inference_time_ms:.2f}")

        drift = float(torch.norm(output).item())
        self.drift_card.set_value(f"{drift:.3f}")

        self.inference_times.append(inference_time_ms)
        self.update_chart()

    # ---- System Monitoring
    def update_system_metrics(self):

        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent

        self.cpu_progress.setValue(int(cpu))
        self.ram_progress.setValue(int(ram))

    # ---- Chart Update
    def update_chart(self):

        if len(self.inference_times) < 2:
            return

        self.chart.removeAllSeries()

        series = QLineSeries()

        for i, val in enumerate(self.inference_times):
            series.append(i, val)

        self.chart.addSeries(series)

        axis_x = QValueAxis()
        axis_x.setRange(0, max(59, len(self.inference_times)))
        axis_x.setTitleText("Samples")

        axis_y = QValueAxis()
        axis_y.setRange(0, max(self.inference_times) * 1.2)
        axis_y.setTitleText("ms")

        self.chart.addAxis(axis_x, Qt.AlignBottom)
        self.chart.addAxis(axis_y, Qt.AlignLeft)

        series.attachAxis(axis_x)
        series.attachAxis(axis_y)
