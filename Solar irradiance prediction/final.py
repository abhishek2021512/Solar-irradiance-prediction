import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLineEdit, QLabel, QHBoxLayout
from PyQt5.QtGui import QFont, QIcon, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

dni_df = pd.read_csv(r"C:\Users\ABHISHEK DHAWAN\Downloads\final_draft\csr_africa_dni.csv")
glo_df = pd.read_csv(r'C:\Users\ABHISHEK DHAWAN\Downloads\final_draft\csr_africa_glo.csv')
tilt_df = pd.read_csv(r'C:\Users\ABHISHEK DHAWAN\Downloads\final_draft\csr_africa_tilt.csv')
dif_df = pd.read_csv(r'C:\Users\ABHISHEK DHAWAN\Downloads\final_draft\csr_africa_dif.csv')

combined_df = dni_df.merge(glo_df, on='PSECELLID', suffixes=('_dni', '_glo')) \
                    .merge(tilt_df, on='PSECELLID', suffixes=('', '_tilt')) \
                    .merge(dif_df, on='PSECELLID', suffixes=('', '_dif'))
X = combined_df[['CJAN', 'CFEB', 'CMAR', 'CAPR', 'CMAY',
                 'CJUN', 'CJUL', 'CAUG', 'CSEP', 'COCT', 'CNOV', 'CDEC']]
y = combined_df['CANN']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

class ModernUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Advanced Solar Irradiance Data Visualization")
        self.setGeometry(100, 100, 900, 700)
        self.setWindowIcon(QIcon('icon.jpg'))

        self.widget = QWidget(self)
        self.setCentralWidget(self.widget)
        self.layout = QVBoxLayout(self.widget)

        top_bar_layout = QHBoxLayout()
        self.cell_id_input = QLineEdit(self)
        self.cell_id_input.setPlaceholderText("Enter PSECELLID")
        self.cell_id_input.setFont(QFont('Arial', 10))
        self.cell_id_input.setFixedWidth(450)  
        top_bar_layout.addWidget(self.cell_id_input)

        self.plot_button = QPushButton("Generate Plot", self)
        self.plot_button.setFont(QFont('Arial', 12))
        self.plot_button.setStyleSheet("background-color: green; color: white; padding: 10px;")
        self.plot_button.setFixedSize(150, 60)  
        self.plot_button.clicked.connect(self.plot_data)
        top_bar_layout.addWidget(self.plot_button)
        
        self.layout.addLayout(top_bar_layout)

        self.result_label = QLabel(f"Mean Squared Error: {mse:.2f}, R-squared: {r2:.2f}", self)
        self.result_label.setFont(QFont('Arial', 9))
        self.result_label.setFixedHeight(20)  
        self.layout.addWidget(self.result_label)

        self.plot_controls_layout = QHBoxLayout()
        self.toggle_dni_button = QPushButton("Toggle DNI", self)
        self.toggle_dni_button.setStyleSheet("background-color: lightblue; border-radius: 10px;") 
        self.toggle_glo_button = QPushButton("Toggle GLO", self)
        self.toggle_glo_button.setStyleSheet("background-color: lightcoral; border-radius: 10px;") 
        self.toggle_tilt_button = QPushButton("Toggle TILT", self)
        self.toggle_tilt_button.setStyleSheet("background-color: lightgreen; border-radius: 10px;")  
        self.toggle_dif_button = QPushButton("Toggle DIF", self)
        self.toggle_dif_button.setStyleSheet("background-color: magenta; border-radius: 10px;")  
        for btn in [self.toggle_dni_button, self.toggle_glo_button, self.toggle_tilt_button, self.toggle_dif_button]:
            btn.clicked.connect(self.toggle_plot)
            btn.setFixedSize(140, 40)  
            self.plot_controls_layout.addWidget(btn)
        self.layout.addLayout(self.plot_controls_layout)


        self.canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.canvas.figure.subplots_adjust(bottom=0.2)  
        self.layout.addWidget(self.canvas)

        self.plot_visibility = {'dni': True, 'glo': True, 'tilt': True, 'dif': True}
    def clear_plot(self):
        """
        Clears the existing plot completely to ensure that plots do not overlap on new input.
        """
        self.canvas.figure.clf()
        self.canvas.draw()
    def plot_data(self):
        self.clear_plot() 
        ax = self.canvas.figure.subplots()

        PSECELLID = int(self.cell_id_input.text())
        filtered_data = combined_df[combined_df['PSECELLID'] == PSECELLID]
        if filtered_data.empty:
            ax.text(0.5, 0.5, 'No data available for the given PSECELLID', horizontalalignment='center', verticalalignment='center')
            self.canvas.draw()
            return

        months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    
        if self.plot_visibility['dni']:
            values_dni = filtered_data[['CJAN_dni', 'CFEB_dni', 'CMAR_dni', 'CAPR_dni', 'CMAY_dni', 'CJUN_dni',
                                    'CJUL_dni', 'CAUG_dni', 'CSEP_dni', 'COCT_dni', 'CNOV_dni', 'CDEC_dni']].values.flatten()
            ax.plot(months, values_dni, label='Direct', color='blue')
    
        if self.plot_visibility['glo']:
            values_glo = filtered_data[['CJAN_glo', 'CFEB_glo', 'CMAR_glo', 'CAPR_glo', 'CMAY_glo', 'CJUN_glo',
                                    'CJUL_glo', 'CAUG_glo', 'CSEP_glo', 'COCT_glo', 'CNOV_glo', 'CDEC_glo']].values.flatten()
            ax.plot(months, values_glo, label='Global', color='red')
    
        if self.plot_visibility['tilt']:
            values_tilt = filtered_data[['CJAN', 'CFEB', 'CMAR', 'CAPR', 'CMAY', 'CJUN', 'CJUL', 'CAUG',
                                     'CSEP', 'COCT', 'CNOV', 'CDEC']].values.flatten()
            ax.plot(months, values_tilt, label='Tilt', color='green')
    
        if self.plot_visibility['dif']:
            values_dif = filtered_data[['CJAN_dif', 'CFEB_dif', 'CMAR_dif', 'CAPR_dif', 'CMAY_dif', 'CJUN_dif',
                                    'CJUL_dif', 'CAUG_dif', 'CSEP_dif', 'COCT_dif', 'CNOV_dif', 'CDEC_dif']].values.flatten()
            ax.plot(months, values_dif, label='Diffuse', color='purple')

        ax.set_xlabel('Months')
        ax.set_ylabel('Irradiance (watt-hr/M2/day)')
        ax.set_title(f'NREL CSR Model Output, Cell ID = {PSECELLID}')
        ax.legend()

        latitude = filtered_data['LAT'].values[0]
        longitude = filtered_data['LON'].values[0]
        ax.text(0, 12000, f'Latitude: {latitude}, Longitude: {longitude}', fontsize=12, ha='center')
        ax.grid(True)

        self.canvas.draw()

    def toggle_plot(self):
        sender = self.sender()
        plot_type_raw = sender.text().split()[-1]  
        plot_type = plot_type_raw.lower()  

        if plot_type in self.plot_visibility:
            self.plot_visibility[plot_type] = not self.plot_visibility[plot_type]
            self.plot_data()  
        else:
            print(f"Attempted to toggle an undefined plot type: {plot_type}")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    main_window = ModernUI()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
