# Импорт необходимых модулей
from flask import Flask, render_template, request
import pandas as pd
from utils import report, visualize
import os

# Настройки
eng_colnames = ['util_per_offer_sim', 'util_per_offer_kk', 'util_per_offer_invest', 'violations', 'game', 'success', 'refuse_pct']
period1_file_path = "p1.xlsx"
period2_file_path = "p2.xlsx"

# Создание объекта приложения Flask
app = Flask(__name__)

# Маршрут для визуализации отчета
@app.route('/visualization')
def visualize_report():
    idx = request.args.get('id')  # Получение идентификатора из запроса
    period1 = pd.read_excel(period1_file_path, index_col=0)  # Загрузка данных первого периода из файла
    period2 = pd.read_excel(period2_file_path, index_col=0)  # Загрузка данных второго периода из файла

    # Генерация изображения SHAP и текстового описания
    shap_values, shap_img = generate_shap(period2.loc[int(idx)])
    text = generate_desription(period2, int(idx), shap_values)

    # Создание отчета
    report_df = report(period1, period2, int(idx))
    html_report = report_df.round(3).to_html()

    # Визуализация прогноза
    prediction_plot = visualize(report_df.T)

    return render_template('visualization.html', id=idx, shap_img=shap_img, description=text, html_report=html_report, prediction_plot=prediction_plot)

# Маршрут для загрузки файлов и создания отчета
@app.route('/upload', methods=['POST'])
def upload():
    if not request.files:
        return 'No file part'

    file1 = request.files['period1']
    file2 = request.files['period2']
    if file1.filename == '' or file2.filename == '':
        return 'No selected file'

    if file1 and file2:
        df1 = pd.read_excel(file1, index_col=1)[eng_colnames]
        df1.index = df1.index.astype(int)
        df2 = pd.read_excel(file2, index_col=1)[eng_colnames]
        df2.index = df2.index.astype(int)
        merge =pd.merge(df1, df2, left_index=True, right_index=True, suffixes=['_1', '_2'])
        column_ids = merge.index
        df1.to_excel(period1_file_path)
        df2.to_excel(period2_file_path)
        return render_template('result.html', column_ids=column_ids)

# Маршрут по умолчанию для отображения главной страницы
@app.route('/')
def index():
    return render_template('index.html')

# Запуск приложения Flask
if __name__ == '__main__':
    app.run(debug=True, port=8001)
