import pandas as pd
import numpy as np
import os
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pickle
import shap
import joblib
import matplotlib.pyplot as plt
import base64
from io import BytesIO



class ModelWrap():
    def __init__(self, model):
        self.model = model
        
    def fit_scaler(self, X, y):
        self.X_scaler = StandardScaler()
        self.Y_scaler = StandardScaler()
        self.X_scaler.fit(X)
        self.Y_scaler.fit(y)
        
    def fit(self, X_train, y_train, **kwargs):
        X_train_scaled = self.X_scaler.transform(X_train)
        y_train_scaled = self.Y_scaler.transform(y_train)
        
        self.model.fit(X_train_scaled, y_train_scaled, **kwargs)
        
    def evaluate(self, X_test, y_test):
        X_test_scaled = self.X_scaler.transform(X_test)
        y_test_scaled = self.Y_scaler.transform(y_test)
        # print(self.model.evaluate(X_test_scaled, y_test_scaled))
    
    def predict(self, X):
        X_scaled = self.X_scaler.transform(X)
        y_pred_scaled = self.model.predict(X_scaled)
        y_pred = self.Y_scaler.inverse_transform(y_pred_scaled)
        lower_limits = np.array([0, 0, 0, 0, 0, 0, 0])
        upper_limits = np.array([1,1,1,1,1,1,1])
        print(y_pred)
        y_clipped = np.clip(y_pred, lower_limits, upper_limits)
        return y_clipped


def load_model_wrap(base_dir):
    # Load the Keras model
    model_path = os.path.join(base_dir, 'model.keras')
    model = load_model(model_path)
    
    # Load the scalers
    scaler_path = os.path.join(base_dir, 'scalers.pkl')
    with open(scaler_path, 'rb') as f:
        X_scaler, Y_scaler = pickle.load(f)
        
    # Create a new ModelWrap instance and set the model and scalers
    model_wrap = ModelWrap(model)
    model_wrap.X_scaler = X_scaler
    model_wrap.Y_scaler = Y_scaler
    
    return model_wrap


eval_model = joblib.load("eval_model.pkl")
predict_model = load_model_wrap("model")

X_train = pd.read_excel("training_data.xlsx", index_col=0)
explainer = shap.KernelExplainer(eval_model.predict, X_train)

colnames = ['Утиль Сим', 'Утиль Кредит', 'Утиль Инвест', 'Нарушения', 'Геймификация', 'Успешность', "Процент отказов"]

def report(period1, period2, id):
    
    period1_row = period1.loc[id, ['util_per_offer_sim', 'util_per_offer_kk',
       'util_per_offer_invest', 'violations', 'game', 'success', 'refuse_pct']]
    period2_row = period2.loc[id, ['util_per_offer_sim', 'util_per_offer_kk',
       'util_per_offer_invest', 'violations', 'game', 'success', 'refuse_pct']]
    period1_row.name = "Период 1"
    period2_row.name = "Период 2"
    prediction = predict_model.predict(pd.concat([period1_row, period2_row]).values.reshape(1, -1))[0]
    
    res = pd.DataFrame([period1_row, period2_row])
    res.loc['Предсказание'] = prediction
    res.columns = colnames
    inp = res.loc[['Период 1', "Период 2", "Предсказание"]].values
    res.loc[['Период 1', "Период 2", "Предсказание"], 'Оценка'] = eval_model.predict(inp)
    # res.loc["Изменение за месяц"] = res.pct_change().loc['Период 2'] * 100
    # res.loc["Ожидаемое изменение"] = res.pct_change().loc['Предсказание'] * 100

    # model.predict(pd.concat([period1_row, period2_row]))
    return res.T

def generate_shap(X):
    shap_values = explainer.shap_values(X)
    print(shap_values)
    def _force_plot_html(explainer, shap_values, X):
        force_plot = shap.plots.force(explainer.expected_value, shap_values, np.round(X, 3),
                     matplotlib=False, feature_names = colnames)
        return force_plot
    _force_plot_html(explainer, shap_values, X).matplotlib(figsize=(10,4),show=False, text_rotation=30)
    buffer = BytesIO()
    plt.savefig(buffer, format='png',bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    return shap_values, image_base64



# shap_values
def generate_desription(X, idx, shap_values):
    mask = pd.Series([1,1,1,-1,1,1,-1], index=X.columns)
    X_tmp = X*mask
    t =(((X_tmp < X_tmp.loc[X.loc[idx].name]).mean(axis=0)))
    max_col, max_value = t.argmax(), int(np.round(t.max() * 100, 0))
    t =(((X_tmp > X_tmp.loc[X.loc[idx].name]).mean(axis=0)))
    min_col, min_value = t.argmax(), int(np.round((t.max()) * 100, 0))
    template = f"Вы лучше {max_value}% сотрудников в этом месяце по показателю {colnames[max_col]}, благодаря этому показателю оценка была изменена на {np.round(shap_values[max_col],2)} балла. Обратите внимание на показатель {colnames[min_col]} – в этом месяце вы были хуже {min_value}% сотрудников по этому показателю, что изменило вашу оценку на {np.round(shap_values[min_col], 2)} балла."
    return template



def visualize(df):
    fig, axs = plt.subplots(4, 2, figsize=(6, 12))
    # Plot each column in a separate subplot
    # fig, axs = plt.subplots(2, 2, figsize=(12, 6), )
    for i, column in enumerate(df.columns):
        ax = axs[i//2, i%2]
        ax.plot(df.index[0:2], df[column][0:2], marker='o', linestyle='-')
        # Make the line from the second to the third dot dashed
        ax.plot(df.index[1:3], df[column][1:3], marker='o', linestyle='--', color='orange')
        # Make the third dot hollow
        ax.plot(df.index[2], df[column][2], marker='o', markersize=10, markerfacecolor='w', markeredgewidth=2, linestyle='None', color='orange')
        ax.set_title(column)
        ax.tick_params(axis='x', labelsize=8)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    # Convert plot to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    return image_base64