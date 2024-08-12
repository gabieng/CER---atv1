"""Arquivo de teste do predict."""
from model_xpto.main import predict


def test_predict_eq_42()->None:
    """Este teste valida retorno = 42."""
    assert predict(None)==42  # noqa: PLR2004

import matplotlib.pyplot as plt  # type: ignore  # noqa: E402, F401, I001
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402, F401
from sklearn.model_selection import train_test_split  # noqa: E402, F401

df2=pd.read_csv(r'/workspace/tests/__pycache__/OP_INMET(RN)_2022.csv')  # noqa: PD901
df_treino=pd.read_csv(r'/workspace/tests/__pycache__/CAL-VAL_INMET(RN)_2022.csv')

print(df2.head())
print(df_treino.head())

print(df_treino.shape)
def detect_outliers_zscore(df, threshold=3):  # noqa: ANN201, D103
    z_scores = stats.zscore(df)
    return df[(z_scores > threshold).any(axis=1)]

df_treino.drop(['Lançamento', 'forecast_hour'], axis=1, inplace=True)
outliers_zscore = detect_outliers_zscore(df_treino)

# Função para detectar outliers usando IQR
def detect_outliers_iqr(df):  # noqa: ANN201, D103
    Q1 = df.quantile(0.25)  # noqa: N806
    Q3 = df.quantile(0.75)  # noqa: N806
    IQR = Q3 - Q1  # noqa: N806
    return df[((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
outliers_iqr = detect_outliers_iqr(df_treino)

# Visualização de boxplots
plt.figure(figsize=(10, 6))
df2.boxplot()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("Outliers Z-Score:")
print(outliers_zscore)
print("Outliers IQR:")
print(outliers_iqr)

# Remoção de outliers baseada em Z-score
def remove_outliers_zscore(df, threshold=3):  # noqa: ANN001, ANN201, D103
    z_scores = stats.zscore(df)
    return df[(z_scores <= threshold).any(axis=1)]

z_scores = stats.zscore(df_treino)
threshold = 2
df_sem_outliers = remove_outliers_zscore(df_treino)
print(df_sem_outliers.head(15))
print(df_sem_outliers.isna().sum())
print(df_sem_outliers.dtypes)

# Tratamento de valores ausentes
media_obs = df2["Observação (m/s)"].mean()
df2["Observação (m/s)"].fillna(media_obs, inplace=True)  # noqa: PD002

classes_horizonte = [3, 6, 9, 12, 15, 18, 21, 24]

# Preparação e modelagem dos dados
df_sem_outliers['Horizonte_classe'] = df_sem_outliers['Horizonte'].apply(lambda x: min(classes_horizonte, key=lambda y: abs(x - y)))  # type: ignore # noqa: E501, F821
X = df_sem_outliers.drop(['Observação (m/s)', 'Horizonte_classe'], axis=1)
y = df_sem_outliers['Observação (m/s)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=df_sem_outliers['Horizonte_classe'])  # noqa: E501

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Cálculo de métricas de avaliação
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print("Métricas de Avaliação:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)

# Análise por horizontes
horizontes = [3, 6, 9, 12, 15, 18, 21, 24]
resultados = []
for horizonte in horizontes:
    X_train_horizonte = X_train[X_train['Horizonte'] == horizonte]
    y_train_horizonte = y_train[X_train['Horizonte'] == horizonte]
    X_test_horizonte = X_test[X_test['Horizonte'] == horizonte]
    y_test_horizonte = y_test[X_test['Horizonte'] == horizonte]
    model.fit(X_train_horizonte, y_train_horizonte)
    y_pred = model.predict(X_test_horizonte)
    mae = mean_absolute_error(y_test_horizonte, y_pred)
    mse = mean_squared_error(y_test_horizonte, y_pred)
    rmse = mean_squared_error(y_test_horizonte, y_pred, squared=False)
    r2 = r2_score(y_test_horizonte, y_pred)
    resultados.append({"Horizonte": horizonte, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2})  # noqa: E501
    print(f"Horizonte {horizonte}: MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}")  # noqa: E501
