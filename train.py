# Инициализируем зависимости
from math import ceil
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import LabelBinarizer
plt.style.use('ggplot') # определяем стиль
warnings.filterwarnings('ignore')# исключение конфигурационных строчек (варнинги)

# Загружаем датасет для обучения
intensity = pd.read_csv('./datasets/datasetnew2.csv', encoding="utf-8", delimiter=";")
# Удаляем строчки в которых есть хоть один NaN
intensity.dropna(inplace=True)
# Преобразуем количественное значение интенсивности в качественное (группировка) с шагом 250
intensity['intensity'] = intensity.intensity.apply(lambda x: ceil(x / 250))
# Создаем новый датасет на основе старого, но без значений интенсивности
intensity_ml = intensity.drop('intensity', axis=1)


# Делаем фильтрацию колонок, указываем те, которые используются в обучении
col = [
    'week_day', 'day_time', 't_air', 't_soil', 't_dew_point', 'partial_pressure',
    'humidity', 'saturation_deficit', 'pressure_station', 'pressure_sea',
    'visibility_VV', 'weather_WW', 'wind_direction', 'wind_speed', 'precipitation',
    'daylight', 'straight_stripes_project', 'straight_lanes_provided',
    'lanes_left', 'lanes_right', 'left_stripe_view', 'right_stripe_view',
    'strip_length_left', 'strip_length_right', 'type_movement',
    'distance_to_parking', 'method_of_setting', 'type_of_parking',
    'longitudinal_slope', 'dead_end_street', 'total_strip_width',
    'total_forward_direction_width', 'narrowing_of_movement', 'dividing_strip',
    'Area', 'distance_to_bus_stop', 'bus_stop_type', 'Intersection_type',
    'traffic_light_regulation',
    # 'intensity'
]
intensity_ml = intensity_ml[col]

# intensity = intensity[col]
# # Высчитываем корреляцию
# corr = intensity.corr()
# plt.subplots(figsize=(20, 15))
# sns.heatmap(corr)
# plt.show()

# Начинаем машинное обучение
# Разбиваем датасет на данные для обучения и данные для теста
X_train, X_test, y_train, y_test = train_test_split(intensity_ml.values,
  intensity['intensity'].values, test_size=0.2, random_state=42)
#(создаем объект классификатора)
random_forest = RandomForestClassifier(n_estimators=200)
y_score = random_forest.fit(X_train, y_train).predict_proba(X_test)            # Узнать, скорее всего вероятность прогноза

# # ROC Трансформируем интенсивность
# label_binarizer = LabelBinarizer().fit(y_train)
# y_onehot_test = label_binarizer.transform(y_test)
#   # (n_samples, n_classes)
#
# # ROC запускаем цикл, узнаем размерность матрици, чтобы создать диапазон итераций. lass_of_interest -   номер группы интенсивности
# for class_of_interest in range(1, y_onehot_test.shape[1]):
#      class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
#      RocCurveDisplay.from_predictions(
#          y_onehot_test[:, class_id],
#          y_score[:, class_id],
#          name=f"{class_of_interest} vs the rest",
#          color="darkorange",
#          plot_chance_level=True)
#      plt.axis("square")
#      plt.xlabel("False Positive Rate")
#      plt.ylabel("True Positive Rate")
#      plt.title("One-vs-Rest ROC curves")
#      plt.legend()
# plt.show()


# Оцениваем точность модели
acc_random_forest = round(random_forest.score(X_test, y_test) * 100, 2)
print("Accuracy", acc_random_forest)

# Оцениваем показатель r2
print(r2_score(random_forest.predict(X_test), y_test))
# Значимоть признаков
importances = random_forest.feature_importances_
indices = np.argsort(importances)[::-1]
ar_f=[]
for f, idx in enumerate(indices):
    ar_f.append([round(importances[idx],4), col[idx]])
print("Значимость признака:")
ar_f.sort(reverse=True)
print(ar_f)
# Строим график значимости признаков
d_first = len(col)
plt.figure(figsize=(8, 8))
plt.title("Значимость признака")
plt.bar(range(d_first), importances[indices[:d_first]], align='center')
plt.xticks(range(d_first), np.array(col)[indices[:d_first]], rotation=90)
plt.xlim([-1, d_first])

# Матрица количества правильно- и ошибочно- угаданных классов
from sklearn.metrics import confusion_matrix
# так же матрица в процентах и более изящном виде
matrix = confusion_matrix(y_test, random_forest.predict(X_test))
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# Build the plot
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)

# Add labels to the plot
class_names = ["0-250", "251-500", "501-750", "751-1000",
"1001-1250", "1251-1500", "1501-1750", "1751-2000", "2001-2250",
"2251-2500", "2501-2750", "2751-3000", "3001-3250", "3251-3500",
"3501-3750", "3751- 4000", "4001- 4250", "4251- 4500", "4501- 4750",
"4751-5000", "5001-5250", "5251-5500", "5501-5750"]
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Спрогнозированные классы')
plt.ylabel('Фактические классы')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()

# Базовые метрики оценки точности модели
from sklearn.metrics import classification_report
print(classification_report(y_test, random_forest.predict(X_test)))

# Сохраняем обученую модель
joblib.dump(random_forest, 'model.sav')
