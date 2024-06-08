import joblib
import numpy as np
import pandas as pd
model = joblib.load('model.sav')
intensity = pd.read_csv('./datasets/dataset_test.csv', encoding="utf-8", delimiter=";")
col = ['week_day', 'day_time', 't_air', 't_soil', 't_dew_point',
'partial_pressure', 'humidity', 'saturation_deficit', 'pressure_station',
'pressure_sea', 'visibility_VV', 'weather_WW', 'wind_direction', 'wind_speed',
'precipitation', 'daylight', 'straight_stripes_project', 'straight_lanes_provided',
'lanes_left', 'lanes_right', 'left_stripe_view', 'right_stripe_view', 'strip_length_left',
'strip_length_right', 'type_movement', 'distance_to_parking', 'method_of_setting',
'type_of_parking', 'longitudinal_slope', 'dead_end_street', 'total_strip_width',
'total_forward_direction_width', 'narrowing_of_movement', 'dividing_strip',
'Area', 'distance_to_bus_stop', 'bus_stop_type', 'Intersection_type',
'traffic_light_regulation']
intensity_ml = intensity[col]
result = model.predict(intensity_ml.values)
r = pd.DataFrame(np.array(result), columns=['intensity_predict'])
r.to_csv('./datasets/predicted.csv')


