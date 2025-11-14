#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Универсальный цикл прогнозирования
Автоматически определяет год в данных и прогнозирует на следующий год
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

def load_model():
    """Загрузка сохранённой модели из пункта 18"""
    try:
        with open('knn_full_components.pkl', 'rb') as f:
            model_components = pickle.load(f)
        
        print(f"✅ Модель загружена: {model_components['model_name']}")
        print(f"Признаки модели: {model_components['selected_features']}")
        return model_components
    
    except FileNotFoundError:
        print("❌ Файл модели 'knn_full_components.pkl' не найден!")
        return None

def load_data_and_detect_year(file_path):
    """Загрузка данных и определение года"""
    try:
        data = pd.read_excel(file_path)
        
        # Определяем годы в данных
        years_in_data = sorted(data['Year'].unique()) if 'Year' in data.columns else []
        
        if not years_in_data:
            # Если Year нет, пытаемся определить из дат
            date_col = data.columns[0]
            if 'Unnamed' not in date_col:
                dates = pd.to_datetime(data[date_col])
                years_in_data = sorted(dates.dt.year.unique())
        
        start_year = min(years_in_data) if years_in_data else None
        end_year = max(years_in_data) if years_in_data else None
        
        print(f"✅ Данные загружены: {len(data)} строк, {len(data.columns)} столбцов")
        print(f"📅 Период данных: {start_year} - {end_year}")
        
        return data, start_year, end_year
    
    except Exception as e:
        raise ValueError(f"Ошибка загрузки файла {file_path}: {e}")

def process_date_column(data):
    """Обработка столбца с датой - создание Year и Month"""
    print("\n📅 Обработка даты...")
    
    date_col = data.columns[0]  # Первый столбец
    
    try:
        if pd.api.types.is_datetime64_any_dtype(data[date_col]):
            dates = data[date_col]
        else:
            dates = pd.to_datetime(data[date_col])
        
        data['Year'] = dates.dt.year
        data['Месяц'] = dates.dt.month
        
        years = sorted(data['Year'].unique())
        months = sorted(data['Месяц'].unique())
        
        print(f"   ✅ Годы: {years}")
        print(f"   ✅ Месяцы: {months}")
        print(f"   ✅ Период: {dates.min().strftime('%Y-%m')} до {dates.max().strftime('%Y-%m')}")
        
        return data, date_col, years[0], years[-1]
        
    except Exception as e:
        print(f"❌ Ошибка при обработке даты: {e}")
        raise ValueError(f"Невозможно обработать дату в столбце '{date_col}'")

def find_column_mapping(data, target_columns):
    """Поиск соответствия столбцов"""
    print("\n🔍 Поиск соответствия столбцов...")
    
    mapping = {}
    available_cols = data.columns.tolist()
    
    for target in target_columns:
        print(f"\n🔍 Ищем соответствие для '{target}':")
        
        # Прямое совпадение
        if target in data.columns:
            mapping[target] = target
            print(f"   ✅ Прямое совпадение: {target}")
            continue
        
        # Поиск по частичному совпадению
        best_match = None
        best_score = 0
        
        for col in available_cols:
            score = 0
            col_clean = col.lower().replace('\xa0', ' ').replace('-', ' ').replace('_', ' ')
            target_clean = target.lower().replace('_', ' ')
            
            # Специальная логика для разных признаков
            if 'оффз' in target.lower() and ('офз' in col_clean or 'доходн' in col_clean):
                score += 3
            elif 'ипц' in target.lower() and ('ипц' in col_clean or 'цен' in col_clean):
                score += 3
            elif 'зп' in target.lower() and ('зп' in col_clean or 'зарплат' in col_clean):
                score += 3
            elif target_clean in col_clean:
                score += 2
            
            if score > best_score:
                best_score = score
                best_match = col
        
        if best_match and best_score > 0:
            mapping[target] = best_match
            print(f"   ✅ Найдено: '{best_match}' (счёт: {best_score})")
        else:
            print(f"   ❌ Соответствие не найдено")
            mapping[target] = None
    
    return mapping

def create_full_features(data, target_col):
    """Создание всех признаков как в обучающем пайплайне"""
    print("\n🔧 Создание всех признаков...")
    
    # 1. Временные признаки
    data['Месяц_sin'] = np.sin(2 * np.pi * (data['Месяц'] - 1) / 12)
    data['Месяц_cos'] = np.cos(2 * np.pi * (data['Месяц'] - 1) / 12)
    data['Квартал'] = (data['Месяц'] - 1) // 3 + 1
    print(f"   ✅ Временные признаки созданы")
    
    # 2. Лаги целевой переменной
    data[f'{target_col}_lag_1'] = data[target_col].shift(1)
    data['target_lag_1'] = data[target_col].shift(1)
    print(f"   ✅ Лаги созданы: {target_col}_lag_1, target_lag_1")
    
    # 3. Технические индикаторы для основных признаков
    feature_mapping = {
        'Доходность_ОФЗ': None,
        'ИПЦ': None, 
        'ЗП': None
    }
    
    # Находим соответствие основных признаков
    for required, _ in feature_mapping.items():
        for col in data.columns:
            if required.replace('_', '').lower() in col.lower().replace(' ', '').replace('_', ''):
                feature_mapping[required] = col
                break
    
    # Добавляем найденные признаки в список для обработки
    main_features = ['Месяц_sin', 'Месяц_cos', 'Квартал']
    for req_feature, actual_feature in feature_mapping.items():
        if actual_feature:
            main_features.append(actual_feature)
            print(f"   ✅ Найден основной признак: {req_feature} -> {actual_feature}")
    
    # Создаем технические индикаторы
    for feature in main_features:
        if feature in data.columns:
            # MA3
            data[f'{feature}_MA3'] = data[feature].rolling(window=3, min_periods=1).mean()
            # std3
            data[f'{feature}_std3'] = data[feature].rolling(window=3, min_periods=1).std().fillna(0)
            print(f"   ✅ Технические индикаторы для {feature}")
    
    return data

def universal_forecast_workflow(file_path="2016_year_data.xlsx"):
    """Универсальный цикл прогнозирования с автоматическим определением года"""
    print("🚀 УНИВЕРСАЛЬНЫЙ ЦИКЛ ПРОГНОЗИРОВАНИЯ")
    print("=" * 60)
    
    # 1. Загрузка модели
    model_components = load_model()
    if not model_components:
        return None
    
    # 2. Загрузка данных и определение года
    data, start_year, end_year = load_data_and_detect_year(file_path)
    
    # 3. Обработка даты
    data, date_col, data_start_year, data_end_year = process_date_column(data)
    
    # Определяем год для прогноза
    forecast_year = data_end_year + 1
    print(f"\n🎯 ЗАДАЧА: Прогноз на {forecast_year} год на основе данных за {data_start_year}-{data_end_year}")
    
    # 4. Создание признаков
    target_col = 'Прирост вкладов физических лиц в рублях (млн руб)'
    data = create_full_features(data, target_col)
    
    # 5. Сортировка по дате
    data = data.sort_values(['Year', 'Месяц']).reset_index(drop=True)
    
    # 6. Подготовка признаков
    print(f"\n🎯 Подготовка признаков для модели...")
    
    # Находим соответствие признаков
    feature_mapping = find_column_mapping(data, model_components['selected_features'])
    
    # Переименовываем столбцы
    rename_mapping = {v: k for k, v in feature_mapping.items() if k != v}
    if rename_mapping:
        data = data.rename(columns=rename_mapping)
        print(f"📝 Переименование:")
        for old, new in rename_mapping.items():
            print(f"   {old} -> {new}")
    
    # Проверяем признаки
    missing_features = [f for f in model_components['selected_features'] if f not in data.columns]
    if missing_features:
        print(f"❌ Отсутствуют признаки: {missing_features}")
        return None
    
    print(f"✅ Все признаки модели подготовлены!")
    
    # 7. Подготовка X и y
    X = data[model_components['selected_features']]
    y = data[target_col]
    
    print(f"📊 Данные для модели: {X.shape}, y: {len(y)}")
    
    # 8. Создание и обучение модели с k=3
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
        ('model', KNeighborsRegressor(n_neighbors=3, p=2))
    ])
    
    # Обучаем модель
    pipeline.fit(X, y)
    print(f"✅ Модель обучена с k=3")
    
    # 9. Прогноз на следующий год
    print(f"\n🔄 Прогноз на {forecast_year} год...")
    
    # Берем последние 12 строк для прогноза
    X_last_12 = X.tail(12)
    
    # Делаем прогноз
    forecast = pipeline.predict(X_last_12)
    
    # 10. Вывод результатов
    forecast_dates = pd.date_range(start=f'{forecast_year}-01-01', periods=12, freq='MS')
    
    print(f"\n📅 ПРОГНОЗ НА {forecast_year} ГОД (модель k=3)")
    print("=" * 70)
    
    for i, (date, value) in enumerate(zip(forecast_dates, forecast)):
        month = i + 1
        print(f"   {forecast_year}-{month:02d}: {value:>12,.2f} млн руб")
    
    # 11. Проверяем разнообразие прогноза
    unique_values = len(np.unique(np.round(forecast, 2)))
    print(f"\n📊 Анализ прогноза:")
    print(f"   Уникальных значений: {unique_values}")
    if unique_values > 2:
        print(f"   ✅ Прогноз разнообразный!")
    else:
        print(f"   ❌ Прогноз слишком однообразный")
    
    # 12. Сохранение результатов
    final_forecast = pd.Series(forecast, index=forecast_dates)
    filename_csv = f"forecast_{forecast_year}_k3_universal.csv"
    filename_xlsx = f"forecast_{forecast_year}_k3_universal.xlsx"
    
    final_forecast.to_csv(filename_csv, header=['Прогноз'])
    final_forecast.to_excel(filename_xlsx, header=['Прогноз'])
    
    print(f"\n💾 Прогноз сохранён:")
    print(f"   - {filename_csv}")
    print(f"   - {filename_xlsx}")
    
    return final_forecast

if __name__ == "__main__":
    try:
        # Можно изменить путь к файлу данных
        data_file = "2016_year_data.xlsx"  # Изменить на нужный файл
        final_forecast = universal_forecast_workflow(data_file)
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

def process_uploaded_file(file_path):
    """Обертка для вызова из Flask"""
    try:
        # Просто вызываем существующую функцию
        final_forecast = universal_forecast_workflow(file_path)
        
        if final_forecast is None:
            return None, "Ошибка обработки файла"
        
        # Извлекаем год из индекса прогноза
        forecast_year = final_forecast.index[0].year
        
        # Подготавливаем результаты для Flask
        results = {
            'forecast_year': forecast_year,
            'data_period': "см. логи",  # Эта информация выводится в консоль
            'forecast_values': [(f"{forecast_year}-{i+1:02d}", float(value)) 
                              for i, value in enumerate(final_forecast.values)],
            'filename_csv': f"forecast_{forecast_year}_k3_universal.csv",
            'filename_xlsx': f"forecast_{forecast_year}_k3_universal.xlsx",
            'unique_values': len(np.unique(np.round(final_forecast.values, 2)))
        }
        
        return results, None
        
    except Exception as e:
        return None, str(e)