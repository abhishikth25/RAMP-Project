#file: combine_external_data.py
#author: Kamiel Fokkink, Sai Abhishikth
#Pre-process data files from different sources, and put them all in one file
#external data, in the correct format.

import pandas as pd
from datetime import datetime, timedelta, date
from geopy.distance import distance
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import holidays
from xgboost import XGBRegressor
import numpy as np

def add_mobility_data(data):
    #Read in mobility data
    mobility20 = pd.read_csv("Found data/2020_FR_Region_Mobility_Report.csv")
    mobility21 = pd.read_csv("Found data/2021_FR_Region_Mobility_Report.csv")
    
    #Append dataframes for 2020 and 2021
    mobility = mobility20.append(mobility21)
    
    #Filter the France dataset to use only data for Paris region
    mobility = mobility[mobility['sub_region_2']=='Paris']
    
    #Convert string field to datetime
    mobility['date'] = pd.to_datetime(mobility['date'])
    
    #Shorten column names
    mobility = mobility.rename(columns={
        'retail_and_recreation_percent_change_from_baseline': 'recreation_mob',
        'grocery_and_pharmacy_percent_change_from_baseline': 'grocery_mob',
        'parks_percent_change_from_baseline': 'parks_mob',
        'transit_stations_percent_change_from_baseline': 'transit_mob',
        'workplaces_percent_change_from_baseline': 'workplaces_mob',
        'residential_percent_change_from_baseline': 'residential_mob'
        })
    
    #Drop useless columns
    mobility = mobility.drop(['country_region_code', 'country_region', 
                'sub_region_1', 'sub_region_2', 'metro_area', 
                'iso_3166_2_code', 'census_fips_code', 'place_id'], axis=1)
    
    #Read in bike counter data
    bike_counters = pd.read_parquet("../../data/train.parquet")
    bike_counters = bike_counters.sort_values('date')
    
    #Make a train dataset, to predict log bike count from different mobility components
    train_data = pd.merge_asof(bike_counters[['date','log_bike_count']],mobility,on='date')
    #Split train data into X all mobility components
    X_train = train_data.drop(['date','log_bike_count'],axis=1)
    #And y the log bike count
    y_train = train_data['log_bike_count']
    
    #Train a linear XGBregressor to predict bike count from mobilities
    regressor = XGBRegressor(booster='gblinear')
    regressor.fit(X_train, y_train)
    #Find linear weights of each mobility component
    coeffs = regressor.coef_
    
    #Compute new column weighted mobility, a linear combination of previous mobilities
    mobility['weighted_mob'] = mobility.drop('date',axis=1).multiply(coeffs, axis=1).sum(axis=1)
    #Drop previous columns to only keep weighted data, and greatly reduce dimension
    mobility = mobility.drop(['recreation_mob','grocery_mob','parks_mob',
                             'transit_mob','workplaces_mob','residential_mob'],axis=1)
    
    #Merge mobility data with original dataframe
    out = pd.merge_asof(data, mobility, on='date')
    
    return out

def add_weather_data(data):
    #Read in weather
    weather = pd.read_csv("Found data/WeatherData.csv", index_col=0)
    
    #Convert two columns into one date columns
    weather['date'] = pd.to_datetime(weather.index + ' ' + weather['Time'])
    
    #Drop useless columns
    weather = weather.drop('Time', axis=1)
    
    #Merge weather data with original dataframe
    out = pd.merge_asof(data, weather, on='date')
    
    return out

def add_car_counts(data):
    #Read in car counting data
    path = "Found data/comptage-velo-donnees-"
    cars20 = pd.read_csv(path+"sites-comptage-2020.csv",sep=';')
    cars21 = pd.read_csv(path+"compteurs-2021.csv",sep=';')
    
    #Drop useless columns
    cars20 = cars20.drop(['Date d\'installation du point de comptage',
            'Nom du point de comptage', 'Lien vers photo du point de comptage',
            'Coordonnées géographiques'], axis=1)
    cars21 = cars21.drop(['Nom du compteur', 'Identifiant du site de comptage',
            'Nom du site de comptage', 'Date d\'installation du site de comptage',
            'Lien vers photo du site de comptage', 'Coordonnées géographiques',
            'Identifiant technique compteur'], axis=1)
    
    #Rename such that names of both dataframes match
    cars20 = cars20.rename(columns={'Identifiant du point de comptage':
                                    'Identifiant du compteur'})
    
    #Append dataframes for 2020 and 2021
    cars = cars20.append(cars21)
    
    #Convert string into datetime column
    cars['date'] = pd.to_datetime(cars['Date et heure de comptage'])
    cars['date'] = cars['date'].apply(lambda row: row.replace(tzinfo=None))
    
    col_names = ['date', '3h_car_count', '1h_car_count']
    #Initiate new dataframe containing total car count in Paris for the
    #preceding 3 hours and 1 hour from a date
    car_counts = pd.DataFrame(columns=col_names)
    
    #Loop over all dates to compute car counts
    for date in data['date']:
        #Select subset of data corresponding to 3 preceding hours
        earlier_3h = cars.loc[(cars['date']<=date) &
                              (cars['date']>=(date-timedelta(hours=3)))]
        #Sum total car count in period
        cars_3h = earlier_3h.sum()['Comptage horaire']
        
        #Select subset of data corresponding to 1 preceding hour
        earlier_1h = earlier_3h[earlier_3h['date']>=date-timedelta(hours=1)]
        cars_1h = earlier_1h.sum()['Comptage horaire']
        
        #Add found car counts to dataframe
        cc = pd.DataFrame(data=[[date, cars_3h, cars_1h]],columns=col_names)
        car_counts = car_counts.append(cc,ignore_index=True)
        
    #Merge car count data with original dataframe
    out = pd.merge_asof(data, car_counts, on='date')
    
    return out
  
def add_covid_data(data):
    #Read in covid data
    covid  = pd.read_csv('Found data/Covid_Data_Exhaustive_France.csv',sep=';')
    
    #Filter to include only relevant dates
    mask = (covid['date'] >= '2020-09-01') & (covid['date'] <= '2021-10-21')
    covid  = covid.loc[mask]
    
    #Convert string into datetime column
    covid['date'] = pd.to_datetime(covid['date'])

    #Relevant data columns to include
    relevant_covid = covid[['date','new_cases','new_deaths','total_vaccinations','people_vaccinated', 
                    'people_fully_vaccinated','stringency_index']]
    
    #Merge covid data with original dataframe
    out = pd.merge_asof(data, relevant_covid, on='date')
    
    return out             

def add_construction_data(data):
    #Read in construction data
    construction = pd.read_csv("Found data/chantiers-perturbants.csv", sep=";")
    
    #Read in bike counter data
    bike_counters = pd.read_parquet("../../data/train.parquet")
    
    dates = pd.date_range(data.min()['date'], data.max()['date'])
    #Columns count whether there was a construction within x metres of a bike counter
    cols = ['500m construction','1000m construction']
    obstruction_counts = pd.DataFrame(index=dates, columns=cols)
    obstruction_counts.fillna(0, inplace=True)
    
    #Extract coordinates of unique bike counters
    counter_locations = bike_counters.groupby(['latitude','longitude']).size().reset_index()
    counter_weights = dict()
    
    #Compute average bikes passing by a counter, to give importance weights to counter
    for index, row in counter_locations.iterrows():
        counter_logs = bike_counters[(bike_counters['latitude']==row['latitude']) & 
                                     (bike_counters['longitude']==row['longitude'])]['bike_count']
        total_count = counter_logs.sum()
        average_count = total_count/counter_logs.shape[0]
        #Fill a dict with as keys the geographical location, and as values average bike count
        counter_weights[(row['longitude'],row['latitude'])] = average_count
    
    #For all construction instances, check if it is close to a bike counter
    for index, row in construction.iterrows():
        #Find location, start and end date of construction
        coords = eval(row[-2])
        start = pd.to_datetime(row[10])
        end = pd.to_datetime(row[11])
        
        #Initiate polygon shape, representing area covered by construction
        if coords['type'] == 'Polygon':
            shape = Polygon(eval(row[-2])['coordinates'][0])
        if coords['type'] == 'MultiPolygon':
            shape = Polygon(eval(row[-2])['coordinates'][0][0])
            
        #For all bike counters check if it is close to construction
        for k, v in counter_weights.items():
            place = Point(k)
            #Multiply distance in degrees by 111, to get distance in km
            dist = shape.distance(place) * 111 
            #Check how close bike counter is from construction area
            if (dist < 0.5):
                #Add average bike count to dataframe, indicating impact of obstruction
                obstruction_counts.loc[(obstruction_counts.index>=start) & 
                    (obstruction_counts.index<=end), '500m construction'] += v
            if (dist < 1):
                obstruction_counts.loc[(obstruction_counts.index>=start) & 
                    (obstruction_counts.index<=end), '1000m construction'] += v
    
    #Merge construction data with original dataframe
    out = pd.merge_asof(data, obstruction_counts, left_on='date', right_index=True)

    return out

def add_holidays(data):
    #Make empty dataframe with 0 for every date in original dataframe
    dates = pd.date_range(data.min()['date'], data.max()['date'])
    holiday = pd.DataFrame(index=dates, columns=['holiday'])
    holiday.fillna(0, inplace=True)
    
    #Load in French holidays in 2020 and 2021
    French_holidays = holidays.FR(years=[2020,2021])
    #Add a binary 1 code for if a date is an official holiday
    for date in dates:
        if date in French_holidays:
            holiday.loc[date] = 1
    
    #Merge holiday data with original dataframe
    out = pd.merge_asof(data, holiday, left_on='date', right_index=True)
    return out

def add_daylight_saving(data):
    dates = pd.date_range(data.min()['date'], data.max()['date'])
    daylights = pd.DataFrame(index=dates,columns=['daylight'])
    for date in dates:
        days = (date - pd.datetime(2000,12,21)).days
        x = 1 - np.tan(np.radians(48.84)) * np.tan(np.radians(23.44) * np.cos(days*2*np.pi/365.25))
        daylight = 24 * np.degrees(np.arccos(1 - np.clip(x, 0, 2)))/180
        daylights.loc[date]['daylight'] = daylight
    
    out = pd.merge_asof(data, daylights, left_on='date', right_index=True)
    return out

if __name__ == "__main__":
    """Main framework that sequentially processes and adds data from different 
    source files into our one dataframe containing all complete external data"""
    
    external = pd.read_csv("external_data.csv")
    external['date'] = pd.to_datetime(external['date'])
    external = external.sort_values('date')
    
    external = add_mobility_data(external)
    external = add_weather_data(external)
    external = add_car_counts(external)
    external = add_covid_data(external)
    external = add_construction_data(external)
    external = add_holidays(external)
    external = add_daylight_saving(external)
    
    external.to_csv("External_processed_data.csv",index=False)