#file: combine_external_data.py
#author: Kamiel Fokkink, Sai Abhishikth
#Pre-process data files from different sources, and put them all in one file
#external data, in the correct format.

import pandas as pd
from datetime import datetime, timedelta

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
    
    #Merge cleaned mobility data with original dataframe
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
    df2  = pd.read_csv('Found data/Covid_Data_Exhaustive_France.csv')
    mask = (df2['date'] >= '2020-09-01') & (df2['date'] <= '2021-09-09')
    df2  = df2.loc[mask]
    data.rename(columns = {'date': 'datetime'}, inplace = True)
    data[['date','time']] = df1_copy.datetime.str.split(" ",expand = True)

    relevant_df2 = df2[['date','new_cases','new_deaths','total_vaccinations','people_vaccinated', 
                    'people_fully_vaccinated','stringency_index']]
    out = relevant_df2.merge(data, how ='left', on ='date')
    out = out.drop('date', axis = 1)
    out = out.drop('time', axis = 1)
    out.rename(columns = {'datetime': 'date'}, inplace = True)
    
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
    
    external.to_csv("External_processed_data.csv",index=False)
