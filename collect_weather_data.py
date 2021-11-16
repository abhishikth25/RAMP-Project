#file: collect_weather_data.py#author: Kamiel Fokkink#Scrape publicly available data from worldweatheronline.com, to have more#detailed data on weather in Paris.import mechanicalsoupimport pandas as pdimport numpy as npimport regex as refrom datetime import date, timedeltadef extract_info_page(page,date):    """From a given webpage, extract all relevant weather information from    the table. Return a dataframe"""        cols = ['Date','Time','Temp','Wind','Rain','Cloud']    df = pd.DataFrame(columns=cols)        divs = page.soup.find_all('section')[1].find_all('div')    #Indexes of html elements corresponding to useful cells in table    div_indexes = np.array([37,39,41,43,45])        for _ in range(8): #There are 8 datapoints per date        #Extract text from divs            data = [divs[index].text for index in div_indexes]         #Take out all non-numerical characters from results        clean_data = [re.sub('\s.*$|%','',res) for res in data]        datarow = [date] + clean_data        df = df.append(pd.DataFrame([datarow], columns = cols))        #Update indexes, to match new row in table        div_indexes += 12          return dfdef dates_looper(dates,browser,url):    """Loop over a range of given dates, find the webpage for weather data on    those dates. Put all collected data into a dataframe and return"""        cols = ['Time','Temp','Wind','Rain','Cloud']    df = pd.DataFrame(columns=cols)    page = browser.get(url)        for date in dates:        soup = page.soup        #Interact with input field, to choose a different date        soup.find_all('input')[3]['value'] = date        #Find a new webpage with information for chosen date        page = browser.submit(soup,url)        #Use previous function to make a dataframe with weather data        result = extract_info_page(page,date)        result.set_index('Date',inplace=True)        df = df.append(result)    return dfif __name__ == '__main__':    """Main framework that executes the whole data collection process. Run    this file once to create a file with weather data"""        start_date = date(2020,9,1) #Earliest date of bike dataset    end_date = date(2021,9,9) #Latest date of bike dataset    dates = pd.date_range(start_date,end_date,freq='d')    browser = mechanicalsoup.Browser()    url = "https://www.worldweatheronline.com/paris-weather-history/ile-de-france/fr.aspx"    data = dates_looper(dates,browser,url)    data.to_csv("WeatherData.csv",index=True) #Save dataframe into a csv file