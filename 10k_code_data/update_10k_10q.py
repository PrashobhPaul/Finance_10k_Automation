#!/usr/bin/env python
# coding: utf-8

# In[14]:


# Import the required library
from bs4 import BeautifulSoup
from collections import Counter
from collections import OrderedDict
import spacy
from string import punctuation
import collections
import numpy as np
import pickle
from sec_edgar_downloader import Downloader
import json
import bs4
import bs4 as bs
import requests
import re
import time
import datetime
from datetime import date
from dateutil.relativedelta import relativedelta
from sec_api import QueryApi
from sec_api import ExtractorApi # https://pypi.org/project/sec-api/
import matplotlib.pyplot as plot
import pytextrank
from fnmatch import fnmatch
import os
import pandas as pd
import numpy as np
import shutil
import unicodedata
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
'en_US.UTF-8'
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import itertools
from time import sleep
from apscheduler.schedulers.background import BackgroundScheduler, BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import schedule


# In[84]:


global DIR_PATH
DIR_PATH = "C:\\Users\\deepali.b\\DL_tensorflow\\10k_document\\"
#company_ticker = str(input("Please enter the company ticker"))
year = date.today().year
form_type_q = '10-Q'
form_type_k = '10-K'
company_ticker = ['MDT','STE','SYK','JNJ','GMED']


# In[6]:


year = int(input("Input year: "))
n = int(input("Enter the number of companies: "))
print("\n")


# In[51]:


year_list = [year-i for i in range(output_range)]


# In[52]:


# scheduling job to run function everyday
scheduler = BackgroundScheduler()
scheduler.start()
trigger = CronTrigger(
    year="*", month="*", day="*", hour="5", minute="0", second="5"
)


# In[53]:


def crete_temp_folder(folder_path):
    #folder_path = '10k_download/'
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        #os.makedirs(newpath)
    else:
        os.makedirs(folder_path)
    return folder_path


# #### 10q download and update

# In[54]:


def get_time_update():
    print('DIR_PATH')
    print(DIR_PATH)
    with open(DIR_PATH+'\\time_update.pkl', 'rb') as f:
        time_updated_data = pickle.load(f)
    time_updated_data["STE"].iloc[0] = '30-08-2022'
    time_update = list(time_updated_data.iloc[0].values)
    date_format = [datetime.datetime.strptime(elem, '%d-%m-%Y') for elem in time_update]
    now = datetime.datetime.now()
    time_diff = [(now-elem).days for elem in date_format]
    time_updated_data.loc[len(time_updated_data.index)] = time_diff
    time_diff1 = time_updated_data.iloc[1]
    time_diff1s = time_diff1.gt(95)
    updated_company = list(time_diff1s[time_diff1s].index.values)
    return updated_company


# In[55]:


def extract_fin_elem(file_location,company_ticker,file,form_type):
    '''This function takes 4 inputs as mentioned in arguments and returns the list of 
    Net sales and Cost of service through regex pattern'''
    #create empty list to append net sales and cost of service values
    filter_net_sales = []
    filter_cost_service = []
    #create empty list to append values with their respective quarters
    quarter_list =[]
    #path for search the text file of 10-Q report
    path = file_location + "sec-edgar-filings\\"+company_ticker +"\\"+form_type+"\\"+ file
    os.chdir(path)
    #open text file
    f = open('full-submission.txt', 'r')
    #Read text file
    content = f.read()
    #word = 'CONFORMED PERIOD OF REPORT:'
    #regex pattern for extract report date
    date_reg = re.findall(r'(?P<name>(CONFORMED PERIOD OF REPORT:)[\W\d]*)',content)[0][0]
    dtreg_lst = date_reg.split()
    for sub1 in dtreg_lst:
        reg12 = re.sub('[^0-9]+', '', sub1)
        if reg12:
            year_date = reg12
    #we only need month from report date.
    month = year_date[4:6]
    #print(year_end)
    # Regex to find <DOCUMENT> tags
    doc_start_pattern = re.compile(r'<DOCUMENT>')
    doc_end_pattern = re.compile(r'</DOCUMENT>')
    # Regex to find <TYPE> tag prceeding any characters, terminating at new line
    type_pattern = re.compile(r'<TYPE>[^\n]+')
    # Create 3 lists with the span indices for each regex
    ### There are many <Document> Tags in this text file, each as specific exhibit like 10-K, EX-10.17 etc
    ### First filter will give us document tag start <end> and document tag end's <start> 
    ### We will use this to later grab content in between these tags
    doc_start_is = [x.end() for x in doc_start_pattern.finditer(content)]
    doc_end_is = [x.start() for x in doc_end_pattern.finditer(content)]

    ### Type filter is interesting, it looks for <TYPE> with Not flag as new line, ie terminare there, with + sign
    ### to look for any char afterwards until new line \n. This will give us <TYPE> followed Section Name like '10-K'
    ### Once we have have this, it returns String Array, below line will with find content after <TYPE> ie, '10-K' 
    ### as section names
    doc_types = [x[len('<TYPE>'):] for x in type_pattern.findall(content)]
    # Create a loop to go through each section type and save only the 10-K section in the dictionary
    document = {}

    for doc_type, doc_start, doc_end in zip(doc_types, doc_start_is, doc_end_is):
        if doc_type == form_type:
            document[doc_type] = content[doc_start:doc_end]
    #Print 10-Q document dict
#     print(document[form_type][0:1000])
    
    #Regex pattern for extract statements of income
    regex = re.compile(r'(Consolidated Statements of Income)|(CONSOLIDATED STATEMENTS OF INCOME)|(CONSOLIDATED STATEMENTS OF EARNINGS)|(CONSOLIDATED STATEMENTS OF OPERATIONS AND COMPREHENSIVE INCOME)|(CONSOLIDATED STATEMENTS OF OPERATIONS)')
    # Use finditer to math the regex
    matches = regex.finditer(document[form_type])
    # Write a for loop to print the matches
#     for match in matches:
#         print(match)

    # Matches
    matches = regex.finditer(document[form_type])
    # Create the dataframe
    test_df = pd.DataFrame([(x.group(), x.start(), x.end()) for x in matches])
    test_df.columns = ['item', 'start', 'end']
    # Display the dataframe
    test_df.head()
    
    #Replacing the end value to none
    test_df['end'] = test_df['end'].replace('','',regex=True,inplace=True)
    #Replacing the financial metric name to parameter
    test_df.loc[0,"item"] = "Parameter"
    test_df.head()
    
    # Drop duplicates
    pos_dat = test_df.sort_values('start', ascending=True).drop_duplicates(subset=['item'])
    # Display the dataframe
    pos_dat
    
    # Set item as the dataframe index
    pos_dat.set_index('item', inplace=True)
    # display the dataframe
    pos_dat
    
    #Get item 8
    item_8_raw = document[form_type][pos_dat['start'].loc['Parameter']:pos_dat['end'].loc['Parameter']]
    
    # First convert the raw text we have to exrtacted to BeautifulSoup object 
    Fin_8A = BeautifulSoup(item_8_raw, 'lxml')
    
    # Finding out metrices using tags
    Fin_headings=[]
    for i in Fin_8A.findAll('table'):
        Fin_headings.append(i.text)
    for i in Fin_8A.findAll('span'):
        Fin_headings.append(i.text)
    for i in Fin_8A.findAll('td'):
        Fin_headings.append(i.text)
    Fin_headings = ''.join(Fin_headings)
    Fin_text = unicodedata.normalize("NFKD",Fin_headings)
    
    ###Process of extract Net sales and cost of service through regex pattern
    try:
        Reg = re.findall(r'(?P<name>(Net.sales|Sales.to.customers|Net.Sales|Sales)[\W\d]*)',Fin_text) #MDT SYK 
#         Reg = re.findall(r'(?P<name>(Sales\sto\scustomers\s\((.*?)\])[\W\d]*)',Fin_text) #JNJ 
        net_sales = Reg[0][0]   
    except:
        pass
    try:
        Reg = re.findall(r'(?P<name>(Total.revenues)[\W\d]*)',Fin_text) #MDT SYK 
        net_sales = Reg[-1][0]
    except:
        pass
    try:
        Reg = re.findall(r'(?P<name>(Cost.of.products.sold,.excluding.amortization.of.intangible.assets|Cost.of.products.sold|Cost.of.goods.sold|Cost.of.sales|Total.cost.of.revenues)[\W\d]*)',Fin_text) #MDT SYK 
        cost_of_service = Reg[0][0]
    except:
        pass
    #split the whole regex string to identify only useful string
    sales_lst = net_sales.split()
    for char in sales_lst:
    #\$[0-9]+   [^$0-9]+
    #regex pattern to extract only numbers(i.e. net sales)
        reg1 = re.sub('[^0-9]+', '', char)
        if reg1:
            #Append the needed string to new list
            filter_net_sales.append(reg1)
            break
        else:
            continue
    #split the whole regex string to identify only useful string
    service_lst = cost_of_service.split()
    for elem in service_lst:
        #regex pattern to extract only numbers(i.e. cost of service)
        reg2 = re.sub('[^$0-9]+', '', elem)
        if reg2:
            #Append the needed string to new list
            filter_cost_service.append(reg2)
            break
        else:
            continue
    

    #For the company- steris and GMED the values will be divide by 1000
    if (company_ticker == 'STE' or company_ticker == 'GMED'):
        for i,x in enumerate(filter_net_sales):
            filter_values = round(int(x)/1000)
            filter_net_sales[i] = filter_values
        for j,y in enumerate(filter_cost_service):
            fil_value = round(int(y)/1000)
            filter_cost_service[i] = fil_value
    #Here we create empty list for append the company ticker with special symbol.
    company_lst = []
    if form_type == '10-Q':
        comp = company_ticker+"(M$)"
    elif form_type == '10-K':
        comp = company_ticker+"(M$)"
    company_lst.append(comp)
#     quarter_list = list(OrderedDict.fromkeys(quarter_list))
    #Create list of list with the parameters of company name and it's respected financial value 
    net_sales_list = list(zip(company_lst,filter_net_sales))
    cost_service_list = list(zip(company_lst,filter_cost_service))
    return net_sales_list,cost_service_list
#%%%


# In[56]:


def calling_function(file_location,company_list,form_type):
    '''This function takes the file location and company list as a input and returns 
    4 dataframe of 4 financial parameters.(i.e.- Net sales,Cost of service,Gross margin,Revenue growth)'''
    #Create empty list of sales,cost of service
    sales_list = []
    service_list = []
    company_list = ["MDT", "STE", "SYK", "JNJ", "GMED"]
    #iterate through all companies to extract financial parameters for all companies 
    for company in company_list:
        path = file_location +"\\sec-edgar-filings\\"+ company +"\\"+form_type
        file_list = os.listdir(path)
        for file in file_list:
            print('file')
            print(file)
            #calling the extract_fin_elem function
            net_sales_list,cost_service_list = extract_fin_elem(file_location,company,file,form_type)
            #Extend the list in parent list.
            sales_list.extend(net_sales_list)
            service_list.extend(cost_service_list)       
    #create dataframe for net sales  
    sales_df = pd.DataFrame(sales_list, columns=['Company', 'Net sales values(M$)'])
#     print(sales_df)
    #create list of all values of Company column
    company_values = sales_df['Company'].tolist()
    #Transpose the dataframe to fulfilled the requirenment 
    sales_trans_df = sales_df.set_index('Company').T
    #create dataframe for cost of service
    service_df = pd.DataFrame(service_list, columns=['Company', 'Cost of service values(M$)'])
    #Transpose the dataframe to fulfilled the requirenment 
    cost_trans_df = service_df.set_index('Company').T
    #To calculate the gross margin we need cost of service value and net sales values so we merged both dataframe
    combined_df = pd.merge(sales_df,service_df,on='Company')
    #Apply function to calculate gross margin
    combined_df['Gross margin(%)'] = combined_df.apply(lambda x: calc_gross_margin(x['Net sales values(M$)'], x['Cost of service values(M$)']), axis=1)
    #Change the column values through lambda expression.
    combined_df['Company']=combined_df['Company'].apply(lambda x: x.replace('(M$)','(%)') if x.endswith('(M$)') else x)
    #We only need Company and gross margin in this dataframe so we created dataframe with these two columns
    combined_df = combined_df[['Company','Gross margin(%)']]
    #Transpose the dataframe to fulfilled the requirenment 
    gross_margin_trans_df = combined_df.set_index('Company').T
    #load the existing pickle file for the use of net sales dataframe
#     print(form_type)
    if form_type == '10-Q':
        with open('net_sales_10q.pkl', 'rb') as f:
            net_sales_10q = pickle.load(f)
        #revenue growth function calling
        revenue_growth_values = calc_revenue_growth(net_sales_10q,sales_df,company_values)
    elif form_type == '10-K':
#         print('YES-10-K')
        with open('net_sales.pkl', 'rb') as f:
            net_sales = pickle.load(f)
        #revenue growth function calling
        revenue_growth_values = calc_revenue_growth(net_sales,sales_df,company_values)
    #Create the list of list for prepare the revenue growth dataframe
    revenue_growth_lst = list(zip(company_values,revenue_growth_values))
    #Revenue growth dataframe
    growth_df = pd.DataFrame(revenue_growth_lst, columns=['Company', 'Revenue growth(%)'])
    #Change the column values through lambda expression.
    growth_df['Company']=growth_df['Company'].apply(lambda x: x.replace('(M$)','(%)') if x.endswith('(M$)') else x)
    #Transpose the dataframe to fulfilled the requirenment 
    rev_growth_df = growth_df.set_index('Company').T
    return sales_trans_df,cost_trans_df,gross_margin_trans_df,rev_growth_df
#%%%


# In[57]:


def calc_revenue_growth(net_sales,sales_df,company_list):
    '''This function takes the current net sales values(i.e.-from sales_df) and
    previous net sales values(i.e.-from net_sales_10q) and company list as a 
    input and return the revenue growth of all companies as a list'''
    revenue_growth_values = []
    print('sales_df in revenue')
    print(sales_df)
    for company in company_list:
        revenue_growth = round((int(sales_df.loc[sales_df['Company'] == company, 'Net sales values(M$)'].iloc[0]) - int(locale.atoi(net_sales.loc[net_sales.index[0], company])))/int(locale.atoi(net_sales.loc[net_sales.index[0], company]))*100,2)
        revenue_growth_values.append(revenue_growth)
    return revenue_growth_values
#%%%


# In[58]:


def calc_gross_margin(net_sales,cost_service):
    '''This function takes the net sales & cost of service values as a input and return 
    the gross margin of each row. Row wise one by one calculate for all the values'''
    gross_margin = round((int(net_sales) - int(cost_service))/int(net_sales)*100,2)
    return gross_margin
#%%%


# In[59]:


def get_quarter(last_q):
    if len(get_time_update())>0:
        # extract from code and update in pickle also update time pickle
        #last_q = net_sales_10q['Quarters/ USD millions'][0]
        print('last_q')
        print(last_q)
        quarter = last_q.split("-")[0]
        print('quarter')
        print(quarter)
        year = int(last_q.split("-")[1])
        if quarter!='Q4 ':
            new_q = int([re.findall(r'(\w+?)(\d+)', last_q.split("-")[0])[0]][0][1])+1
            new_quarter = 'Q' + str(new_q)
            new_quarter_year = new_quarter + ' - ' + str(year)
        else:
            new_quarter_year = 'Q1 ' + ' - ' + str(year+1)
    print('new_quarter')
    print(new_quarter_year)
    return new_quarter_year


# In[60]:


def download_files(file_location,company_ticker, form_type):
    try:
#         if os.path.exists(file_location):
#             shutil.rmtree(file_location)
        dl = Downloader(file_location)
        dl.get(form_type,company_ticker,amount=1)
        print('Download report successfully')
        print(file_location)
    except Exception as e:
        print(e)
        print("Error in downloading file")
    return file_location


# In[61]:


def extract_file_date(file_location,company_ticker):
    
    '''This function returns the form type with their file date'''
    #create empty list to store form type with file date
    form_lst_with_date = {}
    base_getcwd = DIR_PATH
    print('base_getcwd')
    print(base_getcwd)
    #extract forms types for every company at given path
    form_list = os.listdir(base_getcwd +file_location+'\\sec-edgar-filings\\'+company_ticker)
    for form in form_list:
        if(form =='10-K'):
            form = '10-K'
        else:
            form = '10-Q'
        #go to the form type location and extract all files present in that folder 
        file_list = os.listdir(base_getcwd+file_location+'\\sec-edgar-filings\\'+company_ticker+'\\'+form)
        base_path = base_getcwd +file_location+'\\sec-edgar-filings\\'+company_ticker+'\\'+form
        #for loop to iterate over every file present in that folder
        for file in file_list:
            path = base_path+'\\'+file
            #reach to the text file
            os.chdir(path)
            #open text file
            f = open('full-submission.txt', 'r')
            #read the content of text file
            content = f.read()
            #create regex pattern for extract filing date
            date_reg = re.findall(r'(?P<name>(FILED AS OF DATE:)[\W\d]*)',content)[0][0]
            #this variable contain file date with some unnecessary words, so split them for create list
            dtreg_lst = date_reg.split()
            #iterate through every words to find the our needed file date.
            for sub1 in dtreg_lst:
                #create regex pattern to store only file date.
                reg12 = re.sub('[^0-9]+', '', sub1)
                if reg12:
                    year_date = reg12
            #convert that string date to date time object
            date_object = datetime.datetime.strptime(year_date, '%Y%m%d').strftime('%Y-%m-%d')
            filed_date = date_object
            form_lst_with_date.update({form:filed_date})
#             form_lst_with_date.append(filed_date)
    return form_lst_with_date
#%%


# In[62]:


def calling_function_10k(file_location,company_list,form_type):
    '''This function takes the file location and company list as a input and returns 
    4 dataframe of 4 financial parameters.(i.e.- Net sales,Cost of service,Gross margin,Revenue growth)'''
    #Create empty list of sales,cost of service
    sales_list = []
    service_list = []
    #iterate through all companies to extract financial parameters for all companies 
    for company in company_list:
        path = file_location+"\\sec-edgar-filings\\" + company +"\\"+form_type
        file_list = os.listdir(path)
        for file in file_list:
            #calling the extract_fin_elem function
            net_sales_list,cost_service_list = extract_fin_elem(file_location,company,file,form_type)
            #Extend the list in parent list.
            sales_list.extend(net_sales_list)
            service_list.extend(cost_service_list)       
    #create dataframe for net sales  
    sales_df = pd.DataFrame(sales_list, columns=['Company', 'Net sales values(M$)'])
    print('sales_df')
    print(sales_df)
    #create list of all values of Company column
    company_values = sales_df['Company'].tolist()
    #Transpose the dataframe to fulfilled the requirenment 
    sales_trans_df = sales_df.set_index('Company').T
    #create dataframe for cost of service
    service_df = pd.DataFrame(service_list, columns=['Company', 'Cost of service values(M$)'])
    #Transpose the dataframe to fulfilled the requirenment 
    cost_trans_df = service_df.set_index('Company').T
    #load the existing pickle file for the use of net sales dataframe
#     print(form_type)
    pckl_file_path = DIR_PATH
    with open(pckl_file_path + 'net_sales_10q.pkl', 'rb') as f:
        net_sales_10q = pickle.load(f)
    with open(pckl_file_path + 'cost_of_service_10q.pkl', 'rb') as f:
        cost_of_service_10q = pickle.load(f)
    with open(pckl_file_path + 'net_sales.pkl', 'rb') as f:
            net_sales = pickle.load(f)
#     with open(pckl_file_path + 'cost_of_products.pkl', 'rb') as f:
#             cost_of_products.pkl = pickle.load(f)
    q4_sales_lst,q4_cost_lst = q4_calculation(net_sales_10q,cost_of_service_10q,sales_df,service_df,company_values)
    q4_sales_lst_zip = list(zip(company_values,q4_sales_lst))
    q4_cost_lst_zip = list(zip(company_values,q4_cost_lst))
    q4_sales_df = pd.DataFrame(q4_sales_lst_zip, columns=['Company', 'Net sales(M$)'])
    q4_cost_df = pd.DataFrame(q4_cost_lst_zip, columns=['Company', 'Cost Of Service(M$)'])
    q4_sales_trans_df = q4_sales_df.set_index('Company').T
    q4_cost_trans_df = q4_cost_df.set_index('Company').T
    
    #To calculate the gross margin we need cost of service value and net sales values so we merged both dataframe
    combined_df = pd.merge(q4_sales_df,q4_cost_df,on='Company')
    #Apply function to calculate gross margin
    combined_df['Gross margin(%)'] = combined_df.apply(lambda x: calc_gross_margin(x[combined_df.columns.values.tolist()[1]], x[combined_df.columns.values.tolist()[2]]), axis=1)
    #Change the column values through lambda expression.
    combined_df['Company']=combined_df['Company'].apply(lambda x: x.replace('(M$)','(%)') if x.endswith('(M$)') else x)
    #We only need Company and gross margin in this dataframe so we created dataframe with these two columns
    combined_df = combined_df[['Company','Gross margin(%)']]
    #Transpose the dataframe to fulfilled the requirenment 
    gross_margin_trans_df = combined_df.set_index('Company').T
    company_values = q4_sales_df['Company'].tolist()
    if form_type == '10-Q':
        #revenue growth function calling
        revenue_growth_values = calc_revenue_growth(net_sales_10q,sales_df,company_values)
    elif form_type == '10-K':
        #revenue growth function calling
        revenue_growth_values = calc_revenue_growth_q4(net_sales_10q,q4_sales_df,company_values)
    #Create the list of list for prepare the revenue growth dataframe
    revenue_growth_lst = list(zip(company_values,revenue_growth_values))
    #Revenue growth dataframe
    growth_df = pd.DataFrame(revenue_growth_lst, columns=['Company', 'Revenue growth(%)'])
    #Change the column values through lambda expression.
    growth_df['Company']=growth_df['Company'].apply(lambda x: x.replace('(M$)','(%)') if x.endswith('(M$)') else x)
    #Transpose the dataframe to fulfilled the requirenment 
    rev_growth_df = growth_df.set_index('Company').T
    
    #return sales_trans_df,cost_trans_df,gross_margin_trans_df,rev_growth_df,q4_sales_trans_df,q4_cost_trans_df
    return q4_sales_trans_df,q4_cost_trans_df, gross_margin_trans_df,rev_growth_df
#%%%


# In[63]:


# for q4
def q4_calculation(net_sales_10q,cost_of_service_10q,sales_df,service_df,company_values):
    q4_sales_lst = []
    q4_cost_lst = []
    for company in company_values:
        comp = company.replace(" ", "")
#         print(comp+"Sales")
        #First we calculate for net sales
        q3 = int(locale.atoi(net_sales_10q.loc[net_sales_10q.index[2], comp]))
#         print(str(q3) + " Sales"+comp)
        q2 = int(locale.atoi(net_sales_10q.loc[net_sales_10q.index[3], comp]))
        q1 = int(locale.atoi(net_sales_10q.loc[net_sales_10q.index[4], comp]))
        print('sales_df')
        print(sales_df)
        annual_value = int(sales_df.loc[sales_df['Company'] == company, 'Net sales values(M$)'].iloc[0])
        q4_value = annual_value - (q1+q2+q3)
        q4_sales_lst.append(q4_value)
        #Second we calculate for cost of service
        comp1 = company.replace(" ", "")
#         print(str(type(cost_of_service_10q.loc[cost_of_service_10q.index[2], comp1]))+"TYPE")
        if type(cost_of_service_10q.loc[cost_of_service_10q.index[2], comp1])==np.float64:
            q3_service = round(cost_of_service_10q.loc[cost_of_service_10q.index[2], comp1])
#             print(str(q3_service) + " Service")
            q2_service = round(cost_of_service_10q.loc[cost_of_service_10q.index[3], comp1])
            q1_service = round(cost_of_service_10q.loc[cost_of_service_10q.index[4], comp1])
            annual_value_service = int(service_df.loc[service_df['Company'] == company, 'Cost of service values(M$)'].iloc[0])
            q4_value_service = annual_value_service - (q1_service+q2_service+q3_service)
            q4_cost_lst.append(q4_value_service)
        else:
            print('comp1')
            print(comp1)
            print(cost_of_service_10q.loc[cost_of_service_10q.index[2], comp1])
            q3_service = int(locale.atoi(cost_of_service_10q.loc[cost_of_service_10q.index[2], comp1]))
#             print(str(q3_service) + " Service"+comp1)
            q2_service = int(locale.atoi(cost_of_service_10q.loc[cost_of_service_10q.index[3], comp1]))
            q1_service = int(locale.atoi(cost_of_service_10q.loc[cost_of_service_10q.index[4], comp1]))
            annual_value_service = int(service_df.loc[service_df['Company'] == company, 'Cost of service values(M$)'].iloc[0])
            q4_value_service = annual_value_service - (q1_service+q2_service+q3_service)
            q4_cost_lst.append(q4_value_service)
                                         
    return q4_sales_lst,q4_cost_lst


# In[64]:


#use for q4
def calc_revenue_growth_q4(net_sales,sales_df,company_list):
    '''This function takes the current net sales values(i.e.-from sales_df) and
    previous net sales values(i.e.-from net_sales_10q) and company list as a 
    input and return the revenue growth of all companies as a list'''
    revenue_growth_values = []
    for company in company_list:
        revenue_growth = round((int(sales_df.loc[sales_df['Company'] == company, 'Net sales(M$)'].iloc[0]) - int(locale.atoi(net_sales.loc[net_sales.index[0], company])))/int(locale.atoi(net_sales.loc[net_sales.index[0], company]))*100,2)
        revenue_growth_values.append(revenue_growth)
    return revenue_growth_values
#%%%


# In[65]:


def update_10_q():
    #file_location = "C:\\Users\\deepali.b\\DL_tensorflow\\10k_document"
    file_location = '10q_extraction_new'
    #crete_temp_folder(file_location)

    company_to_update = get_time_update()
    company_to_update = ['MDT','STE','SYK','JNJ','GMED']
    if len(company_to_update)>0:
        # extract from code and update in pickle also update time pickl    

        with open(DIR_PATH+'net_sales_10q.pkl', 'rb') as f:
            net_sales_10q = pickle.load(f)
        with open(DIR_PATH+'sequential_quarter_growth.pkl', 'rb') as f:
            sequential_quarter_growth_10q = pickle.load(f)
        with open(DIR_PATH+'gross_margin_10_q.pkl', 'rb') as f:
            gross_margin_10q = pickle.load(f)
        with open(DIR_PATH+'cost_of_service_10q.pkl', 'rb') as f:
            cost_of_service_10q = pickle.load(f)

        last_q = net_sales_10q['Quarters/ USD millions'][0]
        print('last_q')
        print(last_q)
        print()

        if any(net_sales_10q.iloc[0].isna()):
            new_quarter_year = last_q
        else:
            new_quarter_year = get_quarter(last_q)

        column_name_m = [elem+'(M$)' for elem in company_to_update]
        column_name_p = [elem+'(%)' for elem in company_to_update]

        net_sales_dict = {net_sales_10q.columns[0]: [new_quarter_year]}
        net_sales_dict.update({elem:np.nan for elem in column_name_m})
        cost_of_service_dict = {cost_of_service_10q.columns[0]: [new_quarter_year]}
        cost_of_service_dict.update({elem:np.nan for elem in column_name_m})

        revenue_grth_dict = {sequential_quarter_growth_10q.columns[0]: [new_quarter_year]}
        revenue_grth_dict.update({elem:np.nan for elem in column_name_p})
        gross_margin_dict = {gross_margin_10q.columns[0]: [new_quarter_year]}
        gross_margin_dict.update({elem:np.nan for elem in column_name_p})

        sales_trans_df = pd.DataFrame()
        cost_trans_df = pd.DataFrame()
        gross_margin_trans_df = pd.DataFrame()
        rev_growth_df = pd.DataFrame()

        print('company_to_update')
        print(company_to_update)
        for company in company_to_update:
            company_list = []
            #and any(net_sales_10qn['Quarters/ USD millions'].isin([new_quarter_year])):
            if any(pd.isnull(elem) for elem in list(net_sales_dict.values())):
                company_elem = re.sub(r" ?\([^)]+\)", "", company)
                company_list.append(company_elem)
                print('company_list')
                print(company_list)
                print(download_files(DIR_PATH+file_location, company_elem, '10-Q'))
                print(download_files(DIR_PATH+file_location, company_elem, '10-K'))
                latest_date_report = extract_file_date(file_location,company_elem)
                print('latest_date_report')
                print(latest_date_report)
                date_now = datetime.datetime.now()
                if '10-Q' in latest_date_report.keys():
                    date_10_q = datetime.datetime.strptime(latest_date_report['10-Q'], '%Y-%m-%d')
                    date_diff = (date_now-date_10_q).days
                    print('date_diff 10-q')
                    print(date_diff)
                    if date_diff >10: # >10
                        if '10-K' in latest_date_report.keys():
                            print('checking 10-k')
                            date_10_k = datetime.datetime.strptime(latest_date_report['10-K'], '%Y-%m-%d')
                            date_diff = (date_now-date_10_k).days
                            print('Date difference for 10-k')
                            print(date_diff)
                            print("Download 10-k and extract")
                            if 1< date_diff <10:
                                print("Extracting from 10-k")
                                form_type = '10-K'
                                sales_trans_df,cost_trans_df,gross_margin_trans_df,rev_growth_df = calling_function_10k(file_location,company_list,form_type)
                                #k_data = calling_function_10k(file_location,company_list,form_type)
                                #print(k_data)
                                print("Download and extract from 10-k here update for Q4")
                                print(sales_trans_df)
                                print(cost_trans_df)
                                print(gross_margin_trans_df)
                                print(rev_growth_df)
                            else:
                                print("No latest report available")
                    else:
                        form_type = '10-Q'
                        print("Download and extract from 10-q here")
                        print('company_list')
                        print(company_list)


                        sales_trans_df,cost_trans_df,gross_margin_trans_df,rev_growth_df = calling_function(file_location,company_list,form_type)

                if sales_trans_df.empty or cost_trans_df.empty or gross_margin_trans_df.empty or rev_growth_df.empty:
                    print("No update in data")
                else:
                    column_name_m = company+'(M$)'
                    column_name_p = company+'(%)'
                    if any(net_sales_10q['Quarters/ USD millions'].isin([new_quarter_year])):
                        print("Updating existing row")
                        net_sales_10q.loc[0,column_name_m] = sales_trans_df[column_name_m].iloc[0]
                        sequential_quarter_growth_10q.loc[0,column_name_p] = rev_growth_df[column_name_p].iloc[0]
                        gross_margin_10q.loc[0,column_name_p] = gross_margin_trans_df[column_name_p].iloc[0]
                        cost_of_service_10q.loc[0,column_name_m] = cost_trans_df[column_name_m].iloc[0]
    #                     net_sales_10qn = net_sales_10q
    #                     sequential_quarter_growth_10qn = sequential_quarter_growth_10q
    #                     gross_margin_10qn = gross_margin_10q
    #                     cost_of_service_10qn = cost_of_service_10q
                    else:
                        print("creating new row")
                        net_sales_update = pd.DataFrame({net_sales_10q.columns[0]: [new_quarter_year], column_name_m: sales_trans_df[column_name_m].iloc[0]})
                        sequential_update = pd.DataFrame({sequential_quarter_growth_10q.columns[0]: [new_quarter_year], column_name_p: rev_growth_df[column_name_p].iloc[0]})
                        gross_margin_update = pd.DataFrame({gross_margin_10q.columns[0]: [new_quarter_year], column_name_p: gross_margin_trans_df[column_name_p].iloc[0]})
                        cost_of_ser_update = pd.DataFrame({cost_of_service_10q.columns[0]: [new_quarter_year], column_name_m: cost_trans_df[column_name_m].iloc[0]})

                        net_sales_10q = pd.concat([net_sales_update,net_sales_10q],ignore_index=True)
                        sequential_quarter_growth_10q = pd.concat([sequential_update,sequential_quarter_growth_10q],ignore_index=True)
                        gross_margin_10q = pd.concat([gross_margin_update,gross_margin_10q],ignore_index=True)
                        cost_of_service_10q = pd.concat([cost_of_ser_update,cost_of_service_10q], ignore_index=True)

    #                     net_sales_10q.loc[0,elem] = sales_trans_df[column_name_m].iloc[0].values[0]
    #                     sequential_quarter_growth_10q.loc[0,sequential_quarter_growth_10q.columns[0]] = rev_growth_df[column_name_p].iloc[0].values[0]
    #                     gross_margin_10q.loc[0,gross_margin_10q.columns[0]] = gross_margin_trans_df[column_name_p].iloc[0].values[0]
    #                     cost_of_service_10q.loc[0,cost_of_service_10q.columns[0]] = cost_trans_df[column_name_m].iloc[0].values[0]

                    with open('net_sales_10q.pkl', 'wb') as f:
                        pickle.dump(net_sales_10qn, f)
                    with open('sequential_quarter_growth.pkl', 'wb') as f:
                        pickle.dump(sequential_quarter_growth_10qn, f)
                    with open('gross_margin_10_q.pkl', 'wb') as f:
                        pickle.dump(gross_margin_10qn, f)
                    with open('cost_of_service_10q.pkl', 'wb') as f:
                        pickle.dump(cost_of_service_10qn, f)

                    print(sales_trans_df)
                    print('\n')
                    print(cost_trans_df)
                    print('\n')
                    print(gross_margin_trans_df)
                    print('\n')
                    print(rev_growth_df)


    print('new_quarter')
    print(new_quarter_year)


# In[66]:


#######


# ##### 10k download and update

# In[ ]:





# In[149]:


def report_10k(path,company_ticker,File_Type,year):
    try:
        year = str(year)
        input_date = year+'-01' + '-01'
        date_format = datetime.datetime.strptime(input_date, "%Y-%m-%d")
        start_date = str(date_format).split(" ")[0]
        date_next = date_format + relativedelta(months=12)
        end_date = str(date_next).split(" ")[0] 
        dl_Period = Downloader(path)
        print('company_ticker')
        print(company_ticker)
        dl_Period.get(File_Type,company_ticker,after= start_date, before=end_date)
        print("File downloaded successfully at given path")
    except Exception as e:
        print(e)
        print("Error in downloading file")
        
def download_10k_update(path, company_ticker, File_Type, year): 
#     global path
#     path = input('Please input the path:')
#     global company_ticker
#     company_ticker = input('Please input the Company Ticker:')
#     File_Type = input('Please input the File Type:')
#     year = input('Please input the Year:')
    report_10k(path,company_ticker,File_Type,year)
    root = path
    pattern = "*.txt"
    #global path_list
    
    for path, subdirs, files in os.walk(root):
        for name in files:
            if fnmatch(name, pattern):
                path_list = []
                print('os.path.join(path, name')
                print(os.path.join(path, name))
                path_list.append(os.path.join(path, name))
                path_list = str(''.join(path_list))
    path_list
    return path_list 


# In[150]:


def get_content_10k(path_list):
    with open(path_list) as f:
        global raw_10k
        raw_10k = f.read()
        raw_10k

    # Regex to find <DOCUMENT> tags
    doc_start_pattern = re.compile(r'<DOCUMENT>')
    doc_end_pattern = re.compile(r'</DOCUMENT>')
    # Regex to find <TYPE> tag prceeding any characters, terminating at new line
    type_pattern = re.compile(r'<TYPE>[^\n]+')

    # Create 3 lists with the span idices for each regex
    doc_start_is = [x.end() for x in doc_start_pattern.finditer(raw_10k)]
    doc_end_is = [x.start() for x in doc_end_pattern.finditer(raw_10k)]
    doc_types = [x[len('<TYPE>'):] for x in type_pattern.findall(raw_10k)]

    global document
    document = {}
    for doc_type, doc_start, doc_end in zip(doc_types, doc_start_is, doc_end_is):
        if doc_type == '10-K':
            document[doc_type] = raw_10k[doc_start:doc_end]
    #print(document['10-K'][:500])

    #Regex pattern for finance table
    regex = re.compile(r'(Consolidated Statements of Income)|(CONSOLIDATED STATEMENTS OF INCOME)|(CONSOLIDATED STATEMENTS OF EARNINGS)|(CONSOLIDATED STATEMENTS OF OPERATIONS AND COMPREHENSIVE INCOME)|(CONSOLIDATED STATEMENTS OF OPERATIONS)')

    # Use finditer to math the regex
    matches = regex.finditer(document['10-K'])

    # Write a for loop to print the matches
    for match in matches:
        match
        
    # Matches
    matches = regex.finditer(document['10-K'])

    # Create the dataframe
    test_df = pd.DataFrame([(x.group(), x.start(), x.end()) for x in matches])
    test_df.columns = ['item', 'start', 'end']

    #Replacing the end value to none
    test_df['end'] = test_df['end'].replace('','',regex=True,inplace=True)

    #Replacing the financial metric name to parameter
    test_df.loc[0,"item"] = "Parameter"
    #display(test_df.head())

    # Drop duplicates
    pos_dat = test_df.sort_values('start', ascending=True).drop_duplicates(subset=['item'])

    # Set item as the dataframe index
    pos_dat.set_index('item', inplace=True)
    
    #Get item 8
    item_8_raw = document['10-K'][pos_dat['start'].loc['Parameter']:pos_dat['end'].loc['Parameter']]
    #display(item_8_raw[0:1000])

    # First convert the raw text we have to exrtacted to BeautifulSoup object 
    Fin_8A = BeautifulSoup(item_8_raw, 'lxml')
    #display(Fin_8A)

    # Finding out metrices using tags
    Fin_headings=[]
    for i in Fin_8A.findAll('table'):
        Fin_headings.append(i.text)
    for i in Fin_8A.findAll('span'):
        Fin_headings.append(i.text)
    for i in Fin_8A.findAll('td'):
        Fin_headings.append(i.text)
    Fin_headings = ''.join(Fin_headings)
    global Fin_text
    Fin_text = unicodedata.normalize("NFKD",Fin_headings)

def extract_10k(path_list):
    get_content_10k(path_list)
    print('path_list')
    print(path_list)
    path=os.path.dirname(path_list)
    global Ticker
    Ticker = os.path.basename(os.path.dirname(os.path.dirname(path)))
    Ticker
    print("Data Extraction is successfull!")

def Net_Sales():
    results =[]
    try:
        Reg = re.findall(r'(?P<name>(Total.revenues)[\W\d]*)',Fin_text) 
        text = (Reg[-1][0])
        results.append(text)
        return results
    except:
        pass
    
    try:
        Reg = re.findall(r'(?P<name>(Net.sales|Sales.to.customers|Net.Sales|Sales)[\W\d]*)',Fin_text) 
        text = (Reg[0][0])
        results.append(text)
        return results 
        
    except:
        pass


def sales():
    NS = str(Net_Sales())
    Net = re.sub(r',', '', NS)
    Net = re.findall('[\d]*[\d]+',Net)
    global NetSales
    NetSales = pd.DataFrame(Net).transpose()
    Date = re.findall(r'<ACCEPTANCE-DATETIME>\d{4}',raw_10k)
    Year = re.findall(r'\d',str(Date))
    s=[str(i) for i in Year]
    global years
    years = int("".join(s))
    NetSales.columns=[years,years-1,years-2]
    NetSales.insert(0, "YEAR", Ticker +' (M$)')
    if Ticker == 'STE':
        NetSales.iloc[0,1] =round(float(NetSales.iloc[0,1])*0.001)
        NetSales.iloc[0,1] =(str(NetSales.iloc[0,1]))
        NetSales.iloc[0,2] =round(float(NetSales.iloc[0,2])*0.001)
        NetSales.iloc[0,2] =(str(NetSales.iloc[0,2]))
        NetSales.iloc[0,3] =round(float(NetSales.iloc[0,3])*0.001)
        NetSales.iloc[0,3] =(str(NetSales.iloc[0,3]))
    elif Ticker == 'GMED':
        NetSales.iloc[0,1] =round(float(NetSales.iloc[0,1])*0.001)
        NetSales.iloc[0,1] =(str(NetSales.iloc[0,1]))
        NetSales.iloc[0,2] =round(float(NetSales.iloc[0,2])*0.001)
        NetSales.iloc[0,2] =(str(NetSales.iloc[0,2]))
        NetSales.iloc[0,3] =round(float(NetSales.iloc[0,3])*0.001)
        NetSales.iloc[0,3] =(str(NetSales.iloc[0,3]))
    else:
        NetSales.iloc[0,1] =(str(NetSales.iloc[0,1]))
        NetSales.iloc[0,2] =(str(NetSales.iloc[0,2]))
        NetSales.iloc[0,3] =(str(NetSales.iloc[0,3]))
    NetSales = NetSales.transpose()
    NetSales.columns = NetSales.iloc[0]
    NetSales = NetSales[1:]
    return NetSales[0:1]

def Cost_of_Revenue():
    Results = []
    try:
        Reg = re.findall(r'(?P<name>(Cost.of.products.sold,.excluding.amortization.of.intangible.assets|Cost.of.products.sold|Cost.of.goods.sold|Cost.of.sales|Total.cost.of.revenues)[\W\d]*)',Fin_text) #MDT SYK 
        text = (Reg[0][0])
        Results.append(text)
        return Results
    except:
        pass

def Revenuecost():
    Rev = str(Cost_of_Revenue())
    RevC = re.sub(r',', '', Rev)
    REG = re.findall('[\d]*[\d]+',RevC)
    global Revcos
    Revcos = pd.DataFrame(REG).transpose()
    Date = re.findall(r'<ACCEPTANCE-DATETIME>\d{4}',raw_10k)
    Year = re.findall(r'\d',str(Date))
    s=[str(i) for i in Year]
    years = int("".join(s))
    Revcos.columns=[years,years-1,years-2]
    Revcos.insert(0, "YEAR", Ticker +' (M$)')
    if Ticker == 'STE':
        Revcos.iloc[0,1] =round(float(Revcos.iloc[0,1])*0.001)
        Revcos.iloc[0,1] =(str(Revcos.iloc[0,1]))
        Revcos.iloc[0,2] =round(float(Revcos.iloc[0,2])*0.001)
        Revcos.iloc[0,2] =(str(Revcos.iloc[0,2]))
    elif Ticker == 'GMED':
        Revcos.iloc[0,1] =round(float(Revcos.iloc[0,1])*0.001)
        Revcos.iloc[0,1] =(str(Revcos.iloc[0,1]))
        Revcos.iloc[0,2] =round(float(Revcos.iloc[0,2])*0.001)
        Revcos.iloc[0,2] =(str(Revcos.iloc[0,2]))
    else:
        Revcos.iloc[0,1] =(str(Revcos.iloc[0,1]))
        Revcos.iloc[0,2] =(str(Revcos.iloc[0,2]))
    Revcos = Revcos.transpose()
    Revcos.columns = Revcos.iloc[0]
    Revcos = Revcos[1:]
    return Revcos[0:1]


def Gross_Margin():
    sales()
    global Sales
    Sales = NetSales.copy()
    Sales.iloc[0,0] = Sales.iloc[0,0].replace(',', '')
    Sales.iloc[1,0] = Sales.iloc[1,0].replace(',', '')
    Sales.iloc[2,0] = Sales.iloc[2,0].replace(',', '')
    Date = re.findall(r'<ACCEPTANCE-DATETIME>\d{4}',raw_10k)
    Year = re.findall(r'\d',str(Date))
    s=[str(i) for i in Year]
    years = int("".join(s))
    RevC = Revcos.copy()
    RevC.iloc[0,0] = RevC.iloc[0,0].replace(',', '')
    RevC.iloc[1,0] = RevC.iloc[1,0].replace(',', '')
    RevC.iloc[2,0] = RevC.iloc[2,0].replace(',', '')
    Y1 = round(((float(Sales.iloc[0,0])-(float(RevC.iloc[0,0])))/(float(Sales.iloc[0,0]))*100),2)
    Y2 = round(((float(Sales.iloc[1,0])-(float(RevC.iloc[1,0])))/(float(Sales.iloc[1,0]))*100),2)
    Y3 = round(((float(Sales.iloc[2,0])-(float(RevC.iloc[2,0])))/(float(Sales.iloc[2,0]))*100),2)
    grossmargin = pd.DataFrame([[years,Y1],[years-1,Y2],[years-2,Y3]])
    grossmargin.columns=["YEAR", Ticker + ' (%)']
    return grossmargin[0:1]

def Revenue_growth():
    RevSales = NetSales.copy()
    RevSales.iloc[0,0] = RevSales.iloc[0,0].replace(',', '')
    RevSales.iloc[1,0] = RevSales.iloc[1,0].replace(',', '')
    RevSales.iloc[2,0] = RevSales.iloc[2,0].replace(',', '')
    B =round(((((float(RevSales.iloc[0,0]))/(float(RevSales.iloc[1,0])))-1)*100),2)
    C =round(((((float(RevSales.iloc[1,0]))/(float(RevSales.iloc[2,0])))-1)*100),2)
    RG = pd.DataFrame([[years,B],[years-1,C]])
    RG.columns = ["YEAR",Ticker + " (%)"]
    return RG[0:1]
   


# In[166]:


def crete_temp_folder(folder_path):
    #folder_path = '10k_download/'
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        #os.makedirs(newpath)
    else:
        os.makedirs(folder_path)
    return folder_path


# In[167]:


def update_10_k(year,comapny_name,form_type):
    path_10k = DIR_PATH+"10k_downlod_extraction/"
    with open(DIR_PATH+'revenue_growth.pkl', 'rb') as f:
        revenue_growth_df = pickle.load(f)
    with open(DIR_PATH+'gross_margin.pkl', 'rb') as f:
        gross_margin_df = pickle.load(f)
    with open(DIR_PATH+'net_sales.pkl', 'rb') as f:
        net_sales_df = pickle.load(f)

    
    print('company_name')
    print(comapny_name)
    path_list = download_10k_update(path_10k, comapny_name, form_type, year)
    extract_10k(path_list)
    sales_df = sales()
    revenue_cost_df = Revenuecost()
    gross_df = Gross_Margin()
    print('gross_df')
    print(gross_df)
    print(gross_margin_df.iloc[0].values[1])
    revenue_df = Revenue_growth()
    print('revenue_df')
    print(revenue_df)
    print(revenue_df.iloc[0].values[1])
    company_name_m = comapny_name + ' (M$)'
    print('company_name_m')
    print(company_name_m)
    company_name_p = comapny_name + ' (%)'
    print(company_name_p)
    if revenue_growth_df['Year'].isin([year]).any():
        revenue_growth_df.set_index("Year",inplace=True)
        gross_margin_df.set_index("Year",inplace=True)
        net_sales_df.set_index(net_sales_df.columns[0],inplace=True)
        print(net_sales_df.columns)
        print('updating existing row')
        print(revenue_df.iloc[0].values[1])
        print(revenue_df)
        print(sales_df.index)
        print(sales_df.iloc[0].values[0])

        revenue_growth_df.loc[year,company_name_p] = revenue_df.iloc[0].values[1]
        gross_margin_df.loc[year,company_name_p] = gross_df.iloc[0].values[1]
        net_sales_df.loc[year,company_name_m] = int(sales_df.iloc[0].values[0])
        #net_sales_df.loc[year,company_name_m] = 0
        print(net_sales_df)
        revenue_growth_df.reset_index(inplace= True)
        gross_margin_df.reset_index(inplace= True)
        net_sales_df.reset_index(inplace= True)

        #gross_margin_df[company_name_p].iloc[0] = gross_df.iloc[0].values[1]
        #net_sales_df[company_name_m].iloc[0] = int(sales_df.iloc[0].values[0])
    else:
        print('adding new row')
        print(revenue_df.iloc[0].values[0])
        revenue_growth_update = pd.DataFrame({revenue_growth_df.columns[0]: [year], company_name_p: revenue_df.iloc[0].values[1]})
        gross_margin_update = pd.DataFrame({gross_margin_df.columns[0]: [year], company_name_p: gross_df.iloc[0].values[1]})
        net_sales_update = pd.DataFrame({net_sales_df.columns[0]: [year], company_name_m: int(sales_df.iloc[0].values[0])})
        print('net_sales_update')
        print(net_sales_update)
        revenue_growth_df = pd.concat([revenue_growth_update,revenue_growth_df],ignore_index=True)
        gross_margin_df = pd.concat([gross_margin_update,gross_margin_df],ignore_index=True)
        print("-----------------")
        print(net_sales_update.index)
        print(net_sales_df.index)
        net_sales_df = pd.concat([net_sales_update,net_sales_df],ignore_index=True)
        print('net_sales_df new row')
        print(net_sales_df)
    print('updating pickle file')
    with open(DIR_PATH+'revenue_growth.pkl', 'wb') as f:
        pickle.dump(revenue_growth_df, f)
    with open(DIR_PATH+'gross_margin.pkl', 'wb') as f:
        pickle.dump(gross_margin_df, f)
    with open(DIR_PATH+'net_sales.pkl', 'wb') as f:
        pickle.dump(net_sales_df, f)
    crete_temp_folder(path_10k)

    


# In[ ]:





# In[153]:


######


# In[154]:


def get10kurl(year, form_type, company_ticker_name):
    """ Function to get html link of 10-K for given company and year
    """
    queryApi = QueryApi(api_key= API_KEY)
    year = str(year)
    print('company_ticker_name',company_ticker_name)
    try:
        input_date = year+'-01' + '-01'
        #print('input_date', input_date)
        date_format = datetime.datetime.strptime(input_date, "%Y-%m-%d") + relativedelta(years=1)
        date_format_1 = str(date_format).split(" ")[0]
        date_previous = date_format - relativedelta(years=2)
        date_previous_1 = str(date_previous).split(" ")[0]
        
        date_range = '{' + date_previous_1 + ' TO ' + date_format_1 + '}'
        #print('date_range',date_range)
    except Exception:
        print("Check input year is correct or not")
    
    try:
        query = {
          "query": { "query_string": { 
              "query": "ticker"+":" +company_ticker_name +" AND formType:" + "\"" + form_type + "\"" + " AND filedAt:" + date_range  
            } },
          "from": "0",
          "size": "10",
          "sort": [{ "filedAt": { "order": "desc" } }]
        }

        filings = queryApi.get_filings(query)
    except Exception:
        print("Check parameters in query")
        
    return filings


# In[155]:


def get_text_path(input_path):
    root = input_path
    pattern = "*.txt"
    path_list = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            if fnmatch(name, pattern):
                path_list.append(os.path.join(path, name))
    #print('path_list',path_list)
    return path_list        


# In[156]:


def download_10k_risk_factor(path,company_ticker,File_Type,year):
    try:
        year = str(year)
        input_date = year+'-01' + '-01'
        date_format = datetime.datetime.strptime(input_date, "%Y-%m-%d")
        start_date_1 = date_format + relativedelta(years=1)
        start_date = str(start_date_1).split(" ")[0]
        #print('Start date:', start_date)
        date_next = date_format - relativedelta(years=1)
        #print('date_next', date_next)
        end_date = str(date_next).split(" ")[0] 
        #print("End date:", end_date)
        dl_Period = Downloader(path)
        dl_Period.get(File_Type,company_ticker,after= end_date, before=start_date)
        time.sleep(10)
        print("File downloaded successfully at given path")
    except Exception as e:
        print(e)
        print("Error in downloading file")


# In[157]:


def get_content(input_text_file_path):
    with open(input_text_file_path) as f:
        raw_10k = f.read()
    # Regex to find <DOCUMENT> tags
    doc_start_pattern = re.compile(r'<DOCUMENT>')
    doc_end_pattern = re.compile(r'</DOCUMENT>')
    # Regex to find <TYPE> tag prceeding any characters, terminating at new line
    type_pattern = re.compile(r'<TYPE>[^\n]+')
    
    # Create 3 lists with the span idices for each regex


    doc_start_is = [x.end() for x in doc_start_pattern.finditer(raw_10k)]
    doc_end_is = [x.start() for x in doc_end_pattern.finditer(raw_10k)]
    doc_types = [x[len('<TYPE>'):] for x in type_pattern.findall(raw_10k)]
    document = {}

    for doc_type, doc_start, doc_end in zip(doc_types, doc_start_is, doc_end_is):
        if doc_type == '10-K':
            document[doc_type] = raw_10k[doc_start:doc_end]
    return document


# In[158]:


def get_risk_text(input_document):
    regex = re.compile(r'(>Item(\s|&#160;|&nbsp;)(1A|1B|7A|7|8)\.{0,1})|(ITEM\s(1A|1B|7A|7|8))')
    matches = regex.finditer(input_document['10-K'])

    # Create the dataframe
    test_df = pd.DataFrame([(x.group(), x.start(), x.end()) for x in matches])

    test_df.columns = ['item', 'start', 'end']
    test_df['item'] = test_df.item.str.lower()
    
    test_df.replace('&#160;',' ',regex=True,inplace=True)
    test_df.replace('&nbsp;',' ',regex=True,inplace=True)
    test_df.replace(' ','',regex=True,inplace=True)
    test_df.replace('\.','',regex=True,inplace=True)
    test_df.replace('>','',regex=True,inplace=True)
    pos_dat = test_df.sort_values('start', ascending=True).drop_duplicates(subset=['item'], keep='last')
    pos_dat.set_index('item', inplace=True)
    item_1a_raw = input_document['10-K'][pos_dat['start'].loc['item1a']:pos_dat['start'].loc['item1b']]
    riskhtml = bs.BeautifulSoup(item_1a_raw) 
    #riskhtml = item_1a_raw
    risk_headings = []
    for i in riskhtml.find_all('span', style=lambda x: x and 'font-weight:700;' in x):
        risk_headings.append(i.text)
    for i in riskhtml.find_all('span', style=lambda x: x and 'font-weight:bold;' in x):
        risk_headings.append(i.text)
    for i in riskhtml.find_all('font', style=lambda x: x and 'font-weight:bold;' in x):
        risk_headings.append(i.text)
    return risk_headings


# In[159]:


def get_risk_keywords(risk_factor):
    risk_keyword = []
    for i in range(len(risk_factor)):
        text = risk_factor[i].lower()
        # load a spaCy model, depending on language, scale, etc.
        nlp = spacy.load("en_core_web_sm")
        # add PyTextRank to the spaCy pipeline
        nlp.add_pipe("textrank")
        doc = nlp(text)
        # examine the top-ranked phrases in the document
        for phrase in doc._.phrases[:20]:
            risk_keyword.append(phrase.text)
    custom_stopwords = ['we','us','.','u.s.','irs']
    stopword_list = [*STOPLIST, *custom_stopwords]
    
    risk_keyword_list = [elem.lower() for elem in risk_keyword if elem not in stopword_list]
    risk_dict = Counter(risk_keyword_list)
    sorted_keywords = risk_dict.most_common()
    return sorted_keywords


# In[160]:


def arrange_path(list_of_ticker,download_folder):
    path_list = get_text_path(download_folder)
    path_arranged_list = []
    for folder_path in path_list:
        subdirname = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(folder_path))))
        subdirname1 = os.path.basename(os.path.dirname(folder_path))
        subdirname1 = subdirname1.split('-')
        base_year = '20'
        file_year = base_year+subdirname1[1]
        #print(file_year)
        company_dict = {'company_name':subdirname, 'year':file_year, 'file_path':folder_path}
        path_arranged_list.append(company_dict)
        #print(path_arranged_list)
    return path_arranged_list
    


# In[161]:


def risk_factor_compare(list_of_ticker,year):
    folder_path = '10k_download/'
    downloaded_path = crete_temp_folder(folder_path)
    File_type = '10-K'
    for company_ticker in list_of_ticker:
        try:
            download_10k_risk_factor(downloaded_path,company_ticker,File_type,year)
        except Exception as e:
            print(e)
    
    arranged_path_dict = arrange_path(list_of_ticker, downloaded_path)
    
    #text_file_path = get_text_path(downloaded_path)
    print('text_file_path')
    #print(text_file_path)
    path_list = []
    risk_list = []
    risk_keywords = []
    
    for elem in arranged_path_dict:
        
        html_content = get_content(elem['file_path'])
        risk_factors_text = get_risk_text(html_content)
        elem['risk_text'] = [risk_factors_text]
        elem['risk_keywords'] = [get_risk_keywords(risk_factors_text)]
        
    #risk_data = pd.DataFrame(arranged_path_dict)
    
    #risk_data_text = risk_data.copy()
    #risk_data_text1 = risk_data_text.drop(columns=['risk_keywords'])
    #risk_data1 = risk_data.drop(columns=['risk_text'])
    risk_data_text = pd.DataFrame(columns = ["Year",list_of_ticker[0]+"_risk_text",list_of_ticker[1]+"_risk_text"])
    year_list = [elem["year"] for elem in arranged_path_dict]
    year_list1 = year_list[:int(len(year_list)/2)]
    year_list2 = year_list[int(len(year_list)/2):]
  
    risk_data_text["Year"] = year_list1
    risk_data_text[list_of_ticker[0]+"_risk_text"] = [elem["risk_text"] for elem in arranged_path_dict if elem['company_name']==list_of_ticker[0]]
    risk_data_text[list_of_ticker[1]+"_risk_text"] = [elem["risk_text"] for elem in arranged_path_dict if elem['company_name']==list_of_ticker[1]]
    #print('lengths')
    #print(len([elem["year"] for elem in arranged_path_dict]))
    #print(len([elem["risk_text"] for elem in arranged_path_dict if elem['company_name']=='MDT']))
    #print(len([elem["risk_text"] for elem in arranged_path_dict if elem['company_name']=='STE']))
    
    risk_data_keyword = pd.DataFrame(columns = ["Year",list_of_ticker[0]+"_risk_keywords",list_of_ticker[1]+"_risk_keywords"])
    risk_data_keyword["Year"] = year_list2
    risk_data_keyword[list_of_ticker[0]+"_risk_keywords"] = [elem["risk_keywords"] for elem in arranged_path_dict if elem['company_name']==list_of_ticker[0]]
    risk_data_keyword[list_of_ticker[1]+"_risk_keywords"] = [elem["risk_keywords"] for elem in arranged_path_dict if elem['company_name']==list_of_ticker[1]]
      
      
    print("Risk factors extracted successfully")
    print("len of risk_data")
    print(len(risk_data_text))
    return (risk_data_text, risk_data_keyword)   


# #### Download 10-K file to path

# In[162]:


def download_10k(path,company_ticker,File_Type,year, output_range):
    try:
        year = str(year)
        print('path')
        print(path)
        input_date = year+'-01' + '-01'
        date_format = datetime.datetime.strptime(input_date, "%Y-%m-%d")
        start_date = str(date_format).split(" ")[0]
        print('Start date:', start_date)
        #date_next = date_format + relativedelta(months=12)
        date_previous = date_format - relativedelta(years=output_range)
        #print('date_next', date_next)
        end_date = str(date_previous).split(" ")[0] 
        print("End date:", end_date)
        dl_Period = Downloader(path)
        dl_Period.get(File_Type,company_ticker,after=end_date , before=start_date)         
        print("File downloaded successfully at given path")
    except Exception as e:
        print(e)
        print("Error in downloading file")


# In[ ]:





# In[163]:


def update_pickle_riskdf(risk_text_data, risk_keywords_data):
    try:
        with open('risk_text.pkl', 'rb') as f:
            risk_text = pickle.load(f)
        with open('risk_keywords.pkl', 'rb') as f:
            risk_keywords = pickle.load(f)

        risk_text_df_new = risk_text.append(risk_text_data)
        risk_keywords_df_new = risk_keywords.append(risk_keywords_data)

        with open('risk_text.pkl', 'wb') as f:
            pickle.dump(risk_text_df_new, f)

        with open('risk_keywords.pkl', 'wb') as f:
            pickle.dump(risk_keywords_df_new, f)

    except Exception as e:
        print("Error in updating pickle file")
        print(e)


# In[125]:


#static risk extraction


# In[126]:


def business_risk(company, year, clean_text):
    #function to extract risk from clean text
    business_risk = ' '
    return {'Year': [year],company:[business_risk]}


# In[127]:


def regulatory_risk(company, year, clean_text):
    #function to extract risk from clean text
    regulatory_risk = ' '
    return {'Year': [year],company:[regulatory_risk]}


# In[128]:


def aquisition_risk(company, year, clean_text):
    #function to extract risk from clean text
    aquisition_risk = ' '
    return {'Year': [year],company:[aquisition_risk]}


# In[129]:


def jurisdiction_risk(company, year, clean_text):
    #function to extract risk from clean text
    jurisdiction_risk = ' '
    return {'Year': [year],company:[jurisdiction_risk]}


# In[130]:


def economic_risk(company, year, clean_text):
    #function to extract risk from clean text
    business_risk = ' '
    return {'Year': [year],company:[business_risk]}


# In[131]:


def update_static_risk(company,year,clean_text):
    try:
        with open('business_operational_risk.pkl', 'rb') as f:
            business_risk_df = pickle.load(f)
            print('business_risk')

        with open('legal_regulatory_risk.pkl', 'rb') as f:
            regulatory_risk_df = pickle.load(f)

        with open('risk_related_aquisition.pkl', 'rb') as f:
            aquisition_risk_df = pickle.load(f)

        with open('risk_related_jurisdiction.pkl', 'rb') as f:
            jurisdiction_risk_df = pickle.load(f)

        with open('economic_industrial_risk.pkl', 'rb') as f:
            economic_risk_df = pickle.load(f)
    except Exception as e:
        print(e)
        print("Check pickle file is present or not for static risks")
    
    business_risk_dict = business_risk(company, year, clean_text)
    regulatory_risk_dict = regulatory_risk(company, year, clean_text)
    aquisition_risk_dict = aquisition_risk(company, year, clean_text)
    jurisdiction_risk_dict = jurisdiction_risk(company, year, clean_text)
    economic_risk_dict = economic_risk(company, year, clean_text)
    
    
    if business_risk_df['Year'].isin([year]).any():
        print('date present')
        print(business_risk_dict)
        business_risk_df.loc[0,company] = business_risk_dict[company]
        regulatory_risk_df.loc[0,company] = regulatory_risk_dict[company]
        aquisition_risk_df.loc[0,company] = aquisition_risk_dict[company]
        jurisdiction_risk_df.loc[0,company] = jurisdiction_risk_dict[company]
        economic_risk_df.loc[0,company] = economic_risk_dict[company]

    else:
        print('create new row')
        business_risk_dict_df = pd.DataFrame(business_risk_dict)
        business_risk_df = business_risk_dict_df.append(business_risk_df)
        print('regulatory_risk_dict')
        print(regulatory_risk_dict)
        regulatory_risk_dict_df = pd.DataFrame(regulatory_risk_dict)
        regulatory_risk_df = regulatory_risk_dict_df.append(regulatory_risk_df)

        aquisition_risk_dict_df = pd.DataFrame(aquisition_risk_dict)
        aquisition_risk_df = aquisition_risk_dict_df.append(aquisition_risk_df)

        jurisdiction_risk_dict_df = pd.DataFrame(jurisdiction_risk_dict)
        jurisdiction_risk_df = jurisdiction_risk_dict_df.append(jurisdiction_risk_df)
        
        economic_risk_dict_df = pd.DataFrame(economic_risk_dict)
        economic_risk_df = economic_risk_dict_df.append(economic_risk_df)
        
    with open('business_operational_risk.pkl', 'wb') as f:
        pickle.dump(business_risk_df, f)
    with open('legal_regulatory_risk.pkl', 'wb') as f:
        pickle.dump(regulatory_risk_df, f)
    with open('risk_related_aquisition.pkl', 'wb') as f:
        pickle.dump(aquisition_risk_df, f)
    with open('risk_related_jurisdiction.pkl', 'wb') as f:
        pickle.dump(jurisdiction_risk_df, f)
    with open('economic_industrial_risk.pkl', 'wb') as f:
        pickle.dump(economic_risk_df, f)
        


# In[132]:


# Dynamic risk


# In[133]:


def patents_risks(company, year, clean_text):
    #function to extract risk from clean text
    patents_risks = ' '
    return {'Year': [year],company:[patents_risks]}


# In[134]:


def RandD_expense_risks(company, year, clean_text):
    #function to extract risk from clean text
    randd_expense_risks = ' '
    return {'Year': [year],company:[randd_expense_risks]}


# In[135]:


def recall_risk(company, year, clean_text):
    #function to extract risk from clean text
    recall_risk = ' '
    return {'Year': [year],company:[recall_risk]}


# In[136]:


def restructuring_cost_risks(company, year, clean_text):
    #function to extract risk from clean text
    restructuring_cost_risks = ' '
    return {'Year': [year],company:[restructuring_cost_risks]}


# In[137]:


def acquisition_risks(company, year, clean_text):
    #function to extract risk from clean text
    acquisition_risks = ' '
    return {'Year': [year],company:[acquisition_risks]}


# In[138]:


def litigation_risks(company, year, clean_text):
    #function to extract risk from clean text
    litigation_risks = ' '
    return {'Year': [year],company:[litigation_risks]}


# In[139]:


def new_patents_risks(company, year, clean_text):
    #function to extract risk from clean text
    new_patents_risks = ' '
    return {'Year': [year],company:[new_patents_risks]}


# In[140]:


def update_dynamic_risk(company,year,clean_text):
    try:
        with open('patents_risks.pkl', 'rb') as f:
            patent_risk_df = pickle.load(f)
            print('patent_risk_df')

        with open('RandD_expense_risks.pkl', 'rb') as f:
            RandD_expense_risks_df = pickle.load(f)

        with open('recall_risk.pkl', 'rb') as f:
            recall_risk_df = pickle.load(f)

        with open('restructuring_cost_risks.pkl', 'rb') as f:
            restructuring_cost_risks_df = pickle.load(f)

        with open('acquisition_risks.pkl', 'rb') as f:
            acquisition_risks_df = pickle.load(f)

        with open('litigation_risks.pkl', 'rb') as f:
            litigation_risks_df = pickle.load(f)
            
        with open('new_patents_risks.pkl', 'rb') as f:
            new_patents_risks_df = pickle.load(f)
            
    except Exception as e:
        print(e)
        print("Check pickle file is present or not for static risks")
    
    patents_risks_dict = patents_risks(company, year, clean_text)
    RandD_expense_risks_dict = RandD_expense_risks(company, year, clean_text)
    recall_risk_dict = recall_risk(company, year, clean_text)
    restructuring_cost_risks_dict = restructuring_cost_risks(company, year, clean_text)
    acquisition_risks_dict = acquisition_risks(company, year, clean_text)
    litigation_risks_dict = litigation_risks(company, year, clean_text)
    new_patents_risks_dict = new_patents_risks(company, year, clean_text)
    
    
    if patents_risks_dict['YEAR'].isin([year]).any():
        print('date present')
        print(patents_risks_dict)
        patent_risk_df.loc[0,company] = patents_risks_dict[company]
        RandD_expense_risks_df.loc[0,company] = RandD_expense_risks_dict[company]
        recall_risk_df.loc[0,company] = recall_risk_dict[company]
        restructuring_cost_risks_df.loc[0,company] = restructuring_cost_risks_dict[company]
        acquisition_risks_df.loc[0,company] = acquisition_risks_dict[company]
        litigation_risks_df.loc[0,company] = litigation_risks_dict[company]
        new_patents_risks_df.loc[0,company] = new_patents_risks_dict[company]

    else:
        print('create new row')
        patents_risks_dict_df = pd.DataFrame(patents_risks_dict)
        patent_risk_df = patents_risks_dict_df.append(patent_risk_df)
        
        RandD_expense_risks_dict_df = pd.DataFrame(RandD_expense_risks_dict)
        RandD_expense_risks_df = RandD_expense_risks_dict_df.append(RandD_expense_risks_df)

        recall_risk_dict_df = pd.DataFrame(recall_risk_dict)
        recall_risk_df = recall_risk_dict_df.append(recall_risk_df)

        restructuring_cost_risks_dict_df = pd.DataFrame(restructuring_cost_risks_dict)
        restructuring_cost_risks_df = restructuring_cost_risks_dict_df.append(restructuring_cost_risks_df)
        
        acquisition_risks_dict_df = pd.DataFrame(acquisition_risks_dict)
        acquisition_risks_df = acquisition_risks_dict_df.append(acquisition_risks_df)

        litigation_risks_dict_df = pd.DataFrame(litigation_risks_dict)
        litigation_risks_df = litigation_risks_dict_df.append(litigation_risks_df)

        new_patents_risks_dict_df = pd.DataFrame(new_patents_risks_dict)
        new_patents_risks_df = new_patents_risks_dict_df.append(new_patents_risks_df)

    with open('patents_risks.pkl', 'wb') as f:
        pickle.dump(patent_risk_df,f)
        print('patent_risk_df')

    with open('RandD_expense_risks.pkl', 'wb') as f:
        pickle.dump(RandD_expense_risks_df,f)

    with open('recall_risk.pkl', 'wb') as f:
        pickle.dump(recall_risk_df,f)

    with open('restructuring_cost_risks.pkl', 'wb') as f:
        pickle.dump(restructuring_cost_risks_df,f)

    with open('acquisition_risks.pkl', 'wb') as f:
        pickle.dump(acquisition_risks_df,f)

    with open('litigation_risks.pkl', 'wb') as f:
        pickle.dump(litigation_risks_df,f)

    with open('new_patents_risks.pkl', 'wb') as f:
        pickle.dump(new_patents_risks_df,f)
        


# ##### scheduling job to run function of 10q updation everyday

# In[142]:


for company in company_ticker:
    update_static_risk(company,year,' ')


# In[ ]:


for company in company_ticker:
    update_dynamic_risk(company,year,' ')


# In[180]:


#company_list = ['MDT','STE','SYK','JNJ','GMED']
for company in company_ticker:
    update_10_k(year,company,form_type_k)


# In[ ]:


# update 10_q


# In[ ]:


update_10_q()


# In[ ]:



# scheduler = BlockingScheduler()
# scheduler.add_job(func=update_10_q, trigger='interval', hours=24, id='10_q updation')
# scheduler.start()


# In[ ]:




