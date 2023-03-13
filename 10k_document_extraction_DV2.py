#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the required library
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from collections import OrderedDict
import spacy
from string import punctuation
import collections
import numpy as np
import pickle
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import rcParams
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
import pickle
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
'en_US.UTF-8'
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import itertools
from time import sleep
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger


# In[3]:


STOPLIST = stopwords.words('english')


# In[53]:


path = input("Input path to save files: ")  # localpath: D:\\work\\10kform\\10k_download
year = int(input("Input year: "))
n = int(input("Enter the number of companies: "))
print("\n")
company_ticker = list(str(num) for num in input("Enter company ticker separated by space ").strip().split())[:n]
form_type = input("Input form type: ")
output_range = int(input("Input number to get range: "))


# In[54]:


year_list = [year-i for i in range(output_range)]


# In[55]:


def crete_temp_folder(folder_path):
    #folder_path = '10k_download/'
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        #os.makedirs(newpath)
    else:
        os.makedirs(folder_path)
    return folder_path


# #### 10q download and update

# In[56]:


#######


# ##### 10k download and update

# In[57]:


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


# In[58]:


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


# In[59]:


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


# In[60]:


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


# In[61]:


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


# In[62]:


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
    


# In[73]:


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


# In[74]:


def risk_compare_company(risk_data_keywords,column):
    keywords_dict_list = []
    keywords_dict_list.append(dict(zip(risk_data_keywords.Year, risk_data_keywords[column])))
    keywords_dict_list[0]["company"] = column
    keywords_dict_list
    i = 0
    year_risks = []
    for year in list(risk_data_keywords.Year):
        #print(year)
        year_risks.append([elem[0] for elem in keywords_dict_list[i][year][0]])
        #print("year_risks")
        #print(year_risks)
        return_dict = {year: year_risks}
    year_risks
    for year in list(return_dict.keys()):
        for a, b in itertools.combinations(year_risks, 2):
            if a==b:
                repeat_risk = a
                #print(a==b)
                repeat_dict = {"company":column,"risk_repeat":repeat_risk}
            else:
                repeat_risk = []
                #print(a==b)
                repeat_dict = {"company":column,"risk_repeat":repeat_risk}
    return repeat_dict


# In[75]:


def crete_compare_df(risk_data):
    try:
        compared_data = []
        column_list = list(risk_data.columns)
        print("column_list")
        print(column_list)
        try:
            column_list.remove('Year')
        except Exception as e:
            pass
        for column in column_list:
            #print(column)
            compared_data.append(risk_compare_company(risk_data,column))
        #print(compared_data)
        compared_data_df = pd.DataFrame(compared_data)
        compared_data_df_t = compared_data_df.T
        compared_data_df_t.reset_index(drop=True, inplace=True)
        compared_data_df_t.columns = compared_data_df_t.iloc[0] 
        compared_data_df_t = compared_data_df_t[1:]
    except Exception as e:
        print(e)
        compared_data_df_t = pd.DataFrame()
    return compared_data_df_t


# #### Download 10-K file to path

# In[76]:


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





# In[77]:


#Graphs
def bar_graphs(df_gross, df_revenue, df_net):
    #Bar_charts
    plt1 = df_gross_margin.plot.bar(figsize=(8,5),width=0.5,title = "Yearly Gross margin performance",ylabel="Year wise gross margin in percentage(%)")
    plot.savefig('plt1.png')
    plt2 = df_revenue_growth.plot.bar(figsize=(8,5),width=0.5,title = "Yearly Revenue growth performance",ylabel="Year wise Revenue growth in percentage(%)")
    plot.savefig('plt2.png')
    plt3 = df_net_sales.plot.bar(figsize=(8,5),width=0.5,title = "Yearly Net sales performance",ylabel="Year wise net sales in $")
    plot.savefig('plt3.png')
    


# In[78]:


def financial_extraction(year, list_of_ticker, output_range):
    try:
        with open('revenue_growth.pkl', 'rb') as f:
            revenue_growth_df = pickle.load(f)
        with open('gross_margin.pkl', 'rb') as f:
            gross_margin_df = pickle.load(f)
        with open('net_sales.pkl', 'rb') as f:
            net_sales_df = pickle.load(f)
#         with open('costs_of_products.pkl', 'rb') as f:
#             costs_of_products_df = pickle.load(f)
    except Exception as e:
        print(e)
        print("check pickle file is present or not")
    
    try:
#         revenue_growth_df.rename(columns = {'Year':'Company'},inplace = True)
#         revenue_growth_dft = revenue_growth_df.transpose()
#         new_column_r = [str(int(elem)) for elem in revenue_growth_dft.iloc[0].to_list()]
#         revenue_growth_dft.columns = new_column_r
#         revenue_df_new = revenue_growth_dft.iloc[1:]
#         revenue_df_new.index.name = "Company"
#         revenue_df_new.columns.names = ['Years']
# this code needs to uncomment and repeat for all dataframe if want previous format data

        revenue_df_new = revenue_growth_df.copy()
        revenue_df_new.set_index('Year',inplace=True)  

        gross_margin_new = gross_margin_df.copy()
        gross_margin_new.set_index('Year',inplace=True)
        
        net_sales_new = net_sales_df.copy()
        net_sales_new.set_index(net_sales_new.columns[0],inplace=True)
    
    
        #year_list = [str(int(year-i)) for i in range(output_range)]
        year_list = [int(year-i) for i in range(output_range)]

        print('year_list')
        print(year_list)
        ticker_list_new_s = [elem + " ("+"M$)" for elem in list_of_ticker]
        ticker_list_new_p = [elem + " ("+"%"")" for elem in list_of_ticker]

        #revenue_growth_return = revenue_df_new.loc[year_list,ticker_list_new_p]
        revenue_growth_return = revenue_df_new[revenue_df_new.index.isin(year_list)]
        print('financial_extraction revenue growth return')
        print(revenue_growth_return)
        gross_margin_return = gross_margin_new[gross_margin_new.index.isin(year_list)]
        net_sales_return = net_sales_new[net_sales_new.index.isin(year_list)]
        
        # data for graph
        revenue_growth_dfg = revenue_growth_df.melt('Year')
        revenue_growth_dfg.rename(columns = {'variable':'Company','value':'Revenue_growth'},inplace=True)
        revenue_growth_dfgy = revenue_growth_dfg.loc[revenue_growth_dfg['Year'].isin(year_list)]
        revenue_growth_graph = revenue_growth_dfgy.loc[revenue_growth_dfgy['Company'].isin(ticker_list_new_p)]
#         gross_margin_graph['Gross_margin'] = gross_margin_graph['Gross_margin'].astype('Float64')     
        
        gross_margin_dfg = gross_margin_df.melt('Year')
        gross_margin_dfg.rename(columns = {'variable':'Company','value':'Gross_margin'},inplace=True)
        gross_margin_dfgy = gross_margin_dfg.loc[gross_margin_dfg['Year'].isin(year_list)]
        gross_margin_graph = gross_margin_dfgy.loc[gross_margin_dfgy['Company'].isin(ticker_list_new_p)]
#         gross_margin_graph['Gross_margin'] = gross_margin_graph['Gross_margin'].astype('Float64')        

        net_sales_dfg = net_sales_df.melt(net_sales_df.columns[0])
        net_sales_dfg.rename(columns = {'variable':'Company','value':'Net_Sales'},inplace=True)
        net_sales_dfgy = net_sales_dfg.loc[gross_margin_dfg['Year'].isin(year_list)]
        net_sales_graph = net_sales_dfgy.loc[net_sales_dfgy['Company'].isin(ticker_list_new_s)]
#         gross_margin_graph['Gross_margin'] = gross_margin_graph['Gross_margin'].astype('Float64') 
        

#         costs_of_products_return = costs_of_products_df_new.loc[ticker_list_new,year_list]
    except Exception as e:
        print(e)
        revenue_growth_return = pd.DataFrame()
        gross_margin_return = pd.DataFrame()
        net_sales_return = pd.DataFrame()
        revenue_growth_graph = pd.DataFrame()
        gross_margin_graph = pd.DataFrame()
        net_sales_graph = pd.DataFrame()
        
        
#         costs_of_products_return = pd.DataFrame()
        print("Data not present")
    return net_sales_return, revenue_growth_return, gross_margin_return, revenue_growth_graph, gross_margin_graph, net_sales_graph
    


# In[79]:


def extract_10q(year, list_of_ticker):
    with open('net_sales_10q.pkl', 'rb') as f:
        net_sales_10q = pickle.load(f)
    with open('sequential_quarter_growth.pkl', 'rb') as f:
        sequential_quarter_growth_10q = pickle.load(f)
    with open('gross_margin_10_q.pkl', 'rb') as f:
        gross_margin_10q = pickle.load(f)
    with open('cost_of_service_10q.pkl', 'rb') as f:
        cost_of_service_10q = pickle.load(f)
    year_list = [year, year-1]
    try:
        quarter = ["Q1", "Q2", "Q3", "Q4"]
        quarter_year = [[elem + " - "+ str(yearn) for elem in quarter] for yearn in year_list]
        quarter_yearn = [item for elem in quarter_year for item in elem]
        
        list_of_tickerp = [elem+"("+"%"")" for elem in list_of_ticker]
        list_of_tickerp1 = list_of_tickerp
        list_of_tickerp1.insert(0,"Quarters")
        
        list_of_tickerm = [elem+"("+"M$"")" for elem in list_of_ticker]
        list_of_tickerm1 = list_of_tickerm
        list_of_tickerm1.insert(0,"Quarters/ USD millions")

        net_sales_q = net_sales_10q.loc[net_sales_10q['Quarters/ USD millions'].isin(quarter_yearn)]
        net_sales_10q_new = net_sales_q[list_of_tickerm]
        net_sales_10q_new = net_sales_10q_new.set_index("Quarters/ USD millions")
        
        sequential_growth = sequential_quarter_growth_10q.loc[sequential_quarter_growth_10q['Quarters'].isin(quarter_yearn)]
        sequential_growth_q = sequential_growth[list_of_tickerp]
        sequential_growth_q = sequential_growth_q.set_index("Quarters")
        
        gross_margin = gross_margin_10q.loc[gross_margin_10q['Quarters'].isin(quarter_yearn)]
        gross_margin_q = gross_margin[list_of_tickerp]
        gross_margin_q = gross_margin_q.set_index("Quarters")
        
        cost_of_service = cost_of_service_10q.loc[cost_of_service_10q['Quarters/ USD millions'].isin(quarter_yearn)]
        cost_of_service_q = cost_of_service[list_of_tickerm]
        cost_of_service_q = cost_of_service_q.set_index("Quarters/ USD millions")
        
        #data for graph
        net_sales_10qm = net_sales_10q.melt('Quarters/ USD millions')
        net_sales_10qm.rename(columns={'variable':'Company','value':'Net_sales'},inplace=True)
        net_sales_10qms = net_sales_10qm.loc[net_sales_10qm['Quarters/ USD millions'].isin(quarter_yearn)]
        net_sales_10q_graph = net_sales_10qms.loc[net_sales_10qms['Company'].isin(list_of_tickerm)]
        
        sequential_quarter_growth_10qm = sequential_quarter_growth_10q.melt('Quarters')
        sequential_quarter_growth_10qm.rename(columns={'variable':'Company','value':'Sequential Growth'},inplace=True)
        sequential_quarter_growth_10qms = sequential_quarter_growth_10qm.loc[sequential_quarter_growth_10qm['Quarters'].isin(quarter_yearn)]
        sequential_growth_graph = sequential_quarter_growth_10qms.loc[sequential_quarter_growth_10qms['Company'].isin(list_of_tickerp)]

        gross_margin_10qm = gross_margin_10q.melt('Quarters')
        gross_margin_10qm.rename(columns={'variable':'Company','value':'Gross_margin'},inplace=True)
        gross_margin_10qms = gross_margin_10qm.loc[gross_margin_10qm['Quarters'].isin(quarter_yearn)]
        gross_margin_10q_graph = gross_margin_10qms.loc[gross_margin_10qms['Company'].isin(list_of_tickerp)]

        cost_of_service_10qm = cost_of_service_10q.melt('Quarters/ USD millions')
        cost_of_service_10qm.rename(columns={'variable':'Company','value':'Cost_of_service'},inplace=True)
        cost_of_service_10qms = cost_of_service_10qm.loc[cost_of_service_10qm['Quarters/ USD millions'].isin(quarter_yearn)]
        cost_of_service_10q_graph = cost_of_service_10qms.loc[cost_of_service_10qms['Company'].isin(list_of_tickerm)]
        
    except Exception as e:
        print(e)
        net_sales_10q_new = pd.DataFrame()
        sequential_growth_q = pd.DataFrame()
        gross_margin_q = pd.DataFrame()
        cost_of_service_q = pd.DataFrame()
        net_sales_10q_graph = pd.DataFrame()
        sequential_growth_graph = pd.DataFrame()
        gross_margin_10q_graph = pd.DataFrame()
        cost_of_service_10q_graph = pd.DataFrame()
    return net_sales_10q_new, sequential_growth_q,  gross_margin_q, cost_of_service_q, net_sales_10q_graph, sequential_growth_graph, gross_margin_10q_graph, cost_of_service_10q_graph


# In[80]:


def get_risk_text_keywords(year, list_of_ticker):
    try:
        with open('risk_text.pkl', 'rb') as f:
            risk_text = pickle.load(f)
        with open('risk_keywords.pkl', 'rb') as f:
            risk_keywords = pickle.load(f)
    except Exception as e:
        print(e)
        print("check pickle file is present or not")
        
    try:        
        risk_text.set_index("Year", inplace=True)
        risk_keywords.set_index("Year", inplace=True)

        year_list = [year, year-1]
        columns_risk_text = [elem+"_risk_text" for elem in list_of_ticker]
        columns_risk_keywords = [elem+"_risk_keywords" for elem in list_of_ticker]


        risk_text_subset = risk_text.loc[year_list,columns_risk_text]
        risk_keywords_subset = risk_keywords.loc[year_list,columns_risk_keywords]
    except Exception as e:
        print(e)
        risk_text_subset = pd.DataFrame()
        risk_keywords_subset = pd.DataFrame()
    return (risk_text_subset, risk_keywords_subset)
    


# In[81]:


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


# In[82]:


def static_risk(ticker_list, year, output_range):
    try:
        with open('business_operational_risk.pkl', 'rb') as f:
            business_risk = pickle.load(f)
            print('business_risk')

        with open('legal_regulatory_risk.pkl', 'rb') as f:
            regulatory_risk = pickle.load(f)

        with open('risk_related_aquisition.pkl', 'rb') as f:
            aquisition_risk = pickle.load(f)

        with open('risk_related_jurisdiction.pkl', 'rb') as f:
            jurisdiction_risk = pickle.load(f)

        with open('economic_industrial_risk.pkl', 'rb') as f:
            economic_risk = pickle.load(f)
    except Exception as e:
        print(e)
        print("Check pickle file is present or not for static risks")

    try:
        year_list = [int(year-i) for i in range(output_range)]
        print('year_list')
        print(year_list)
        business_risk.set_index("Year",inplace=True)
        business_risk_sub = business_risk.loc[year_list,ticker_list]

        regulatory_risk.set_index("Year",inplace=True)
        regulatory_risk_sub = regulatory_risk.loc[year_list,ticker_list]


        aquisition_risk.set_index("Year",inplace=True)
        aquisition_risk_sub = aquisition_risk.loc[year_list,ticker_list]

        jurisdiction_risk.set_index("Year",inplace=True)
        jurisdiction_risk_sub = jurisdiction_risk.loc[year_list,ticker_list]

        economic_risk.set_index("Year",inplace=True)
        economic_risk_sub = economic_risk.loc[year_list,ticker_list]
    except Exception as e:
        print("Check subset of data is correct or not")
        business_risk_sub = pd.DataFrame()
        regulatory_risk_sub = pd.DataFrame()
        aquisition_risk_sub = pd.DataFrame()
        economic_risk_sub = pd.DataFrame()
    return business_risk_sub, regulatory_risk_sub, aquisition_risk_sub, economic_risk_sub   


# In[83]:


def download_10Q(path,company_ticker,File_Type,year, output_range):
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


# In[84]:


# dynamic risk


# In[85]:


def dynamic_risk(ticker_list, year, output_range):
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

    try:
        year_list = [int(year-i) for i in range(output_range)]
        print('year_list')
        print(year_list)
        patent_risk_df.set_index("YEAR",inplace=True)
        patent_risk_sub = patent_risk_df.loc[year_list,ticker_list]

        RandD_expense_risks_df.set_index("YEAR",inplace=True)
        RandD_expense_risks_sub = RandD_expense_risks_df.loc[year_list,ticker_list]

        recall_risk_df.set_index("YEAR",inplace=True)
        recall_risk_sub = recall_risk_df.loc[year_list,ticker_list]

        restructuring_cost_risks_df.set_index("YEAR",inplace=True)
        restructuring_cost_risks_sub = restructuring_cost_risks_df.loc[year_list,ticker_list]

        litigation_risks_df.set_index("YEAR",inplace=True)
        litigation_risks_sub = litigation_risks_df.loc[year_list,ticker_list]
        
        new_patents_risks_df.set_index("YEAR",inplace=True)
        new_patents_risks_sub = new_patents_risks_df.loc[year_list,ticker_list]

        acquisition_risks_df.set_index("YEAR",inplace=True)
        acquisition_risks_sub = acquisition_risks_df.loc[year_list,ticker_list]
        patent_risk_sub, RandD_expense_risks_sub ,recall_risk_sub ,restructuring_cost_risks_sub ,litigation_risks_sub ,new_patents_risks_sub ,acquisition_risks_sub
    except Exception as e:
        print("Check subset of data is correct or not")
        patent_risk_sub = pd.DataFrame()
        RandD_expense_risks_sub = pd.DataFrame()
        restructuring_cost_risks_sub = pd.DataFrame()
        litigation_risks_sub = pd.DataFrame()
        new_patents_risks_sub = pd.DataFrame()
        acquisition_risks_sub = pd.DataFrame()
        recall_risk_sub = pd.DataFrame()

    return patent_risk_sub, RandD_expense_risks_sub ,recall_risk_sub ,restructuring_cost_risks_sub ,litigation_risks_sub ,new_patents_risks_sub ,acquisition_risks_sub
  


# #### Calling functions

# ##### Calling download function

# In[ ]:





# In[86]:


if form_type == '10-K':
    print("Downloading 10-K document")
    for ticker in company_ticker:
        download_10k(path,ticker,form_type,year, output_range)
elif form_type == '10-Q':
    print("Downloading 10-Q document")
    for ticker in company_ticker:
        download_10Q(path,ticker,form_type,year, output_range)
else:
    print("Please pass correct form type")


# ##### Calling revenue extraction

# In[87]:


if form_type == '10-K':

    df_net_sales, df_revenue_growth, df_gross_margin, revenue_growth_graph, gross_margin_graph, net_sales_graph = financial_extraction(year, company_ticker, output_range)
    print('net sales at first step')
    print(df_net_sales)
    new_year_list = []
    for elem in year_list:
        if elem not in (df_net_sales.index & df_revenue_growth.index & df_gross_margin.index):
            print("Need to update pickle files")
    
    print("df net sales\n")
    print(df_net_sales)
    print("\ndf revenue growth\n")
    print(df_revenue_growth)
    print("\ndf gross margin\n")
    print(df_gross_margin)
    print("\n net sales for graph\n")
    print(net_sales_graph)
    print("\n revenue growth for graph\n")
    print(revenue_growth_graph)
    print("\n gross margin for graph\n")
    print(gross_margin_graph)
     
elif form_type == '10-Q':
    net_sales_10q_new, sequential_growth_q,  gross_margin_q, cost_of_service_q, net_sales_10q_graph, sequential_growth_graph, gross_margin_10q_graph, cost_of_service_10q_graph = extract_10q(year,company_ticker)
    print("df net sales_10Q in $\n")
    print(net_sales_10q_new)
    
    print("\n sequential quarter growth 10Q in %\n")
    print(sequential_growth_q)
    
    print("\n gross margin 10Q in %\n")
    print(gross_margin_q)
    
    print("\n cost of service 10Q in $ \n")
    print(cost_of_service_q)
    
    print("df net sales_10Q for graph $\n")
    print(net_sales_10q_graph)
    
    print("\n sequential quarter growth 10Q for graph in %\n")
    print(sequential_growth_graph)
    
    print("\n gross margin 10Q for graph in %\n")
    print(gross_margin_10q_graph)
    
    print("\n cost of service 10Q for graph in $ \n")
    print(cost_of_service_10q_graph)    


# In[88]:


net_sales_10q_new


# ##### Calling risk factor comparison

# In[89]:


#Risk_data = risk_factor_compare(company_ticker, path, year)
if form_type == '10-K':
    risk_data = get_risk_text_keywords(year, company_ticker)
    Risk_data_text = risk_data[0]
    Risk_data_text.reset_index(inplace=True)
    print('Risk_data_text')
    print(Risk_data_text)
    Risk_data_keywords = risk_data[1]
    Risk_data_keywords.reset_index(inplace=True)
    
    print('Extracting static risk')
    business_risk_sub, regulatory_risk_sub, aquisition_risk_sub, economic_risk_sub = static_risk(company_ticker, year, output_range)
    print("Business and operational risk")
    print(business_risk_sub)
    print("Legal regulatory risk")
    print(regulatory_risk_sub)
    print("Aquisition risk")
    print(aquisition_risk_sub)
    print("Economic and industrial risk")
    print(economic_risk_sub)
    if Risk_data_text.empty or Risk_data_keywords.empty:
        print("calling code to extract risks")
        Risk_data = risk_factor_compare(company_ticker,year)
        Risk_data_text = Risk_data[0]
        Risk_data_keywords = Risk_data[1]
        update_pickle_riskdf(Risk_data_text, Risk_data_keywords)    
    else:
        pass
elif form_type == '10-Q':
    print("Risk data is not available for 10-Q")
    Risk_data_text = pd.DataFrame()
    Risk_data_keywords = pd.DataFrame()
    Risk_data = (Risk_data_text, Risk_data_keywords)    
else:
    pass


# # Risk data keywords

# In[91]:


Risk_data_keywords


# In[92]:


Risk_data_text


# #### Comparison of risk factors

# In[93]:


comparison_risk_df = crete_compare_df(Risk_data_keywords)
comparison_risk_df


# In[94]:


print('business_risk_sub')
business_risk_sub


# In[95]:


print('regulatory_risk_sub')
regulatory_risk_sub


# In[96]:


print('aquisition_risk_sub')
aquisition_risk_sub


# In[97]:


print('economic_risk_sub')
economic_risk_sub


# In[98]:


patent_risk_sub, RandD_expense_risks_sub ,recall_risk_sub ,restructuring_cost_risks_sub ,litigation_risks_sub ,new_patents_risks_sub ,acquisition_risks_sub = dynamic_risk(company_ticker, year, output_range)
# if data is empty update pickle files


# In[99]:


print('patent_risk')
patent_risk_sub


# In[100]:


print('RandD_expense_risks')
RandD_expense_risks_sub


# In[101]:


print('recall_risk')
recall_risk_sub


# In[102]:


print('restructuring_cost_risks')
restructuring_cost_risks_sub


# In[103]:


print('litigation_risks')
litigation_risks_sub


# In[104]:


print('new_patents_risks')
new_patents_risks_sub


# In[105]:


print('acquisition_risks')
acquisition_risks_sub


# In[ ]:





# In[ ]:




