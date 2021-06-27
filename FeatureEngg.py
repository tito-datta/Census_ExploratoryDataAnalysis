import operator
from numpy import NaN
import pandas as pd
from pandas.core.common import not_none
from pandas.core.dtypes.missing import notna
from pandas.core.frame import DataFrame
from pandas.core.series import Series

# census_xlsx = 'C:\Tito\Being Adult\Self Dev\BITS AIML course\FeatureEngineering\Census-1.xlsx'

# census_df: DataFrame = pd.read_excel(io=census_xlsx,sheet_name='C-08')

# census_df = census_df.drop(labels=4, axis=0)
# census_df = census_df.dropna(axis=0,how='all')

# # Function to merge the headers and delete redundant entries 
# def merge_headers_and_delete(data: DataFrame, iters: int, merge_position: int):   
#     cols_data = []
#     for col in data.columns:
#         cols_data.append(int(col[9:]))
#     for it in range(1,iters):
#         cur_iter = merge_position + it
#         for col in cols_data:
#             if pd.isnull(data.iloc[cur_iter,col]) == False and pd.isna(data.iloc[cur_iter,col]) == False:
#                 if pd.isna(data.iloc[merge_position,col]):
#                     data.iloc[merge_position,col] = data.iloc[cur_iter,col]
#                 else:
#                     data.iloc[merge_position,col] = data.iloc[merge_position,col] + ' ' + data.iloc[cur_iter,col]                                             
#     data = data.drop(labels=range((merge_position+1),iters), axis=0)
#     return data           

# census_df = merge_headers_and_delete(census_df,4,0)

# census_df = census_df.dropna(axis=1)

# # Assign correct headers
# headers_census = census_df.iloc[0,:].to_numpy()

# def cleanup_headers(headers):
#     head_i = 3
#     substr = ''
#     cur_idx = 0
#     for head in headers:
#         if head.find('Person') != -1:
#             substr = head[0:-7]   
#             head_i -= 1     
#         if head_i > 0 and head_i < 3 and head.find('Person') == -1:
#             headers[cur_idx] = substr + ' ' + head
#             head_i -= 1
#         if head_i == 0: head_i = 3
#         cur_idx += 1
#     return headers

# print(cleanup_headers(headers_census))

# census_df = pd.DataFrame(census_df.iloc[1:,:].to_numpy(), columns=headers_census)

# list_states_areas = census_df.iloc[:,1:4]
# list_states_areas = list_states_areas.drop_duplicates()


# dic_state_data = {}
# for state in list_states_areas['Area Name']:
#     dic_state_data[state] = census_df[(census_df['Area Name'] == state)]  

# df_india = dic_state_data.get('INDIA')
# df_rural = df_india[((df_india['Total/ Rural/ Urban/'] == 'Rural') & (df_india['Age-group'] != 'All ages'))]
# df_rural['Age-group'] = df_rural['Age-group'].astype(str)
# df_urban = df_india[((df_india['Total/ Rural/ Urban/'] == 'Urban') & (df_india['Age-group'] != 'All ages'))]
# df_urban['Age-group'] = df_urban['Age-group'].astype(str)

# import seaborn as sns
# import matplotlib.pyplot as plt

# def plot_simple(data: DataFrame, x_col, y_col, x_dscr, y_dscr,label, title, dim_x, dim_y):
#     plt.figure(num=None, figsize=(dim_y, dim_x), dpi=80)
#     plt.plot(df_rural[x_col],df_rural[y_col],label=label)
#     plt.legend(loc='best',fancybox=True, shadow=True)
#     plt.ylabel(y_dscr)
#     plt.xlabel(x_dscr)
#     plt.title(title)
#     plt.show()

# def plot_graph(data: DataFrame, x_col, y_col, x_dscr, y_dscr,label, title, dim_x, dim_y):
#     plt.figure(num=None, figsize=(dim_y, dim_x), dpi=100)
#     plt.plot(data[x_col],data[y_col],label=label)
    

# def plot_two_graphs(data: DataFrame, x_col, y_col, x_dscr, y_dscr,label, title, dim_x, dim_y,y2_col='',label2=''):
#     plot_graph(data,x_col,y_col,x_dscr,y_dscr,label,title,dim_x,dim_y)
#     if np.isnan(y2_col) == False:
#         plot_graph(data,x_col,y2_col,x_dscr,y_dscr,label2,title,dim_x,dim_y)       
#     plt.legend(loc='best',fancybox=True, shadow=True)
#     plt.ylabel(y_dscr)
#     plt.xlabel(x_dscr)
#     plt.title(title)    
#     plt.show()

# def plot_multiple_graphs(data: DataFrame, x_col, y_col, x_dscr, y_dscr,label1, title, dim_x, dim_y, graphs={}):
#     plt.figure(num=None, figsize=(dim_y, dim_x), dpi=100)
#     plt.plot(data[x_col],data[y_col],label=label1)
#     for key, value in graphs.items():
#         plt.plot(data[x_col], data[key], label=value) 
#     plt.legend(loc='best',fancybox=True, shadow=True)
#     plt.ylabel(y_dscr)
#     plt.xlabel(x_dscr)
#     plt.title(title)    
#     plt.show()        
        
# # plot_simple(df_rural,'Literate Persons','Illiterate Persons','Age Group','Literate Persons','Literates by age group in rural India','A lokk at how the number of literates across different age groups of rural India',10,35)

# df_india = df_india[((df_india['Total/ Rural/ Urban/'] == 'Total') & (df_india['Age-group'] != 'All ages'))]
# # df_india['Age-group'] = df_urban['Age-group'].astype(str)

# cols = ['Age-group',
#        'Educational level Literate without educational level  Males',
#        'Educational level Literate without educational level  Females',
#        'Below primary  Males',
#        'Below primary  Females', 
#        'Primary  Males',
#        'Primary  Females', 
#        'Middle  Males',
#        'Middle  Females',
#        'Matric/Secondary  Males', 
#        'Matric/Secondary  Females',      
#        'Higher secondary/Intermediate Pre-University/Senior secondary  Males',
#        'Higher secondary/Intermediate Pre-University/Senior secondary  Females',
#        'Non-technical diploma or certificate not equal to degree  Males',
#        'Non-technical diploma or certificate not equal to degree  Females',
#        'Technical diploma or certificate not equal to degree   Males',
#        'Technical diploma or certificate not equal to degree   Females',
#        'Graduate & above  Males',
#        'Graduate & above  Females']
# df_india_literacy_lvl = df_india[cols]

# df_india_literacy_lvl = df_india_literacy_lvl.T
# headers = df_india_literacy_lvl.iloc[0,:].to_numpy()
# df_india_literacy_lvl = pd.DataFrame(df_india_literacy_lvl.iloc[1:,:].to_numpy(), columns=headers)
# idx = pd.Index(cols[1:])

# df_india_literacy_lvl.set_index(idx)
# df_india_literacy_lvl.head()

# list_states_areas = census_df.iloc[:,1:4]
# list_states_areas = list_states_areas.drop_duplicates()

# dic_state_data = {}
# for state in list_states_areas['Area Name']:
#     dic_state_data[state] = census_df[(census_df['Area Name'] == state)]  

# # Clean up and split data in smaller frames
# cols = [
#     'Area Name',
#     'Age-group',
#     'Higher secondary/Intermediate Pre-University/Senior secondary Persons',
#     'Non-technical diploma or certificate not equal to degree Persons',
#     'Technical diploma or certificate not equal to degree  Persons',
#     'Graduate & above Persons', 
# ]

# states = ['INDIA', 'State - JAMMU & KASHMIR', 'State - HIMACHAL PRADESH', 'State - PUNJAB', 'State - CHANDIGARH', 'State - UTTARAKHAND', 'State - HARYANA', 'State - NCT OF DELHI', 'State - RAJASTHAN', 'State - UTTAR PRADESH', 'State - BIHAR', 'State - SIKKIM', 'State - ARUNACHAL PRADESH', 'State - NAGALAND', 'State - MANIPUR', 'State - MIZORAM', 'State - TRIPURA', 'State - MEGHALAYA', 'State - ASSAM', 'State - WEST BENGAL', 'State - JHARKHAND', 'State - ODISHA', 'State - CHHATTISGARH', 'State - MADHYA PRADESH', 'State - GUJARAT', 'State - DAMAN & DIU', 'State - DADRA & NAGAR HAVELI', 'State - MAHARASHTRA', 'State - ANDHRA PRADESH', 'State - KARNATAKA', 'State - GOA', 'State - LAKSHADWEEP', 'State - KERALA', 'State - TAMIL NADU', 'State - PUDUCHERRY', 'State - ANDAMAN & NICOBAR ISLANDS']

# def prepare_state_data(data: DataFrame, state):
#     data = data[cols]
#     #  Sum the state data age groups since we are concerned with the highest across ages
#     prepare_state_data = data.sum()
#     return prepare_state_data

# prepared_state_data = dic_state_data
# for state in states:
#     if state != 'INDIA':
#         prepared_state_data[state] = prepared_state_data(prepared_state_data.get(state), state)

# prepared_state_data


dict_random = {
    'West Bengal': 1234,
    'Chattisgarh': 1232,
    'Jaamu & Kashmir': 1233,
    'Sikkim': 1232
}

print('Un sorted: ')
print(dict_random)
dict_random = sorted(dict_random.items(), key=operator.itemgetter(1))
print('Sorted')
print(dict_random)


plt.pie(np.fromiter(prepared_state_data.values(), dtype=float), labels=prepared_state_data.keys())


states = dic_state_data.keys()

prepared_state_data = {}
for state in states:
    if state != 'INDIA':
        # Pick only desired columns
        state_data = dic_state_data.get(state)        
        state_data = state_data[cols] 
        # Since we are concerned with literacy rate we can use the all ages record 
        state_data = state_data[(state_data['Age-group'] == 'All ages')]                            
        # Sum across the columns since we care about literacy rate of the state
        state_data = state_data.assign(SumOfLiterates = state_data[['Higher secondary/Intermediate Pre-University/Senior secondary Persons', 'Non-technical diploma or certificate not equal to degree Persons', 'Technical diploma or certificate not equal to degree  Persons', 'Graduate & above Persons']].sum(axis=1))
        state_data = state_data.assign(LiteracyRate= (state_data['SumOfLiterates']/state_data['    Total Persons']) * 100) 
        prepared_state_data[state] = state_data[['LiteracyRate']].sum()[0]

prepared_state_data


['Age-group',
       '    Total  Males', 
       'Below primary  Males', 
       'Primary  Males', 
       'Middle  Males',
       'Matric/Secondary  Males',
       'Higher secondary/Intermediate Pre-University/Senior secondary  Males',
       'Non-technical diploma or certificate not equal to degree  Males',
       'Technical diploma or certificate not equal to degree   Males',
       'Graduate & above  Males',
       'Unclassified  Males',
       '    Total  Females', 
       'Below primary  Females', 
       'Primary  Females', 
       'Middle  Females',
       'Matric/Secondary  Females',
       'Higher secondary/Intermediate Pre-University/Senior secondary  Females',
       'Non-technical diploma or certificate not equal to degree  Females',
       'Technical diploma or certificate not equal to degree   Females',
       'Graduate & above  Females',
       'Unclassified  Females']


def find_literacy_rate(data: DataFrame, total_column):
    # Since we are concerned with literacy rate we can use the all ages record 
    data = data[(data['Age-group'] == 'All ages')]
    # Sum across the columns since we care about literacy rate of the state
    cols_sum = data.columns[2:]    
    sumOfLiterates = data[cols_sum].sum(axis=1)
    literacy_rates = (sumOfLiterates/data[cols_sum[total_column]]) * 100
    data = data.assign(LiteracyRates = literacy_rates)
    return data  