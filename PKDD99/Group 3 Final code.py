
# Import libraries
import numpy as np
import pandas as pd
#   !!!!!  pip3 --no-cache-dir install seaborn !!!!!
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import colors
import datetime
from matplotlib import figure as fig

##############
#### LOAN#####
##############

# Each record describes a loan granted for a given account
loan = pd.read_csv('C:/Users/yliu10/Desktop/Python Group/Origin/data_berka/loan.asc', sep=';')
loan.head()

loan.duplicated(subset= "account_id",keep = False).sum()
loan['date'] =  pd.to_datetime(loan['date'], format='%y%m%d')


######################
#### ORDER ###########
######################


# Each record describes characteristics of a payment order
order = pd.read_csv('C:/Users/yliu10/Desktop/Python Group/Origin/data_berka/order.asc', sep=';')
order.head()

#pivoting orders table by k_symbol and suming ammount per each k_symbol.
order_pivot1 = order.pivot_table(index= ("account_id"),columns= ("k_symbol") , values = ("amount"), aggfunc= ('sum'))
order_pivot1.reset_index(level=0, inplace=True)
order_pivot1.columns = ["account_id", "Other", "Leasing","Insurance_Pay", "Household_Pay","Loan_Pay"]
order_pivot1.head()

#Pivoting table by k_symbol and counting the number of orders per each k_symbol type
order_pivot2 = order.pivot_table(index= ("account_id"),columns= ("k_symbol") , values = ("order_id"), aggfunc= ('count'))
order_pivot2.reset_index(level=0, inplace=True)
order_pivot2.columns = ["account_id", "nbr_Other", "nbr_Leasing","nbr_Insurance_Pay", "nbr_Household_Pay","nbr_Loan_Pay"]


order_final = pd.merge(order_pivot1, order_pivot2, how= "left" , on=["account_id"])


######################
#### transactions ####
######################

# Each record describes one transaction on an account
trans = pd.read_csv('C:/Users/yliu10/Desktop/Python Group/Origin/data_berka/trans.asc', sep=';', low_memory=False)

trans.head()

trans['Q1-1993'] = np.where(trans['date']<= 930331, 1, 0)
trans['Q2-1993'] = np.where((trans['date']>= 930401) & (trans['date']<= 930631) , 1, 0)
trans['Q3-1993'] = np.where((trans['date']>= 930701) & (trans['date']<= 930931) , 1, 0)
trans['Q4-1993'] = np.where((trans['date']>= 930801) & (trans['date']<= 931231) , 1, 0)

trans['Q1-1994'] = np.where((trans['date']>= 940101) & (trans['date']<= 940331) , 1, 0)
trans['Q2-1994'] = np.where((trans['date']>= 940401) & (trans['date']<= 940631) , 1, 0)
trans['Q3-1994'] = np.where((trans['date']>= 940701) & (trans['date']<= 940931) , 1, 0)
trans['Q4-1994'] = np.where((trans['date']>= 940801) & (trans['date']<= 941231) , 1, 0)

trans['Q1-1995'] = np.where((trans['date']>= 950101) & (trans['date']<= 950331) , 1, 0)
trans['Q2-1995'] = np.where((trans['date']>= 950401) & (trans['date']<= 950631) , 1, 0)
trans['Q3-1995'] = np.where((trans['date']>= 950701) & (trans['date']<= 950931) , 1, 0)
trans['Q4-1995'] = np.where((trans['date']>= 950801) & (trans['date']<= 951231) , 1, 0)

trans['Q1-1996'] = np.where((trans['date']>= 960101) & (trans['date']<= 960331) , 1, 0)
trans['Q2-1996'] = np.where((trans['date']>= 960401) & (trans['date']<= 960631) , 1, 0)
trans['Q3-1996'] = np.where((trans['date']>= 960701) & (trans['date']<= 960931) , 1, 0)
trans['Q4-1996'] = np.where((trans['date']>= 960801) & (trans['date']<= 961231) , 1, 0)

trans['Q1-1997'] = np.where((trans['date']>= 970101) & (trans['date']<= 970331) , 1, 0)
trans['Q2-1997'] = np.where((trans['date']>= 970401) & (trans['date']<= 970631) , 1, 0)
trans['Q3-1997'] = np.where((trans['date']>= 970701) & (trans['date']<= 970931) , 1, 0)
trans['Q4-1997'] = np.where((trans['date']>= 970801) & (trans['date']<= 971231) , 1, 0)

trans['Q1-1998'] = np.where((trans['date']>= 980101) & (trans['date']<= 980331) , 1, 0)
trans['Q2-1998'] = np.where((trans['date']>= 980401) & (trans['date']<= 980631) , 1, 0)
trans['Q3-1998'] = np.where((trans['date']>= 980701) & (trans['date']<= 980931) , 1, 0)
trans['Q4-1998'] = np.where((trans['date']>= 980801) & (trans['date']<= 981231) , 1, 0)

#creating quarters table
quarters= pd.pivot_table(trans, index=['account_id'], values=['Q1-1993', 'Q1-1994', 
                                                          'Q1-1995','Q1-1996',
                                                         'Q1-1997','Q1-1998','Q2-1993', 'Q2-1994', 
                                                          'Q2-1995','Q2-1996',
                                                         'Q2-1997','Q2-1998','Q3-1993', 'Q3-1994', 
                                                          'Q3-1995','Q3-1996',
                                                         'Q3-1997','Q3-1998','Q4-1993', 'Q4-1994', 
                                                          'Q4-1995','Q4-1996',
                                                         'Q4-1997','Q4-1998'], 
                     aggfunc= ('sum'))

#pivoting transactions table by type and suming the amount for each type of operation
trans_pivot1 = trans.pivot_table(index= ("account_id"),columns= ("type") , values = ("amount"), aggfunc= ('sum'))
trans_pivot1.reset_index(level=0, inplace=True)
trans_pivot1['Withdrawals'] = trans_pivot1['VYDAJ']+ trans_pivot1['VYBER']
trans_pivot1= trans_pivot1.drop(['VYDAJ', 'VYBER'],axis=1)
trans_pivot1.columns= ['account_id', 'Credits','Withdrawals']

trans_pivot1['Credits'].astype(float).convert_objects()
trans_pivot1['Withdrawals'].astype(float).convert_objects()


#pivoting transactions table by operation and suming the amount for each type of operation
trans_pivot2 = trans.pivot_table(index= ("account_id"),columns= ("operation") , values = ("amount"), aggfunc= ('sum'))
trans_pivot2.reset_index(level=0, inplace=True)
trans_pivot2.columns= ['account_id', 'remittance_another_bank',
                       'collection_another_bank','credit_cash',
                      'withdrawal_cash', 'creditcard_withdrawal']

trans_pivot2.info()

#merging: transactions per type, transactions per operation type and quarters.
transpivot_final = pd.merge(trans_pivot1, trans_pivot2, how= "left" , on=["account_id"])

trans_final = pd.merge(transpivot_final,quarters, how= 'left', on=['account_id'])

trans_final.head()
trans_final.info()
order_final.info()

trans_pivot1.info()
trans_pivot2.info()
quarters.info()

transpivot_final.info()
trans_final.info()


######################
####Account ##########
######################

# Each record describes static characteristics of an account
account = pd.read_csv('C:/Users/yliu10/Desktop/Python Group/Origin/data_berka/account.asc', sep=';')
account.head()
account.info()

account.duplicated(subset= "account_id",keep = False).sum()
account.info()

######################
#### DISPOSITION #####
######################

# Each record relates together a client with an account i.e. this relation describes the rights
# of clients to operate accounts
disp = pd.read_csv('C:/Users/yliu10/Desktop/Python Group/Origin/data_berka/disp.asc', sep=';')
disp.head(20)


#####################
#### CARD ###########
#####################

# Each record describes a credit card issued to an account
card = pd.read_csv('C:/Users/yliu10/Desktop/Python Group/Origin/data_berka/card.asc', sep=';')
card.head(20)

card['issued2'] = card.issued.astype(str).str[0:6].astype(int)
card['issued2'] = 19000000 + card['issued2']
card['iyear'] = card.issued2.astype(str).str[:4].astype(int)
card['imonth'] = card.issued2.astype(str).str[4:6].astype(int)
card['idate'] = card.issued2.astype(str).str[6:8].astype(int)
card['issued_date']= card['iyear'].astype(str) + '-' + card['imonth'].astype(str) + '-' + card['idate'].astype(str)
card['issued_date'] = pd.to_datetime(card['issued_date'])

card = card.drop("iyear",1)
card = card.drop("issued2",1)
card = card.drop("imonth",1)
card = card.drop("idate",1)
card = card.drop("issued",1)

#############
#Demographics
#############

# Each record describes demographic characteristics of a district.
district = pd.read_csv('C:/Users/yliu10/Desktop/Python Group/Origin/data_berka/district.asc', sep=';')
district.rename(columns={'A1':'district_id', 'A2':'district_name','A3':'region','A4':'nbr_inhabitants','A5':'nbr_municipalities_inhabitants < 499','A6':'nbr_municipalities_inhabitants 500-1999','A7':'nbr_municipalities_inhabitants 2000-9999','A8':'nbr_municipalities_inhabitants  >10000','A9':'nbr_cities','A10':'ratio_urban_inhabitants','A11':'average_salary','A12':'unemploymant_rate95','A13':'unemploymant_rate96','A14':'nbr_enterpreneurs_1000inhabitants','A15':'nbr_crimes95','A16':'nbr_crimes96'},inplace=True)


######################
####Client data ######
######################

# Each record describes characteristics of a client
client = pd.read_csv('C:/Users/yliu10/Desktop/Python Group/Origin/data_berka/client.asc', sep=';')
pd.merge(client,district, how='inner',)
Client_District=pd.merge(client,district,how='inner', on= 'district_id')



def get_mid2_dig(x):
    return int(x/100) % 100

# returns the month of birth_number.
def get_month(x):
    mth = get_mid2_dig(x)
    if mth > 50:
        return mth - 50
    else:
        return mth

# returns the month of birth_number.
def get_day(x):
    return x % 100

# returns the year of birth_number.
def get_year(x):
    return int(x/10000)

# reference dates.
start_date = datetime.datetime(1993,1,1)
end_date = datetime.datetime(2000,1,1)
# function to convert a date to age at end_date.
def convert_to_age_days(x):
    td = end_date - x
    return td.days

# function to convert a date to days after start_date.
def convert_date_to_days(x):
    td = x - start_date
    return td.days

# converts the birth_number into a date.
def convert_int_to_date(x):
    yr = get_year(x) + 1900
    mth = get_month(x)
    day = get_day(x)
    return datetime.datetime(yr, mth, day)

# converts birth_number into age.
def convert_birthday_to_age(x):
    yr = get_year(x) + 1900
    mth = get_month(x)
    day = get_day(x)
    return convert_to_age_days(datetime.datetime(yr,mth,day))/365


#Converting Birth Number To Age 
Client_District['client_age'] = Client_District['birth_number'].map(convert_birthday_to_age).round()


#Correcting the format of unemployment_rate95 and nbr_crimes95
def convert_question_marks(x, typ):
    if x == '?':
        return -1
    elif typ == 'float':
        return float(x)
    else:
        return int(x)
    
Client_District['unemploymant_rate95'] = Client_District['unemploymant_rate95'].apply(convert_question_marks, args=('float',))
Client_District['nbr_crimes95'] = Client_District['nbr_crimes95'].apply(convert_question_marks, args=('int',))

Client_District = Client_District[['client_id','client_age']]


#Joining dataset
da = disp.merge(account, on= 'account_id', how = 'left')
dac = da.merge(card, on= 'disp_id', how = 'left')
dacc = dac.merge(client, on= 'client_id', how = 'left')
dacc.rename(columns = {'district_id_y':'district_id'}, inplace=True)

daccd1 = dacc.merge(district, on= 'district_id', how = 'left')
daccd = daccd1.merge(Client_District, on= 'client_id', how = 'left')
daccd['A_district_id'] = daccd['district_id']

#Adding gender
daccd['gender'] = daccd.birth_number.astype(str).str[2:4].astype(int)
daccd['gender'] = np.where(daccd['gender']>12, 'female', 'male')

#Recreate birthdate
daccd['byear']= 1900 + daccd.birth_number.astype(str).str[:2].astype(int)
daccd['bmonth'] = np.where(daccd.birth_number.astype(str).str[2:4].astype(int) >50, daccd.birth_number.astype(str).str[2:4].astype(int)-50, daccd.birth_number.astype(str).str[2:4].astype(int))
daccd['bdate'] = daccd.birth_number.astype(str).str[4:6].astype(int)
daccd['birthdate']= daccd['byear'].astype(str) + '-' + daccd['bmonth'].astype(str) + '-' + daccd['bdate'].astype(str)
daccd['birthdate'] = pd.to_datetime(daccd['birthdate'])

daccd.info()

#Merge all tables together to create customer data mart

data_mart =pd.merge(daccd,trans_final, how= 'left',on='account_id')
data_mart = data_mart.drop("district_id_x",1)

#Renaming type of disposition variable
data_mart.rename(columns = {'type_x':'type_disp'}, inplace=True)

# Changing definitions for frequency
data_mart['frequency'] = np.where(data_mart['frequency']== 'POPLATEK MESICNE', 'Monthly', 
                             np.where(data_mart['frequency']== 'POPLATEK TYDNE', 'Weekly','After_transaction'))
data_mart['frequency'].unique()

# Cleaning date of creation of the account variable
data_mart['date'] =  pd.to_datetime(data_mart['date'], format='%y%m%d')
data_mart.rename(columns = {'date':'Creation_account'}, inplace=True)
data_mart.rename(columns = {'type_y':'Type_card'}, inplace=True)

#Dropping unsed variables
data_mart = data_mart.drop("birth_number",1)
data_mart = data_mart.drop("A_district_id",1)
data_mart = data_mart.drop("bdate",1)
data_mart = data_mart.drop("byear",1)
data_mart = data_mart.drop("bmonth",1)


#Addin loan and permanent orders.
data_mart =pd.merge(data_mart,order_final, how= 'left',on='account_id')
data_mart =pd.merge(data_mart,loan, how= 'left',on='account_id')


data_mart.to_csv('C:/Users/yliu10/Desktop/Final_python_Group 3/datamart.csv',index=False)










##################
# ANALYZING DATA #
##################

# Age distribution
data_mart['client_age'].plot.hist(title="Age distribution")
plt.show()

# Clientes per gender
plotgender= data_mart.groupby('gender')['client_id'].count().plot(kind='pie',figsize=(5,5),
                              title="Clientes per gender",autopct='%1.0f%%')
plt.show()


# Clients per disposition type
plotdisp= data_mart.groupby('type_disp')['client_id'].count().plot(kind='pie',figsize=(5,5),
                              title="Disposition type",autopct='%1.0f%%')
plt.show()

# Cards per year
cardissued= data_mart.groupby('date')['card_id'].count()
cardissued = cardissued.reset_index()
cardissued['year'] = pd.DatetimeIndex(cardissued['date']).year
cardissuedb = cardissued.groupby('year')['card_id'].sum().plot(kind='bar',figsize=(5,5),title="Cards issued by year")
cardissuedb.set_xlabel("Year")
cardissuedb.set_ylabel("Number of cards")
plt.show()


#Count of client by region
ax = data_mart['region'].value_counts().plot(kind='bar',figsize=(10,10),title=" Count of client by region")
ax.set_xlabel("region")
ax.set_ylabel("Count of client")
plt.show()


#Count of type card by region
plt.figure(figsize=(10,10))
plt.suptitle('Count of type card by region')
sns.countplot(data=data_mart, x='region', hue='Type_card')
plt.show()

#Average_salary by region
plt.figure(figsize=(10,10))
plt.suptitle('Average salary by region')
sns.barplot(data=data_mart, x='region', y='average_salary', hue='region')
plt.show()

#   !!!!!  pip3 --no-cache-dir install seaborn !!!!!
#Quarterly count of trans. per account each year 1993 to 1998
plt.figure(figsize=(10,10))
plt.suptitle('Count of trans. 1993')
plt.subplot(2,2,1)
sns.scatterplot(data=data_mart, x='client_id', y='Q1-1993', color='r')
plt.ylim(0,70)
plt.subplot(2,2,2)
sns.scatterplot(data=data_mart, x='client_id', y='Q2-1993', color='b')
plt.ylim(0,70)
plt.subplot(2,2,3)
sns.scatterplot(data=data_mart, x='client_id', y='Q3-1993', color='g')
plt.ylim(0,70)
plt.subplot(2,2,4)
sns.scatterplot(data=data_mart, x='client_id', y='Q4-1993', color='k')
plt.ylim(0,70)
plt.savefig('C:/Users/yliu10/Desktop/png/CountTransbyQuarter1993.png', transparent=True)
plt.show()

plt.figure(figsize=(10,10))
plt.suptitle('Count of trans. 1994')
plt.subplot(2,2,1)
sns.scatterplot(data=data_mart, x='client_id', y='Q1-1994', color='r')
plt.ylim(0,70)
plt.subplot(2,2,2)
sns.scatterplot(data=data_mart, x='client_id', y='Q2-1994', color='b')
plt.ylim(0,70)
plt.subplot(2,2,3)
sns.scatterplot(data=data_mart, x='client_id', y='Q3-1994', color='g')
plt.ylim(0,70)
plt.subplot(2,2,4)
sns.scatterplot(data=data_mart, x='client_id', y='Q4-1994', color='k')
plt.ylim(0,70)
plt.savefig('C:/Users/yliu10/Desktop/png/CountTransbyQuarter1994.png', transparent=True)
plt.show()

plt.figure(figsize=(10,10))
plt.suptitle('Count of trans. 1995')
plt.subplot(2,2,1)
sns.scatterplot(data=data_mart, x='client_id', y='Q1-1995', color='r')
plt.ylim(0,70)
plt.subplot(2,2,2)
sns.scatterplot(data=data_mart, x='client_id', y='Q2-1995', color='b')
plt.ylim(0,70)
plt.subplot(2,2,3)
sns.scatterplot(data=data_mart, x='client_id', y='Q3-1995', color='g')
plt.ylim(0,70)
plt.subplot(2,2,4)
sns.scatterplot(data=data_mart, x='client_id', y='Q4-1995', color='k')
plt.ylim(0,70)
plt.savefig('C:/Users/yliu10/Desktop/png/CountTransbyQuarter1995.png', transparent=True)
plt.show()

plt.figure(figsize=(10,10))
plt.suptitle('Count of trans. 1996')
plt.subplot(2,2,1)
sns.scatterplot(data=data_mart, x='client_id', y='Q1-1996', color='r')
plt.ylim(0,70)
plt.subplot(2,2,2)
sns.scatterplot(data=data_mart, x='client_id', y='Q2-1996', color='b')
plt.ylim(0,70)
plt.subplot(2,2,3)
sns.scatterplot(data=data_mart, x='client_id', y='Q3-1996', color='g')
plt.ylim(0,70)
plt.subplot(2,2,4)
sns.scatterplot(data=data_mart, x='client_id', y='Q4-1996', color='k')
plt.ylim(0,70)
plt.savefig('C:/Users/yliu10/Desktop/png/CountTransbyQuarter1996.png', transparent=True)
plt.show()

plt.figure(figsize=(10,10))
plt.suptitle('Count of trans. 1997')
plt.subplot(2,2,1)
sns.scatterplot(data=data_mart, x='client_id', y='Q1-1997', color='r')
plt.ylim(0,70)
plt.subplot(2,2,2)
sns.scatterplot(data=data_mart, x='client_id', y='Q2-1997', color='b')
plt.ylim(0,70)
plt.subplot(2,2,3)
sns.scatterplot(data=data_mart, x='client_id', y='Q3-1997', color='g')
plt.ylim(0,70)
plt.subplot(2,2,4)
sns.scatterplot(data=data_mart, x='client_id', y='Q4-1997', color='k')
plt.ylim(0,70)
plt.savefig('C:/Users/yliu10/Desktop/png/CountTransbyQuarter1997.png', transparent=True)
plt.show()

plt.figure(figsize=(10,10))
plt.suptitle('Count of trans. 1998')
plt.subplot(2,2,1)
sns.scatterplot(data=data_mart, x='client_id', y='Q1-1998', color='r')
plt.ylim(0,70)
plt.subplot(2,2,2)
sns.scatterplot(data=data_mart, x='client_id', y='Q2-1998', color='b')
plt.ylim(0,70)
plt.subplot(2,2,3)
sns.scatterplot(data=data_mart, x='client_id', y='Q3-1998', color='g')
plt.ylim(0,70)
plt.subplot(2,2,4)
sns.scatterplot(data=data_mart, x='client_id', y='Q4-1998', color='k')
plt.ylim(0,70)
plt.savefig('C:/Users/yliu10/Desktop/png/CountTransbyQuarter1998.png', transparent=True)
plt.show()


#Output gif
import os
import imageio
png_dir = 'C:/Users/yliu10/Desktop/png' #png folder path
images = []
for file_name in os.listdir(png_dir):
    if file_name.endswith('.png'):
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))
imageio.mimsave('C:/Users/yliu10/Desktop/png/0.gif', images, duration=1) #save path and file name, duration = speed in second


# Distribution of duration of loans 
data_mart.groupby('duration')['duration'].count().round().plot(kind='pie',figsize=(5,5),
                              title= 'Loan duration', autopct='%1.0f%%' )
data_mart.groupby('duration')['duration'].count()
plt.show()

#Amounts and paymens per loan 
plotamount = data_mart.groupby('date')['amount','payments'].sum()
plotamount = plotamount.reset_index()
plotamount['year'] = pd.DatetimeIndex(plotamount['date']).year
plotamount['month'] = pd.DatetimeIndex(plotamount['date']).month
plotamount.groupby('month')['amount','payments'].sum().plot(kind='line' ,subplots=True, xticks= (1,2,3,4,5,6,7,8,9,10,11,12))
plt.show()

plotamount.groupby('year')['amount','payments'].sum()
dateloan = data_mart.groupby('date')["loan_id"].count().reset_index()
dateloan['year'] = pd.DatetimeIndex(dateloan['date']).year
dateloan.groupby("year")["loan_id"].count().plot(kind="line", title="Loans granted by year")
plt.show()

# loan status
status = data_mart.groupby('status')['loan_id'].count().reset_index()
labels = status['status']
sizes = status['loan_id']
plt.title("Loan status")
plt.pie(sizes,  labels=labels, 
        autopct='%1.1f%%', shadow=False, startangle=140)
 
plt.axis('equal')
plt.show()


# Number of payments
plotcredits = data_mart.groupby('region')['nbr_Other','nbr_Leasing','nbr_Insurance_Pay','nbr_Household_Pay' ,'nbr_Loan_Pay'].sum().plot(kind='bar',figsize=(8,8),title= 'Number of payments per region and type' )
plotcredits.set_xlabel("Region")
plotcredits.set_ylabel("Number")
plt.show()


# Credits and withdrawals
plotcredits = data_mart.groupby('region')['Credits','Withdrawals'].sum().plot(kind='bar',figsize=(5,5),title= 'Credits and withdrawals' ,stacked=True)
plotcredits.set_xlabel("Region")
plotcredits.set_ylabel("Amount")
line_up, = plt.plot([1,2,3], label='Line 2')
line_down, = plt.plot([3,2,1], label='Line 1')
plt.legend([line_up, line_down], ['Credits', 'Withdrawals'])
plt.show()

# accounts open per year 
accopen = data_mart.groupby('date')['account_id'].count()
accopen = accopen.reset_index()
accopen['year'] = pd.DatetimeIndex(accopen['date']).year
accopenb = accopen.groupby('year')['account_id'].sum().plot(kind='bar',figsize=(5,5),title="Accounts opened by year")
accopenb.set_xlabel("Year")
accopenb.set_ylabel("Number of accounts")
plt.show()


# Mode of transactions
plotmodetrans = data_mart.groupby('region')['remittance_another_bank',
                       'collection_another_bank','credit_cash',
                      'withdrawal_cash', 'creditcard_withdrawal'].sum().plot(kind='bar',figsize=(8,8),title= 'Mode of transactions per region', legend= True)
plotcredits.set_xlabel("Region")
plotcredits.set_ylabel("Amount")
plt.show()


# Type of client.

typeclient = data_mart.groupby('type_disp')['client_id'].count().plot(kind='pie',figsize=(5,5),
                              title= 'Type of client' , autopct='%1.0f%%')
plt.show()

