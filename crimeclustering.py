import pandas as pd
import numpy as np
import os, sys
import warnings

from sklearn.cluster import DBSCAN
import hdbscan


class Clustering:

	'''
		Remove columns different from lat and long
	'''
	def encode(self, data):
		data = data.drop(['type'], axis=1).drop(['hour'], axis=1)
		return data

	'''
		Clusterize crime data
	'''
	def clusterize(self, data):
		data_formated = self.encode(data.copy())
		clustering = DBSCAN(eps=0.01, min_samples=4).fit_predict(data_formated)
		#clustering = hdbscan.HDBSCAN(min_cluster_size=10).fit_predict(data_formated)
		data['cluster'] = clustering
		return data.sort_values('cluster')

######################################################################


class CrimeClustering:


	MONTHS = {
				1  : 'January',
				2  : 'February',
				3  : 'March',
				4  : 'April',
				5  : 'May',
				6  : 'June',
				7  : 'July',
				8  : 'August',
				9  : 'September',
				10 : 'October',
				11 : 'November',
				12 : 'December',
	     	}

	def remove_invalid_coord(self, df): #[-90; 90]
		return df.query('lat >= -90 & lat <= 90').query('lon >= -90 & lat <= 90')

	def read_data(self, day):

		data_file = open('data/' + day + '/crimes.csv', 'r')

		crime_list = []

		for line in data_file:
			line = line.strip().split(',')

			item = {}
			item['datetime'] = pd.to_datetime(str(line[0]), format='%Y/%m/%d %H:%M')
			item['hour'] = pd.to_datetime(str(line[0]), format='%Y/%m/%d %H:%M').hour
			item['minute'] = pd.to_datetime(str(line[0]), format='%Y/%m/%d %H:%M').minute
			item['lat'] = float(line[1])
			item['lon'] = float(line[2])
			item['type'] = line[3].strip()

			crime_list.append(item)

		df = pd.DataFrame(crime_list)
		df.set_index('datetime', inplace=True)

		return self.remove_invalid_coord(df)

	def make_gauss(self, N=1, sig=1, mu=0):
		return lambda xt: N/(sig * (2*np.pi)**.5) * np.e ** (-(xt-mu)**2/(1000 * sig**2))

	def calculate_difference(self, hour, minute, ref_hour, ref_minute):

		time = hour*60 + minute
		ref_time = ref_hour*60 + ref_minute

		xt = time - ref_time

		score = self.make_gauss()(xt)

		#print('%.10f' % (score))
		#input(';')

		return float('%.10f' % (score))
		

	def calculate_score(self, crimes_filtered):

		window_scores = []

		for hour in range(0, 24):

			for minute in range(0, 60, 10):

				state_score = []
				for index, row in crimes_filtered.iterrows():
					state_score.append(self.calculate_difference(row['hour'], row['minute'], hour, minute))

				window_scores.append(np.sum(state_score))

		return window_scores



	def clusterize(self):

		clustering = Clustering()

		month = 1
		day = 'monday'

		for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:

			day_crimes = self.read_data(day)

			for month in range(1, 13):

				df_crimes = day_crimes['2018-' + str(month)]
				crimes = df_crimes.groupby('type').all().index

				for crime in crimes:

					crimes_filtered = df_crimes.query("type == '%s'" % crime)
					crimes_filtered = clustering.clusterize(crimes_filtered).query('cluster != -1')	

					if not crimes_filtered.empty:
						
						window_scores = self.calculate_score(crimes_filtered)

							
				exit()

		
######################################################################

def main():

	cc = CrimeClustering()
	cc.clusterize()

if __name__ == '__main__':
	warnings.simplefilter("ignore")
	main()