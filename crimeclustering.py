import pandas as pd
import numpy as np
import os, sys
import warnings
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
#import hdbscan
from scipy.signal import find_peaks


class Clustering:

	'''
		Remove columns different from lat and long
	'''
	def encode(self, data):
		data = data.drop(['type', 'hour', 'minute'], axis=1)
		return data

	'''
		Clusterize crime data
	'''
	def clusterize(self, data):
		data_formated = self.encode(data.copy())
		clustering = DBSCAN(eps=0.01, min_samples=3).fit_predict(data_formated)
		#clustering = hdbscan.HDBSCAN(min_cluster_size=10).fit_predict(data_formated)
		data['cluster'] = clustering
		return data.sort_values('cluster')


######################################################################

class Util:

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

	def format_digits(self, number):

		if len(str(number)) < 2:
			number = '0' + str(number)
		return str(number)

	def format_clusters(self, data):

		clusters = []
		clusters.append([])
		lastid = 0

		data = data.query('cluster > -1')

		for indx, row in data.iterrows():
			if row['cluster'] > lastid:
				clusters.append([])
				lastid = row['cluster']
			clusters[-1].append((row['lat'], row['lon']))

		return clusters

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

######################################################################


'''
	Initial code for crime clustering considering minutes interval and gaussian distribution
'''
class CrimeClustering:

	def __init__(self):
		self.u = Util()

	def make_gauss(self, N=1, sig=1, mu=0):
		return lambda xt: N/(sig * (2*np.pi)**.5) * np.e ** (-(xt-mu)**2/(1000 * sig**2))

	def calculate_difference(self, hour, minute, ref_hour, ref_minute):

		time = hour*60 + minute
		ref_time = ref_hour*60 + ref_minute

		xt = time - ref_time

		score = self.make_gauss()(xt)

		return float('%.5f' % (score))


	def calculate_score(self, crimes_filtered):

		window_scores = []

		for hour in range(0, 24):

			for minute in range(0, 60, 10):

				state_score = []
				for index, row in crimes_filtered.iterrows():
					state_score.append(self.calculate_difference(row['hour'], row['minute'], hour, minute))

				window_scores.append(np.sum(state_score))

		return window_scores

	def plot_to_see(self, window_scores):

		plt.figure(1)
		plt.subplot(211)

		#for w in window_scores:
		plt.plot(range(0, 144), window_scores, '--')
		plt.xticks(np.arange(0, 145, 6), np.arange(0, 25))
		plt.grid(True)

		plt.show()

	def identify_window(self, window_scores, peaks):


		iterpeaks = iter(peaks)
		next(iterpeaks)
		last_peak = peaks[0]

		apeaks = []

		for peak in iterpeaks:

			apeaks.append(last_peak + np.argmin(window_scores[last_peak:peak]))
			last_peak = peak

		return [0] + apeaks

	def get_window(self, start, end, crimes_filtered):

		start_hour = start * 10 // 60
		start_minute = start * 10 % 60

		end_hour = end * 10 // 60
		end_minute = end * 10 % 60

		# Filter the closed interval
		crimes_opened = crimes_filtered.query('hour > {0} & hour < {1}'.format(start_hour, end_hour))

		# Filter the opened interval
		crimes_closed_low = crimes_filtered.query('hour == {0} & minute >= {1}'.format(start_hour, start_minute))
		crimes_closed_high = crimes_filtered.query('hour == {0} & minute < {1}'.format(end_hour, end_minute))

		return pd.concat([crimes_opened, crimes_closed_low, crimes_closed_high])

	# Write data

	'''
		Save clusters
	'''
	def write_clusters(self, clusters, month, day, start, crime):

		if not os.path.exists('clusters'):
			os.makedirs('clusters')

		output_file = open('clusters/{0}_{1}_{2}_{3}_clusters.txt'.format(self.u.MONTHS[month], str(day), crime, self.format_digits(str(start))), 'w')

		for cluster in clusters:
			for point in cluster:
				output_file.write(str(point[0]) + ' ' + str(point[1]) + '; ')

			output_file.write('\n')
		output_file.close()

	def clusterize(self):

		clustering = Clustering()

		for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:

			day_crimes = self.u.read_data(day)

			print('### ' + day)

			for month in range(1, 13):

				print('#' + self.u.MONTHS[month])

				df_crimes = day_crimes['2018-' + str(month)]
				crimes = df_crimes.groupby('type').all().index

				for crime in crimes:
					
					crimes_filtered = df_crimes.query("type == '%s'" % crime)
					crimes_filtered = clustering.clusterize(crimes_filtered).query('cluster != -1')	

					if not crimes_filtered.empty:
						
						window_scores = self.calculate_score(crimes_filtered)

						peaks = find_peaks(window_scores, distance=9)[0]

						if len(peaks) > 0:
						
							window = self.identify_window(window_scores, peaks)

							if len(window) > 0:
								iterwindow = iter(window)
								next(iterwindow)
								last_window = window[0]
								for iw in iterwindow:
									
									crimes_window = self.get_window(last_window, iw, crimes_filtered)

									cluster_crime = clustering.clusterize(crimes_window)
									clusters = self.u.format_clusters(cluster_crime)

									#self.write_clusters(clusters, month, day, last_window, crime)
									
									last_window = iw

		
######################################################################


class FixedWindowClustering:

	def __init__(self, size):
		self.size = size

	def get_window(self, df, start, end):
		return df.query('hour >= ' + str(start) + ' & hour < ' + str(end))

	def convert_to_minutes(self, hour, minutes):
		return hour * 60 + minutes


	def metric_max_interval(self, clusters):

		interval = 0
		indx_cluster = clusters['cluster'].max()

		if not np.isnan(indx_cluster):

			for i in range(0, indx_cluster):

				cluster_interval = 0
				last_datetime = None

				crime_clusters = clusters.query('cluster == ' + str(i)).sort_values(by=['minute', 'hour'])

				for index, row in crime_clusters.iterrows():

					if cluster_interval == 0:
						last_datetime = self.convert_to_minutes(row['hour'], row['minute'])
						cluster_interval = 1

					else:
						diff = self.convert_to_minutes(row['hour'], row['minute']) - last_datetime

						if diff > cluster_interval:
							cluster_interval = diff

						last_datetime = self.convert_to_minutes(row['hour'], row['minute'])

				if cluster_interval > interval:
					interval = cluster_interval

		return interval

	def metric_close_crimes(self, clusters, start, end):
		return 0

	def clusterize(self, month_crimes, clustering):

		result_max, result_close = [], []
	
		for i in range(0, 24, self.size):
			window_crime = self.get_window(month_crimes, i, i+self.size)
			clusters = clustering.clusterize(window_crime).query('cluster != -1')
			result_max.append(self.metric_max_interval(clusters))
			result_close.append(self.metric_close_crimes(clusters, i, i+self.size))

		return np.max(result_max), np.max(result_close)



class TimeMinutesClustering:

	def __init__(self):
		self.u = Util()

	def make_gauss(self, N=1, sig=1, mu=0):
		return lambda xt: N/(sig * (2*np.pi)**.5) * np.e ** (-(xt-mu)**2/(1000 * sig**2))

	def calculate_difference(self, hour, minute, ref_hour, ref_minute):

		time = hour*60 + minute
		ref_time = ref_hour*60 + ref_minute

		xt = time - ref_time

		score = self.make_gauss()(xt)

		return float('%.5f' % (score))


	def calculate_score(self, crimes_filtered):

		window_scores = []

		for hour in range(0, 24):

			for minute in range(0, 60, 10):

				state_score = []
				for index, row in crimes_filtered.iterrows():
					state_score.append(self.calculate_difference(row['hour'], row['minute'], hour, minute))

				window_scores.append(np.sum(state_score))

		return window_scores

	def identify_window(self, window_scores, peaks):

		iterpeaks = iter(peaks)
		next(iterpeaks)
		last_peak = peaks[0]

		apeaks = []

		for peak in iterpeaks:

			apeaks.append(last_peak + np.argmin(window_scores[last_peak:peak]))
			last_peak = peak

		return [0] + apeaks

	def get_window(self, start, end, crimes_filtered):

		start_hour = start * 10 // 60
		start_minute = start * 10 % 60

		end_hour = end * 10 // 60
		end_minute = end * 10 % 60

		# Filter the closed interval
		crimes_opened = crimes_filtered.query('hour > {0} & hour < {1}'.format(start_hour, end_hour))

		# Filter the opened interval
		crimes_closed_low = crimes_filtered.query('hour == {0} & minute >= {1}'.format(start_hour, start_minute))
		crimes_closed_high = crimes_filtered.query('hour == {0} & minute < {1}'.format(end_hour, end_minute))

		return pd.concat([crimes_opened, crimes_closed_low, crimes_closed_high])

	def convert_to_minutes(self, hour, minutes):
		return hour * 60 + minutes

	def metric_max_interval(self, clusters):

		interval = 0
		indx_cluster = clusters['cluster'].max()

		if not np.isnan(indx_cluster):

			for i in range(0, indx_cluster):

				cluster_interval = 0
				last_datetime = None

				crime_clusters = clusters.query('cluster == ' + str(i)).sort_values(by=['minute', 'hour'])

				for index, row in crime_clusters.iterrows():

					if cluster_interval == 0:
						last_datetime = self.convert_to_minutes(row['hour'], row['minute'])
						cluster_interval = 1

					else:
						diff = self.convert_to_minutes(row['hour'], row['minute']) - last_datetime

						if diff > cluster_interval:
							cluster_interval = diff

						last_datetime = self.convert_to_minutes(row['hour'], row['minute'])

				if cluster_interval > interval:
					interval = cluster_interval

		return interval
	
	def metric_close_crimes(self, clusters, start, end):
		return 0

	def clusterize(self, month_crimes, clustering):

		result_max, result_close = [0], [0]
		crimes = month_crimes.groupby('type').all().index

		for crime in crimes:
			
			crimes_filtered = month_crimes.query("type == '%s'" % crime)
			crimes_filtered = clustering.clusterize(crimes_filtered).query('cluster != -1')

			if not crimes_filtered.empty:
				
				window_scores = self.calculate_score(crimes_filtered)

				peaks = find_peaks(window_scores, distance=9)[0]

				if len(peaks) > 0:
				
					window = self.identify_window(window_scores, peaks)

					if len(window) > 0:

						iterwindow = iter(window)
						next(iterwindow)
						last_window = window[0]
						for iw in iterwindow:
							
							crimes_window = self.get_window(last_window, iw, crimes_filtered)

							cluster_crime = clustering.clusterize(crimes_window).query('cluster != -1')
							result_max.append(self.metric_max_interval(cluster_crime))
							result_close.append(self.metric_close_crimes(cluster_crime, last_window, iw))

							last_window = iw

		return np.max(result_max), np.max(result_close)


######################################################################

class CompareClustering:

	def __init__(self):
		self.u = Util()

	def plot_max_metric(self, result_max):
		
		fig, ax = plt.subplots()

		x = [x for x in range(len(result_max[0]))]

		labely = ['Fixed 1', 'Fixed 2', 'Fixed 4', 'Fixed 8', 'Fixed 12', 'Time Minutes']

		for indx, result in enumerate(result_max):
			ax.plot(x, result, 'o--', label=labely[indx])

		labels = []
		for month in range(1, 13):
			labels.append(self.u.MONTHS[month])
			for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
				labels.append(day)
		
		plt.xticks(x, labels)
		ax.grid()

		plt.show()

	def clusterize(self):

		clustering = Clustering()

		result_strategy = {'max': [[], [], [], [], [], []], 'close': [[], [], [], [], [], []]}

		for month in range(1, 13):

			print('#' + self.u.MONTHS[month])

			for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:

				print('### ' + day)

				day_crimes = self.u.read_data(day)

				month_crimes = day_crimes['2018-' + str(month)]

				for indx, strategy in enumerate([FixedWindowClustering(1), FixedWindowClustering(2), FixedWindowClustering(4), FixedWindowClustering(8), FixedWindowClustering(12),\
					TimeMinutesClustering()]):

					maxi, close = strategy.clusterize(month_crimes, clustering)
					result_strategy['max'][indx].append(maxi)
					result_strategy['close'][indx].append(close)

		self.plot_max_metric(result_strategy['max'])


######################################################################


def main():

	cc = CompareClustering()
	cc.clusterize()

if __name__ == '__main__':
	warnings.simplefilter("ignore")
	main()