import pandas as pd
import numpy as np
import os, sys
import warnings

import matplotlib.pyplot as plt
from mlxtend.plotting import ecdf

from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
import hdbscan

import threading
import time


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
	def clusterize(self, data, ep=0.01):

		data_formated = self.encode(data.copy())
		#clustering = DBSCAN(eps=ep, min_samples=3).fit_predict(data_formated)
		clustering = hdbscan.HDBSCAN(min_cluster_size=3).fit_predict(data_formated)
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

				crime_clusters = clusters.query('cluster == ' + str(i)).sort_values(by=['hour', 'minute'])

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

	def metric_cluster(self, clusters):

		cl = 0
		if not clusters is None and not np.isnan(clusters['cluster'].max()):
			cl = clusters['cluster'].max()

		return cl

	def clusterize(self, month_crimes, clustering):

		result_max = [0]
		result_cluster = {}

		crimes = month_crimes.groupby('type').all().index

		for crime in crimes:
			
			crimes_filtered = month_crimes.query("type == '%s'" % crime)

			if not crimes_filtered.empty:
	
				for i in range(0, 24, self.size):
					
					window_crime = self.get_window(crimes_filtered, i, i+self.size)

					clusters = None

					if len(window_crime) >= 3:
						clusters = clustering.clusterize(window_crime).query('cluster != -1')
						
						max_interval = self.metric_max_interval(clusters)
						result_max.append(max_interval)

					cluster = self.metric_cluster(clusters)
					if crime not in result_cluster:
						result_cluster[crime] = []
					result_cluster[crime].append(cluster)

		return np.max(result_max), result_cluster


class TimeMinutesClustering:

	def __init__(self):
		self.u = Util()

	def make_gauss(self, N=1, sig=1, mu=0):
		#return lambda x: N/(sig * (2*np.pi)**.5) * np.e ** (-(x-mu)**2/(410 * sig**2))
		return lambda x: N/(sig * (2*np.pi)**.5) * np.e ** (-(x-mu)**2/(105 * sig**2))

	def calculate_difference(self, hour, minute, ref_hour, ref_minute):

		time = hour*60 + minute
		ref_time = ref_hour*60 + ref_minute

		xt = time - ref_time

		score = self.make_gauss()(xt)

		return float('%.5f' % (score))

	def normalize(self, window_scores):

		maxi = np.amax(window_scores)
		mini = np.amin(window_scores)

		for indx, w in enumerate(window_scores):
			window_scores[indx] = (w - mini) / (maxi - mini)

		return window_scores


	def calculate_score(self, crimes_filtered):

		window_scores = []

		for hour in range(0, 24):

			for minute in range(0, 60, 10):

				state_score = []
				for index, row in crimes_filtered.iterrows():
					state_score.append(self.calculate_difference(row['hour'], row['minute'], hour, minute))

				window_scores.append(np.sum(state_score))

		return self.normalize(window_scores)

	def identify_window(self, window_scores, peaks):

		last_peak = 0
		apeaks = []

		peaks.append(len(window_scores)-1)

		for peak in peaks:

			mini = last_peak + np.argmin(window_scores[last_peak:peak])

			if window_scores[mini] == 0.0:
				apeaks.append(mini)

				zero_indx = mini
				while zero_indx < len(window_scores) and window_scores[zero_indx] == 0.0:
					zero_indx += 1
				
				apeaks.append(zero_indx-1)

			else:
				apeaks.append(mini)

			last_peak = peak

		if apeaks[0] < 3:
			apeaks[0] = 0
		if apeaks[0] != 0:
			apeaks.insert(0, 0)

		if apeaks[-1] > len(window_scores) - 3:
			apeaks[-1] = len(window_scores) - 1
		elif apeaks[-1] != len(window_scores)-1:
			apeaks.append(len(window_scores)-1)

		return apeaks

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

				crime_clusters = clusters.query('cluster == ' + str(i)).sort_values(by=['hour', 'minute'])

				cluster_interval = 0
				last_datetime = self.convert_to_minutes(crime_clusters.iloc[0]['hour'], crime_clusters.iloc[0]['minute'])

				for index, row in crime_clusters.iterrows():

					this_datetime = self.convert_to_minutes(row['hour'], row['minute'])

					diff = this_datetime - last_datetime

					if diff > cluster_interval:
						cluster_interval = diff

					last_datetime = this_datetime

				if cluster_interval > interval:
					interval = cluster_interval

		return interval

	def metric_cluster(self, clusters):

		cl = 0
		if clusters is not None and not np.isnan(clusters['cluster'].max()):
			cl = clusters['cluster'].max()

		return cl

	def calculate_percentage_max(self, interval, windowsize):
		return 100 * interval / (windowsize*10)

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

	def plot_to_see(self, window_scores):

		plt.clf()
		plt.figure(1)
		plt.subplot(211)

		#for w in window_scores:
		#window_scores.insert(0, 0)
		plt.plot(range(0, 144), window_scores, '--')

		plt.xticks(np.arange(0, 145, 6), np.arange(0, 25))
		plt.grid(False)
		plt.xlabel('Horas do dia')
		plt.ylabel('Score')

		plt.show()
		#plt.savefig('windows.pdf', bbox_inches="tight", format='pdf')

	def plot_to_save(self, window_scores, name):

		if not os.path.exists('plots'):
			os.makedirs('plots')

		plt.clf()
		plt.subplot(211)

		plt.plot(range(0, 144), window_scores, '--')
			
		plt.xticks(np.arange(0, 145, 12), np.arange(0, 25, 2))
		#plt.grid(True)
		plt.xlabel('Horas do dia')
		plt.ylabel('Distribuição de crimes')

		#plt.grid('off', axis='x')
		#plt.grid('on', axis='y')

		plt.savefig('plots/' + str(name) + '.pdf', bbox_inches="tight", format='pdf')

	def clusterize(self, month_crimes, clustering):

		result_max = [0]
		result_cluster = {}
		crimes = month_crimes.groupby('type').all().index

		#windows = []

		for crime in crimes:
			
			crimes_filtered = month_crimes.query("type == '%s'" % crime)
				
			window_scores = self.calculate_score(crimes_filtered)
			#windows.append(window_scores)
			if crime in ['ASSAULT', 'BATTERY', 'BURGLARY', 'CRIMINAL DAMAGE', 'MOTOR VEHICLE THEFT',
						'ROBBERY', 'THEFT']:
				print(crime)
				self.plot_to_save(window_scores, crime)
				#input(';')

			peaks = find_peaks(window_scores, distance=3)[0].tolist()

			if len(peaks) > 0:
			
				window = self.identify_window(window_scores, peaks)

				if len(window) > 0:

					iterwindow = iter(window)
					next(iterwindow)
					last_window = window[0]
					for iw in iterwindow:

						crimes_window = self.get_window(last_window, iw, crimes_filtered)

						cluster_crime = None

						if len(crimes_window) >= 3:
							cluster_crime = clustering.clusterize(crimes_window).query('cluster != -1')

							max_interval = self.metric_max_interval(cluster_crime)
							result_max.append(max_interval)

						cluster = self.metric_cluster(cluster_crime)
						if crime not in result_cluster:
							result_cluster[crime] = []
						result_cluster[crime].append((cluster, last_window, iw))

						last_window = iw

		#self.plot_to_see(windows)

		return np.max(result_max), result_cluster


######################################################################

class CallClusterize(threading.Thread):

	def __init__(self, indx, strategy, month_crimes, clustering):
		threading.Thread.__init__(self)
		self.indx = indx
		self.strategy = strategy
		self.month_crimes = month_crimes
		self.clustering = clustering

		self.maxi = 0
		self.cluster = 0

	def run(self):
		maxi, cluster = self.strategy.clusterize(self.month_crimes, self.clustering)
		self.maxi = maxi
		self.cluster = cluster

	def get_maxi(self):
		return self.maxi

	def get_cluster(self):
		return self.cluster


class CompareClustering:

	def __init__(self):
		self.u = Util()

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

	def count_crime(self, month_crimes, result_crime):

		crimes = month_crimes.groupby('type').all().index
		for crime in crimes:
			crimes_filtered = month_crimes.query("type == '%s'" % crime)

			if crime not in result_crime:
				result_crime[crime] = []

			if not crimes_filtered.empty:
				start_minute = 0
				for end_minute in range(1, 144):

					result_crime[crime].extend([self.get_window(start_minute, end_minute, crimes_filtered).shape[0]] * 10)
					start_minute = end_minute

			else:
				result_crime[crime].extend([0]*1440)


	def plot_ecdf(self, result_max):

		plt.clf()
		axis = None

		strategy_icon = ['1', '|', '_', '.', 'o', '2']

		for indx, result in enumerate(result_max):
			ax, _, _ = ecdf(x=result, ecdf_marker=strategy_icon[indx])
			axis = ax 

		plt.legend(['Fixed 1', 'Fixed 2', 'Fixed 4', 'Fixed 8', 'Fixed 12', 'MARTINI'], 
			loc='upper center', ncol=3, fancybox=True, bbox_to_anchor=(0.5, 1.15))
		plt.xlabel('Time Interval (minutes)')

		plt.savefig('metric_max_ecdf.pdf', bbox_inches="tight", format='pdf')

	def plot_max_metric(self, result_max):
		
		plt.clf()
		fig, ax = plt.subplots()

		x = [x for x in range(len(result_max[0]))]
		
		labely = ['Fixed 1', 'Fixed 2', 'Fixed 4', 'Fixed 8', 'Fixed 12', 'MARTINI']
		strategy_icon = ['1', '|', '_', '.', 'o', '2']
		
		for indx, result in enumerate(result_max):
			ax.plot(x, result, strategy_icon[indx] + '--', label=labely[indx], alpha=0.7, markersize=6.5)
		ax.legend(loc='upper center', ncol=3, fancybox=True, bbox_to_anchor=(0.5, 1.15))

		labels = []
		for month in range(1, 13):
			for day in ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday']:
				if day is 'monday':
					labels.append(self.u.MONTHS[month])
				else:
					labels.append('')

		plt.ylabel('Time Interval (minutes)')
		plt.xticks(np.arange(0, len(result_max[0])), labels, rotation=50)
		
		ax.grid('off', axis='x')
		ax.grid('on', axis='y')

		plt.savefig('metric_max.pdf', bbox_inches="tight", format='pdf')

	def format_to_min_fixed(self, result_cluster):

		cluster_list = {}

		for i in range(0, 5):

			for fragment in result_cluster[i]:

				for cl in fragment:

					if cl not in cluster_list:
						cluster_list[cl] = []

					size = len(fragment[cl])
					count = (24 // size) * 60

					fcl_list = []
					for fcl in fragment[cl]:
						fcl_list.extend([fcl]*count)

					cluster_list[cl].append(fcl_list)

		return cluster_list

	def format_to_min_timeminutes(self, cluster_list, result_cluster):

		for fragment in result_cluster[5]:

			for cl in fragment:

				if cl not in cluster_list:
					cluster_list[cl] = []

				fcl_list = []
				for fcl in fragment[cl]:
					fcl_list.extend([fcl[0]]*((fcl[2] - fcl[1])*10))

				cluster_list[cl].append(fcl_list)

	def normalize(self, x, mini, maxi):
		return (x - mini) / (maxi - mini)

	def normalize_clusters(self, cluster_list):

		for crime in cluster_list:

			mini, maxi = 0, 0
			for indx, fragment in enumerate(cluster_list[crime]):

				if np.amin(fragment) < mini:
					mini = np.amin(fragment)

				if np.amax(fragment) > maxi:
					maxi = np.amax(fragment)

			if maxi != 0:
				for indx, fragment in enumerate(cluster_list[crime]):
					fragment = [self.normalize(x, mini, maxi) for x in fragment]
					cluster_list[crime][indx] = fragment

	def normalize_crimes(self, crime_list):

		for crime in crime_list:

			mini = np.amin(crime_list[crime])
			maxi = np.amax(crime_list[crime])

			if maxi != 0:
				crime_list[crime] = [self.normalize(x, mini, maxi) for x in crime_list[crime]]

	def plot_cluster(self, result_cluster, result_crime):

		if not os.path.exists('clusters_plot'):
			os.makedirs('clusters_plot')

		cluster_list = self.format_to_min_fixed(result_cluster)
		self.format_to_min_timeminutes(cluster_list, result_cluster)

		self.normalize_clusters(cluster_list)
		self.normalize_crimes(result_crime)

		label = ['One Hour', 'Two Hours', 'Four Hours', 'Eight Hours', 'Twelve Hours', 'MARTINI']

		for crime in cluster_list:

			plt.clf()
			fig, ax = plt.subplots(2)
			
			ax[1].bar(np.arange(0,len(result_crime[crime])), result_crime[crime])

			strategy_icon = ['1', '|', '_', '.', 'o', '2']
			for indx, strategy in enumerate(cluster_list[crime]):
				#strategy = np.convolve(strategy, np.ones((3,))/3, mode='valid')
				#ax[0].plot(strategy, '.', label=label[indx], alpha=0.9, markersize=4)
				ax[0].plot(strategy[0::20], strategy_icon[indx], label=label[indx], alpha=0.9, markersize=6.5)
			ax[0].legend(loc='upper center', ncol=3, fancybox=True, bbox_to_anchor=(0.5, 1.33))

			plt.sca(ax[0])
			#plt.xticks(np.arange(0, 1441, 60), ['']*24)
			plt.xticks(np.arange(0, 73, 3), ['']*24)

			plt.ylabel('Cluster Quantity')

			plt.sca(ax[1])
			plt.xticks(np.arange(0, 1441, 60), ['']*24)

			plt.ylabel('Crime Quantity')
			plt.xlabel('Hours of the day')

			#ax.grid('off', axis='x')
			#ax.grid('on', axis='y')

			plt.savefig('clusters_plot/'+ crime + '.pdf', bbox_inches="tight", format='pdf')


	def clusterize(self):

		result_maxi = [[], [], [], [], [], []]
		result_cluster = [[], [], [], [], [], []]
		result_crime = {}

		for month in range(1, 13):

			print('#' + self.u.MONTHS[month])

			for day in ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday']:

				print('### ' + day)

				day_crimes = self.u.read_data(day)
				month_crimes = day_crimes['2018-' + str(month)]

				threads = []

				for indx, strategy in enumerate([FixedWindowClustering(1), FixedWindowClustering(2), FixedWindowClustering(4), FixedWindowClustering(8), FixedWindowClustering(12),\
					TimeMinutesClustering()]):
				
					thread = CallClusterize(indx, strategy, month_crimes.copy(), Clustering())
					thread.start()

					threads.append(thread)

				for indx, t in enumerate(threads):
					t.join()
					result_maxi[indx].append(t.get_maxi())
					result_cluster[indx].append(t.get_cluster())

				self.count_crime(month_crimes, result_crime)

				# TimeMinutesClustering().clusterize(month_crimes, Clustering())
				# exit()

		self.plot_max_metric(result_maxi)
		self.plot_ecdf(result_maxi)
		self.plot_cluster(result_cluster, result_crime)


######################################################################


def main():

	cc = CompareClustering()
	cc.clusterize()

if __name__ == '__main__':
	warnings.simplefilter("ignore")
	main()