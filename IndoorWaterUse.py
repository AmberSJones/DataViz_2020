##########################################
# Irrigation Cost Scenarios
##########################################
# Created by: Amber Jones
# Date: 3 Dec 2020
# This script focuses on indoor water use including several shower scenarios.
# Hose use is disregarded for this analysis.
# The script imports labeled event data, groups by label, subsets into indoor use.
# The duration range of each use is compared in one plot. Another plot examines the hourly volume for each use.
# There is little variation in uses other than shower, so shower durations are examined at each hour.
# Three scenarios are considered for reducing shower volumes:
# Scenario 1: Use ultra low flow showerheads (1.26 gpm).
# Scenario 2: Keep all showers less than 10 minutes.
# Scenario 3: Reduce both flow rate and duration.
# All of these are compared to the current.

# Import Libraries
#####################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#####################

# Import Classified/Labeled Event Data
#####################
df_events = pd.read_csv('Classified_Events.csv', engine='python', header=0, parse_dates=True, infer_datetime_format=True)

#####################
# Variability in Uses
#####################

# Organize Data
#####################
# get indoor data, group and label
df_indoor_events = df_events[(df_events.Label != 'irrigation') & (df_events.Label != 'hose')]
df_indoor_grouped = df_indoor_events.groupby(['Label'], as_index=False).agg({'Volume(gal)': np.sum, 'Duration(min)': ['mean', 'min', 'max']})
df_indoor_grouped.columns = ['Label', 'Volume_tot', 'Duration_mean', 'Duration_min', 'Duration_max']
df_indoor_grouped = df_indoor_grouped.sort_values('Volume_tot')
df_indoor_grouped['Label'] = ['Faucet', 'Clothes\nWasher', 'Shower', 'Toilet']

# Plotting duration ranges for each use
#####################
# this could also be done for volume ranges.
# this is similar to violin or boxplot but removes the frequency element for simplicity.
fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(1, 1, 1)
colors = ['#004c6d', '#6996b3', '#F5793A', '#A95AA1']
ax.hlines(y=df_indoor_grouped['Label'], xmin=df_indoor_grouped['Duration_min'], xmax=df_indoor_grouped['Duration_max'], color=colors[0], alpha=0.7)
ax.scatter(df_indoor_grouped['Duration_min'], df_indoor_grouped['Label'], color=colors[3], alpha=0.8, label='Minimum')
ax.scatter(df_indoor_grouped['Duration_mean'], df_indoor_grouped['Label'], marker='P', s=70, color=colors[1], label='Average')
ax.scatter(df_indoor_grouped['Duration_max'], df_indoor_grouped['Label'], color=colors[2], alpha=0.8, label='Maximum')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.legend(frameon=False)
ax.set_facecolor(plt.cm.tab20c(19))
ax.tick_params(top=False, bottom=True, left=False, right=False, labelleft=True, labelbottom=True)
plt.xlabel('Duration (min)')
plt.title('Variation in Duration for Indoor Uses')

# to save
plt.savefig('indoor_durations.png', bbox_inches='tight')

#####################
# Hourly Aggregation
#####################

# Organize Data
#####################
# Add hour of event
df_hourly = df_events[['Volume(gal)', 'Duration(min)', 'Label']].copy()
df_hourly['Hour'] = pd.to_datetime(df_events['StartTime']).dt.hour
# Group data into total volume and mean/min/max duration for each hour/label
df_grouped = df_hourly.groupby(['Label', 'Hour'], as_index=False).agg({'Volume(gal)': np.sum, 'Duration(min)': ['mean','min','max']})
df_grouped.columns = ['Label', 'Hour', 'Volume_tot', 'Duration_mean', 'Duration_min', 'Duration_max']
# Pivot table based on hourly volume
df_hr_vol = pd.pivot_table(df_grouped, values='Volume_tot', index=['Hour'], columns=['Label'], fill_value=0)
df_hr_vol = df_hr_vol[['irrigation', 'hose', 'shower', 'toilet', 'clothwasher', 'faucet']]
# Subset indoor use
df_hr_vol = df_hr_vol.drop(columns=['irrigation', 'hose'], index=[0, 3, 4, 5, 6])
df_hr_vol = df_hr_vol.rename(columns={'shower': 'Shower', 'toilet':'Toilet', 'clothwasher': 'Clothes Washer', 'faucet': 'Faucet'})

# Plot Hourly Indoor Uses
#####################
fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(1, 1, 1)
labels = list(df_hr_vol.columns)
colors = ['#004c6d', '#6996b3', '#F5793A', '#A95AA1']
N = len(df_hr_vol)
bottom = np.zeros(N)
width = 0.95
for elem, color in zip(labels, colors):
    ax.bar(df_hr_vol.index, df_hr_vol[elem], bottom=bottom, color=color, width=width, edgecolor='w', label=elem)
    bottom += df_hr_vol[elem]
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], frameon=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.tick_params(top=False, bottom=False, left=True, right=False, labelleft=True, labelbottom=True)
plt.ylabel('Volume (gal)')
plt.xticks(df_hr_vol.index)
plt.xlabel('Hour of Day')
plt.title('Indoor Hourly Water Use')
plt.show()

# to save
plt.savefig('indoor_volumes.png', bbox_inches='tight')

# Plot Hourly Shower Durations
#####################
# this could also be done for volume ranges.
# this is similar to violin or boxplot but removes the frequency element for simplicity.
df_hr_shower = df_grouped[df_grouped['Label'] == 'shower']

fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(1, 1, 1)
colors = ['#004c6d', '#6996b3', '#F5793A', '#A95AA1']
ax.vlines(x=df_hr_shower['Hour'], ymin=df_hr_shower['Duration_min'], ymax=df_hr_shower['Duration_max'], color=colors[0], alpha=0.7)
ax.scatter(df_hr_shower['Hour'], df_hr_shower['Duration_max'], color=colors[2], alpha=0.8, label='Maximum')
ax.scatter(df_hr_shower['Hour'], df_hr_shower['Duration_mean'],  marker='P', s=70, color=colors[1], label='Average')
ax.scatter(df_hr_shower['Hour'], df_hr_shower['Duration_min'], color=colors[3], alpha=0.8, label='Minimum')
ax.scatter(df_hr_shower['Hour'], df_hr_shower['Duration_mean'],  marker='P', s=70, color=colors[1])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.legend(frameon=False)
ax.set_facecolor(plt.cm.tab20c(19))
ax.tick_params(top=False, bottom=False, left=True, right=False, labelleft=True, labelbottom=True)
plt.ylabel('Duration (min)')
plt.xticks(df_hr_vol.index)
plt.xlabel('Hour of Day')
plt.title('Hourly Range of Shower Durations')

# to save
plt.savefig('shower_durations.png', bbox_inches='tight')

#####################
# Shower Scenarios
#####################
# Subset data for showers only
df_shower = df_events[df_events['Label'] =='shower']

# Current Scenario
#####################
current_daily_vol = df_shower['Volume(gal)'].sum()/14
# 42 gal/day
current_daily_dur = df_shower['Duration(min)'].sum()/14
# 22.4 minutes/day
current_gpm = df_shower['Volume(gal)'].sum()/df_shower['Duration(min)'].sum()
# 1.88 gpm

# Reduce Shower Duration
#####################
max_duration = 10  # set maximum shower duration (min)
df_shower['ShortShowerDuration'] = np.where(df_shower['Duration(min)'] >= max_duration, max_duration, df_shower['Duration(min)'])
short_dur = df_shower['ShortShowerDuration'].sum()/14
# 18.9 minutes/day
short_dur_vol = short_dur * current_gpm
# 35.1 gal/day
short_dur_save = current_daily_vol - short_dur_vol
# saving 7 gal/day

# Ultra Low Flow Shower Head
#####################
low_flow = 1.26  # set low flow shower rate (gpm)
low_flow_vol = current_daily_dur * low_flow
# 33.6 gal/day
low_flow_save = current_daily_vol - low_flow_vol
# saving 8.5 gal/day

# Reduce Both Flow and Duration
#####################
both_vol = short_dur * low_flow
# 28 gal/day
both_save = current_daily_vol - both_vol
# saving 14 gal/day

# Plotting
#####################
# this is analogous to the bathtub illustration
Scenario = ['Current Showering', 'Shorter Showers', 'Ultra Low Flow', 'Shorter and Low Flow']
Volume = [current_daily_vol, short_dur_vol, low_flow_vol, both_vol]

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(1, 1, 1)
colors = ['#004c6d', '#3d708f', '#75a1be', '#94bed9']
width = 0.95
# Bars
p1 = plt.bar(Scenario, Volume, bottom=0, color=colors, width=width, edgecolor='w')
# Volume annotations
for i in range(len(Volume)):
    plt.annotate('{:.0f}'.format(Volume[i]) + ' gal', xy=(i, Volume[i] + 1), rotation=0, color='k', ha='center', va='center', alpha=0.7, fontsize=9)
# Extras
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.tick_params(top=False, bottom=False, left=True, right=False, labelleft=True, labelbottom=True)
plt.ylabel('Daily Volume (gal)')
plt.title('Shower Scenario Daily Volumes')
plt.show()

# to save
plt.savefig('shower_scenarios.png', bbox_inches='tight')

##########################################

