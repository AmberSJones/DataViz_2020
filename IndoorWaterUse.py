# Import Libraries
#####################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#####################

# Import Classified/Labeled Event Data
#####################
df_events = pd.read_csv('Classified_Events.csv', engine='python', header=0, parse_dates=True, infer_datetime_format=True)

# Organize data into groups
#####################
groups = df_events.groupby('Label')
df_sort = df_events.sort_values('Volume(gal)', ascending=False)
# two distinct types of irrigation events
df_sub = df_events[df_events['Volume(gal)'] < 2000]
groups = df_sub.groupby('Label')

#####################
# Volume and Duration ranges
#####################
# Create Fig and gridspec
fig = plt.figure(figsize=(16, 10), dpi=80)
grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)
# Define the axes
ax_main = fig.add_subplot(grid[:-1, :-1])
ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])
# Scatterplot on main ax
for name, group in groups:
    ax_main.plot(group[group['Duration(min)'], 'Volume(gal)'], marker='o', linestyle='', label=name)
    # ax_main.sns.scatterplot(x="Duration(min)", y="Volume(gal)", data=df_events, hue="Label")
ax_main.legend()
# histogram on the bottom
ax_bottom.hist(df_sub['Duration(min)'], 40, histtype='stepfilled', orientation='vertical', color='gray')
ax_bottom.invert_yaxis()
# histogram on the right
ax_right.hist(df_sub['Volume(gal)'], 40, histtype='stepfilled', orientation='horizontal', color='gray')
# Decorations
ax_main.set(title='Duration vs Volume', xlabel='Volume(gal)', ylabel='Duration(min)')
ax_main.title.set_fontsize(20)
for item in ([ax_main.xaxis.label, ax_main.yaxis.label] + ax_main.get_xticklabels() + ax_main.get_yticklabels()):
    item.set_fontsize(14)
xlabels = ax_main.get_xticks().tolist()
ax_main.set_xticklabels(xlabels)
plt.show()


#####################
# Hourly Aggregation
#####################

# Organize Data
#####################
# Add hour of event
df_hourly = df_events[['Volume(gal)', 'Duration(min)', 'Label']].copy()
df_hourly['Hour'] = pd.to_datetime(df_events['StartTime']).dt.hour
# Add total volume and mean/min/max duration
df_grouped = df_hourly.groupby(['Label', 'Hour'], as_index=False).agg({'Volume(gal)': np.sum, 'Duration(min)': ['mean', 'min', 'max']})
df_grouped.columns = ['Label', 'Hour', 'Volume_tot', 'Duration_mean', 'Duration_min', 'Duration_max']

# Pivot table based on hourly volume
df_pvt = pd.pivot_table(df_grouped, values='Volume_tot', index=['Hour'], columns=['Label'], fill_value=0)
df_pvt = df_pvt[['irrigation', 'hose', 'shower', 'toilet', 'clothwasher', 'faucet']]
# Subset indoor use
df_indoor = df_pvt.drop(columns=['irrigation', 'hose'], index=[0, 3, 4, 5, 6])

# Stacked Bar Plot Indoor Uses
#####################
Labels = list(df_indoor.columns)
colors = [plt.cm.Spectral(i/float(len(Labels)-1)) for i in range(len(Labels))]
N = len(df_indoor)
bottom = np.zeros(N)
width = 0.5
for elem, color in zip(Labels, colors):
    plt.bar(df_indoor.index, df_indoor[elem], bottom=bottom, color=color, width=width)
    bottom += df_indoor[elem]
plt.ylabel('Volume (gal)')
plt.xlabel('Hour of Day')
plt.title('Total Hourly Water Use')
plt.xticks(df_indoor.index)
plt.legend(Labels)
plt.show()

# Shower Durations
#####################
df_hourly_shower = df_hourly[(df_events.Label == 'shower')]

sns.stripplot(x="Hour", y="Duration(min)", data=df_hourly_shower, palette="Pastel1")
sns.boxplot(x="Hour", y="Duration(min)", data=df_hourly_shower, palette="Pastel1", width=0)

sns.boxplot(x='site', y='value', hue='label', data=df)
sns.stripplot(x='site', y='value', hue='label', data=df,
              jitter=True, split=True, linewidth=0.5)
plt.legend(loc='upper left')







#Shower Duration
df_pvt_duration = pd.pivot_table(df_grouped, values='Duration_mean', index=['Hour'], columns=['Label'], fill_value=0)
df_pvt_duration = df_pvt_duration[['irrigation', 'hose', 'shower', 'toilet', 'clothwasher', 'faucet']]
df_drop_duration = df_pvt_duration.drop(columns=['irrigation', 'hose'], index=[0, 3, 4, 5, 6])

df_pvt_duration = pd.pivot_table(df_grouped, values='Duration_max', index=['Hour'], columns=['Label'], fill_value=0)
df_pvt_duration = df_pvt_duration[['irrigation', 'hose', 'shower', 'toilet', 'clothwasher', 'faucet']]
df_drop_duration = df_pvt_duration.drop(columns=['irrigation', 'hose'], index=[0, 3, 4, 5, 6])

plt.bar(df_drop_duration.index, df_drop_duration['shower'], width=1, edgecolor='white', color='#2d7f5e')
plt.ylabel('Duration (min)')
plt.xlabel('Hour of Day')
plt.title('Hourly Shower Duration')
plt.xticks(df_drop.index)
plt.show()


# use box plots? lines of ranges? violin plots?

#####################
# Grouped Violin Plot
#####################
df_indoor_events = df_events[(df_events.Label != 'irrigation') & (df_events.Label != 'hose')]

sns.violinplot(x="Label", y="Volume(gal)", data=df_indoor_events, palette="Pastel1")
sns.violinplot(x="Label", y="Duration(min)", data=df_indoor_events, palette="Pastel1")
sns.plt.show()

# Shower Hourly Violin Plot
df_hourly_shower = df_hourly[(df_events.Label == 'shower')]
sns.violinplot(x="Hour", y="Duration(min)", data=df_hourly_shower, palette="Pastel1", cut =0)
sns.boxplot(x="Hour", y="Duration(min)", data=df_hourly_shower, palette="Pastel1")


#####################

# Shower
df_events['Flowrate(gpm)'][df_events['Label'] == 'shower'].mean()
# 1.758 gpm

df_events['Volume(gal)'][df_events['Label'] == 'shower'].sum()/14
# 589 gal
# 42 gal

df_events['Duration(min)'][df_events['Label'] == 'shower'].sum()
# 313 minutes
# 22.4 minutes

(df_events['Duration(min)'][df_events['Label'] == 'shower']*1.5).sum()/14
# 470.7 gal
# 33.6 gal

df_events['Duration(min)'][df_events['Label'] == 'shower'].mean()
#9.5

# new dataframe
df_shower = df_events[df_events['Label'] =='shower']
df_shower['Duration(min)'].sum()
# 313.8 min
df_shower['Volume(gal)'].sum()
# 589.3 gal

# 589.3/313.8 = 1.88 gpm

# 313.8*1.5 = 470.7 gal/14 = 33.6 gal/day
# saving 8.5 gal/day

df_shower['ShortShowerDuration'] = np.where(df_shower['Duration(min)'] >= 10, 10, df_shower['Duration(min)'])
df_shower['ShortShowerDuration'].sum()
# 261.7 min * 1.88 gpm = 492 gallons/14 = 35.1 gal/day
# saving 7 gal/day

#261.7 min * 1.5 gpm = 392 gallons/14 = 28 gal/day
# saving 14 gal/day

plt.hist(df_shower['Duration(min)'])

