
# Import Libraries
#####################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey
import squarify
import seaborn as sns
from pywaffle import Waffle
#####################

# Import Classified/Labeled Event Data
#####################
df_events = pd.read_csv('Classified_Events.csv', engine='python', header=0, parse_dates=True, infer_datetime_format=True)

#####################
# Treemaps
#####################
# Group data by label
df = df_events.groupby('Label', as_index=False)['Volume(gal)'].sum()
df['Volume(gal)'] = df['Volume(gal)'].astype(int)
# Reorder rows
df = df.reindex([3, 2, 5, 4, 0, 1])
# Concatonate label + volume for plotting
df['label+vol'] = df['Label'] + '\n(' + df['Volume(gal)'].astype(str) + ' gallons)'
# Add columns for daily averages
df['daily'] = (df['Volume(gal)']/14).astype(int)
df['daily_label+vol'] = df['Label'] + '\n(' + df['daily'].astype(str) + ' gallons)'
# Subset for indoor water use
df_indoor = df[(df.Label != 'irrigation') & (df.Label != 'hose')]

# Overall Treemap
#####################
labels = df['label+vol']
sizes = df['Volume(gal)']
colors = [plt.cm.Spectral(i/float(len(labels))) for i in range(len(labels))]

plt.figure(figsize=(12, 8), dpi=80)
squarify.plot(sizes=sizes, label=labels, color=colors, alpha=.8)
plt.title('Overall Water Use')
plt.axis('off')
plt.show()

# Daily Average Treemap
#####################
labels = df['daily_label+vol']
sizes = df['daily']
colors = [plt.cm.Spectral(i/float(len(labels))) for i in range(len(labels))]

plt.figure(figsize=(12, 8), dpi=80)
squarify.plot(sizes=sizes, label=labels, color=colors, alpha=.8)
plt.title('Average Daily Water Use')
plt.axis('off')
plt.show()

# Indoor Treemap
#####################
labels = df_indoor['label+vol']
sizes = df_indoor['Volume(gal)']
colors = [plt.cm.Spectral(i/float(len(labels))) for i in range(len(labels))]

plt.figure(figsize=(12, 8), dpi=80)
squarify.plot(sizes=sizes, label=labels, color=colors, alpha=.8)
plt.title('Indoor Water Use')
plt.axis('off')
plt.show()

# Daily Average Indoor Treemap
#####################
labels = df_indoor['daily_label+vol']
sizes = df_indoor['daily']
colors = [plt.cm.Spectral(i/float(len(labels))) for i in range(len(labels))]

plt.figure(figsize=(12, 8), dpi=80)
squarify.plot(sizes=sizes, label=labels, color=colors, alpha=.8)
plt.title('Average Daily Indoor Water Use')
plt.axis('off')
plt.show()


#####################
# Sankey Plots
#####################
# Organize data as in/out flows
df_flows = df_events.groupby('Label', as_index=False)['Volume(gal)'].sum().sort_values('Volume(gal)', ascending=False)
df_flows['Volume(gal)'] = -df_flows['Volume(gal)'].astype(int)
df_flows = df_flows.append(-df_flows.sum(numeric_only=True), ignore_index=True)
df_flows.iloc[6, 0] = 'Total Volume'
# Daily averages
df_flows['daily'] = (df_flows['Volume(gal)']/14).astype(int)

# Overall Plots
#####################
flows = df_flows['Volume(gal)']
labels = df_flows['Label']
Sankey(head_angle=150, unit='gal', scale=1/24524, offset=0.2,
       flows=flows, labels=labels,
       orientations=[0, -1, 1, -1, -1, -1, 0],
       pathlengths=[0.6, 0.25, 0.25, 0.25, 0.25, 0.25, 0.6]).finish()
plt.title('Flows')
plt.show()

# Daily Sankey Plots
#####################
flows = df_flows['daily']
labels = df_flows['Label']
Sankey(head_angle=150, unit='gal', scale=1/1751, offset=0.2,
       flows=flows, labels=labels,
       orientations=[0, -1, 1, -1, -1, -1, 0],
       pathlengths=[0.6, 0.25, 0.25, 0.25, 0.25, 0.25, 0.6]).finish()
plt.title('Daily flows')
plt.show()

# Indoor Sankey Plots
#####################
df_flows = df_events.groupby('Label', as_index=False)['Volume(gal)'].sum().sort_values('Volume(gal)', ascending=False)
df_indoor_flows = df_flows[(df_flows.Label != 'irrigation') & (df_flows.Label != 'hose')]
df_indoor_flows['Volume(gal)'] = -df_indoor_flows['Volume(gal)'].astype(int)
df_indoor_flows = df_indoor_flows.append(-df_indoor_flows.sum(numeric_only=True), ignore_index=True)
df_indoor_flows.iloc[4, 0] = 'Total \n Volume'
# Daily averages
df_indoor_flows['daily'] = (df_indoor_flows['Volume(gal)']/14).astype(int)

# Indoor Plot
#####################
flows = df_indoor_flows['Volume(gal)']
labels = df_indoor_flows['Label']
Sankey(head_angle=150, unit='gal', scale=1/1839, offset=0.2,
       flows=flows, labels=labels,
       orientations=[0, -1, 1, -1, 0],
       pathlengths=[0.6, 0.25, 0.25, 0.25, 0.6]).finish()
plt.title('Flow diagram')
plt.show()

# Daily Indoor Plot
#####################
flows = df_indoor_flows['daily']
labels = df_indoor_flows['Label']
Sankey(head_angle=150, unit='gal', scale=1/131, offset=0.2,
       flows=flows, labels=labels,
       orientations=[0, -1, 1, -1, 0],
       pathlengths=[0.6, 0.25, 0.25, 0.25, 0.6],
       #color='b'
       ).finish()
plt.title('Daily flows')
plt.show()

fig = plt.figure(figsize=(16, 6), dpi=80)
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[],
                     title='Daily flows')
sankey = Sankey(ax=ax, scale=1/131, head_angle=120, unit='gal', shoulder=0.075, margin=0.2, offset=0.15)
sankey.add(flows=flows, labels=labels, orientations=[0, -1, 1, -1, 0],
           pathlengths=[0.25, 0.25, 0.25, 0.25, 1],
           fill = False)  # Arguments to matplotlib.patches.PathPatch
diagrams = sankey.finish()
diagrams[0].texts[-1].set_color('r')
diagrams[0].text.set_fontweight('bold')






#####################
# Marginal Histogram
#####################
# Create Fig and gridspec
fig = plt.figure(figsize=(16, 10), dpi=80)
grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)

# Define the axes
ax_main = fig.add_subplot(grid[:-1, :-1])
ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])

# Scatterplot on main ax
groups = df_events.groupby('Label')
for name, group in groups:
    ax_main.plot(group['Volume(gal)'], group['Duration(min)'], marker='o', linestyle='', label=name)
ax_main.legend()
#ax_main.scatter('Duration(min)', 'Volume(gal)', c=df_events.Label.astype('category').cat.codes, alpha=.9, data=df_events, cmap="tab10", linewidths=.5)
#ax_main.sns.scatterplot(x="Duration(min)", y="Volume(gal)", data=df_events, hue="Label")

# histogram on the bottom
ax_bottom.hist(df_events['Duration(min)'], 40, histtype='stepfilled', orientation='vertical', color='gray')
ax_bottom.invert_yaxis()

# histogram on the right
ax_right.hist(df_events['Volume(gal)'], 40, histtype='stepfilled', orientation='horizontal', color='gray')

# Decorations
ax_main.set(title='Duration vs Volume', xlabel='Volume(gal)', ylabel='Duration(min)')
ax_main.title.set_fontsize(20)
for item in ([ax_main.xaxis.label, ax_main.yaxis.label] + ax_main.get_xticklabels() + ax_main.get_yticklabels()):
    item.set_fontsize(14)

xlabels = ax_main.get_xticks().tolist()
ax_main.set_xticklabels(xlabels)
plt.legend()
plt.show()

# Refined Marginal Histogram
#####################
df_sort = df_events.sort_values('Volume(gal)', ascending=False)
# two distinct types of irrigation events
df_sub = df_events[df_events['Volume(gal)'] < 2000]

# Create Fig and gridspec
fig = plt.figure(figsize=(16, 10), dpi=80)
grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)

# Define the axes
ax_main = fig.add_subplot(grid[:-1, :-1])
ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])

# Scatterplot on main ax
groups = df_sub.groupby('Label')
for name, group in groups:
    ax_main.plot(group['Volume(gal)'], group['Duration(min)'], marker='o', linestyle='', label=name)
ax_main.legend()
#ax_main.scatter('Duration(min)', 'Volume(gal)', c=df_events.Label.astype('category').cat.codes, alpha=.9, data=df_events, cmap="tab10", linewidths=.5)
#ax_main.sns.scatterplot(x="Duration(min)", y="Volume(gal)", data=df_events, hue="Label")

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

df_hourly = df_events[['Volume(gal)', 'Duration(min)', 'Label']].copy()
df_hourly['Hour'] = pd.to_datetime(df_events['StartTime']).dt.hour
df_grouped = df_hourly.groupby(['Label', 'Hour'], as_index=False)['Volume(gal)'].sum()
df_grouped = df_hourly.groupby(['Label', 'Hour'], as_index=False).aggregate({'Volume(gal)': np.sum, 'Duration(min)': np.average})
df_grouped = df_hourly.groupby(['Label', 'Hour'], as_index=False).agg({'Volume(gal)': np.sum, 'Duration(min)': ['mean', 'min', 'max']})
df_grouped.columns = ['Label', 'Hour', 'Volume_tot', 'Duration_mean', 'Duration_min', 'Duration_max']

# Volume
df_pvt = pd.pivot_table(df_grouped, values='Volume_tot', index=['Hour'], columns=['Label'], fill_value=0)
df_pvt = df_pvt[['irrigation', 'hose', 'shower', 'toilet', 'clothwasher', 'faucet']]

Labels = list(df_pvt.columns)
colors = [plt.cm.Spectral(i/float(len(Labels)-1)) for i in range(len(Labels))]
N = len(df_pvt)
bottom = np.zeros(N)
width = 0.5
for elem, color in zip(Labels, colors):
    plt.bar(df_pvt.index, df_pvt[elem], bottom=bottom, color=color, width=width)
    bottom += df_pvt[elem]

plt.ylabel('Volume (gal)')
plt.xlabel('Hour of Day')
plt.title('Total Hourly Water Use')
plt.xticks(df_pvt.index)
plt.legend(Labels)
plt.show()

# Without Irrigation
df_drop = df_pvt.drop(columns=['irrigation', 'hose'], index=[0, 3, 4, 5, 6])
Labels = list(df_drop.columns)
colors = [plt.cm.Spectral(i/float(len(Labels)-1)) for i in range(len(Labels))]
N = len(df_drop)
bottom = np.zeros(N)
width = 0.5
for elem, color in zip(Labels, colors):
    plt.bar(df_drop.index, df_drop[elem], bottom=bottom, color=color, width=width)
    bottom += df_drop[elem]

plt.ylabel('Volume (gal)')
plt.xlabel('Hour of Day')
plt.title('Total Hourly Water Use')
plt.xticks(df_drop.index)
plt.legend(Labels)
plt.show()

#Side by Side
bottom = np.zeros(N)
wide = 0
for elem, color in zip(Labels, colors):
    plt.bar(df_drop.index + wide, df_drop[elem], bottom=bottom, color=color, width=0.15)
    wide += 0.15

plt.ylabel('Volume (gal)')
plt.xlabel('Hour of Day')
plt.title('Total Hourly Water Use')
plt.xticks(df_drop.index)
plt.legend(Labels)
plt.show()

#Only Shower
plt.bar(df_drop.index, df_drop['shower'], width=1, edgecolor='white', color='#2d7f5e')
plt.ylabel('Volume (gal)')
plt.xlabel('Hour of Day')
plt.title('Hourly Shower Volume')
plt.xticks(df_drop.index)
plt.show()

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
# $23.25 for 10,000 gallons of water.
# 75 cents per 1,000 gallons if residents use anywhere from 10,001 to 50,000 gallons.
# 1.50 per 1,000 gallons for all usage over 50,000 gallons

df['monthly'] = df['daily'] * 30


# 27154 gallons/acre 0.28*2/3 acres = 5000 gal is 1 inch of water
# 5000 gallons/week = 20000 gallons/month


Labels = ['Indoor', 'Outdoor']
Values1 = [(5160),10000, ]
Values2 = [(48600), 40000,]
Values3 =
colors = [plt.cm.Spectral(i/float(len(Labels)-1)) for i in range(len(Labels))]
N = len(df['Label'])
width = 0.5
bottom = 0
for elem, color in zip(Labels, colors):
    plt.bar(0, Values, bottom=bottom, color=color, width=width)
    bottom += Values

plt.ylabel('Volume (gal)')
plt.title('Total Monthly Water Use')
plt.legend(Labels)
plt.show()







#####################


# Overall Waffle Plot
# Prepare Data
df = df_events.groupby('Label', as_index=False)['Volume(gal)'].sum()
df = df[(df.Label != 'irrigation') & (df.Label != 'hose')]
n_categories = df.shape[0]
colors = [plt.cm.inferno_r(i/float(n_categories)) for i in range(n_categories)]

# Draw Plot and Decorate
fig = plt.figure(
    FigureClass=Waffle,
    plots={
        '111': {
            'values': df['Volume(gal)'],
            'labels': ["{0} ({1})".format(n[0], n[1]) for n in df[['Label', 'Volume(gal)']].itertuples()],
            'legend': {'loc': 'upper left', 'bbox_to_anchor': (1.05, 1), 'fontsize': 12},
            #'title': {'label': '# Vehicles by Class', 'loc': 'center', 'fontsize':18}
        },
    },
    rows=4,
    colors=colors,
    figsize=(16, 9)
)
