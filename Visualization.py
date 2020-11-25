#####################
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey
import squarify
import seaborn as sns
from pywaffle import Waffle
#####################

df_events = pd.read_csv('Classified_Events.csv',
                      engine='python',
                      header=0,
                      parse_dates=True,
                      infer_datetime_format=True)


#####################
# Overall Treemap
#####################
df = df_events.groupby('Label', as_index=False)['Volume(gal)'].sum()
df['Volume(gal)'] = df['Volume(gal)'].astype(int)
df['label+vol'] = df['Label'] + '\n(' + df['Volume(gal)'].astype(str) + ' gallons)'
labels = df['label+vol']
sizes = df['Volume(gal)']
colors = [plt.cm.Spectral(i/float(len(labels))) for i in range(len(labels))]

plt.figure(figsize=(12, 8), dpi=80)
squarify.plot(sizes=sizes, label=labels, color=colors, alpha=.8)
plt.title('Overall Water Use')
plt.axis('off')
plt.show()

# Indoor Treemap
#####################
df_indoor = df[(df.Label != 'irrigation') & (df.Label != 'hose')]
labels = df_indoor['label+vol']
sizes = df_indoor['Volume(gal)']
colors = [plt.cm.Spectral(i/float(len(labels))) for i in range(len(labels))]

plt.figure(figsize=(12, 8), dpi=80)
squarify.plot(sizes=sizes, label=labels, color=colors, alpha=.8)
plt.title('Indoor Water Use')
plt.axis('off')
plt.show()

#####################
# Overall Sankey Plot
#####################
df_flows = df_events.groupby('Label', as_index=False)['Volume(gal)'].sum().sort_values('Volume(gal)', ascending=False)
df_flows['Volume(gal)'] = -df_flows['Volume(gal)'].astype(int)
df_flows = df_flows.append(-df_flows.sum(numeric_only=True), ignore_index=True)
df_flows.iloc[6, 0] = 'Total Volume'
flows = df_flows['Volume(gal)']
labels = df_flows['Label']
Sankey(head_angle=150, unit='gal', scale=1/24524, offset=0.2,
       flows=flows, labels=labels,
       orientations=[0, -1, 1, -1, -1, -1, 0],
       pathlengths=[0.6, 0.25, 0.25, 0.25, 0.25, 0.25, 0.6]).finish()
plt.title('Flow diagram')
plt.show()

# Indoor Sankey Plot
#####################
df_flows = df_events.groupby('Label', as_index=False)['Volume(gal)'].sum().sort_values('Volume(gal)', ascending=False)
df_indoor_flows = df_flows[(df_flows.Label != 'irrigation') & (df_flows.Label != 'hose')]
df_indoor_flows['Volume(gal)'] = -df_indoor_flows['Volume(gal)'].astype(int)
df_indoor_flows = df_indoor_flows.append(-df_indoor_flows.sum(numeric_only=True), ignore_index=True)
df_indoor_flows.iloc[4, 0] = 'Total Volume'
flows = df_indoor_flows['Volume(gal)']
labels = df_indoor_flows['Label']
Sankey(head_angle=150, unit='gal', scale=1/1839, offset=0.2,
       flows=flows, labels=labels,
       orientations=[0, -1, 1, -1, 0],
       pathlengths=[0.6, 0.25, 0.25, 0.25, 0.6]).finish()
plt.title('Flow diagram')
plt.show()

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
# Grouped Violin Plot
#####################
sns.violinplot(x="Label", y="Duration(min)", data=df_events, palette="Pastel1")
sns.violinplot(x="Label", y="Volume(gal)", data=df_events, palette="Pastel1")
sns.plt.show()













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
