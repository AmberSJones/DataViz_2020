# Import Libraries
#####################
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey
#####################

# Import Classified/Labeled Event Data
#####################
df_events = pd.read_csv('Classified_Events.csv', engine='python', header=0, parse_dates=True, infer_datetime_format=True)

# Organize data as in/out flows
#####################
df_flows = df_events.groupby('Label', as_index=False)['Volume(gal)'].sum().sort_values('Volume(gal)', ascending=False)
# Daily averages
df_flows['daily'] = (df_flows['Volume(gal)']/14).astype(int)
# Class indoor/outdoor
df_in_out_flows = pd.DataFrame({'Label': ('Indoor', 'Outdoor', 'Average\nDaily\nWater Use'),
                                'Daily Volume': (-df_flows['daily'][(df_flows.Label != 'irrigation') & (df_flows.Label != 'hose')].sum(),
                                                 -df_flows['daily'][(df_flows.Label == 'irrigation') | (df_flows.Label == 'hose')].sum(),
                                                 df_flows['daily'].sum()
                                                 )})

# Daily Flow Plots
#####################
flows = df_in_out_flows['Daily Volume']
labels = df_in_out_flows['Label']
fig = plt.figure(figsize=(10, 6), dpi=80)
ax = fig.add_subplot(1, 1, 1)
ax.axis('off')
in_out_flows = Sankey(ax=ax, head_angle=120, unit='gal', scale=1/1751, offset=0.3, margin=0.2, shoulder=0.1,)
in_out_flows.add(flows=flows, labels=labels, orientations=[-1, 0, 0], pathlengths=[0.25, 0.6, 0.6], fill=False, trunklength=2)
diagrams = in_out_flows.finish()
plt.show()

# Indoor Use
#####################
df_indoor_flows = df_flows[(df_flows.Label != 'irrigation') & (df_flows.Label != 'hose')]
df_indoor_flows['daily'] = -df_indoor_flows['daily'].astype(int)
df_indoor_flows = df_indoor_flows.append(-df_indoor_flows.sum(numeric_only=True), ignore_index=True)
df_indoor_flows.iloc[4, 0] = 'Average\nDaily\nIndoor\nUse'

# Daily Indoor Plot
#####################
flows = df_indoor_flows['daily']
labels = df_indoor_flows['Label']
fig = plt.figure(figsize=(10, 6), dpi=80)
ax = fig.add_subplot(1, 1, 1)
ax.axis('off')
indoor_flows = Sankey(ax=ax, head_angle=120, unit='gal', scale=1/130, offset=0.25, margin=0.2, shoulder=0.1,)
indoor_flows.add(flows=flows, labels=labels, orientations=[0, -1, 1, -1, 0], pathlengths=[0.5, 0.25, 0.25, 0.25, 0.25], fill=False, trunklength=2)
diagrams = indoor_flows.finish()
diagrams[0].texts[-1].set_color('r')
plt.show()
