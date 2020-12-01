# Import Libraries
#####################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#####################

# Import Classified/Labeled Event Data
#####################
df_events = pd.read_csv('Classified_Events.csv', engine='python', header=0, parse_dates=True, infer_datetime_format=True)

# Organize data
#####################
# Group data by label
df = df_events.groupby('Label', as_index=False)['Volume(gal)'].sum()
df['Volume(gal)'] = df['Volume(gal)'].astype(int)
# Reorder rows
df = df.reindex([3, 2, 5, 4, 0, 1])
# Add columns for daily and monthly
df['daily'] = (df['Volume(gal)']/14).astype(int)
df['monthly'] = df['daily'] * 30

# Irrigation scenarios
#####################
# Reduce irrigation volume
# Recommended irrigation for turfgrass: 1 inch water = 27154 gallons/acre (see https://www.lowes.com/n/how-to/watering-tips)
lot_fraction = 2/3 #removing house and paved
one_inch = 27154 #gallons/acre per week
reduce_watering = one_inch * lot_size * lot_fraction * 30/7
# 27154 * 0.28 * 2/3 * (30 days/month) / (7 days/week) = 21723 gallons/month
current = df['monthly'][df['Label'] == 'irrigation'].sum() #monthly water use undercurrent scenario
current_inches = current*7/30*1/lot_fraction*1/lot_size*1/one_inch #weekly watering depth under current scenario
# 47340 * 7/30 * 3/2 * 1/0.28 *1/27154 = 2.18 inches/week

# Reduce irrigated area
lot_reduce = 0.5
reduce_lawn = current * lot_reduce
reduce_both = one_inch * lot_size * lot_fraction * lot_reduce * 30/7
# 27154 gallons/acre 0.28*2/3*0.5 acres = 2534.5 gal is 1 inch of water
# 27154 * 0.28 * 2/3 * 30/7 * 0.5 = 10862 gallons/month

indoor = df['monthly'][df['Label'] != 'irrigation'].sum()

# Define scenarios
Labels = ['Indoor', 'Sprinklers']
Scenario = ['Current Watering', 'Reduce Lawn', 'Reduce Watering', 'Reduce Both']
Indoor = [indoor, indoor, indoor, indoor]
Sprinklers = [current, reduce_lawn, reduce_watering, reduce_both]

# Pricing
#####################
# Providence City water rate tiers:
# $23.25 for 10,000 gallons of water.
# $0.75 per 1,000 gallons from 10,001 to 50,000 gallons.
# $1.50 per 1,000 gallons over 50,000 gallons.
tier1 = 10000
tier2 = 50000
flat_rate = 23.25
rate_tier1 = 0.75
rate_tier2 = 1.5

# Create dataframe with pricing information
pricing = pd.DataFrame({'Scenario': Scenario, 'Indoor': Indoor, 'Sprinklers': Sprinklers})
pricing['Total'] = pricing['Indoor'] + pricing['Sprinklers']
pricing['FirstTier'] = np.where(pricing['Total'] >= tier2, tier2-tier1, pricing['Total']-tier1)
pricing['SecondTier'] = np.where(pricing['Total'] >= tier2, pricing['Total']-tier2, 0)
pricing['FirstTierCost'] = pricing['FirstTier']*rate_tier1/1000
pricing['SecondTierCost'] = pricing['SecondTier']*rate_tier2/1000
pricing['TotalCost'] = pricing['FirstTierCost'] + pricing['SecondTierCost'] + flat_rate

# Plotting
#####################

colors = [plt.cm.Spectral(i/float(len(Labels)-1)) for i in range(len(Labels))]
width = 0.75

# Bars
p1 = plt.bar(Scenario, Indoor, bottom=0, color=colors[0], width=width)
p2 = plt.bar(Scenario, Sprinklers, bottom=Indoor, color=colors[1], width=width)

# Lines and labels
p3 = plt.axhline(y=tier1, linewidth=1.5, linestyle='--', color='k')
# plt.text(x=4.5, y=10500, s='Flat Rate Tier', fontweight='bold', color='k')
p4 = plt.axhline(y=tier2, linewidth=1.5, linestyle='--', color='k')
# plt.text(x=4.5, y=50500, s='Irrigation Tier', fontweight='bold', color='gray')

# Cost and volume annotations
for i, rows in pricing.iterrows():
    plt.annotate('{:,}'.format(rows['Total']) + ' gal', xy=(i, rows['Total']+1000), rotation=0, color='k', ha='center', va='center', alpha=0.7, fontsize=9)
    plt.annotate('${:,.2f}'.format(rows['TotalCost']), xy=(i, 12000), rotation=0, color='k', ha='center', va='center', fontsize=10, fontweight='bold', bbox=dict(boxstyle='square', fc='white'))

# Brackets for Rate Ranges
plt.annotate('Flat Rate\n$23.25', xy=(3.35, 5000), xytext=(3.425, 5000), annotation_clip=False, rotation=0,
            fontsize=9, ha='left', va='center',
            bbox=dict(boxstyle='square', fc='white'),
            arrowprops=dict(arrowstyle='-[, widthB=2.4, lengthB=0.9', lw=1))
plt.annotate(' $0.75/\n1000gal', xy=(3.35, 30000), xytext=(3.425, 30000), annotation_clip=False, rotation=0,
            fontsize=9, ha='left', va='center',
            bbox=dict(boxstyle='square', fc='white'),
            arrowprops=dict(arrowstyle='-[, widthB=10.35, lengthB=0.9', lw=1))
plt.annotate(' $1.50/\n1000gal', xy=(3.35, 52600), xytext=(3.425, 52600), annotation_clip=False, rotation=0,
            fontsize=9, ha='left', va='center',
            bbox=dict(boxstyle='square', fc='white'),
            arrowprops=dict(arrowstyle='-[, widthB=1.15, lengthB=0.9', lw=1))

# Extras
plt.ylabel('Volume (gal)')
plt.title('Summer Season Monthly Water Use and Cost')
plt.legend((p1, p2), Labels, loc='upper center', ncol=2)
plt.show()

