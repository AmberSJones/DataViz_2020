# Visualization of Household Water Use Data

By monitoring household water use with smart sensors and associated algorithms, homeowners, residents, and utilities can better understand not only the quantity of water used, but how and when the water is used. Increased information can help target uses and times for water conservation and cost savings. This repository contains code and images for visualizing water use for a single household. The overall water use is considered along with separate focus on outdoor and indoor uses. Several scenarios are presented as potential opportunities for water conservation. While these graphics correspond to a single, specific household, the code should be generalizable to any home with a smart water sensor and categorized use data.

This repository was created December 2020 as a submission for a [data visualization challenge](https://github.com/UCHIC/CIWS-VisChallenge/) sponsored by the Cyberinfrastructure for Intelligent Water Systems Research Group.

## Overview and Input Data
The code consists of three main scripts: FlowPlots.py, IrrigationCostScenarios.py, and IndoorWaterUse.py, each of which are described below. FlowPlots provides an overview of water use, IrrigationCostScenarios focuses on outdoor water use and cost, and IndoorWaterUse examines categories and timing of indoor water use.

Input data for all of these scripts is a comma separated values files of classified and labeled water use events. Each row corresponds to a water use event. Required columns are StartTime, Label, Volume, and Duration. The input file supplied for the challenge contained data for a single household in Providence City, UT for 2 weeks during the summer.

## Overall Water Use
The flow plots script imports data, aggregates all events by category, and determines daily averages. Data are further organized as in/out flows for plotting. Two plots are constructed and put together for a single visualization: one to illustrate the split between indoor and outdoor use, and another to show the proportions of indoor use. A separate plot is used for indoor because, for the timeframe for this example, outdoor water use overshadowed indoor use. All numbers in the plots are daily averages.

![flowplots](/Images/flowplots.png)

As shown here, a vast majority of water used by the household in this period goes to outdoor uses - generally irrigation of turfgrass by automated sprinkler system along with a relatively small amount of hose use. Indoors, toilet flushes and showers consume the most water, with some faucet and clothes washer. Note that 'faucet' includes sink use and automatic dishwashing.

## Outdoor Water Use
Because of the dominance of outdoor water use, possibilities for reduction are explored and compared to the current baseline. The IrrigationCostScenarios script determines monthly watering volume and cost for three scenarios (irrigation = sprinklers = watering.) Note that this analysis is really only applicable to the summer irrigation season.

###Scenarios
Three scenarios are considered:
1. Reducing the rate of irrigation: The need for watering depends on weather as well as soil type, so it can be difficult to broadly prescibe rates and a schedule. To conserve water, one [source](https://www.lowes.com/n/how-to/watering-tips) recommends 1 inch of water per week. The script determines the monthly water use for 1 inch/week as well as the depth/week for the current baseline using the lot size and assuming that 2/3 of the lot area is watered (because of the house and paved surfaces). 
2. Reducing the irrigated area: This scenario explores water use if the irrigated area were reduced by swapping turfgrass for xeriscaped plants and other elements that do not require watering. For this case, the irrigated area was reduced by half
3. Both: A final scenario considers reducing both the rate of irrigation as well as the irrigated area.

### Pricing
The City of Providence, UT uses a tiered pricing structure to charge for water use with rates as follows: 
- $23.25 for 10,000 gallons of water (flat rate).
- $0.75 per 1,000 gallons from 10,001 to 50,000 gallons.
- $1.50 per 1,000 gallons over 50,000 gallons.
Using these rates, the pricing for each scenario was determined.

### Plots and Illustrations
The water use and pricing information for each scenario is visualized in a bar chart. Each bar represents total monthly household water use for each scenario. Indoor use is constant while outdoor use varies. Dashed lines indicate the pricing tiers, and total monthly cost is shown for each scenario. Note that this is only applicable to summer irrigation season.

![outdoorscenarios](/Images/outdoor_scenarios.png)

These plots indicate the savings that can be realized with increased water conservation.

 The below visualizes the impact of each of the scenarios. In the upper figure, the current baseline weekly watering depth is shown next to the recommended depth overlaid on an image of grass to illustrate the opportunity for reduction. In the lower figure, the relative portions of irrigated area are shown as a full lawn and a partially xeriscaped lawn to show reduced irrigated area. 
 
 ![outdoorillustration](/Images/outdoor_illustration.png)
 
 ## Indoor Water Use
For this house and time period, outdoor water use dominates and presents the greatest opportunity for conservation; however, outdoor use will not be significant for portions of the year, may not be as relevant for some households, and some consumers may want to go beyond outdoor conservation. The IndoorWaterUse script explores various indoor uses including timing and variability and potential paths for conservation. The below plot illustrates the ranges of event duration for each category. Because volume is directly related to duration, the range of durations also indicates the range of volumes. 

![indoordurations](/Images/indoor_durations.png)

For faucet, clothes washer, and toilet, all events are of similar duration. On the other hand, the duration of shower events vary widely. Compared to the other uses, showers offer the greatest opportunity for conservation, so three scenarios are considered for reducing shower water use. (Note that toilet flushes use more overall water than showers, so opportunity for reduction could occur if residents are willing to flush less, however, it is questionable whether messaging on toilet flush conservation will be welcomed (e.g., "if it's yellow, let it mellow").

### Timing
To visualize when water is being used in different areas of the house, the data are split into hours when the event occurred and plotted.

![indoorvolumes](/Images/indoor_volumes.png)

This visualization indicates that ... As mentioned, there is greatest opportunity for reduction with showers, so  information on the typical shower duration for each hour can help target specific times or individuals. The below plot illustrates the range of shower duration for each hour of the day.

![showerdurations](/Images/shower_durations.png)

This plot shows that xx and xx are hours when relatively lengthy showers are occurring, so the household can focus its reduction efforts on showers that occur during those times of the day. That is the focus of one of the scenarios.

### Shower Scenarios
As shown by the previous visualizations, shower events are the best opportunity for reducing indoor water use. Three scenarios are considered:
1. Reduce Shower Duration: A maximum shower duration is set. For this case, 10 minutes was selected as a maximum shower length. Any showers over 10 minutes are set to 10 minutes and the total daily volume determined.
2. Ultra Low Flow: There is a wide range of low flow fixtures available to consumers. This scenario sets a low flow rate and determines total daily volume used for the shower events in the dataset. Low flow fixtures can impose conservation when shorter showers are not welcomed by household members. (Note that for this household, the average flow rate was 1.88 gpm, which is considered within the range of low flow. Recent technology reports an effective flow rate of 1.26.)
3. Both: A final scenario implements both shorter showers and a reduced shower flow rate. 

### Plots and Illustrations
The water used by each shower scenario is visualized in a bar chart. Each bar represents total daily water use for each scenario. The same numbers are presented in an illustration with daily volumes shown as water depth scaled to a bathtub with the amount of water conserved labeled.

![showerscenarios](/Images/shower_scenarios.png)

![tub illustration](/Images/Tub_illustration.png)












 

