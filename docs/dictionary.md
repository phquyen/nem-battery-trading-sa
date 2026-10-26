### Data Dictionary 

This project uses the AEMO Aggregated Price and Demand data for South Australia region. 

#### Dataset overview 
Feature's description: 

1. `REGION`: region code 
2. `SETTLEMENTDATE`: end of the 5-minute interval, in market time (UTC+10, no daylight saving) - 2026/09/01 00:05:00 means the interval 00:00-00:05
3. `TOTALDEMAND`: operational demand: electricity drawn from the SA grid. Excludes rooftop solar
4. `RRP`: region reference price: the wholesale spore price, excl GTS (S/MWh)
5. `PERIODTYPE`: type of interval

**Important note**
1. Timestamps are period-ending. 12:05 describes what happened between 12:00 and 12:05, not after 12:05.
2. Timestamps are in market time, not Adelaide time. Market time is UTC+10 all year round. Adelaide is UTC+9:30 (winter) or UTC+10:30 (summer). Timestamps are kept in market time to avoid misalignment.
3. Only data after 1 October 2021 is used. On that date the NEM moved from 30-minute to 5-minute settlement. Using only post-change data keeps every price at the same granularity.
4. Prices can be negative. The market price floor is −$1,000/MWh. Negative prices are common in SA at midday, when solar and wind supply exceeds demand.
5. TOTALDEMAND excludes rooftop solar. Household solar reduces the electricity drawn from the grid, so midday demand appears low.