# Analysis of analogic data in Industrial Filter Presses
## Objectives
The objective of this small library is to provide some function to analyze some analogic data of a filter press provided in a specific way, described below.
## Filter Press
A filter press is a piece of equipment used in liquid/solid separation. 
Specifically, the filter press separates the liquids and solids using pressure filtration, 
wherein a slurry is pumped into the filter press and is dewatered under pressure. 
Basically, each filter press is designed based on the volume and type of slurry that needs to be dewatered.
### Working
The working principle of filter presses is that slurry is pumped into the machine such that solids are distributed evenly during the fill cycle. 
Solids build up on the filter cloth, forming the filter cake; 
the filtrate exits the filter plates through the corner ports into the manifold, yielding clean filtered water.
Filter presses are a pressure filtration method and as such, as the filter press feed pump builds pressure, 
solids build within the chambers until they are completely chock-full of solids, forming the cake. 
Once the chambers are full, the cycle is complete and the filter cakes are ready to be released. 
In many higher capacity filter presses, fast action automatic plate shifters are employed, speeding cycle time.
To watch an informative video of how a filter press works, press [here](https://www.youtube.com/watch?v=UguqOosjrTc&ab_channel=Prolific3DTech).

## Data
The data should be arranged in this way: a dataframe with any data recorded with the time of record, saved as bson files divided by number of cycle and number of phase in the cycle for the phase variables
### Cycle
A cycle is the ensemble of process that happens between the moment in which the slurry is started to pump, to the moment in with each cake is extracted from the filter press.
### Phase
each cycle is divided in any phases, the phases of interest for the analogic variables are the the second (phase 2) and the third (phase 3), that are the two phases of feeding (when the slurry is pumped in the filter press)
## Analysis 
