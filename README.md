# Analysis of analogic data in Industrial Filter Presses
## Objectives
The objective of this small library is to provide some function to analyze some analogic data of a filter press provided by the application made with Mindsphere (from Siemens) and by 40 Factory, described below.
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
The data provided by the application are  arranged in this way: dataframe with any data recorded with the time of record, saved as bson (binary json) files divided by number of cycle and number of phase in the cycle for the phase variables. The variables are divided in Analogic and Phase Analogic data, the firsts referred to all the the cycle, the seconds referred to single phases of the cycle (the meaning of cycle and phase are described below). In the Phase analogic data the bson file contain 2 dataframes, the first referred to the max values of all analogic measured and the second that contain the record of the variables of interest.
### Cycle
A cycle is the ensemble of process that happens between the moment in which the slurry is started to pump, to the moment in with each cake is extracted from the filter press.
### Phase
Each cycle is divided in any phases, the phases of interest for the analogic variables are the the second (phase 2) and the third (phase 3), that are the two phases of feeding (when the slurry is pumped in the filter press). In particular, phase 2 is the phase in which the slurry is filling the machine and phase 3 is the phase in which the slurry is pumped after the machine is full, so the pumped slurry increase the cakes and pump out the liquid component.  In Phase 3 after few seconds the pressure becomes almost constant 


![Figure_1](https://user-images.githubusercontent.com/48355728/162576060-57c991f9-d761-4f83-85aa-c32780a16325.png)



while the flow is rapidly dropping


## Analysis 
The main focus of the functions of this library is the residual humidity, that is the residual mass of water in the cake, and the time/volume-volume curve.
### Residual Humidity
The residual humidity is the residual mass ow the liquid component (in general water) in the cake. Each filter press, separating the liquid and solid part in a mechanical way, can't eliminate all the liquid component from the cake. The objective of each filter press is to reduce under a significant level the residual humidity
(typically around 15-20 % of total mass of the cake) in a reasonable time. Here are provided some functions to analyzed the residual humidity of the cake 
![Figure_4](https://user-images.githubusercontent.com/48355728/162576143-0a0576bd-e43c-4ed2-b1cc-3152643b5c91.png)

### Time/volume-volume curve
During the phase 3 of the filtration process, the filter press is in a situation of approximately constant pressure. In this condition, theory said that the time of alimentation t and the volume pumped V are in a specific relation: the ratio t/V vs. V should be in a linear relation (for a detaied physical explanation press [here](https://core.ac.uk/reader/48625383) ). This is no true in the the first data analyzed, where there is a change of slope in the data analyzed, so there are some 
functions to analyze and plot the behaviour of this curve.

![figure_3](https://user-images.githubusercontent.com/48355728/162017792-a0e6e50b-3e6e-44b2-ba3c-13dd6f6a794f.png)

## Structure of this Project

In the file tirocinio.py there are all the functions. First of all there is the need to call the function **change_global_names** to define the names of the analog variables in the dataframe.
