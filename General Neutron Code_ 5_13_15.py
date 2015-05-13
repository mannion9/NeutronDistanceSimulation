import math as m
import matplotlib.pyplot as plt
import random
import numpy as np


#############################
#### General User Inputs ####
#############################

#### General Inputs For Running Code ####
EnergyMax = 20E6   # Highest energy to plot
ThermalWalk = 0    # 1 to consider thermal walk 0 to ignore
VaryAngle = 0      # 1 to consider varying thermal walk 0 to ignore
Resonance = 0      # 1 to consider nuclear resoances 0 to ignore
VaryEnergyLoss = 0 # 1 to consdier varying energy loss 0 to ignore

############################################
#### Substance Substance Specific Input ####
############################################

#### Specific Properties Of Substance ####
Density = 2.94                        # Density of substence in g/cm^3
A = [136, 134]                        # Atomic weight in atomic mass units of substances
Number = [0.80 , 0.20]                # Percentage of each substance
ThermalAbsorptionXC = [ .26 , .265]   # Thermal absorption cross section of each substance in units of barns

#### Cross Section Fit Of Substance ####

### Xe-136 Fits ###
# 50 MeV - 13.07 MeV parameters
y_1_0=5.1824
A_1=-1.7744
x_1_0=37.588 
width_1=0.64387
# 13.07 MeV - 1.231 MeV parameters
k_2_0=2.0236
k_2_1=9.0551
k_2_2=-6.7402
k_2_3=2.5277 
k_2_4=-0.57594 
k_2_5=0.083509
k_2_6=-0.007686 
k_2_7=0.00043296
k_2_8=-1.3584e-05
k_2_9=1.8161e-07
# 1.231 MeV - .0265 MeV parameters
y_3_0=11.091
A_3=-6.3557
x_3_0=0.17283
width_3=3.3266
# .0265 MeV - .0059 MeV parameters
y_4_0=12.882 
A_4=-6.4768
x_4_0=0.022182
width_4=8.7966
# .0059 MeV - 1/40 eV parameters
k_5_0=8.2975
k_5_1=-2850.1
k_5_2=3.337e+06
k_5_3=-2.3996E9
k_5_4=1.0062e+12 
k_5_5=-2.4572E14
k_5_6=3.422E16
k_5_7=-2.5069E18
k_5_8=7.4657E19

### Cross Section Fit Function Xe-136 ###
def Xe_136_Fit(Energy):
    Energy = Energy*1E-6 # Converts energy to MeV so because thats how fit functions where done
    if Energy > 13.07:
        XC = y_1_0+A_1*(m.exp(-(m.log(Energy/x_1_0)/width_1)**2))
        return XC
    elif Energy > 1.231:
        XC = k_2_0+(k_2_1*Energy)+(k_2_2*(Energy**2))+(k_2_3*(Energy**3))+(k_2_4*(Energy**4))+(k_2_5*(Energy**5))+(k_2_6*(Energy**6))+(k_2_7*(Energy**7))+(k_2_8*(Energy**8))+(k_2_9*(Energy**9))
        return XC
    elif  Energy > .0265:
        XC = y_3_0+A_3*(m.exp(-(m.log(Energy/x_3_0)/width_3)**2))
        return XC
    elif Energy > .0059:
        XC = XC_test = y_4_0+A_4*(m.exp(-(m.log(Energy/x_4_0)/width_4)**2))
        return XC
    else :
        XC = k_5_0+(k_5_1*Energy)+(k_5_2*(Energy**2))+(k_5_3*(Energy**3))+(k_5_4*(Energy**4))+(k_5_5*(Energy**5))+(k_5_6*(Energy**6))+(k_5_7*(Energy**7))+(k_5_8*(Energy**8))
        return XC

### Cross Section Fit Function Xe-134 Fits ###
def Xe_134_Fit(energy):
    if energy <= 3.37952E-3:
        exponent = -(energy-1E-11)
        cross_section = 4.461 + 20.692*m.exp(exponent/2.2301E-11) + 8.9504*m.exp(exponent/4.2726E-10)
        return cross_section
    elif energy < 1:
        exponent= - ((m.log(energy/0.1381)/(8.0352))**2)
        cross_section = 27.861 + -22.116 * m.exp(exponent)
        return cross_section
    else:
        cross_section = 4.5013 + 2.8375*(energy**-1.0621)
        return cross_section

###### Nuclear Resoance Data ####
# Resoance for each substance (Ordering must match that of must match that of the PercentEnergyLoss list)
res = [[2153.0],[339E-6,1E-3,2.186E-3,6.315E-3,7.260E-3]] 
        
###  Resoance Cross Section Fitting ###
## Xe-136 ##
# Absorption XC Fit
y_res_1=0.049687
A_res_1=16.718
x_o_res_1=0.002154
width_res_1=1.2847e-06
# Scattering XC Fit
y_res_scat_1=6.894
A_res_scat_1=8.3395
x_o_res_scat_1=0.002154
width_res_scat_1=1.279e-06
# Functions
def abs_xc_1(energy):
    cross_section = y_res_1+(A_res_1*m.exp(-((energy-x_o_res_1)/width_res_1)**2))
    return cross_section
def scat_xc_1(energy):
    cross_section = y_res_scat_1+(A_res_scat_1*m.exp(-((energy-x_o_res_scat_1)/width_res_scat_1)**2))
    return cross_section
    
### Xe-134 ###
#########################
### ADD ME HERE ########
#########################


#### Thermal Data ####
ThermalCrossSection = [ Xe_136_Fit(1/40) , Xe_134_Fit(1/40)]
TotalCrossSection = (A[0]*ThermalCrossSection[0] + A[1]*ThermalCrossSection[1])
ThermalProbability = [ A[0]*ThermalCrossSection[0] , A[1]*ThermalCrossSection[1] ]

#### Absorption Choice Function ####
def Absorption(atom,energy,index):
    if atom == 0:
       absorption_xc = abs_xc_1(energy)
       scatter_xc = scat_xc_1(energy)
    else:
        return(0) # Change this when you have the thermal cross section fits for Xe-134 
    absorption_probability = absorption_xc / (absorption_xc + scatter_xc)
    scatter_probability = scatter_xc / (absorption_xc+scatter_xc)
    return ChoiceFunc(scatter_probability,absorption_probability)

#### Collision Choice ####
def CollisionChoice(energy):
    CrossSections = [Xe_136_Fit(energy),Xe_134_Fit(energy)]
    Xe_136Probability = Number[0]*CrossSections[0] / (Number[0]*CrossSections[0]+Number[1]*CrossSections[1])
    Xe_134Probability =  Number[1]*CrossSections[1] / (Number[0]*CrossSections[0]+Number[1]*CrossSections[1])
    choice = ChoiceFunc(Xe_136Probability,Xe_134Probability)
    if VaryEnergyLoss == 0:
        energy_loss = energy * PercentEnergyLoss[choice]
    else:
        energy_loss = energy * random.uniform(EnergyLossLow[choice],EnergyLossHigh[choice])
    if choice == 0:
        XC.append(CrossSections[0])
    elif choice == 1:
        XC.append(CrossSections[1])
    Collision_Record.append(choice)
    return(energy_loss,choice)



#####################
#### Simulation #####
#####################

#### Constants ####
Avagadro = 6.022E23         # Avagadros number
A_matrix = np.matrix(A)
N = Density*Avagadro/A_matrix      # Number Densities atom/cm^3
M = 939.56E6                # Mass of neuton (eV/c^2)
mue = 2/(3*A_matrix)               # Mean cosine of scattering angle (from papper)
v_thermal=2200*100  # velocity of thermal neutron in cm/s

#### Lists ####
list_counter=[]
list_walk = []

# Resoance bounds
# Create upper and lower bounds of resoanance to 
res_low  = [[],[]]
res_high = [[],[]]
for atom in res:
    indexer = res.index(atom)
    for resonance in atom:
        res_low[indexer].append(resonance-resonance*.001)
        res_high[indexer].append(resonance+resonance*.001)
        
# Energy Loss in collision
def Energy_loss(A):
    alpha = ((A-1)/(A+1))**2
    return m.exp(1+(alpha*m.log(alpha)/(1-alpha)))/100

#### Energy Loss ####
PercentEnergyLoss = []
for a in A:
    PercentEnergyLoss.append(Energy_loss(a))
EnergyLossHigh=[]
EnergyLossLow=[]
for energy_loss in PercentEnergyLoss:
    EnergyLossHigh.append(energy_loss + 0.01*energy_loss)
    EnergyLossLow.append(energy_loss - 0.01*energy_loss)
    
#### Data Lists ####
Energies = []
DistanceTraveled = []
TimeTraveled = []
AbsorptionEnergy = []
ThermalDistance = []
ThermalTime = []

#### Choice Function ####

def ChoiceFunc(Prob1,Prob2):
    # This function is used later to pick which reaction will occur by knowning
    # the probability of what will occur. It works essentially by creating a list
    # with 1's and 2's with a 1 correpsonding to reaction 1 and 2 to reaction 2
    # We then go into the list at some random index and read the number. There
    # are more 1's in the list if reaction 1 is more probable or more 2's if reaction
    # 2 is more probable.

    Precision = 3
    
    Prob1 = round(Prob1,Precision)
    Prob2 = 1 - Prob1

    option_length = 10**Precision

    options_list = [0]*option_length
    
    max_1=Prob1*option_length
    max_2=Prob2*option_length

    counter_1=0
    counter_2=0
    index=0

    while counter_1 < max_1:
        options_list[index]=0
        index+=1
        counter_1+=1
    while counter_2 < max_2:
        if index == option_length:
            break
        options_list[index]=1
        index+=1
        counter_2+=1

    index_choice=random.randrange(0,option_length)
    choice=options_list[index_choice]
    
    return choice

#### Angle Choice
def AngleChoice(energy,atom):
    if energy > 1E3:
        sigma=5*m.pi/180
        mue_particular = random.gauss(mue[0,atom],sigma)
    else:
        sigma=10*m.pi/180
        mue_particular = random.gauss(mue[0,atom],sigma)
    return mue_particular

#### Resonance Aborption Choice ####
def AbsorptionChoice(energy,atom):
    index = 0
    choice = 0
    while index < len(res[atom]):
        if res_low[atom][index] <= energy and energy < res_high[atom][index]:
            choice=Absorption(atom,energy,index)
        index += 1
    return choice

#### Thermal Calculations ####

if ThermalWalk == 1:
    ScatteringProbability = []
    AbsorptionProbability = []
    ThermalDistance = []
    ThermalTimes = []
   
    for i in ThermalCrossSection:
        index=ThermalCrossSection.index(i)
        ScatteringProb = (i-ThermalAbsorptionXC[index])/i
        ThermalDist = 1/N[0,index]/i/10E-24
        ScatteringProbability.append(ScatteringProb)
        AbsorptionProbability.append(1-ScatteringProb)
        ThermalDistance.append(ThermalDist)
        ThermalTimes.append(ThermalDist/v_thermal)
        

#### Initial Energies To Simulate####
while EnergyMax > (1/40):
    Energies.append(EnergyMax)
    EnergyMax -= EnergyMax*.02

#### Core Calculations ####
for Energy in Energies:
    Absorber = 0
    XC = []
    Energy_Record = []
    Collision_Record = []
    Absorption_Record = []
    MeanFreePaths=[]
    Velocities=[]
    Times=[]
    while Energy > (1/40):
        # Determines the energies this neutron will exist at
        Energy_Record.append(Energy)
        energy_loss , atom = CollisionChoice(Energy)
        Energy -= energy_loss
        Collision_Record.append(atom)
        if Resonance == 1:
            if Absorber == 0:
                Absorber = AbsorptionChoice(Energy,atom)
            else:
                Absorption_Record.append(Energy)
                break
    for xc in XC:    # Convert cross section to cm
        indexer = XC.index(xc)
        XC[indexer] = xc * 10E-24
        MeanFreePaths.append(1/N[0,Collision_Record[indexer]]/XC[indexer])  # Calculate mean free path for that cross section

    for energy in Energy_Record: # Calculate velocity in m/s
        Velocities.append(((2*energy/M)**(1/2))*3E8 )
    
    Timer = 0
    for velocity in Velocities:
        index = Velocities.index(velocity)
        distance = MeanFreePaths[index]/100 # cm--->m
        Timer += distance/velocity
    Times.append(Timer) 

    lambda_1 = 0
    lambda_2 = sum(MeanFreePaths)-MeanFreePaths[-1]
    lambda_3 = 0
    for distance in MeanFreePaths:
        lambda_1 += distance**2
        index = MeanFreePaths.index(distance)
        if VaryAngle == 1:
            mue_particular = AngleChoice(Energy_Record[index],Collision_Record[index])
            lambda_3 += distance*mue_particular
        else:
            lambda_3 += distance*mue[0,Collision_Record[index]]

    R_squared = 2*(lambda_1+lambda_2*lambda_3)
    R = abs((R_squared**(1/2)))

    if ThermalWalk == 1:
        counter = 0
        timing = 0
        x = 0
        y = 0
        while Absorber == 0:
            if counter > 1000:
                break
            else:
                counter += 1
                energy_loss , atom = CollisionChoice(1/40)
                Absorber = ChoiceFunc(ScatteringProbability[atom],AbsorptionProbability[atom])
                angle = random.randrange(0,360)*m.pi/180
                x += ThermalDistance[atom]*m.cos(angle)
                y += ThermalDistance[atom]*m.sin(angle)
                timing += ThermalTimes[atom]
        
        Absorption_Record.append(1/40)
        lambda_final = m.sqrt(x**2+y**2)
        R += lambda_final
        Timer += timing

        list_counter.append(counter)
        list_walk.append(m.sqrt(x**2+y**2))
        
        
    DistanceTraveled.append(R)
    TimeTraveled.append(Timer*1000)


#################
### Plotting ####
#################
    
print('The Max distance traveled is',max(DistanceTraveled),'in cm')
print('The longest time is',max(Times),'in seconds')
plt.figure(1)
plt.scatter(Energies,DistanceTraveled)
if ThermalWalk == 1:
    plt.title('Distance To Absorption vs Initial Energy ')
else:
    plt.title('Distance to Thermal vs Initial Energy ')
plt.xlabel('Energies (eV)')
plt.ylabel('Distance (cm)')
plt.xscale('log')
plt.text(max(Energies)-1E2,min(DistanceTraveled),'New Code\n(80% Xe-136 20% Xe-134)\n(Constant Angle-Yes)\n(Thermal walk-No)\n(Resonance-No)', ha='center', va='center')
plt.figure(2)
plt.scatter(Energies,TimeTraveled)
if ThermalWalk == 1:
    plt.title('Distance To Absorption vs Initial Energy ')
else:
    plt.title('Distance to Thermal vs Initial Energy ')
plt.xlabel('Energies (eV)')
plt.ylabel('Time (ms)')
plt.xscale('log')
plt.text(max(Energies)-1E2,min(TimeTraveled),'New Code\n(80% Xe-136 20% Xe-134)\n(Constant Angle-Yes)\n(Thermal walk-No)\n(Resonance-No)', ha='center', va='center')
plt.show()
