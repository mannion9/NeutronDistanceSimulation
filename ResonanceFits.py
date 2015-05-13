import matplotlib.pyplot as plt
import math as m

#### Scattering ####
def scattering_resonance(energy):
    def res_1(energy):
        #### 339 -6 MeV Resonance ####
        # 338E-6---- 340E-6
        y_0 = 4.1061
        A = 1.6218
        x_0 = 0.0003395
        width = 5.4E-7
        return y_0 + A*m.exp(-((energy-x_0)**2/width))

    def res_2(energy):
        #### 1E-3 MeV Resonance ####
        # .9E-6---- 1.002E-6
        y_0 = 3.9691
        A = 42.943
        x_0 = 0.0001
        width = 9.22E-7
        return y_0 + A*m.exp(((energy-x_0)**2/width))

    def res_3(energy):
       #### 2.186 E-3 MeV Resonance ####
        # 2.183 E-3 ---- 2.189E-3
        y_0 = 6.0119
        A = 141.63
        x_0 = 0.002186
        width = 1.4083E-6
        return y_0 + A*m.exp(((energy-x_0)**2/width))
    
    def res_4(energy):
        #### 6.315E-3 MeV Resonance ####
        # 6.312E-3 ---- 6.1318E-3
        y_0 = 4.7749
        A = 49.531
        x_0 = 0.006315
        width = 2.355E-6
        return y_0 + A*m.exp(((energy-x_0)**2/width))
    
    def res_5(energy):
        #### 7.260E-3 MeV Resonance ####
        # 7.254E-3 ---- 7.264E-3
        y_0 = 4.2887
        A = 14.698
        x_0 = 0.00726
        width = 2.4066E-6
        return y_0 + A*m.exp(((energy-x_0)**2/width))
#### Gamma ####
def gamma_resonance(energy):

    def res_1(energy):
        #### 339 -6 MeV Resonance ####
        # 338E-6---- 340E-6
        y_0 = 0.45376
        A = 28.557
        x_0 = 0.000339
        width = 5.29E-7
        return y_0 + A*m.exp(((energy-x_0)**2/width))
    
    def res_2(energy):
        #### 1E-3 MeV Resonance ####
        # .9E-6---- 1.002E-6
        y_0 = 0.87949
        A = 70.784
        x_0 = 0.0010006
        width = 9.084e-07
        return y_0 + A*m.exp(((energy-x_0)**2/width))
    
    def res_3(energy):
        #### 2.186 E-3 MeV Resonance ####
        # 2.183 E-3 ---- 2.189E-3
        y_0 = 0.94865
        A = 42.191
        x_0 = 0.002186
        width = 1.3871E-6
        return y_0 + A*m.exp(((energy-x_0)**2/width))
    
    def res_4(energy):
        #### 6.315E-3 MeV Resonance ####
        # 6.312E-3 ---- 6.1318E-3
        y_0 = 0.25496
        A = 9.5438
        x_0 = 0.006315
        width = 2.3065e-06
        return y_0 + A*m.exp(((energy-x_0)**2/width))

    def res_5(energy):
        #### 7.260E-3 MeV Resonance ####
        # 7.254E-3 ---- 7.264E-3
        y_0 = 0.10517
        A = 11.425
        x_0 = 0.00726
        width = 2.3936e-06
        return y_0 + A*m.exp(((energy-x_0)**2/width))
