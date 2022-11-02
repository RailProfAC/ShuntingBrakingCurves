# Import packages for calculation and plotting
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Time package for performance measurements
import time
# Statistics package for rare event simulation
from scipy.stats import norm
# Pandas for convenient table export
import pandas as pd
# Settings for seaborn plots
sns.set()
sns.set_context("paper")
sns.set_style("whitegrid")

#########################################################
# Friction value function according to Karwatzki formula
def muBr(v,F):
    # Karwatzki parameter (composite blocks)
    k1 = 0.055
    k2 = 200000 #N
    k3 = 50000 #N
    k4 = 150/3.6 #m/s
    k5 = 75/3.6 #m/s
    return k1 * (F+k2)/(F+k3) * (v+k4)/(v+k5)


def ShuntingBrakeCalculation(data):
    c = 250 # brake propagation velocity
    #FL = FLoco #Loco braking force
    dt = 0.05
    s = 0
    t = 0
    # Import data from input
    v = data['v0']
    mloco = data['mloco']
    Floco = data['Floco']
    lloco = data['lloco']
    tfloco = data['tfloco']
    gloco = data['Gloco']
    m = data['m']
    mrot = data['mrot']
    tf = data['tf']
    gmode = data['G']
    cdia = data['Cdia']
    p = data['p']
    l = data['l']
    i = data['i']
    n = data['n']
    eta = data['eta']
    muc = data['muc']
    # Correct filling time according to position in train
    for k in range(len(l)):
        tf[k] = tf[k] + (lloco + np.sum(l[0:k]))/c

    while v > 0:
        F = [] # List for vehicle Forces
        # Treat locomotive
        if t < tfloco: # Respect filling time
                # 10% immediate pressure in brake mode G
            if bool(gloco):
                fill = 0.1+0.9*(t/tfloco)
            else: # Linear behaivour in P
                fill = t/tfloco
        else:
            fill = 1 #Fully filled after tf
        F.append(Floco*fill)
        for k in range(len(l)): # Process wagon k
            #Cylinder force
            Fc = p[k]*np.pi*cdia[k]*cdia[k]/4-1500 #Return force
            if t < tf[k]: # Respect filling time
                # 10% immediate pressure in brake mode G
                if bool(gmode[k]):
                    fill = 0.1+0.9*(t/tf[k])
                else: # Linear behaivour in P
                    fill = t/tf[k]
                #Force for all blocks
                Fblocks = Fc*fill*i[k]
                #Friction according to Karwatzki formula
                F.append(Fblocks*muBr(v, Fblocks/n[k])*eta[k]*muc[k])
            else:
                Fblocks = Fc*i[k]
                F.append(Fblocks*muBr(v, Fblocks/n[k])*eta[k]*muc[k])
        # Braking force from individual wagons and Strahl ride resistance
        FB = np.sum(F) + (1.6e-3+5.7e-3*(v/27.8)**2)*np.sum(m)*9.81
        # Update, wagon mass increased by dynamical augment
        v = v - (Floco + FB)/(mloco + np.sum(np.multiply(mrot, m)))*dt
        s = s + v*dt
        t = t + dt
    return s
################################
def main():
    #############################
    # Loco Braking
    mLoco = 53000
    lambdaLoco = np.floor(56/53*100)
    aLoco = np.round(0.0075*lambdaLoco+0.076, decimals = 2)
    FLoco = mLoco*aLoco

    ##############################
    # Monte Carlo
    start = time.time()
    np.random.seed(42) # Fix seed
    v0 = 25/3.6
    N = 5 # Number of wagons
    M = 1e3 # Number of MC iterations
    # Critical distances
    distcrit = [64, 65, 66, 67, 68, 69]
    slist = [] # List for storage of individual distances
    for k in range(int(M)):
        # Generate random data
        data = {'v0' : v0,
                'mloco': mLoco,
                'Floco': FLoco,
                'lloco': 10.5,
                'tfloco': 24,
                'Gloco': 1,
                'm' : 90000*np.ones(N), # Mass of individual wagons
                'mrot': 1.04*np.ones(N), # Dynamic augment
                'G' : np.ones(N), # G mode of individual wagon
                'Cdia': 0.406*np.ones(N), # Cylinder diameter
                'p' : np.random.normal(loc = 3.8e5, scale = 5e3, size = N),
                'l' : 14*np.ones(N),
                'i' : 5.65*np.ones(N), #4 axle, loaded
                'eta' : np.random.normal(loc = 0.83, scale = 0.02, size = N),
                'tf' : np.random.normal(loc = 24, scale = 2, size = N),
                'n' : 16*np.ones(N), # Number of brake blocks
                'muc': np.random.normal(loc = 1, scale = 0.025, size = N) # Friction correction factor
               }
        # Perform brake calculation
        s = ShuntingBrakeCalculation(data)
        # Append result to list
        slist.append(s)
        #print(k%10, end = ' ')
    elapsed = time.time()-start
    # Plot histogram
    # sns.histplot(slist, kde = True, cumulative=True, stat = 'probability',
    #              color = 'seagreen', label = 'Cumulative Density', bins = 20)
    # sns.histplot(slist, kde = True, stat = 'probability',
    #              color = 'navy', label = 'Probability Density', bins = 20)
    # plt.ylabel('Probability')
    # plt.xlabel('s/m')
    # # Uncomment to save figure
    # plt.savefig('Class363' + str(N) +
    #             'WagonsG'+str(round(v0))+
    #             'M' + str(M) +'.pdf')
    # # Print statistical values
    # print('Mean: ' + str(np.mean(slist)))
    # print('SD  : ' + str(np.std(slist)))
    # ssorted = np.sort(slist)
    # if M >= 1e7:
    #     print('p = 1e-5 for: ' + str(ssorted[-101]))
    # print('Longest: ' + str(ssorted[-1]))
    # # Analyse for relative frequency of braking distance above threshold
    # df = pd.DataFrame(columns = ['s', 'p'])
    # for d in distcrit:
    #     p, b = np.histogram(slist, bins = [0, d, 1e6])
    #     p = p/M
    #     print('p(s> ' + str(d) + 'm) = ' + str(p[1]))
    #     df.loc[len(df.index)] = [d, p[1]]
    # # Uncomment to save data
    # df.to_excel('Class363' + str(N) +
    #             'WagonsG'+str(round(v0))+
    #             'M' + str(M) +'.xlsx', index=False)
    print('Time: ' + str(np.round(elapsed, decimals = 2)) + ' s')
    # df2 = pd.DataFrame(data = slist, columns = ['s'])
    # # Uncomment to save data
    # # Save data to file
    # df2.to_csv('Class363' + str(N) +
    #             'WagonsG'+str(round(v0))+
    #             'M' + str(M) +'Distances.csv')

main()
