import cv2 as cv
import numpy as np
import random as rnd
from scipy.stats import entropy

def stretch(I):
    I = I.astype(np.float64)
    for i in range(I.shape[2]):
        I[:,:,i] = (I[:,:,i] - np.min(I[:,:,i]))/(np.max(I[:,:,i])-np.min(I[:,:,i]))*255
        
    return I.astype(np.uint8)
    
def gamma_correction(I,gamma):
    I = I/255.0
    I = I**gamma
    I = I*255.0
    
    return np.clip(I,0,255)

def enhance_image(I_in,alpha,beta,gamma):
    
    I_eq = cv.equalizeHist(I_in)    
    I_cp = gamma_correction(I_in,gamma)
    I_ex = gamma_correction(I_in,1/gamma)

    I_en = alpha/(1+beta)*I_cp + (1-alpha)/(1+alpha)*I_ex + beta/(1+beta)*I_eq
    
    return np.clip(I_en,0,255).astype(np.uint8)

def image_entropy(I):

    bins = 128
    hist, _ = np.histogram(I.ravel(),bins=bins,range=(0,255))
    
    p = hist/hist.sum()
    
    return entropy(p,base=2)

def objective_function(I_en,I_in):
       
    return -np.var(I_en)/np.mean(I_en)*(image_entropy(I_en)-image_entropy(I_in))

def initialization(I_in,mins,maxs,N):
    
    n = len(mins)
    
    X = np.zeros((N,n))
    V = np.zeros((N,n))
    
    low_x = mins
    high_x = maxs
    
    low_v = -np.abs(maxs-mins)
    high_v = np.abs(maxs-mins)
    for k in range(N):
            X[k,:] = np.random.uniform(low=low_x,high=high_x,size=n)
            V[k,:] = np.random.uniform(low=low_v,high=high_v,size=n)
            
    P_best = X
    
    G = X[0,:]
    G = update_G(I_in,G,P_best,N)
    
    return X, V, P_best, G

def update_P_best(I_in,N,P_best,X):
    for i in range(N):
        if objective_function(enhance_image(I_in,P_best[i,0],P_best[i,1],P_best[i,2]),I_in) < objective_function(enhance_image(I_in,X[i,0],X[i,1],X[i,2]),I_in):
            P_best[i,:] = X[i,:]
    
    return P_best

def update_G(I_in,G,P_best,N):
    f_max = objective_function(enhance_image(I_in,G[0],G[1],G[2]),I_in)
    for i in range(N):
        curr = objective_function(enhance_image(I_in,P_best[i,0],P_best[i,1],P_best[i,2]),I_in)
        if curr > f_max:
            G = P_best[i,:]
            f_max = curr
    
    return G
    
def swarm_particle_optimization(I_in,mins,maxs,N,max_epochs):
    
    X, V, P_best, G = initialization(I_in,mins,maxs,N)
    
    # https://doi.org/10.1162/EVCO_r_00180 - reference
    w = 0.715
    phi_p = 1.7
    phi_g = 1.7
   
    for epoch in range(max_epochs):
        
        if (epoch+1) % 5 == 0:
            print(epoch+1)
        
        rp = np.random.rand(N,3)
        rg = np.random.rand(N,3)
        
        # update velocity
        V = w*V + phi_p*rp*(P_best - X) + phi_g*rg*(G - X)
        # update position
        X = X + V
        
        for i in range(3):
            X[:,i] = np.clip(X[:,i],mins[i],maxs[i])
           
        # update best known position for each particle
        P_best = update_P_best(I_in,N,P_best,X)
        # update best known position overall
        G = update_G(I_in,G,P_best,N)
        
    return G

def grid_search(I_in,mins,maxs,step):
    f_max = 0
    G = np.zeros(3)
    
    alpha = mins[0]
    beta = mins[1] 
    gamma = mins[2]

    
    while alpha < maxs[0]:
        beta = mins[1]
        while beta < maxs[1]:
            gamma = mins[2]
            while gamma < maxs[2]:
                
                I_en = enhance_image(I_in,alpha,beta,gamma)
                
                f_curr = objective_function(I_en, I_in) 
                
                if f_curr > f_max:
                    f_max = f_curr
                    G[0] = alpha
                    G[1] = beta
                    G[2] = gamma
                    
                gamma += step
            beta += step
        alpha += step
        
    return G, f_max
                    
I = cv.imread('image2.jpg')
I = stretch(I)

I_hsv = cv.cvtColor(I,cv.COLOR_BGR2HSV)
H = I_hsv[:,:,0]
S = I_hsv[:,:,1]

I_in = I_hsv[:,:,2] # V

alpha_min = 0
alpha_max = 1

beta_min = 0
beta_max = 4

gamma_min = 1
gamma_max = 5

mins = np.array([alpha_min, beta_min, gamma_min])
maxs = np.array([alpha_max, beta_max, gamma_max])

alpha = (alpha_max - alpha_min)*rnd.random() + alpha_min
beta  = (beta_max - beta_min)*rnd.random() + beta_min
gamma = (gamma_max - gamma_min)*rnd.random() + gamma_min

#G = [alpha, beta, gamma]

N = 20
max_epochs = 100
# G = swarm_particle_optimization(I_in, mins, maxs, N, max_epochs)

step = 0.1
G, f_max = grid_search(I_in, mins, maxs,step)

#%%

I_en = enhance_image(I_in,G[0],G[1],G[2])

I_hsv_out = np.zeros(I_hsv.shape,dtype = np.ubyte)
np.copyto(I_hsv_out[:,:,0],I_hsv[:,:,0])
np.copyto(I_hsv_out[:,:,1],I_hsv[:,:,1])
np.copyto(I_hsv_out[:,:,2],I_en)

I_final = cv.cvtColor(I_hsv_out,cv.COLOR_HSV2BGR)

cv.imwrite('out.jpg',I_final)
print(f"J = {objective_function(I_en,I_in)}")