# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Infinite Square Well Potential
# Here are some plots and code related to the PHAS0004: Atoms, Stars and the Universe course as taught in 2020 to the first year students of UCL Physics and Astronomy. In this note book we are going to be looking at the infinite square well potential.

# +
import numpy as np  #import the numpy library as np
import matplotlib.pyplot as plt #import the pyplot library as plt
import matplotlib.style #Some style nonsense
import matplotlib as mpl #Some more style nonsense
import math #Import math so that math.pi can be used

#Set default figure size
#mpl.rcParams['figure.figsize'] = [12.0, 8.0] #Inches... of course it is inches
mpl.rcParams["legend.frameon"] = False #Turn off the box around the legend
mpl.rcParams['figure.dpi']=150 # dots per inch


# -

# ## Our first potential
#
# During the course we discussed the infinite square well potential, which had a potential of the form
# $$V(x)=\begin{cases}
# 0 & \text{if } 0 \leq x \leq L \\
# \infty & \text{otherwise}
# \end{cases}$$
#
# Now we will try and plot this function... obviously we can't plot infinity so to plot it below we have just picket an arbitrary number (1000)

# +
#Define a function which will operate on an array of x values all at once
def infiniteV(x):
    conds = [x < 0, (x >= 0) & (x <= 1), x > 1]   #The three regions of x
    funcs = [lambda x: 1000, lambda x: 0,  #0 in the middle and 1000 outsides
            lambda x: 1000]  #the lambda keyword is allowing us to define a quick function
    return np.piecewise(x, conds, funcs)  #Now do the piecewise calculation and return it
    
    

# +
#Now let's plot our potential
fig, ax = plt.subplots()  #I like to make plots using this silly fig,ax method but plot how you like
x = np.linspace(-0.5, 1.5, 1000)  #x values between -0.5 and 1.5
ax.fill_between(x,infiniteV(x),0)  #Fill between the V(x) values and 0
ax.set_title(r"Infinite Square Well")  #Set the plot title
ax.set_xlabel("$x / L$")  # Set the x-axis label
ax.set_ylabel("$V(x)$") # Set the y-axis label
ax.set_yticklabels([]) #Remove the y-axis label as we are faking that it goes to infinity
ax.annotate(r'$V(x)=\infty$', xy=(1.25,500), xycoords="data",  #Label on right-hand side
                  va="center", ha="center",  #horizontal and vertical alignment
                  bbox=dict(boxstyle="round", fc="w")) #White round box about text

ax.annotate(r'$V(x)=0$', xy=(0.5,500), xycoords="data",  #Label on center
                  va="center", ha="center",  #horizontal and vertical alignment
                  bbox=dict(boxstyle="round", fc="w")) #White round box about text

ax.annotate(r'$V(x)=\infty$', xy=(-0.25,500), xycoords="data",  #Label on left-hand side
                  va="center", ha="center",  #horizontal and vertical alignment
                  bbox=dict(boxstyle="round", fc="w")) #White round box about text


# -

# ## Wavefunctions in the Infinite Square Well  
# In the course lectures we showed that the the solution to the infinite square well potential was
# $$\psi(x)=\begin{cases}
# \sqrt{\frac{2}{L}} \sin \left(\frac{n \pi x}{L}\right) & \text{if } 0 \leq x \leq L \\
# 0 & \text{otherwise}
# \end{cases}$$
#
#

# +
#Define a function which will operate on an array of x values all at once
def infinitePsi(x,n):
    conds = [x < 0, (x >= 0) & (x <= 1), x > 1]   #The three regions of x
    funcs = [lambda x: 0, lambda x: np.sqrt(2)*np.sin(n*math.pi*x),  # sqrt(2)*sin(n pi x) in the middle and 0 outsides
            lambda x: 0]  #the lambda keyword is allowing us to define a quick function
    return np.piecewise(x, conds, funcs)  #Now do the piecewise calculation and return it
    
    
# -

# ### Let's plot those wavefunctions

#Now let's plot our first wavefunction
fig, ax = plt.subplots()  #I like to make plots using this silly fig,ax method but plot how you like
x = np.linspace(-0.25, 1.25, 1000) # 1000 x values between -0.25 and 1.25
cmap = plt.get_cmap("tab10")  #This mysterious object is the default colormap... can use color=cmap(3)
ax.plot(x,infinitePsi(x,1),linewidth=3,label="n=1") #Plot x vs psi(x) for n=1
ax.set_title(r"Infinite Square Well") #Plot title
ax.set_xlabel("$x / L$") # Set x-axis label
ax.set_ylabel("$\psi(x) * \sqrt{L}$") # Set y-axis label
plt.legend()  # Add a legend to the plot
ax.set_ylim(-1.5,1.5)  # Set the axis limits from -1.5 to 1.5

#Now let's plot our first 4 wavefunction
fig, ax = plt.subplots()  #I like to make plots using this silly fig,ax method but plot how you like
x = np.linspace(-0.25, 1.25, 1000) # 1000 x values between -0.25 and 1.25
ax.plot(x,infinitePsi(x,1),linewidth=1,label="n=1")  #Plot x vs psi(x) for n=1
ax.plot(x,infinitePsi(x,2),linewidth=1,label="n=2")  #Plot x vs psi(x) for n=2
ax.plot(x,infinitePsi(x,3),linewidth=1,label="n=3")  #Plot x vs psi(x) for n=3
ax.plot(x,infinitePsi(x,4),linewidth=1,label="n=4")  #Plot x vs psi(x) for n=4
ax.set_title(r"Infinite Square Well")  #Plot title
ax.set_xlabel("$x / L$") # Set x-axis label
ax.set_ylabel("$\psi(x) * \sqrt{L}$") # Set y-axis label
plt.legend()  # Add a legend to the plot

# ## What about probability distributions?
# The probability density function $\rho(x)$ is defined as
# $$\rho(x) = \left| \psi(x) \right|^2$$
#
# What do the probability density functions look like in the infinite square well?

#Now let's plot our first 4 probability density functions
fig, ax = plt.subplots()  #I like to make plots using this silly fig,ax method but plot how you like
x = np.linspace(-0.25, 1.25, 1000) # 1000 x values between -0.25 and 1.25
ax.plot(x,infinitePsi(x,1)**2,linewidth=1,label="n=1")  #Plot x vs psi(x)^2 for n=1
ax.plot(x,infinitePsi(x,2)**2,linewidth=1,label="n=2")  #Plot x vs psi(x)^2 for n=2
ax.plot(x,infinitePsi(x,3)**2,linewidth=1,label="n=3")  #Plot x vs psi(x)^2 for n=3
ax.plot(x,infinitePsi(x,4)**2,linewidth=1,label="n=4")  #Plot x vs psi(x)^2 for n=4
ax.set_title(r"Infinite Square Well") #Plot title
ax.set_xlabel("$x / L$") # Set x-axis label
ax.set_ylabel(r"$\rho(x) * L$") # Set y-axis label
plt.legend()  # Add a legend to the plot



