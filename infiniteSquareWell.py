# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
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
import math 

#Set default figure size
#mpl.rcParams['figure.figsize'] = [12.0, 8.0] #Inches... of course it is inches
mpl.rcParams["legend.frameon"] = False
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
    funcs = [lambda x: 1000, lambda x: 0,  # sin(pi x) in the middle and 0 outsides
            lambda x: 1000]  #the lambda keyword is allowing us to define a quick function
    return np.piecewise(x, conds, funcs)  #Now do the piecewise calculation and return it
    
    

# +
#Now let's plot our potential
fig, ax = plt.subplots()  #I like to make plots using this silly fig,ax method but plot how you like
x = np.linspace(-0.5, 1.5, 1000)
ax.fill_between(x,infiniteV(x),0)
ax.set_title(r"Infinite Square Well")
ax.set_xlabel("$x / L$")
ax.set_ylabel("$V(x)$")
ax.set_yticklabels([])
ax.annotate(r'$V(x)=\infty$', xy=(1.25,500), xycoords="data",
                  va="center", ha="center",
                  bbox=dict(boxstyle="round", fc="w"))

ax.annotate(r'$V(x)=0$', xy=(0.5,500), xycoords="data",
                  va="center", ha="center",
                  bbox=dict(boxstyle="round", fc="w"))

ax.annotate(r'$V(x)=\infty$', xy=(-0.25,500), xycoords="data",
                  va="center", ha="center",
                  bbox=dict(boxstyle="round", fc="w"))


# -

# ## Wavefunctions in the Infinite Square Well  
# In the course lectures we showed that the the solution to the infinite square well potential was
# $$\psi(x)=\begin{cases}
# \sqrt\frac{{2}{L}} \sin \left(\frac{n \pi x}{L}\right) & \text{if } 0 \leq x \leq L \\
# 0 & \text{otherwise}
# \end{cases}$$
#
#

# +
#Define a function which will operate on an array of x values all at once
def infinitePsi(x,n):
    conds = [x < 0, (x >= 0) & (x <= 1), x > 1]   #The three regions of x
    funcs = [lambda x: 0, lambda x: np.sqrt(2)*np.sin(n*math.pi*x),  # sin(n pi x) in the middle and 0 outsides
            lambda x: 0]  #the lambda keyword is allowing us to define a quick function
    return np.piecewise(x, conds, funcs)  #Now do the piecewise calculation and return it
    
    
# -

# ### Let's plot those wavefunctions

#Now let's plot our first wavefunction
fig, ax = plt.subplots()  #I like to make plots using this silly fig,ax method but plot how you like
x = np.linspace(-0.25, 1.25, 1000)
cmap = plt.get_cmap("tab10")
ax.plot(x,infinitePsi(x,1),linewidth=3,label="n=1")
ax.set_title(r"Infinite Square Well")
ax.set_xlabel("$x / L$")
ax.set_ylabel("$\psi(x) * \sqrt{L}$")
plt.legend()
ax.set_ylim(-1.5,1.5)

#Now let's plot our first 4 wavefunction
fig, ax = plt.subplots()  #I like to make plots using this silly fig,ax method but plot how you like
x = np.linspace(-0.25, 1.25, 1000)
ax.plot(x,infinitePsi(x,1),linewidth=1,label="n=1")
ax.plot(x,infinitePsi(x,2),linewidth=1,label="n=2")
ax.plot(x,infinitePsi(x,3),linewidth=1,label="n=3")
ax.plot(x,infinitePsi(x,4),linewidth=1,label="n=4")
ax.set_title(r"Infinite Square Well")
ax.set_xlabel("$x / L$")
ax.set_ylabel("$\psi(x) * \sqrt{L}$")
plt.legend()

# ## What about probability distributions?
# The probability density function $\rho(x)$ is defined as
# $$\rho(x) = \left| \psi(x) \right|^2$$
#
# What do the probability density functions look like in the infinite square well?

#Now let's plot our first 4 probability density functions
fig, ax = plt.subplots()  #I like to make plots using this silly fig,ax method but plot how you like
x = np.linspace(-0.25, 1.25, 1000)
ax.plot(x,infinitePsi(x,1)**2,linewidth=1,label="n=1")
ax.plot(x,infinitePsi(x,2)**2,linewidth=1,label="n=2")
ax.plot(x,infinitePsi(x,3)**2,linewidth=1,label="n=3")
ax.plot(x,infinitePsi(x,4)**2,linewidth=1,label="n=4")
ax.set_title(r"Infinite Square Well")
ax.set_xlabel("$x / L$")
ax.set_ylabel("$\psi(x) * \sqrt{L}$")
plt.legend()



