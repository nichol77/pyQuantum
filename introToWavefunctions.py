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

# # Wavefunction plots
#
# Here are some plots and code related to the PHAS0004: Atoms, Stars and the Universe course as taught in 2020 to the first year students of UCL Physics and Astronomy.

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

# ## Our first wavefunction
#
# The first wavefunction we introduced was a piecewise function that satisifed the following:
# $$\psi(x)=\begin{cases}
# \sin(\pi x) & \text{if } 0 \leq x \leq 2 \\
# 0 & \text{otherwise}
# \end{cases}$$
#
# We will use the python numpy library to code up this function using the piecewise method. Note that the $x$ below will refer to an array of numbers rather than just a single number, which means we have to be careful how we define the conditions and the function.

# +
#Define a function which will operate on an array of x values all at once
def sinpix(x):
    conds = [x < 0, (x >= 0) & (x <= 2), x > 2]   #The three regions of x
    funcs = [lambda x: 0, lambda x: np.sin(math.pi*x),  # sin(pi x) in the middle and 0 outsides
            lambda x: 0]  #the lambda keyword is allowing us to define a quick function
    return np.piecewise(x, conds, funcs)  #Now do the piecewise calculation and return it
    
    
# -

#Now we will use the linspace function to get 100 numbers
# linearly spaced between -2 and 4
x = np.linspace(-2, 4, 100)
print("x=",x)  #Just print the numbers, since -2 and 4 are in the 100 numbers the step size is not 0.06
print("sinpix(x)=",sinpix(x)) #Print the wavefunction numbers

#Now let's create out first plot
fig, ax = plt.subplots()  #I like to make plots using this silly fig,ax method but plot how you like
ax.plot(x,sinpix(x),linewidth=3)
ax.set_title(r"$\psi(x)=\sin(\pi x)$ for $0\leq x \leq 2$")
ax.set_xlabel("$x$")
ax.set_ylabel("$\psi(x)$")
ax.grid()

# ### Probability density
# We saw in the lectures that the probability density associated with the waveform is:
# $$ \rho(x) = \left| \psi(x) \right|^2$$
# so in our case we have
# $$\rho(x)=\left| \psi(x) \right|^2 =\begin{cases}
# \sin^2(\pi x) & \text{if } 0 \leq x \leq 2 \\
# 0 & \text{otherwise}
# \end{cases}$$
#
# So now we can plot this function

#Now let's create our next plot, and zoom in to the non-zero region
x = np.linspace(-0.5, 2.5, 100)
fig, ax = plt.subplots()  #I like to make plots using this silly fig,ax method but plot how you like
ax.plot(x,sinpix(x)**2,linewidth=3)
ax.set_title(r"$\rho(x)=\sin^2(\pi x)$ for $0\leq x \leq 2$")
ax.set_xlabel("$x$")
ax.set_ylabel(r"$\rho(x)$")
ax.grid()

# ## de Broglie Wavefunctions
#
# In the lecture we saw that our toy wavefunctions above are actually examples of de Broglie wavefunctions which represent particles in 1-D space with momentum $p$:
# $$ \psi(x) = \sin \left( \frac{p x}{\hbar} \right) $$

#Now let's create our next plot, and zoom in to the non-zero region
x = np.linspace(-20, 20, 2000)
fig, ax = plt.subplots()  #I like to make plots using this silly fig,ax method but plot how you like
ax.plot(x,np.sin(x),linewidth=1)
ax.set_title(r"$\psi(x)=\sin(p x / \hbar)$")
ax.set_xlabel("$x (p/\hbar)$")
ax.set_ylabel("$\psi(x)$")
ax.grid()
ax.set_xlim(-20,20)

# ## Can you normalise a de Broglie wavefunction
# This wavefunction is defined as a sin function for all values of $x$. So the probability density will also be defined for all values of $x$:
# $$ \rho(x) = \sin \left( \frac{p x}{\hbar} \right)^2 $$
# and will be positive definite for all values (except for when it is 0).
#
# Which means that if we integrate this wavefunction from $-\infty$ to $+\infty$ the area under the curve will be $\infty$!!!

#Obviously this plot isn't infinite... but if it were then the area under the curve would be infinite
fig, ax = plt.subplots()  #I like to make plots using this silly fig,ax method but plot how you like
x = np.linspace(-20, 20, 2000)
ax.fill_between(x,y1=np.sin(x)**2,y2=0)
ax.set_title(r"$\rho(x)=\sin(p x / \hbar)$ ")
ax.set_xlabel(r"$x (p/\hbar)$")
ax.set_ylabel(r"$\rho(x)$")
ax.grid()
ax.set_xlim(-20,20)


# ## Wavepackets

# +
def addSin(x,n,delta):
    y=np.sin(x)
    f=1
    if n>0:
        for i in range(n):
            f=f-delta
            y=y+np.sin(f*x)
    return y/(n+1)

fig, ax = plt.subplots()  #I like to make plots using this silly fig,ax method but plot how you like
x = np.linspace(-20, 20, 2000)
for i in range(50):
    ax.plot(x,addSin(x,i,-0.1))
ax.set_title(r"Multiple $\sin$s")
ax.set_xlabel("$x$")
ax.set_ylabel(r"$\psi(x)$")
ax.grid()

# -


