import numpy as np
import matplotlib.pyplot as plt
import math

##Uniform
def uniform(a,b):                                                  # b>a
  return 1/(b-a)
a1= float(input())
b1= float(input())
N1=round(1000*(1/(b1-a1)))
X1=np.linspace(a1,b1,N1)
Y1=np.array([uniform(a1,b1) for x in X1])

a2= float(input())
b2= float(input())
N2=round(1000*(1/(b2-a2)))
X2=np.linspace(a2,b2,N2)
Y2=np.array([uniform(a2,b2) for x in X2])

plt.plot(X1,Y1,label=f"{a1},{b1}")
plt.plot(X2,Y2,label=f'{a2},{b2}')

plt.xlabel('x')
plt.ylabel('P[x]')
plt.legend()
plt.title("Uniform Distribution")
plt.show()


##Gaussian
def gaussian(x,a,b):                                 ## a=  mu  b=  sigma^2
  p=1/(pow(2*math.pi,0.5)*b)
  q= math.exp(-((x-a)**2)/2*b)
  Y=p*q
  return Y
a1= float(input("enter mu"))
b1= float(input("enter sigma^2"))
a2=float(input("enter mu"))
b2=float(input("sigma^2"))
n=float(input())
X1=np.linspace(-(n*pow(b1,0.5)),n*pow(b1,0.5),100000)
X2=np.linspace(-(n*pow(b2,0.5)),n*pow(b2,0.5),100000)
plt.subplot(2,1,1)
Y1=np.array([gaussian(x,a1,b1) for x in X1])
plt.fill_between(X1,Y1,alpha=0.5)
plt.xlabel('x')
plt.ylabel('P[x]')
plt.plot(X1,Y1)

plt.subplot(2,1,2)
Y2=np.array([gaussian(x,a2,b2)for x in X2])
plt.fill_between(X2,Y2,color="orange")
plt.plot(X2,Y2)
plt.xlabel('x')
plt.ylabel('P[x]')
plt.title("Gaussian Distribution")
plt.tight_layout()
plt.show()

##exponential
def exponential(x,l):                  ## l= lambda
  return l*math.exp(-l*x)
l1=float(input())
l2=float(input())
n=10*min(l1,l2)
X=np.linspace(0,n ,100000)
Y1=np.array([exponential(x,l1) for x in X])
Y2=np.array([exponential(x,l2) for x in X])
plt.plot(X,Y1,label=f'l1={l1}')
plt.plot(X,Y2,label=f'l2={l2}')
plt.legend()
plt.xlabel('x')
plt.ylabel('P[x]')
plt.title("Exponential distribution")
plt.show()

##Cauchy
def cauchy(x,x0,gamma):
  p= pow((x-x0)/gamma,2)
  q= math.pi*gamma*(1+p)
  return 1/q
for i in range(2):
  x0=float(input("Enter x0"))
  gamma= float(input("Enter Gamma"))
  X=np.linspace(-10,10 ,1000)
  Y=np.array([cauchy(x,x0,gamma) for x in X])
  plt.plot(X,Y,label= f'x0={x0}, gamma={gamma}')
plt.title("Cauchy Distribution")
plt.xlabel('x')
plt.ylabel('P[x]')
plt.legend()
plt.show()


##Laplacian
def laplacian(x,mu,b):
  return (1/(2*b))*math.exp(-abs(x-mu)/b)
for i in range(2):
  mu=float(input("Enter mu"))
  b= float(input("Enter b"))
  X=np.linspace(-10,10 ,1000)
  Y=np.array([laplacian(x,mu,b) for x in X])
  plt.plot(X,Y,label= f'mu={mu}, b={b}')
plt.title("Laplacian Distribution")
plt.xlabel('x')
plt.ylabel('P[x]')
plt.legend()
plt.show()


#Rayleigh
def rayleigh(x,sigma):
  return x/pow(sigma,2)* math.exp(-pow(x,2)/(2*pow(sigma,2)))
for i in range(3):
  sigma=float(input("Enter sigma"))
  X=np.linspace(0,10 ,1000)
  Y=np.array([rayleigh(x,sigma) for x in X])

  plt.plot(X,Y,label= f'sigma={sigma}')
plt.title("Rayleigh  Distribution")
plt.xlabel('x')
plt.ylabel('P[x]')
plt.legend()
plt.show()


#CDF UNIFORM DISTRIBUTION
def cuniform(x,a,b):
  if x<=a:
    return 0
  elif x<=b:
     return (x-a)/(b-a)
  else :
    return 0

a=float(input())
b=float(input())
X=np.linspace(0,b,10000)
Y=np.array([cuniform(x,a,b) for x in X])                       #for cdf

Y1=np.array([uniform(a,b) for x in X])
Z=Y1/np.sum(Y1)
CY1=np.cumsum(Z)

plt.subplot(2,1,1)
plt.xlabel('x')
plt.ylabel('P[X<=x]')
plt.title("CDF Uniform Distribution")
plt.plot(X,Y)

plt.subplot(2,1,2)
plt.plot(X,CY1)
plt.title("CDF Uniform Distribution: cumsum")
plt.xlabel('x')
plt.ylabel('P[X<=x]')
plt.tight_layout()
plt.show()

## CDF GAUSSIAN DISTRIBUTION
def cgaussian(x,mu,sigmasq):                                 ## a=  mu  b=  sigma^2
  return 0.5*(1 + math.erf((x-mu)/pow(2*sigmasq,0.5)))
mu= float(input("ente mu"))
sigmasq= float(input("enter sigma^2"))
# n=float(input())
X=np.linspace(-4,4,1000)
Y=np.array([cgaussian(x,mu,sigmasq) for x in X])             #for cdf
Y1=np.array([gaussian(x,mu,sigmasq) for x in X])
Z=Y1/np.sum(Y1)
CY1=np.cumsum(Z)

plt.subplot(2,1,1)
plt.plot(X,Y)
plt.xlabel('x')
plt.ylabel('P[X<=x]')
plt.title("CDF Gaussian Distribution")

plt.subplot(2,1,2)
plt.plot(X,CY1)
plt.xlabel('x')
plt.ylabel('P[X<=x]')
plt.title("CDF Gaussian Distribution: cumsum")

plt.tight_layout()
plt.show()

#CDF EXPONENTIAL DISTRIBUTION
def cexponential(x,l):                       #l= lambda
    return 1-math.exp(-x*l)
l1=float(input())
l2=float(input())
n=10*min(l1,l2)
X=np.linspace(0,n ,100000)
Y1=np.array([cexponential(x,l1) for x in X])
Y2=np.array([cexponential(x,l2) for x in X])

cumY1=np.array([exponential(x,l1) for x in X])
cumY2=np.array([exponential(x,l2) for x in X])
Z1=cumY1/np.sum(cumY1)
CY1=np.cumsum(Z1)
Z2=cumY2/np.sum(cumY2)
CY2=np.cumsum(Z2)

plt.subplot(2,1,1)
plt.plot(X,Y1,label=f'l1={l1}')
plt.plot(X,Y2,label=f'l2={l2}')
plt.xlabel('x')
plt.ylabel('P[X<=x]')
plt.legend()
plt.title("CDF Exponential distribution")

plt.subplot(2,1,2)
plt.plot(X,CY1,label=f'l1={l1}')
plt.plot(X,CY2,label=f'l2={l2}')
plt.xlabel('x')
plt.ylabel('P[X<=x]')
plt.legend()
plt.title("CDF Exponential distribution: cumsum")

plt.tight_layout()
plt.show()

##CDF CAUCY DISTRIBUTION
def ccauchy(x,x0,gamma):
  return (1/math.pi)*math.atan((x-x0)/gamma) + 0.5

for i in range(2):
  x0=float(input("Enter x0"))
  gamma= float(input("Enter Gamma"))
  X=np.linspace(-4,4 ,1000)
  Y=np.array([ccauchy(x,x0,gamma) for x in X])
  Y1=np.array([cauchy(x,x0,gamma) for x in X])
  plt.subplot(2,1,1)
  plt.plot(X,Y,label= f'x0={x0}, gamma={gamma}')
  plt.xlabel('x')
  plt.ylabel('P[X<=x]')
  plt.title("CDF Cauchy Distribution")
  plt.legend()

  plt.subplot(2,1,2)
  Y1=Y1/np.sum(Y1)
  Z=np.cumsum(Y1)
  plt.plot(X,Z,label= f'x0={x0}, gamma={gamma}')
  plt.xlabel('x')
  plt.ylabel('P[X<=x]')
  plt.title("CDF Cauchy Distribution:cumsum")
  plt.legend()
  plt.tight_layout()
plt.show()


##CDF LAPLACIAN DISTRIBUTION
def Claplacian(x,mu,b):
  if x<mu:
    return 0.5*(math.exp((x-mu)/b))
  else:
    return 1-0.5*(math.exp((x-mu)/-b))
for i in range(2):
  mu=float(input("Enter mu"))
  b= float(input("Enter b"))
  X=np.linspace(-10,10 ,1000)
  Y=np.array([Claplacian(x,mu,b) for x in X])
  Y1=np.array([laplacian(x,mu,b) for x in X])

  plt.subplot(2,1,1)
  plt.plot(X,Y,label= f'mu={mu}, b={b}')
  plt.title("CDF Laplacian Distribution")
  plt.xlabel('x')
  plt.ylabel('P[X<=x]')
  plt.legend()

  plt.subplot(2,1,2)
  Y1=Y1/np.sum(Y1)
  Z=np.cumsum(Y1)
  plt.plot(X,Z,label=f'mu={mu}, b={b}')
  plt.title("CDF laplacian:cumsum")
  plt.xlabel('x')
  plt.ylabel('P[X<=x]')
  plt.legend()
  plt.tight_layout()
plt.show()

##CDF RAYLEIGH DISTRIBUTION
def Crayleigh(x,sigma):
  return 1- math.exp(-pow(x/sigma,2)/2)
for i in range(4):
  sigma=float(input("Enter sigma"))
  X=np.linspace(0,10 ,1000)
  Y=np.array([Crayleigh(x,sigma) for x in X])
  Y1=np.array([rayleigh(x,sigma) for x in X])

  plt.subplot(2,1,1)
  plt.plot(X,Y,label= f'sigma={sigma}')
  plt.title(" CDF Rayleigh Distribution")
  plt.xlabel('x')
  plt.ylabel('P[X<=x]')
  plt.legend()

  plt.subplot(2,1,2)
  Z=Y1/np.sum(Y1)
  CY=np.cumsum(Z)
  plt.plot(X,CY,label= f'sigma={sigma}')
  plt.title("CDF Rayleigh Distribution: cumsum")
  plt.xlabel('x')
  plt.ylabel('P[X<=x]')
  plt.legend()
  plt.tight_layout()
plt.show()

##CDF of Dicrete functions
def fact(z):
  factorial=1
  for i in range(1,z+1):
    factorial*=i
  return factorial

#Discrete Uniform Distribution CDF
a=int(input())
b=int(input())
x=np.arange(a,b+1)
k=b-a+1
y=np.array([(1/(k))for a in x])
y1=np.cumsum(y)
plt.plot(x,y1,drawstyle='steps-post')
plt.title("Discrete Uniform Distribution CDF")
plt.xlabel("x")
plt.ylabel("P[X<=x]")
plt.show()


#Bernoullis Random Variable CDF
p=float(input())
def bernoulli(p):
  x=[0,1]
  y=[1-p,p]
  y=y/np.sum(y)
  z=np.cumsum(y)
  plt.plot(x,z,drawstyle='steps-post')
  plt.xlabel('x')
  plt.ylabel('P[X<=x]')
  plt.title('Bernoullis Random Variable CDF')
bernoulli(p)
plt.show()


#Binomial CDF
M=int(input())
p=float(input())
a=[]
b=[]
def Binomial(p,M):
  for k in range(0,M+1):
    a.append(k)
    c=fact(M)/(fact(k)*fact(M-k))
    y=c*(pow(p,k))*(pow(1-p,M-k))
    b.append(y)
Binomial(p,M)
x=np.array(a)
y=np.array(b)
y/=np.sum(y)
z=np.cumsum(y)
plt.xlabel('x')
plt.ylabel('P[X<=x]')
plt.plot(x,z,drawstyle='steps-post')
plt.title('Binomial CDF')
plt.show()


#Poisson CDF
a=[]
b=[]
def poisson(M,p):
  k=np.arange(M)
  for i in k:
    y=(pow(p*M,i)*np.exp(-(p*M)))/fact(i)
    a.append(i)
    b.append(y)
M=int(input())
p=float(input())
poisson(M,p)
x=np.array(a)
y=np.array(b)
y/=np.sum(y)
z=np.cumsum(y)
plt.xlabel('x')
plt.ylabel('P[X<=x]')
plt.plot(x,z,drawstyle="steps-post")
plt.title('Poisson CDF')
plt.show()

#Geometric CDF
c=[]
b=[]
def Geometric(p,k):
  a=np.arange(1,k+1)
  for i in a:
    y=pow(1-p,i-1)*p
    c.append(i)
    b.append(y)
p=float(input())
k=int(input())
Geometric(p,k)
x=np.array(c)
y=np.array(b)
y/=np.sum(y)
z=np.cumsum(y)
plt.xlabel('x')
plt.ylabel('P[X<=x]')
plt.plot(x,z,drawstyle="steps-post")
plt.title('Geometric CDF')
plt.show()