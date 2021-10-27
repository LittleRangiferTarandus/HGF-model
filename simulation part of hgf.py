import math 
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

#HUMAN NEUROSCIENCE上面的文章A Bayesian foundation for individual learning under uncertainty，
# #讨论了HGF模型，模拟她我写出了一个仿真模型

#下面-1这个值纯属占位，是用不上的
# rawData = [-1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,0,0,0,0,1,0,0,1,0,
# 0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,1,1,0,1,1,1,0,0,1,0,0,1,0,1,1,0,1,1,1,1,1,1,0,0,1,1,1,
# 1,0,1,1,1,1,0,1,0,0,0,1,0,1,1,0,1,1,1,1,1,0,1,0,1,1,1,0,1,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,
# 1,1,0,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,0,1,1,1,0,1,1,1,0,0,0,1,0,0,0,0,1,1,0,0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,
# 0,1,0,1,1,1,1,1,0,1,0,0,1,0,1,1,0,0,1,1,1,1,1,1,0,1,0,0,0,0,1,0,1,1,0,0,1,1,1,0,0,1,1,1,0,1,1,0,0,1,1,0,1,1,0,1,1,
# 0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0]
#HGF toolbox的模拟数据，先放着吧

#下面-1这个值纯属占位，是用不上的
rawData = [-1]
rawData +=  np.random.binomial(1, 0.5, 100).tolist()
rawData += np.random.binomial(1, 0.9, 20).tolist()
rawData += np.random.binomial(1, 0.1, 20).tolist()
rawData += np.random.binomial(1, 0.9, 20).tolist()
rawData += np.random.binomial(1, 0.1, 20).tolist()
rawData += np.random.binomial(1, 0.9, 20).tolist()
rawData += np.random.binomial(1, 0.1, 20).tolist()
rawData += np.random.binomial(1, 0.5, 100).tolist()
#根据文章生成的模拟数据↑

def realP(x):#真实概率
  if x<100:
    return 0.5
  elif x<120:
    return 0.9
  elif x<140:
    return 0.1
  elif x<160:
    return 0.9
  elif x<180:
    return 0.1
  elif x<200:
    return 0.9
  elif x<220:
    return 0.1
  else:
    return 0.5

def sigmoid(x):
    return 1.0/(1+math.exp(-x))

class HGF:
  def __init__(self, kappa , omega,theta): 
    self.theta = theta
    self.kappa=(kappa)
    self.omega=omega
    
    self.muhat=np.full((3,len(rawData)), np.nan)
    self.pihat=np.full((3,len(rawData)), np.nan)
    self.sigmahat=np.full((3,len(rawData)), np.nan)
    self.mu=np.full((3,len(rawData)), np.nan)
    self.pi=np.full((3,len(rawData)), np.nan)
    self.sigma=np.full((3,len(rawData)), np.nan)
    self.delta=np.full((3,len(rawData)), np.nan)
    self.r=np.full((3,len(rawData)), np.nan)
    self.w=np.full((3,len(rawData)), np.nan)

    self.initialize()
  def initialize(self):
    self.muhat[0,0] = 1
    self.mu[0,0] = 0
    self.mu[1,0] = 0
    self.mu[2,0] = 0
    self.sigma[1,0] = 1
    self.sigma[2,0] = 1

  def update(self):
    for k in range(1,len(rawData)):
      self.mu[0,k] = rawData[k]

      #level 2
      #-sigma2
      #--sigma2hat&pi2hat
      self.sigmahat[1,k] = self.sigma[1,k-1]+math.exp(self.kappa*self.mu[2,k-1]+self.omega)
      self.pihat[1,k] = 1/self.sigmahat[1,k]
      #--sigma1hat
      #---mu1hat
      self.muhat[0,k] = sigmoid(self.mu[1,k-1])
      #--sigma1hat&pi1hat
      self.sigmahat[0,k] = self.muhat[0,k-1]*(1-self.muhat[0,k-1])
      self.pihat[0,k] = np.inf if(self.sigmahat[0,k]==0) else 1/self.sigmahat[0,k]
      #-sigma2&pi2
      self.sigma[1,k] = 1/(1/self.sigmahat[1,k]+self.sigmahat[0,k])
      self.pi[1,k] = 1/self.sigma[1,k]
      #-mu2
      #--delta1
      self.delta[0,k] = self.mu[0,k] - self.muhat[0,k]
      #-mu2
      self.mu[1,k] = self.mu[1,k-1] + self.sigma[1,k]*self.delta[0,k]

      #level 3
      #-sigma3&pi3
      #--sigma3hat&pi3hat
      self.pihat[2,k] = 1/(self.sigma[2,k-1]+self.theta)
      self.sigmahat[2,k] = 1/self.pihat[2,k]
      #--w2
      self.w[1,k] = math.exp(self.kappa*self.mu[2,k-1]+self.omega)/(self.sigma[1,k-1]+math.exp(self.kappa*self.mu[2,k-1]+self.omega))
      #--r2
      self.r[1,k] = (math.exp(self.kappa*self.mu[2,k-1]+self.omega)-self.sigma[1,k-1])/(self.sigma[1,k-1]+math.exp(self.kappa*self.mu[2,k-1]+self.omega))
      #--delta2
      self.delta[1,k] = (self.sigma[1,k]+(self.mu[1,k]-self.mu[1,k-1])**2)/(self.sigma[1,k-1]+math.exp(self.kappa*self.mu[2,k-1]+self.omega))-1
      #-pi3&sigma3
      self.pi[2,k] = self.pihat[2,k] + self.kappa**2/2 * self.w[1,k]*(self.w[1,k]+self.r[1,k]*self.delta[1,k])
      self.sigma[2,k] = 1/self.pi[2,k]
      #-mu3
      self.mu[2,k] = self.mu[2,k-1] + self.sigma[2,k] * self.kappa/2 *self.w[1,k] *self.delta[1,k]


  def output(self):
    return {
      "muhat":self.muhat,
      "pihat":self.pihat,
      "sigmahat":self.sigmahat,
      "mu":self.mu,
      "pi":self.pi,
      "sigma":self.sigma,
      "delta":self.delta,
      "r":self.r,
      "w":self.w
    }

if __name__ == '__main__':
  instance =   HGF(1.4,-2.2,0.05)#文章里面这是自由参数，但是可以用一种算法估计出最优的自由参数，暂略
  instance.update()

  ans = instance.output() 
 # print(ans)
  length = len(ans["mu"][0,1:])
  x=range(length)
  tempIndex = 0
  y=[]
  while(tempIndex<length):
    y.append(realP(x[tempIndex]))
    tempIndex += 1

  plt.scatter(x,ans["mu"][0,1:])
  plt.plot(x,ans["muhat"][0,1:])
  plt.title("μ1=inputData,μ1hat=s(μ2)")
  plt.plot(x,y)
  plt.show()

  plt.plot(x,ans["mu"][1,1:])
  plt.title("μ2")
  plt.ylim(-4,4)
  plt.show()

  plt.plot(x,ans["mu"][2,1:])
  plt.title("μ3")
  plt.ylim(-4,4)
  plt.show()

    