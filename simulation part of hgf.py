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
    
    self.miu3 = []
    self.miu2 = []
    self.miu3.append(0)
    self.miu2.append(0) #
    
    self.sigma2=[]
    self.sigma2.append(1)#同上，

    self.miu1P=[]
    self.miu1P.append(1) #同上

    self.sigma3=[]
    self.sigma3.append(1) #同上


    self.step = 1

  def miu3Update(self):
    kappa = self.kappa

    tempVal = math.exp(kappa*self.miu3[self.step-1]+self.omega)

    w2 = tempVal/(self.sigma2[self.step-1]+tempVal)

    delta2 = (self.sigma2[self.step]+(self.miu2[self.step]-self.miu2[self.step-1])**2)/(self.sigma2[self.step-1]+tempVal)-1

    #update the sigma3 list
    #pi3P
    pi3P = 1/(self.sigma3[self.step-1]+self.theta)
    #r2
    r2 =( tempVal-self.sigma2[self.step-1])/(self.sigma2[self.step-1]+tempVal)

    #pi3
    pi3 = pi3P+kappa**2/2*w2*(w2+r2*delta2)
    
    tempSigma3 = 1/pi3

    self.sigma3.append(tempSigma3)

    differenceOfmiu3=tempSigma3*kappa*w2*delta2/2
    self.miu3.append(self.miu3[self.step-1]+differenceOfmiu3)
    
  def miu2Update(self): 
    sigma2P=self.sigma2[self.step-1]+math.exp(self.kappa*self.miu3[self.step-1]+self.omega)
    
    miu1P = sigmoid(self.miu2[self.step-1])
    self.miu1P.append(miu1P)
    
    sigma1P = self.miu1P[self.step-1]*(1-self.miu1P[self.step-1])

    tempSigma2 = 1/(1/sigma2P+sigma1P)
    self.sigma2.append(tempSigma2)

    delta1 = rawData[self.step]-self.miu1P[self.step]

    differenceOfmiu2 = tempSigma2*delta1

    self.miu2.append(self.miu2[self.step-1]+differenceOfmiu2)
  
  
  def classUpdate(self):
    while(self.step<len(rawData)):
      self.miu2Update()
      self.miu3Update()
      self.step+=1
  def output(self):
    return {
      "mu1_rawData":[rawData[1:]],
      "mu1_hat":[self.miu1P[1:]],
      "mu2_3":[self.miu2[1:],self.miu3[1:]],
      "sigma2_3":[self.sigma2[1:],self.sigma3[1:]]
    }

if __name__ == '__main__':
  instance =   HGF(1.4,-2.2,0.5)#文章里面这是自由参数，但是可以用一种算法估计出最优的自由参数，暂略
  instance.classUpdate()

  ans = instance.output() 
  length = len(ans["mu1_rawData"][0])
  x=range(length)
  tempIndex = 0
  y=[]
  while(tempIndex<length):
    y.append(realP(x[tempIndex]))
    tempIndex += 1

  plt.scatter(x,ans["mu1_rawData"][0])
  plt.plot(x,ans["mu1_hat"][0])
  plt.plot(x,y)
  plt.show()

  plt.plot(x,ans["mu2_3"][0])
  
  plt.plot(x,ans["sigma2_3"][0]) 
  plt.show()

  plt.plot(x,ans["mu2_3"][1])
  plt.plot(x,ans["sigma2_3"][1])
  plt.show()

    