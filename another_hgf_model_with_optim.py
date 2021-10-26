import math 
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

#这是根据HGF工具箱写的代码，使用了tapas_hgf_binary_config和tapas_bayes_optimal_binary_config这两个文件里面的模型，仿照demo的第一个例子

#下面nan这个值纯属占位，是用不上的
rawData = [np.nan,1,1,1,1,0,1,1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,0,0,0,0,1,0,0,1,0,
0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,1,1,0,1,1,1,0,0,1,0,0,1,0,1,1,0,1,1,1,1,1,1,0,0,1,1,1,
1,0,1,1,1,1,0,1,0,0,0,1,0,1,1,0,1,1,1,1,1,0,1,0,1,1,1,0,1,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,
1,1,0,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,0,1,1,1,0,1,1,1,0,0,0,1,0,0,0,0,1,1,0,0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,
0,1,0,1,1,1,1,1,0,1,0,0,1,0,1,1,0,0,1,1,1,1,1,1,0,1,0,0,0,0,1,0,1,1,0,0,1,1,1,0,0,1,1,1,0,1,1,0,0,1,1,0,1,1,0,1,1,
0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0]
#HGF toolbox的模拟数据，先放着吧

def sigmoid(x):
  return 1.0/(1+math.exp(-x))

class HGF:
  def __init__(self, kappa , omega,theta):
    self.kappa = (kappa)
    self.omega = omega
    self.theta = math.exp(theta)
    #sa_0: [NaN 0.1000 1]

    self.muhat=np.full((3,len(rawData)), np.nan)
    self.pihat=np.full((3,len(rawData)), np.nan)
    self.mu=np.full((3,len(rawData)), np.nan)
    self.pi=np.full((3,len(rawData)), np.nan)
    self.da=np.full((3,len(rawData)), np.nan)
    self.v=np.full((3,len(rawData)), np.nan)
    self.w=np.full((3,len(rawData)), np.nan)

    self.initialize()
  
  def initialize(self):
    self.mu[1,0] = 0
    self.mu[2,0] = 1
    self.pi[0,0] = np.inf
    self.pi[1,0] = 10 
    self.pi[2,0] = 1

  def update(self):
    for i in range(1,len(rawData)):
      self.muhat[1,i] = self.mu[1,i-1]

      self.muhat[0,i] = sigmoid(self.kappa*self.muhat[1,i])
      self.pihat[0,i] = 1/(self.muhat[0,i]*(1-self.muhat[0,i]))

      self.pi[0,i] = np.inf
      self.mu[0,i] = rawData[i]

      self.da[0,i] = self.mu[0,i] - self.muhat[0,i]

      self.pihat[1,i] = 1/(1/self.pi[1,i-1]+math.exp(self.kappa*self.mu[2,i-1]+self.omega))

      self.pi[1,i] = self.pihat[1,i] + self.kappa**2/self.pihat[0,i]
      self.mu[1,i] = self.muhat[1,i] + self.kappa/self.pi[1,i]*self.da[0,i]

      self.da[1,i] = (1/self.pi[1,i] + (self.mu[1,i]-self.muhat[1,i])**2) * self.pihat[1,i]-1

      self.muhat[2,i] = self.mu[2,i-1]  

      self.pihat[2,i] = 1/(1/self.pi[2,i-1]+self.theta)

      self.v[2,i] = self.theta
      self.v[1,i] = math.exp(self.kappa*self.mu[2,i-1]+self.omega)
      self.w[1,i] = self.v[1,i]*self.pihat[1,i]

      self.pi[2,i] = self.pihat[2,i] +0.5*self.kappa**2 * self.w[1,i]*(self.w[1,i]+(2*self.w[1,i]-1)*self.da[1,i])

      self.mu[2,i] = self.muhat[2,i] +0.5 /self.pi[2,i] *self.kappa* self.w[1,i]*self.da[1,i]
      
      self.da[2,i] = (1/self.pi[2,i]+(self.mu[1,i]-self.muhat[1,i])**2)*self.pihat[2,i] -1

  def output(self):
    return {
      "muhat":self.muhat,
      "pihat":self.pihat,
      "mu":self.mu,
      "pi":self.pi,
      "da":self.da,
      "v":self.v,
      "w":self.w,
    }

class optim:

  def parameterUpdate(self,omega2=-3,omega3=-6):
    """
    #toolbox的hgf binary代码看的我一脸懵逼，总体来说是酱紫的：只有omega2、3（或者叫做omega和theta）是会更新的（即lawson的omega2和omega3），kappa===1，rho（本程序没有提到）===0
    #这部分是根据toolbox写的
    #
    #c.ommu = [NaN,  -3,  -6];
    #c.omsa = [NaN, 4^2, 4^2];
    #ptrans_prc(prc_idx)===r.c_prc.priormus(prc_idx)  (?)#存疑
    """
    #logPrcPriors = -1/2.*log(8*atan(1).*r.c_prc.priorsas(prc_idx)) - 1/2.*(ptrans_prc(prc_idx) - r.c_prc.priormus(prc_idx)).^2./r.c_prc.priorsas(prc_idx);
    #logPrcPrior  = sum(logPrcPriors);
    def logPrcPriors(mu,sigma):
      return -0.5*math.log(8*math.atan(1)*sigma) -0.5*(mu-mu)**2/sigma
      #mu-mu....我应该没看错，原来的代码是把ptrans_prc赋值为init，init等于prc.priormus
    logPrcPrior = logPrcPriors(-3,8)+logPrcPriors(-6,8)

    #logObsPriors = -1/2.*log(8*atan(1).*r.c_obs.priorsas(obs_idx)) - 1/2.*(ptrans_obs(obs_idx) - r.c_obs.priormus(obs_idx)).^2./r.c_obs.priorsas(obs_idx);
    #logObsPrior  = sum(logObsPriors); 和上面一样，lawson的模型里面，这一项应该是0(?)
    logObsPrior=0

    #new 一个 HGF实例对象（找不到对象？不会自己new一个吗？）
    #trialLogLls = obs_fun(r, infStates, ptrans_obs); 用到的只有mu1_hat这一列数据
    instance = HGF(1,omega2,omega3)
    instance.update()
    ans = instance.output()
    x=ans["muhat"][0][1:]
    u=ans["mu"][0][1:]
    
    
    LogLl = 0
    for i in range(len(x)):
      LogLl+=u[i]*math.log(x[i])+(1-u[i])*math.log((1-x[i]))
    
    #LogJoint
    LogJoint = LogLl+logPrcPrior+logObsPrior

    return -LogJoint
      

  ######模拟退火
  """
      说明：事实证明，这个目标函数parameterUpdate，跑出来的最优值，有极小可能性，可能是一些很奇怪的数，例如负几百的或者十分接近0的x
      因为模拟退火算法可以全局随机搜索近似最优解，所以有可能遇到上述这些很奇怪的结果
      如果遇到这种情况，xNew的更新范围写小点，然后请多跑几遍。
      或许这里用其他算法好一些，喵🐱
  """
  def SAA(self):
    
    T=10#initiate temperature
    Tmin=1 #minimum value of terperature
    x=[-3,-6]#initiate x
    k=50 #times of internal circulation 
    y=0#initiate result
    t=0#time
    while T>=Tmin:
      for i in range(k):
        #calculate y
        y=opt.parameterUpdate(x[0],x[1])
        #generate a new x in the neighboorhood of x by transform function
        
        while True:
          xNew=[x[0]+np.random.uniform(low=-0.0055,high=+0.0055)*T,x[1]+np.random.uniform(low=-0.0055,high=+0.0055)*T]

          try:#防止随机出一些越界的变量x，例如math.log和math.exp越界
            yNew=opt.parameterUpdate(xNew[0],xNew[1])
          except :
            pass
          else :
            break
        
        print("xOrigin = ")
        print(x)
        print("yOrigin = ")
        print(y)
        print("xNew = ")
        print(xNew)
        print("yNew = ")
        print(yNew)
        print("-----------------------分割线-----------------------\n")

        if yNew-y<0:
          x=xNew
        else:
          #metropolis principle
          p=math.exp(-(yNew-y)/T)
          r=np.random.uniform(low=0,high=1)
          if r<p:
            x=xNew
      t+=1
      T=10/(1+t)
    return {"x":x,"y":opt.parameterUpdate(x[0],x[1])}

if __name__ == '__main__':

  # #opitm👇
  opt = optim()
  opV = opt.SAA()
  print(opV["y"])
  instance =   HGF(1,opV["x"][0],opV["x"][1])
  instance.update()
  ans = instance.output()

  length = len(ans["mu"][0])
  x=range(length)

  plt.scatter(x,ans["mu"][0])
  plt.plot(x,ans["muhat"][0])

  plt.show()

  # # #测试simulation和optimization的代码↑
  


  