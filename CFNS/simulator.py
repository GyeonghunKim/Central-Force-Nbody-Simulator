# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from CFNS.particles import setup
import re
import unittest
""" 궤도에 색 gradient를 넣을 때 사용합니다. 본 프로젝트의 범위(numpy and matplotlib)을 벗어나서 주석처리 했습니다. 
from cycler import cycler
"""
from mpl_toolkits.mplot3d import axes3d


class simulator:
    '''
    주어진 setup에 대해서 시뮬레이션 계산을 하는 부분입니다.
    setup이 particle들의 초기 상태를 지정해주고, 시뮬레이션에 사용할 시간 간격을 지정해주는 부분이었다면,
    simulator에서는 힘의 형태를 셋업해줄 수 있고 이를 기반으로 상호작용을 계산해서 time evolution을 시켜줍니다.
    '''
    def __init__(self, setup):
        """ simulator class의 constructor입니다. 
        constructor는 setup만을 입력받도록 했습니다. 초기 힘은 "-1/(x**2)"을 default로 두었고, 이어지는 함수에서
        :param setup setup: 사전의 시뮬레이션에 사용 될 입자들의 초기 조건이 명시 된 setup class입니다.
        """
        self.setup = setup
        # 디폴트 힘입니다.
        self.force_string = "-1/(x**2)"

    def set_force(self, force_string, params = {}):
        """set_force는 물체간에 상호작용하는 중심력의 크기를 구하는 함수를 지정해주는 부분입니다.
        입력 parameter로는 force_string과 force_string에 들어가는 파라미터를 넣어주는 방식으로 params dictionary를 받습니다.

        :param string force_string: 힘을 계산할 때 사용할 식을 파이썬 문법에 맞게 입력한 string입니다. 이때 두 물체 사이의 거리는 'x'를 이용해야 합니다. 아래 params의 key로 주어진 한 글자 문자를 parameter로 사용할 수 있습니다. 이때 상호작용하는 두 물체의 질량을 나타내는 문자는 m1, m2로 고정되어 있습니다.
        :param dictionary params: 사전의 시뮬레이션에 사용 될 입자들의 초기 조건이 명시 된 setup class입니다. 파라미터는 편의를 위해서, 또 일반적인 경우에 그러므로, 한 글자라고 가정했습니다.
        """
        self.params = params
        # params 딕셔네리를 기반으로 식을 수정(정규표현식을 이용)
        for one_key in params.keys():
            force_string = re.sub(one_key, str(params[one_key]), force_string)
        self.force_string = force_string
        # task: parameter를 입력받을 수 있게 해야합니다.
        # 예를 들어서 force_string = "-k * x**2)"이라면, k를 입력받을 수 있게 해주어야 합니다.


    def __force__(self, x1, x2): # 1이 2에게 주는 힘
        """두 위치에 대해서 힘을 계산하는 클래스입니다.
        외부에서 사용하면 문제를 일으킬 수 있습니다. exec메소드 내부에서만 사용합시다.
        :param numpy array x1: 첫 번째 입자의 위치
        :param numpy array x2: 두 번째 입자의 위치
        """
        # 각 질량을 가져옵니다. 이는 이 메소드에서 사용이 안되는 것 처럼 보일 지 모르지만, force string에 m1, m2가 들어있을 경우에
        # 사용됩니다. 위의 constructor에서는 self.m1과 self.m2가 없으나, exec메소드에서 이 메소드를 사용하기 직전에 상호작용하는 물체에
        # 따라서 self.m1과 self.m2를 지정해줍니다.
        m1 = self.m1
        m2 = self.m2
        tp1 = np.array(x2) - np.array(x1) # 두 위치의 차이벡터입니다. (변위 벡터 입니다.)
        x = np.sqrt((tp1**2).sum()) # 두 위치의 차이의 크기입니다.
        force_size = eval(self.force_string) # 두 위치의 차이의 크기를 바탕으로 힘의 크기를 구했습니다.
        direction =  tp1 / x # [x2[i] - x1[i] for i in range(len(x1))] # 두 위치의 차이를 바탕으로 힘의 방향 단위벡터를 구했습니다.
        return direction * force_size # 힘의 크기와 방향벡터를 곱해서 힘 벡터를 구했습니다.

    # 
    # 
    def exec(self, iter):
        """실제로 입자들을 상호작용에 따라서 운동 시키는 부분입니다. 지정해준 횟수(iter)만큼 반복합니다.
        :param int iter: 계산을 반복할 횟수 
        """
        self.N_ptl = self.setup.getNparticle() # 입자의 수
        mass = np.array([particle.mass*np.ones(self.setup.dim) for particle in self.setup.particles])
        self.mass = mass # self.mass는 test를 위해서 self.mass로 하나 만들어 둡니다.
        # 각 입자의 질량으로 구성된 행렬입니다.
        # 질량 1인 입자 하나, 질량 2인 입자 하나가 있다면, [[1,1,1],[2,2,2]]와 같습니다.
        for i in range(iter):
            # force_matrix는 (i,j)성분이 j번째 입자가 i번째 입자에 주는 힘 벡터인 행렬입니다.
            # 구조로 보면 3차원 행렬이 되는 것 입니다.
            # 이하의 첫 줄에서 그러한 구조를 가진 행렬을 0으로 초기화 해주고, 이중 포문에서 자리를 채웁니다.
            force_matrix = [[np.zeros(self.setup.dim) for i in range(self.N_ptl)] for j in range(self.N_ptl)]
            for j in range(self.N_ptl):
                # 여기서 m1을 지정해줍니다. 이는 추후에 self.__force__메소드에서 사용하기 위함입니다.
                # 자세한 설명은 __force__ 메소드를 참조하시길 바랍니다.
                self.m1 = self.setup.particles[j].mass
                for k in range(self.N_ptl):
                    # 자기자신에게 주는 힘은 0입니다.
                    if j == k:
                        continue
                    # 여기서 m2을 지정해줍니다. 이는 추후에 self.__force__메소드에서 사용하기 위함입니다.
                    # 자세한 설명은 __force__ 메소드를 참조하시길 바랍니다.
                    self.m2 = self.setup.particles[k].mass
                    # 행렬 성분에 상호작용하는 힘 벡터를 넣어줍니다.
                    force_matrix[j][k] = np.array(self.__force__(self.setup.particles[k].location.data[i],
                                                             self.setup.particles[j].location.data[i]))
            # force_matrix를 np array로 바꾸어줍니다.
            force_matrix = np.array([np.array(line) for line in force_matrix])
            # force matrix를 두 번째 축으로 더하면 각 물체에 작용하는 알짜힘이 됩니다.
            net_force = force_matrix.sum(axis = 1)
            # 이렇게 만든 알짜힘 행렬(벡터의 벡터)는 앞의 질량 행렬과 같은 shape입니다. 이 둘을 componentwise하게 나누면
            # 가속도 벡터의 벡터가 나옵니다.
            acce = (net_force / mass)
            # 위의 가속도에 시간 간격을 곱해서 원래 속도에 더해주면, 나중 속도를 구할 수 있습니다.
            # 이때 setup 클래스에 설명이 있겠지만, setup.getvelocities()를 통해서 현재 속도를 가속도 벡터와 같은 shape으로 불러옵니다.
            vel = self.setup.getvelocities() + acce * self.setup.dt
            # vel을 각 입자의 속력으로 교환해줍니다.
            self.setup.setvelocities(vel)
            # 위의 속도에 시간 간격을 곱해서 원래 위치에 더해주면, 나중 위치를 구할 수 있습니다.
            # 이때 setup 클래스에 설명이 있겠지만, setup.getlastloc()를 통해서 현재 위치를 속도 벡터와 같은 shape으로 불러옵니다.
            loc = self.setup.getlastloc() + vel * self.setup.dt
            # vel과 같은 shape을 각 입자의 위치에 append해 줍니다.
            self.setup.appendloc(loc)

    """ cycler를 필요로 해서 주석 처리 했습니다. 
    def color_gradient(self):
        ''' 입자들의 움직임을 시간에 따라 다른 색깔을 가지는(그라데이션) 점으로 표시합니다.
        '''
        #2차원 벡터인 경우
        if self.setup.dim == 2:
            x_all = [] #각 벡터의 x좌표 변화만 모아놓을 list입니다. ex. x_all = [[1,2,3,4], [4,5,6,7], [7,8,9,10]]; 첫 번째 입자의 x좌표가 1,2,3,4으로 변화한 것입니다. 입자는 총 3개인 경우입니다.
            y_all = [] #각 벡터의 y좌표 변화만 모아놓을 list입니다.
            N_ptl = self.setup.getNparticle() # 입자의 수.

            #self에 저장되어 있는 x,y좌표를 각각 x_all, y_all에 저장합니다.
            for i in range(N_ptl):
                x_all.append([d[0] for d in self.setup.particles[i].location.data]) #x_all[i]: (i+1)번째 입자의 x좌표 변화가 저장된 벡터
                y_all.append([d[1] for d in self.setup.particles[i].location.data]) #y[i]: (i+1)번째 입자의 y좌표 변화가 저장된 벡터

            #plot을 생성합니다.
            fig_gradient, ax_gradient = plt.subplots()
            #colormap 4종류를 설정했습니다.
            MAP1 = 'winter'; MAP2 = 'spring'; MAP3 = 'autumn'; MAP4 = 'summer' #더 많은 종류를 설정해도 좋으나 벡터가 5종류 이상일 경우 시각적으로 구분이 힘들어질 수 있습니다.
            #colormap의 색깔을 cycle하며 색깔을 정하는 변수를 설정합니다. 이후 plotting에서 쓰입니다.
            cm1 = plt.get_cmap(MAP1); cm2 = plt.get_cmap(MAP2); cm3 = plt.get_cmap(MAP3); cm4 = plt.get_cmap(MAP4) #정한 colormap을 불러옵니다.
            #x[i]의 점들에 각각 colormap의 다른 색깔을 지정해줄 것입니다.
            #이때, x[i]의 길이들이 모두 같다고 가정했습니다. 설령 길이가 같지 않더라도, 연속적인 colormap에서 큰 차이를 보이지 않을 것입니다.
            cy = cycler('color',[cm1(1.*i/(len(x_all[0])-1)) for i in range(len(x_all[0])-1)] \
            + [cm2(1.*j/(len(x_all[0])-1)) for j in range(len(x_all[0])-1)] + [cm3(1.*j/(len(x_all[0])-1)) for j in range(len(x_all[0])-1)]\
            + [cm4(1.*j/(len(x_all[0])-1)) for j in range(len(x_all[0])-1)])
            ax_gradient.set_prop_cycle(cy) #축에 cy를 설정합니다.

            #각 벡터별로 위치 변화를 표시합니다.
            #같은 벡터는 같은 colormap의 그라데이션으로 표시됩니다.
            for i in range(N_ptl):
                for j in range(len(x_all[i])-1):
                    ax_gradient.plot(x_all[i][j:j+2],y_all[i][j:j+2],'.')
            plt.show() #plot을 화면에 띄웁니다.

        #3차원인 벡터인 경우
        elif self.setup.dim == 3:
            x_all = [] #각 벡터의 x좌표 변화만 모아놓을 list입니다. ex. x_all = [[1,2,3,4], [4,5,6,7], [7,8,9,10]]; 첫 번째 입자의 x좌표가 1,2,3,4으로 변화한 것입니다. 입자는 총 3개인 경우입니다.
            y_all = [] #각 벡터의 y좌표 변화만 모아놓을 list입니다.
            z_all = []
            N_ptl = self.setup.getNparticle() # 입자의 수.

            #self에 저장되어 있는 x,y좌표를 각각 x_all, y_all에 저장합니다.
            for i in range(N_ptl):
                x_all.append([d[0] for d in self.setup.particles[i].location.data]) #x_all[i]: (i+1)번째 입자의 x좌표 변화가 저장된 벡터
                y_all.append([d[1] for d in self.setup.particles[i].location.data]) #y_all[i]: (i+1)번째 입자의 y좌표 변화가 저장된 벡터
                z_all.append([d[2] for d in self.setup.particles[i].location.data]) #z_all[i]: (i+1)번째 입자의 z좌표 변화가 저장된 벡터

            #plot을 생성합니다.
            fig_gradient = plt.figure()
            ax_gradient = fig_gradient.add_subplot(111,projection='3d')
            #colormap 4종류를 설정했습니다.
            MAP1 = 'winter'; MAP2 = 'spring'; MAP3 = 'autumn'; MAP4 = 'summer' #더 많은 종류를 설정해도 좋으나 벡터가 5종류 이상일 경우 시각적으로 구분이 힘들어질 수 있습니다.
            #colormap의 색깔을 cycle하며 색깔을 정하는 변수(cy)를 설정합니다. 이후 plotting에서 쓰입니다.
            cm1 = plt.get_cmap(MAP1); cm2 = plt.get_cmap(MAP2); cm3 = plt.get_cmap(MAP3); cm4 = plt.get_cmap(MAP4) #설정한 colormap을 불러옵니다.
            #입자의 각 점에 colormap의 다른 색깔을 지정해줄 것입니다.
            #이때, 각 입자에 저장된 위치 개수가 모두 같다고 가정했습니다. 설령 길이가 같지 않더라도, 연속적인 colormap에서 큰 차이를 보이지 않을 것입니다.
            cy = cycler('color',[cm1(1.*i/(len(x_all[0])-1)) for i in range(len(x_all[0])-1)] \
            + [cm2(1.*j/(len(x_all[0])-1)) for j in range(len(x_all[0])-1)] + [cm3(1.*j/(len(x_all[0])-1)) for j in range(len(x_all[0])-1)]\
            + [cm4(1.*j/(len(x_all[0])-1)) for j in range(len(x_all[0])-1)])
            ax_gradient.set_prop_cycle(cy) #축에 cy를 설정합니다.

            #각 벡터별로 위치 변화를 표시합니다.
            #같은 벡터는 같은 colormap의 그라데이션으로 표시됩니다.
            for i in range(N_ptl):
                for j in range(len(x_all[i])-1):
                    ax_gradient.plot(x_all[i][j:j+2],y_all[i][j:j+2],z_all[i][j:j+2])
            plt.show() #plot을 화면에 띄웁니다.
        """

    def trajectory(self, projection = False):
        """ 가시화를 위해서 사용합니다. 
        :param bool projection: 3차원에서 True일 경우 각 평면에 사영된 경로도 같이 출력합니다. defalult 는 False
        """
        #2차원 벡터인 경우
        if self.setup.dim == 2:
            fig=plt.figure() #plot을 생성합니다.
            ax=fig.add_subplot(111) 
            x = [0 for i in range(self.N_ptl)] #x는 각 입자의 x좌표들의 리스트로 구성합니다; [[각 입자의 x좌표들]]
            y = [0 for i in range(self.N_ptl)] #y는 각 입자의 y좌표들의 리스트로 구성합니다; [[각 입자의 y좌표들]]
            cmap = [0 for i in range(self.N_ptl)] #각 입자에 지정할 색깔을 위해 colopmap 리스트를 생성합니다.
            for i in range(self.N_ptl):
                x[i] = [d[0] for d in self.setup.particles[i].location.data] #각 입자의 시간에 따른 x좌표들을 저장합니다.
                y[i] = [d[1] for d in self.setup.particles[i].location.data] #각 입자의 시간에 따른 y좌표들을 저장합니다.               
                ax.scatter(x[i], y[i], label=str(i) + 'th particle') #각 입자의 시간에 따른 위치를 점으로 표시합니다.                
            plt.xlabel('x (A.U.)')
            plt.ylabel('y (A.U.)')
            plt.legend()
            plt.show() #plot을 화면에 띄웁니다.
            
        #3차원인 벡터인 경우 #z좌표가 추가됩니다.
        elif self.setup.dim == 3:
            fig=plt.figure() #plot을 생성합니다.
            ax=fig.add_subplot(111,projection='3d')
            x = [0 for i in range(self.N_ptl)] 
            y = [0 for i in range(self.N_ptl)] 
            z = [0 for i in range(self.N_ptl)] 
            cmap = [0 for i in range(self.N_ptl)] 
            for i in range(self.N_ptl):
                x[i] = [d[0] for d in self.setup.particles[i].location.data] 
                y[i] = [d[1] for d in self.setup.particles[i].location.data] 
                z[i] = [d[2] for d in self.setup.particles[i].location.data] 
                sc = ax.scatter(x[i],y[i],z[i], label=str(i) + 'th particle')
                cmap[i] = sc.get_facecolor() 
            if projection: #3차원 공간으로 나타냅니다.
                for i in range(self.N_ptl):
                    xflat = np.full_like(x[i], min(ax.get_xlim())) 
                    yflat = np.full_like(y[i], max(ax.get_ylim()))
                    zflat = np.full_like(z[i], min(ax.get_zlim()))

                    ax.scatter(xflat, y[i], z[i], c = cmap[i], alpha=0.01)
                    ax.scatter(x[i], yflat, z[i], c = cmap[i], alpha=0.01)
                    ax.scatter(x[i], y[i], zflat, c = cmap[i], alpha=0.01)
            ax.set_xlabel('x (A.U.)')
            ax.set_ylabel('y (A.U.)')
            ax.set_zlabel('z (A.U.)')
            plt.legend()
            plt.show() #plot을 화면에 띄웁니다.

class simulator_test(unittest.TestCase):
    """ simulator class를 테스트 하기 위한 unittest class입니다. 
    """
    def setUp(self):
        """ test에 사용할 환경을 미리 구축합니다. 
        """
        self.table = setup(0.1, 3)
        self.table.add_particle(1, [1, 1, 0],  [0, 0, 0])
        self.table.add_particle(2, [0, 0, 0],  [0, 0, 0])
        self.table.add_particle(3, [-1, 1, 0], [0, 0, 0])
        self.sim = simulator(self.table)

    def test_set_force_parameter(self):
        """ set_force 함수에서 파라미터를 잘 받는지 확인합니다.
        """
        self.sim.set_force("-G*m1*m2/(x**2)", {"G": 3})
        self.assertEqual(self.sim.force_string ,"-3*m1*m2/(x**2)", "force_string parameter works wrong")

    
    def test_mass_matrix(self):
        """내부에서 mass matrix가 잘 형성되었는지 확인합니다.
        """
        self.sim.exec(1)
        self.assertTrue((self.sim.mass ==
                        np.array([np.ones(3)*(i+1) for i in range(3)])).all(),
                        "mass matrix is wrong parameter works wrong")

    def test_without_interaction(self):
        """ 상호작용이 없는 경우 잘 움직이는지 확인했습니다. 이 경우에 가속도가 0이기 때문에 등속으로 각 물체가 직진할 것으로 예상됩니다.
        """
        table2 = setup(0.1, 3)
        table2.add_particle(10232, [1, 0, 0],  [0, 10, 0])
        table2.add_particle(234, [0, 0, 0],  [0, 10, 10])
        sim2 = simulator(table2)
        sim2.set_force("0")
        sim2.exec(10)
        self.assertTrue((sim2.setup.getvelocities() ==
                        np.array([[0,10,0], [0,10,10]])).all(),
                        "free-evolution velocity changes")

        self.assertTrue((sim2.setup.getlastloc() ==
                        np.array([[1,10,0], [0,10,10]])).all(),
                        "free-evolution location is weird")

    # 상호작용이 있는 경우는 값으로 에러를 확인하기 어려워 jupyter notebook에서 유명한 문제를 잘 푸는지로 확인했습니다.
    # 예를 들어서 두 물체나, 세 물체가 원 궤도를 그리며 도는 경우에 대해서 이론적 계산에서 예상되는 결과를 잘 설명합니다.

if __name__ == "__main__":
    unittest.main(exit = False)
