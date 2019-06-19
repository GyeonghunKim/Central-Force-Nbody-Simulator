import numpy as np
import matplotlib.pyplot as plt
from data_types import location_series
import unittest

# 입자 하나를 나타내는 클래스입니다. 앞에서 만든 location_series와 속도, 질량을 원소로 합니다.
class particle:
    '''입자 하나를 나타내는 클래스입니다. 위치와 속도 그리고 질량에 대한 정보를 가지고 있습니다.
    위치에 대해서는 앞의 location series 형태로 데이터를 저장하고 속도와 질량은 실수형으로 가집니다. 
    일반적으로 setup class의 원소로 사용되는 클래스로, 직접 particle class의 인스턴스를 선언할 일은 없습니다. 
    '''
    def __init__(self, dt, dim, mass, loc, vel): #dt: 시간 간격, dim: 차원, mass: 질량, loc: 초기 위치, vel: 초기 속도 입니다.
        ''' particle class의 constructor입니다. 
        :param float dt: 입자가 움직이는 궤적을 계산할 때 사용되는 시간 간격입니다. 
        :param int dim: 입자가 움직이는 공간의 차원입니다. 
        :param float mass: 입자의 질량입니다. 
        :param numpy array loc: 입자의 초기 위치입니다. 
        :param numpy array vel: 입자의 초기 속력입니다.
        '''
        self.mass = mass #질량을 설정해줍니다.
        self.location = location_series(dt, dim, loc) #초기 위치와 차원, 시간 간격을 기반으로 location series를 만들어 줍니다. 
        self.velocity = vel #초기 속도를 설정해줍니다.


    def print_info(self):
        ''' 입자에 대한 정보를 출력합니다. #질량, 초기 위치, 초기 속도를 출력합니다.'''
        vec = self.location.get_init_loc() #해당 입자의 초기 위치가 list 형식으로 나옵니다.
        # 차원에 따라 출력하는 메시지를 분류했습니다.
        if len(vec) == 1: #입자가 1차원 벡터일 때
            print("Mass : ", self.mass, "kg\nInitial location : X coordinate =", vec[0],\
                    "\nInitial Velocity : ", self.velocity,"m/s")
        elif len(vec) == 2: #입자가 2차원 벡터일 때
            print("Mass : ", self.mass, "kg\nInitial location : X coordinate =", vec[0],\
                    ", Y coordinate =", vec[1],"\nInitial Velocity : ", self.velocity,"m/s")
        elif len(vec) == 3: #입자가 3차원 벡터일 때
            print("Mass : ", self.mass, "kg\nInitial location : X coordinate =", vec[0],\
            ", Y coordinate =", vec[1],", Z coordinate =",vec[2],"\nInitial Velocity : ", self.velocity,"m/s")
        else: #len(vec) >= 4; 입자가 4차원 이상의 벡터일 때
            print("Mass : ", self.mass, "kg\nInitial location : ", end="")
            for i, value in enumerate(vec, start=1):
                print("{}th coordinate = {}, ".format(i, value), end="") #(1st, 2nd, 3rd가 편의상 1st, 2st, 3st로 출력됩니다.)
            print("\nInitial Velocity : ", self.velocity,"m/s")
        #위치 벡터의 각 좌표를 출력합니다.

    
    def set_velocity(self, vel):
        ''' 입자의 기존 속도를 새로운 속도로 바꾸어 줍니다. 
        :param numpy array vel: 갱신할 속도입니다. 
        '''
        self.velocity = vel

class setup:
    '''
    시뮬레이션의 셋업을 나타내는 클래스입니다. 초기 조건이 설정 된 입자들을 클래스로 가집니다. 
    기존의 입자 클래스는 이 클래스 안에서 불러와 지게 됩니다. 이는 입자들 마다 시간 간격(dt)이나 차원이 달라지는 것을 막기 위함입니다. 
    '''
    def __init__(self, dt, dim):
        ''' setup class의 constructor입니다. 
        :param float dt: 입자가 움직이는 궤적을 계산할 때 사용되는 시간 간격입니다. 
        :param int dim: 입자가 움직이는 공간의 차원입니다. 
        '''
        self.dt = dt #시간 간격
        self.dim = dim #차원
        self.particles = [] #입자들이 저장될 빈 list를 만들어줍니다.

    # 입자를 추가합니다.
    def add_particle(self, mass, loc, vel): #질량, 위치, 속도를 정해줍니다.
        ''' setup에 입자를 추가하는 메소드입니다. 
        :param float mass: 입자가 움직이는 궤적을 계산할 때 사용되는 시간 간격입니다. 
        :param numpy array loc: 입자의 초기 위치 입니다. 
        :param numpy array vel: 입자의 초기 속력 입니다.
        '''
        tmp = particle(self.dt, self.dim, mass, loc, vel) #처음 setup에서 정해준 dt와 dim을 갖는 입자를 생성합니다.
        self.particles.append(tmp) #particles list에 입자를 추가합니다.

    # 입자를 하나 지웁니다. particles list에서 number번째(index는 number-1) 입자를 지웁니다.
    def delete_particle(self, number):
        ''' setup에서 입자를 지우는 메소드입니다. 
        :param int number: 지울 입자의 번호입니다. 입자의 번호는 이하의 print_list메소드를 이용하여 확인할 수 있습니다.
        '''
        del self.particles[number-1]


    def getNparticle(self):
        ''' 총 입자의 수를 반환하는 메소드입니다. '''
        return len(self.particles)

    def getvelocities(self):
        ''' 입자의 속력으로 구성된 numpy array를 반환하는 메소드입니다.
        입자가 세 개 들어있고, 각각의 속력이 [10, 2, 1], [2, 1, 12], [3, 21, 1]이라면, [[10, 2, 1], [2, 1, 12], [3, 21, 1]]를 반환합니다. 
        ''' 
        return np.array([particle.velocity for particle in self.particles])

    def setvelocities(self, velocities):
        '''입자들의 속도를 일괄적으로 갱신하는 메소드 입니다. 
        입자가 3개 일 때, velocity parameter가 [[10, 2, 1], [2, 1, 12], [3, 21, 1]]와 같이 들어오면
        입자 0의 속력은 [10, 2, 1]으로 갱신되고, 입자 1은 [2, 1, 12], 입자 3은 [3, 21, 1]으로 갱신됩니다. 
        :param numpy array velocities: 갱신 될 속력
        '''
        for i, particle in enumerate(self.particles):
            particle.set_velocity(velocities[i])

    def getlastloc(self):
        '''입자들의 마지막 위치들로 구성 된 numpy array가 반환되는 메소드 입니다. 
        입자가 세 개이고, 각각의 location series의 마지막 원소가 [10, 2, 1], [2, 1, 12], [3, 21, 1]이라면, 
        [[10, 2, 1], [2, 1, 12], [3, 21, 1]]이 반환됩니다. 
        '''
        return np.array([particle.location.get_last_loc() for particle in self.particles])

    def appendloc(self, loc):
        '''입자들의 위치를 일괄적으로 추가하는 메소드 입니다, 
        입자가 3개 일 때, loc parameter가 [[10, 2, 1], [2, 1, 12], [3, 21, 1]]와 같이 들어오면
        입자 0의 location series에는 [10, 2, 1]가 append되고, 입자 1에는 [2, 1, 12], 입자 3에는 [3, 21, 1]이 append 됩니다. 
        :param numpy array loc: 갱신 될 위치
        '''
        for i, particle in enumerate(self.particles):
            particle.location.append(loc[i])

    def print_list(self):
        '''입자들의 일련번호와 정보를 출력하는 메소드 입니다. 
        '''
        for i, ptl in enumerate(self.particles):
            print("particle number:", i)
            ptl.print_info() #partcles에 있는 각 입자들에 대해서 질량, 초기 위치, 초기 속도가 출력됩니다.

    # 입자들의 현재 위치를 좌표평면에 표시하고 각 입자의 현재 속도를 화살표로 나타냅니다.
    # 2차원 벡터에 대해서만 가능합니다.
    def print_particles(self):
        '''입자들의 초기 상태를 가시화 해주는 메소드입니다. 
        입자들의 위치를 점으로, 초기 속력을 화살표로 표현해줍니다. 
        3차원에서는 효과적인 가시화가 되지 않아서 2차원에서만 지원하도록 만들었습니다. 
        '''
        if self.dim == 3:
            raise ValueError("We only support this method for 2D setup")

        #quiver 함수를 쓰기 위해 x좌표벡터, y좌표벡터, x방향벡터, y방향벡터를 생성합니다.
        #좌표 벡터 생성.
        x_pos = [] #각 입자(위치 벡터)의 x좌표끼리 모아놓을 빈 list입니다. x_position.
        y_pos = [] #각 입자(위치 벡터)의 y좌표끼리 모아놓을 빈 list입니다.
        for i in range(len(self.particles)): #particles에 들어있는 입자 개수만큼 반복합니다.
            xy_pos = self.particles[i].location.get_last_loc() ##각 입자의 마지막 위치(곧 현재 위치)가 [x,y]의 형태로 저장됩니다.
            x_pos.append(xy_pos[0]) #x좌표를 x_pos에 추가합니다.
            y_pos.append(xy_pos[1]) #y좌표를 y_pos에 추가합니다.
        #print(x_pos, y_pos)

        #방향 벡터 생성.
        x_direct = [] #각 입자의 속도벡터의 x좌표끼리 모아놓을 빈 list입니다. x_direction.
        y_direct = [] #각 입자의 속도벡터의 y좌표끼리 모아놓을 빈 list입니다.
        for i in range(len(self.particles)): #particles에 들어있는 입자 개수만큼 반복합니다.
            xy_direct = self.particles[i].velocity #각 입자의 속도벡터가 [x,y]의 형태로 저장됩니다.
            x_direct.append(xy_direct[0]) #x방향을 x_direct에 추가합니다.
            y_direct.append(xy_direct[1]) #y방향을 y_direct에 추가합니다.
        #print(x_direct, y_direct)

        #quiver함수로 각 입자의 위치와 속도를 표시합니다.
        fig_quiver, ax_quiver = plt.subplots()
        ax_quiver.quiver(x_pos, y_pos, x_direct, y_direct)#, scale = 0.1) #scale을 조정할 수 있습니다.
        ax_quiver.axis([-10,10,-10,10]) #좌표평면에 나타나는 x축, y축 범위를 조정할 수 있습니다.
        plt.show() #화면에 plot을 띄웁니다.

class TestParticleSetup(unittest.TestCase): 
    """ Particle class와 setup class를 테스트 하기 위한 unittest class입니다. """
    def test_particle(self):
        """ particle 클래스 인스턴스가 정상적으로 만들어지는지 확인합니다. """
        particle1 = particle(0.01, 3, 10, [0,1,-1], [-3,4,5])
        self.assertEqual(particle1.mass, 10, "particle mass has a problem.")
        self.assertEqual(particle1.location.dt, 0.01, "location dt has a problem.")
        self.assertEqual(particle1.location.dim, 3, "location dim has a problem.")
        self.assertTrue((particle1.location.data == np.array([0,1,-1])).all() , "location data has a problem.")
        self.assertEqual(particle1.velocity, [-3,4,5], "particle velocity has a problem.")
        self.assertTrue(len(particle1.location.data[0])== particle1.location.dim, "particle dimension has a problem.") #입력된 차원만큼 벡터가 입력되었는지 확인.

    #속도 설정 테스트.
    def test_set_velocity(self):
        """ particle 클래스 set_velocity 메소드가 정상적으로 만들어지는지 확인합니다. """
        particle1 = particle(0.01, 3, 10, [1,2,3], [-3,4,5])
        particle1.set_velocity([3,-6,1])
        self.assertEqual(particle1.velocity, [3,-6,1], "particle velocity has a problem.") #속도가 제대로 입력되는지 확인.

    #setup 클래스 테스트.
    def test_setup(self):
        """ setup 클래스 인스턴스가 정상적으로 만들어지는지 확인합니다. """
        setup1 = setup(0.01, 3)
        self.assertEqual(setup1.dt, 0.01, "particle dt has a problem.") #dt 확인.
        self.assertEqual(setup1.dim, 3, "particle dimension has a problem.") #dim 확인.
        self.assertEqual(setup1.particles, [], "particles list has a problem.") #particles 확인.

    #입자 추가 테스트.
    def test_add_particle(self):
        """ setup 클래스 인스턴스에 particle추가하는게 잘 되는지 확인합니다.  """
        setup1 = setup(0.01, 3)
        tmp = particle(setup1.dt, setup1.dim, 10, [1,2,3], [-1,-3,5])
        setup1.add_particle(10, [1,2,3], [-1,-3,5])
        self.assertTrue(setup1.particles[0].mass == tmp.mass, "the particle is not added properly 1.") #tmp가 particles 리스트에 추가되었는지 확인.
        self.assertTrue((setup1.particles[0].location.get_init_loc() == tmp.location.get_init_loc()).all(), "the particle is not added properly 2.") #tmp가 particles 리스트에 추가되었는지 확인.
        self.assertTrue(setup1.particles[0].velocity == tmp.velocity, "the particle is not added properly 3.") #tmp가 particles 리스트에 추가되었는지 확인.

    #입자 삭제 테스트.
    def test_delete_particle(self):
        """ setup 클래스 인스턴스에 particle삭제하는게 잘 되는지 확인합니다.  """
        setup1 = setup(0.01, 3)
        setup1.add_particle(10, [1,2,3], [-1,-3,5]) #삭제하기 위한 입자 추가
        setup1.delete_particle(1)
        self.assertTrue(setup1.particles==[], "the particle is not deleted properly.") #입자(tmp)가 삭제되었는지 확인.

    #입자 수 테스트.
    def test_getNparticle(self):
        """ setup 클래스 getNparticle()이 잘 작동하는지 확인합니다.  """
        setup1 = setup(0.01, 3)
        tmp = particle(setup1.dt, setup1.dim, 10, [1,2,3], [-1,-3,5])
        setup1.particles.append(tmp) #현재 입자 수 1개
        self.assertTrue(setup1.getNparticle()==1, "the number of particles has a problem.") #입자 수가 제대로 반환되는지 확인.

    #입자 속도 테스트.
    def test_getvelocities(self):
        """ setup 클래스 getvelocities()가 잘 작동하는지 확인합니다.  """
        setup1 = setup(0.01, 3)
        setup1.add_particle(0.01, [0,1,-1], [-3,4,5])
        setup1.add_particle(0.01, [0,1,-1], [1,-6,2]) #두 개의 입자를 추가함.
        self.assertTrue((setup1.getvelocities() == np.array([[-3,4,5],[1,-6,2]])).all(), "the list of particle velocity has a problem.") #두 입자의 속도가 제대로 반환되는지 확인.


if __name__ == "__main__":
    unittest.main(exit = False)
