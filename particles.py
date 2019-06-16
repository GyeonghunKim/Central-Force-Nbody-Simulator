import numpy as nump
import matplotlib.pyplot as plt
from data_types import *
import unittest

# 입자 하나를 나타내는 클래스입니다. 앞에서 만든 location_series와 속도, 질량을 원소로 합니다.
class particle:
    def __init__(self, dt, dim, mass, loc, vel): #dt: 시간 간격, dim: 차원, mass: 질량, loc: 초기 위치, vel: 초기 속도 입니다.
        self.mass = mass #질량을 설정해줍니다.
        self.location = location_series((dt, dim, loc))
        self.velocity = vel #초기 속도를 설정해줍니다.

        # 입자에 대한 정보를 출력합니다. #질량, 초기 위치, 초기 속도를 출력합니다.
    def print_info(self):
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

    #새로운 속도를 설정해줍니다.
    def set_velocity(self, vel):
        self.velocity = vel

# setup 클래스는 실험을 하기 위한 셋업이라고 생가하면 됩니다.
# dt(시간 간격)를 정해주고, 보고자 하는 차원을 정해주고, 입자들을 추가할 수 있습니다.
class setup:
    def __init__(self, dt, dim):
        self.dt = dt #시간 간격
        self.dim = dim #차원
        self.particles = [] #입자들이 저장될 빈 list를 만들어줍니다.

    # 입자를 추가합니다.
    def add_particle(self, mass, loc, vel): #질량, 위치, 속도를 정해줍니다.
        tmp = particle(self.dt, self.dim, mass, loc, vel) #처음 setup에서 정해준 dt와 dim을 갖는 입자를 생성합니다.
        self.particles.append(tmp) #particles list에 입자를 추가합니다.

    # 입자를 하나 지웁니다. particles list에서 number번째(index는 number-1) 입자를 지웁니다.
    def delete_particle(self, number):
        del self.particles[number-1]

    # 총 입자 수를 리턴합니다.
    def getNparticle(self):
        return len(self.particles)

    # 입자들의 속력으로 구성된 np array가 리턴됩니다.
    # 입자가 세 개라면 [[10, 2, 1], [2, 1, 12], [3, 21, 1]]와 같이 출력될 것 입니다.
    def getvelocities(self):
        return np.array([particle.velocity for particle in self.particles])

    # 입자들의 속력이 [[10, 2, 1], [2, 1, 12], [3, 21, 1]]와 같이 들어오면
    # 입자 0은 [10, 2, 1]를 속력으로 넣어주고 입자 1은 [2, 1, 12], 입자 3은 [3, 21, 1]처럼 넣어줍니다.
    def setvelocities(self, velocities):
        for i, particle in enumerate(self.particles):
            particle.set_velocity(velocities[i])

    # 입자들의 마지막 위치로 구성된 np array가 리턴됩니다.
    # 입자가 세 개라면 [[10, 2, 1], [2, 1, 12], [3, 21, 1]]와 같이 출력될 것 입니다. 내부의 세 어레이는 각각 각 입자의 마지막 위치를 말합니다.
    def getlastloc(self):
        return np.array([particle.location.get_last_loc() for particle in self.particles])

    # 입자들의 위치가 [[10, 2, 1], [2, 1, 12], [3, 21, 1]]와 같이 들어오면
    # 입자 0은 [10, 2, 1]를 위치에 추가해서 넣어주고 입자 1은 [2, 1, 12], 입자 3은 [3, 21, 1]을 넣어줍니다.
    def appendloc(self, loc):
        for i, particle in enumerate(self.particles):
            particle.location.append(loc[i])

    # 입자들의 각 정보를 출력합니다.
    def print_list(self):
        for ptl in self.particles:
            ptl.print_info() #partcles에 있는 각 입자들에 대해서 질량, 초기 위치, 초기 속도가 출력됩니다.

    # 입자들의 현재 위치를 좌표평면에 표시하고 각 입자의 현재 속도를 화살표로 나타냅니다.
    # 2차원 벡터에 대해서만 가능합니다.
    def print_particles(self):
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

#test class: particle 클래스와 setup 클래스에 대해 유닛테스트를 실행하는 테스트 클래스입니다.
class TestParticleSetup(unittest.TestCase): #unittest.Testcase를 상속받습니다.
    #particle 클래스 테스트.
    def test_particle(self):
        particle1 = particle(0.01, 3, 10, [0,1,-1], [-3,4,5])
        self.assertEqual(particle1.mass, 10, "particle mass has a problem.")
        self.assertEqual(particle1.location.dt, 0.01, "location dt has a problem.")
        self.assertEqual(particle1.location.dim, 3, "location dim has a problem.")
        self.assertTrue((particle1.location.data == np.array([0,1,-1])).all() , "location data has a problem.")
        self.assertEqual(particle1.velocity, [-3,4,5], "particle velocity has a problem.")
        self.assertTrue(len(particle1.location.data[0])== particle1.location.dim, "particle dimension has a problem.") #입력된 차원만큼 벡터가 입력되었는지 확인.

    #속도 설정 테스트.
    def test_set_velocity(self):
        particle1 = particle(0.01, 3, 10, [1,2,3], [-3,4,5])
        particle1.set_velocity([3,-6,1])
        self.assertEqual(particle1.velocity, [3,-6,1], "particle velocity has a problem.") #속도가 제대로 입력되는지 확인.

    #setup 클래스 테스트.
    def test_setup(self):
        setup1 = setup(0.01, 3)
        self.assertEqual(setup1.dt, 0.01, "particle dt has a problem.") #dt 확인.
        self.assertEqual(setup1.dim, 3, "particle dimension has a problem.") #dim 확인.
        self.assertEqual(setup1.particles, [], "particles list has a problem.") #particles 확인.

    #입자 추가 테스트.
    def test_add_particle(self):
        setup1 = setup(0.01, 3)
        tmp = particle(setup1.dt, setup1.dim, 10, [1,2,3], [-1,-3,5])
        setup1.particles.append(tmp)
        self.assertTrue(setup1.particles[0]==tmp, "the particle is not added properly.") #tmp가 particles 리스트에 추가되었는지 확인.

    #입자 삭제 테스트.
    def test_delete_particle(self):
        setup1 = setup(0.01, 3)
        tmp = particle(setup1.dt, setup1.dim, 10, [1,2,3], [-1,-3,5])
        setup1.particles.append(tmp) #삭제하기 위한 입자 추가
        setup1.delete_particle(1)
        self.assertTrue(setup1.particles==[], "the particle is not deleted properly.") #입자(tmp)가 삭제되었는지 확인.

    #입자 수 테스트.
    def test_getNparticle(self):
        setup1 = setup(0.01, 3)
        tmp = particle(setup1.dt, setup1.dim, 10, [1,2,3], [-1,-3,5])
        setup1.particles.append(tmp) #현재 입자 수 1개
        self.assertTrue(setup1.getNparticle()==1, "the number of particles has a problem.") #입자 수가 제대로 반환되는지 확인.

    #입자 속도 테스트.
    def test_getvelocities(self):
        setup1 = setup(0.01, 3)
        setup1.add_particle(0.01, [0,1,-1], [-3,4,5])
        setup1.add_particle(0.01, [0,1,-1], [1,-6,2]) #두 개의 입자를 추가함.
        self.assertTrue((setup1.getvelocities() == np.array([[-3,4,5],[1,-6,2]])).all(), "the list of particle velocity has a problem.") #두 입자의 속도가 제대로 반환되는지 확인.


if __name__ == "__main__":
    unittest.main(exit = False)
