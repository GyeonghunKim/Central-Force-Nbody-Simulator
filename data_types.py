import numpy as np
import matplotlib.pyplot as plt 
import unittest
class vector_series:
    # 가변적인 입력의 갯수를 위해서 args를 이용합니다. args의 첫 원소는 들어갈 벡터의 차원.
    # vector_series([3, [1, 2, 3]])이런 형식이나 vector_series([3]) 이런형식으로 들어감.
    def __init__(self, args): 
    # len(args)==1인 경우 초기 데이터를 0으로 생각, ex) a = vector_series(3)인 경우 data = [0,0,0]으로 생각
        if len(args) == 1:
            #차원은 args[0]값
            self.dim = args[0]
            # self.data는 리스트의 리스트로 0으로 생각 
            self.data = [np.array([0 for x in range(self.dim)])]
            #데이터 갯수
            self.len = 1
        #len(args)==2인 경우는 초기 데이터가 args[1]에 저장되어 있음.
        elif len(args) == 2:
            #차원은 args[0]의 값
            self.dim = args[0]
            # 초기 데이터 = args[1]
            self.data = [np.array(args[1])]
            #len은 데이터의 갯수
            self.len = 1
        else:
        #더 많은 변수를 입력받는 경우 error 표시
            raise ValueError("constructor input is weird!")
    
    # 벡터를 추가하는 경우 append 사용, 추가 데이터는 one_data 변수에 들어감
    def append(self, one_data):
        #one_data는 [[a,b,c]] or [[a,b,c],[d,e,f]]의 형식이여야함. 즉, 리스트의 리스트 형식.
        #one_data를 np.array로 설정
        #one_data = np.array([one_data])
        #one_data를 data 뒤에 np.r_ 함수를 이용해서 이어붙임.
        self.data.append(one_data)
        #초기 데이터 갯수 1에서 n을 더해서 총 데이터 갯수를 설정
        self.len+=1

# vector_series + some visualization methods
# 이는 앞의 vectorseries보다 dt라는게 하나 추가됩니다. 
class location_series(vector_series):
    def __init__(self, args): # args[0] = dt, args[1] = dim, args[2] = inital value
        if len(args) == 2:
            self.dt = args[0]
            self.dim = args[1]
            self.data = [np.array([0 for x in range(self.dim)])]
            self.len = 1
        elif len(args) == 3:
            self.dt = args[0]
            self.dim = args[1]
            self.data = [np.array(args[2])]
            self.len = 1
        else:
            raise ValueError("constructor input is weird!")
 
    # 초기 위치를 리턴하는 함수
    def get_init_loc(self):
        return self.data[0]
 
    # 저장되어 있는 마지막 위치를 리턴하는 함수
    def get_last_loc(self):
        return self.data[-1]
 
    # 특정 차원(0: x축, 1: y축, 2: z축)에 대해서 시간에 대해서 변화를 출력하는 함수. 
    def plot_timeseries(self, whichdim):
        # 처음에 설정된 dt에서 data의 갯수만큼 dt*i를 하여 시간을 계산함
        t = [self.dt * i for i in range(self.len)]
        # x축, y축, z축으로 그래프를 그리는 경우,
        if whichdim < 3:
            #x는 [1,2,3] 형식의 데이터에서 whichdim==0이면 x-axis, ==1이면 y-axis, ==2이면 z-axis
            x = [d[whichdim] for d in self.data]
            #나중에 y축에 나타낼 문자가 whichdim 숫자에 따라 달라지기 때문에 각각에 대해서 다르게 지정함
            if whichdim==0:
                letter = "X-axis"
            elif whichdim==1:
                letter="Y-axis"
            elif whichdim==2:
                letter="Z-axis"
        #만약 whichdim==3인경우, 원점으로부터의 거리를 측정
        elif whichdim == 3:
            letter = "distance from the origin"
            x = [np.sqrt((d**2).sum()) for d in self.data]
        # whichdim==4인 경우, x축에 대한 각도를 측정
        elif whichdim == 4: # only for 2d
            letter = "degree in X-axis"
            x = [np.arctan(d[1]/d[0]) for d in self.data]
        else:
            #그 외의 경우 error
            raise ValueError("dimension input is weird")
        #t와 x 관계를 그래프로 그림. 각각의 점에 대해서는 '.'으로 마킹하고, 그래프의 제목을 설정함
        plt.plot(t, x, marker = '.',label = 'Changes as time changes')
        # x축 라벨 설정
        plt.xlabel("time")
        # y축 라벨 설정
        plt.ylabel(letter)
        # 그래프 새 화면에 창띄우기
        plt.show()

class data_types_Test(unittest.TestCase):
    def test_1(self):
        a=vector_series([3,[1,1,1]])
        c=np.array([1,1,1])
        self.assertEqual(a.dim,3, "dimension problem")
        self.assertEqual(a.len,1, "len problem")
        self.assertTrue((a.data == c).all(), "data problem")

    def test_2(self):
        b=vector_series([3])
        c=np.array([0,0,0])
        self.assertEqual(b.dim, 3, "dimension problem")
        self.assertEqual(b.len, 1, "len problem")
        self.assertTrue((b.data == c).all(), "data problem")

    def test_3(self):
        a=vector_series([3,[1,1,1]])
        a.append(np.array([3,4,5]))
        c=[np.array([1,1,1]), np.array([3,4,5])]
        self.assertEqual(a.len, 2, "len problem")
        self.assertTrue((np.array(a.data) == np.array(c)).all(), "data problem")

    def test_4(self):
        a=location_series([0.1,3,[1,1,1]])
        c=np.array([1,1,1])
        self.assertEqual(a.dt,0.1,"dt problem")
        self.assertEqual(a.dim,3, "dimension problem")
        self.assertEqual(a.len,1, "len problem")
        self.assertTrue((a.data == c).all(), "data problem")

    def test_5(self):
        b=location_series([0.1,3])
        c=np.array([0,0,0])
        self.assertEqual(b.dt,0.1,"dt problem")
        self.assertEqual(b.dim,3, "dimension problem")
        self.assertEqual(b.len,1, "len problem")
        self.assertTrue((b.data == c).all(), "data problem")

    def test_6(self):
        a=location_series([0.1,3,[1,1,1]])
        c=np.array([1,1,1])
        self.assertTrue((a.get_init_loc()==np.array([1,1,1])).all(), "get_init_loc() problem")
        self.assertTrue((a.get_init_loc()==np.array([1,1,1])).all(), "get_last_loc() problem")


if __name__=="__main__":
    unittest.main(exit=False)
