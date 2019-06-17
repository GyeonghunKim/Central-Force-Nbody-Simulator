import numpy as np
import matplotlib.pyplot as plt 
import unittest
class vector_series:
    """ vector series class는 벡터들의 series를 표현하는 클래스입니다. 
    초기 데이터를 기반으로 append하는 방식으로 사용됩니다.
    """ 
    def __init__(self, dim, data = []): 
        """ vector series의 constructor입니다. 
        :param int dim: 벡터의 차원입니다. 3차원 벡터의 series를 표현하고 싶다면 3을 입력하면 됩니다.
        :param numpy array data: 초기 데이터입니다. 이 값의 차원과 위의 dim입력의 차원이 맞지 않으면 에러가 납니다.
        """
        self.dim = dim
        if not data:
            self.data = [np.array([0 for x in range(self.dim)])]
        elif len(data) != dim:
            raise ValueError("initial value를 확인하세요!")
        else:
            self.data = [np.array(data)]
        #데이터 갯수
        self.len = 1
    
    # 벡터를 추가하는 경우 append 사용, 추가 데이터는 one_data 변수에 들어감
    def append(self, one_data):
        """ vector series의 constructor입니다. 
        :param int dim: 벡터의 차원입니다. 3차원 벡터의 series를 표현하고 싶다면 3을 입력하면 됩니다.
        :param numpy array data: 초기 데이터입니다. 이 값의 차원과 위의 dim입력의 차원이 맞지 않으면 에러가 납니다.
        """
        self.data.append(one_data)
        self.len+=1


class location_series(vector_series):
    """ vector series를 상속받아서 만들어진 클래스입니다. 기존의 vector series에 시계열이라는 의미가 추가되었고, 시화 메소드가
    추가되었습니다. 
    """
    def __init__(self, dt, dim, data = []):
        """location_series class의 constructor입니다. 
        :param float dt: vector들 사이의 시간 간격 입니다. 
        :param int dim: 벡터의 차원입니다. 3차원 벡터의 series를 표현하고 싶다면 3을 입력하면 됩니다.
        :param numpy array data: 초기 데이터입니다. 이 값의 차원과 위의 dim입력의 차원이 맞지 않으면 에러가 납니다.
        """
        vector_series.__init__(self, dim, data)
        self.dt = dt

    def get_init_loc(self):
        """ 초기 위치를 반환하는 메소드 입니다. 첫 벡터를 반환합니다. 
        """
        return self.data[0]
 
    def get_last_loc(self):
        """ 마지막 위치를 반환하는 메소드 입니다. 마지막 벡터를 반환합니다. 
        """
        return self.data[-1]
 
    def plot_timeseries(self, whichdim):
        """ 특정 차원(0: x축, 1: y축, 2: z축)의 값 또는 극좌표에서의 값을 시간에 대해서 변화를 출력하는 함수. 
        :param int whichdim: 어떤 축을 기준으로 그래프를 그릴지를 알려주는데 사용하는 인자입니다. 0: x축, 1: y축, 2: z축, 3: 극좌표 반지름, 4: 극좌표 각도 입니다. 
        """
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
    """ vector_series class와 location_series class를 테스트 하기 위한 unittest class입니다. """
    def test_1(self):
        a=vector_series(3,[1,1,1])
        c=np.array([1,1,1])
        self.assertEqual(a.dim,3, "dimension problem")
        self.assertEqual(a.len,1, "len problem")
        self.assertTrue((a.data == c).all(), "data problem")

    def test_2(self):
        b=vector_series(3)
        c=np.array([0,0,0])
        self.assertEqual(b.dim, 3, "dimension problem")
        self.assertEqual(b.len, 1, "len problem")
        self.assertTrue((b.data == c).all(), "data problem")

    def test_3(self):
        a=vector_series(3,[1,1,1])
        a.append(np.array([3,4,5]))
        c=[np.array([1,1,1]), np.array([3,4,5])]
        self.assertEqual(a.len, 2, "len problem")
        self.assertTrue((np.array(a.data) == np.array(c)).all(), "data problem")

    def test_4(self):
        a = location_series(0.1, 3, [1,1,1])
        c = np.array([1,1,1])
        self.assertEqual(a.dt,0.1,"dt problem")
        self.assertEqual(a.dim,3, "dimension problem")
        self.assertEqual(a.len,1, "len problem")
        self.assertTrue((a.data == c).all(), "data problem")

    def test_5(self):
        b=location_series(0.1,3)
        c=np.array([0,0,0])
        self.assertEqual(b.dt,0.1,"dt problem")
        self.assertEqual(b.dim,3, "dimension problem")
        self.assertEqual(b.len,1, "len problem")
        self.assertTrue((b.data == c).all(), "data problem")

    def test_6(self):
        a=location_series(0.1,3,[1,1,1])
        self.assertTrue((a.get_init_loc()==np.array([1,1,1])).all(), "get_init_loc() problem")
        self.assertTrue((a.get_init_loc()==np.array([1,1,1])).all(), "get_last_loc() problem")


if __name__=="__main__":
    unittest.main(exit=False)
