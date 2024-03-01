from djitellopy import Tello
from pathlib import Path
import sys
# 获取当前文件所在目录的上两级目录的路径
two_up = Path(__file__).resolve().parents[1]
sys.path.append(str(two_up))
from others.params import *


class TelloParams:
    def __init__(self):
        self.distance = None
        self.inter_frame = None
        self.mpresults = None
        self.command = None
        self.label_for_system = args.system_label
        self.label_for_system.sort()

    def _convert_label_for_system(self, command):
        self.command = self.label_for_system[command]


class TelloController:
    def __init__(self) -> None:
        self._registered_command()
        self.tello = Tello()
        self.tello_params = TelloParams()
        self.coordinate_info = None
        self.has_connect = False
        self.rate = 0.1
        self.rotate_rate = 0.5

    def tello_connect(self):
        try:
            self.tello.connect()
            self.has_connect = True
        except:
            print("无人机连接失败 ！ ")
            self.has_connect = False

    def _registered_command(self):
        self.takeoff = [24]
        self.land = [33]

        self.move_left = [70, 80]
        self.move_right = [71, 81]
        self.move_up = [3]
        self.move_down = [2]

        self.move_back = [60]
        self.move_forward = [61]

        self.flip_right = [0, 77]
        self.flip_left = [1, 76]

        self.flip_forward = [4]
        self.flip_back = [5]

        self.rotate_clockwise = [13]
        self.rotate_counter_clockwise = [14]

    def _take_off(self, command):
        if self.has_connect:
            self.tello.takeoff()
        print("无人机 起飞")

    def _set_speed(self, speed):
        """Set speed to x cm/s.
        Arguments:
            x: 10-100
        """
        if self.has_connect:
            self.tello.set_speed(int(speed))

    def _land(self, command):
        if self.has_connect:
            self.tello.land()
        print("无人机 降落")
    
    def _move_left(self, command, distance):
        if self.has_connect:
            self.tello.move_left(distance)
        print("无人机 向左飞 {} 厘米".format(distance))

    def _move_right(self, command, distance):
        if self.has_connect:
            self.tello.move_right(distance)
        print("无人机 向右飞 {} 厘米".format(distance))

    def _move_forward(self, command, distance):
        if self.has_connect:
            self.tello.move_forward(distance)
        print("无人机 向前飞 {} 厘米".format(distance))

    def _move_back(self, command, distance):
        if self.has_connect:
            self.tello.move_back(distance)
        print("无人机 向后飞 {} 厘米".format(distance))

    def _move_up(self, command, distance):
        if self.has_connect:
            self.tello.move_up(distance)
        print("无人机 向上飞 {} 厘米".format(distance))

    def _move_down(self, command, distance):
        if self.has_connect:
            self.tello.move_down(distance)
        print("无人机 向下飞 {} 厘米".format(distance))

    def _rotate_clockwise(self, command, x):
        """Rotate x degree clockwise.
        Arguments:
            x: 1-360, x 输入的是时间，时间越短，转的角度越少
        """
        if self.has_connect:
            self.tello.rotate_clockwise(x)
        print("无人机 顺时针旋转 {} 度".format(x))

    def _rotate_counter_clockwise(self, command, x):
        if self.has_connect:
            self.tello.rotate_counter_clockwise(x)
        print("无人机 逆时针旋转 {} 度".format(x))

    def _flip_left(self):
        if self.has_connect:
            self.tello.flip_left()
        print("无人机 向左翻转")

    def _flip_right(self):
        if self.has_connect:
            self.tello.flip_right()
        print("无人机 向 右 翻转")

    def _flip_forward(self):
        if self.has_connect:
            self.tello.flip_forward()
        print("无人机 向前翻转")

    def _flip_back(self):
        if self.has_connect:
            self.tello.flip_back()
        print("无人机 向后翻转")

    def _check_distance(self, distance):
        if distance < 20:
            distance = 20
        elif distance > 500:
            distance = 500
        return distance

    def _check_rotate(self, rotate):
        if rotate < 0:
            rotate = 0
        elif rotate > 360:
            rotate = 360
        return rotate

    def _check_speed(self, speed):
        if speed < 10:
            speed = 10
        elif speed > 100:
            speed = 99
        return speed

    def _get_speed(self):
        speed = int(self.tello_params.distance * 1000 / self.tello_params.inter_frame)
        return speed

    def _change_speed(self):
        speed = self._get_speed()
        speed = self._check_speed(speed)
        self._set_speed(speed)
        return speed

    def _map_rotate(self, x):
        """输入的x一般为 8 到 50 帧
         ((原始值 - 8) / (50 - 8)) * (360 - 1) + 1
         您可以使用线性映射来将8到50之间的值映射到1到360之间。具体计算方法如下：
        如果您希望将范围从8到50映射到360到1之间，您可以反转目标范围的顺序，并进行线性映射。具体计算方法如下：

        假设原始范围为 [8, 50]，目标范围为 [360, 1]。

        计算原始范围的长度：50 - 8 = 42
        计算目标范围的长度：1 - 360 = -359 （注意这里是反转的范围）
        计算斜率：目标范围长度 / 原始范围长度 = -359 / 42 ≈ -8.5476
        计算截距：360 - 斜率 * 8 = 360 - (-8.5476) * 8 ≈ 426.3808
        因此，将原始值 x 映射到目标值 y 的公式为：y = -8.5476 * x + 426.3808

        例如，当 x = 20 时，代入公式计算得到 y ≈ -8.5476 * 20 + 426.3808 ≈ 245.1176，因此在这个线性映射下，20映射到约245.12。
        """
        x = -8.5476 * x + 426.3808
        x = self._check_rotate(x)
        x *= self.rotate_rate
        return int(x)

    def _map_distance(self, distance):
        """输入的 x 一般为 0.05 到 0.85 之间， 无人机是 20-500
         ((原始值 - 20) / (0.85 - 0.05)) * (500 - 20) + 1
         您可以使用线性映射来将0.05到0.85之间的值映射到20到500之间。具体计算方法如下：

        假设原始范围为 [0.05, 0.85]，目标范围为 [20, 500]。

        计算原始范围的长度：0.85 - 0.05 = 0.8
        计算目标范围的长度：500 - 20 = 480
        计算斜率：目标范围长度 / 原始范围长度 = 480 / 0.8 = 600
        计算截距：20 - 斜率 * 0.05 = 20 - 600 * 0.05 = 20 - 30 = -10
        因此，将原始值 x 映射到目标值 y 的公式为：y = 600 * x - 10

        例如，当 x = 0.5 时，代入公式计算得到 y = 600 * 0.5 - 10 = 290，因此在这个线性映射下，0.5映射到290。
        """
        distance = 600 * distance - 10
        distance = self._check_distance(distance)
        distance = int(distance * self.rate)
        return distance

    def control(self):
        self.coordinate_info = self.tello_params.mpresults.multi_hand_landmarks
        command = self.tello_params.command
        print("\n无人机移动帧数: ", int(self.tello_params.inter_frame))
        print("无人机移动距离: ", self.tello_params.distance)
        distance = self._map_distance(self.tello_params.distance)
        rotate = self._map_rotate(self.tello_params.inter_frame)
        speed = self._change_speed()
        print("无人机实际运动距离：", distance)
        print("无人机实际转动角度：", rotate)
        print("无人机实际运动速度：", speed)
        if command in self.takeoff:
            self._take_off(command)
        elif command in self.land:
            self._land(command)
        elif command in self.move_left:
            self._move_left(command, distance)
        elif command in self.move_right:
            self._move_right(command, distance)
        elif command in self.move_forward:
            self._move_forward(command, distance)
        elif command in self.move_back:
            self._move_back(command, distance)
        elif command in self.move_up:
            self._move_up(command, distance)
        elif command in self.move_down:
            self._move_down(command, distance)
        elif command in self.flip_left:
            self._flip_left()
        elif command in self.flip_right:
            self._flip_right()
        elif command in self.flip_forward:
            self._flip_forward()
        elif command in self.flip_back:
            self._flip_back()
        elif command in self.rotate_clockwise:
            self._rotate_clockwise(command, rotate)
        elif command in self.rotate_counter_clockwise:
            self._rotate_counter_clockwise(command, rotate)

