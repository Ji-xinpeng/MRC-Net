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
        self.command  = self.label_for_system[command]


class TelloController:
    def __init__(self) -> None:
        self._registered_command()
        self.tello = Tello()
        self.tello_params = TelloParams()
        self.coordinate_info = None
        self.has_connect = False
        self.rate = 30

    def tello_connect(self):
        try:
            self.tello.connect()
            self.has_connect = True
        except:
            print("无人机连接失败 ！ ")
            self.has_connect = False

    def _registered_command(self):
        self.takeoff = [62]
        self.land = [63]

        self.move_left = [80]
        self.move_right = [81]
        self.move_up = [3]
        self.move_down = [2]

        self.move_back = [60]
        self.move_forward = [61]

        self.flip_right = [77]
        self.flip_left = [76]

        self.flip_forward = [4]
        self.flip_back = [5]

        self.rotate_clockwise = [9, 13]
        self.rotate_counter_clockwise = [10, 14]

    def _get_finger_tip_distance(self):
        # 获取食指点的位移
        return self.tello_params.distance

    def _take_off(self, command):
        if self.has_connect:
            self.tello.takeoff()
        print("无人机 起飞")

    def _land(self, command):
        if self.has_connect:
            self.tello.land()
        print("无人机 降落")
    
    def _move_left(self, command):
        distance = self._get_finger_tip_distance()
        distance = (distance * self.rate)
        if self.has_connect:
            self.tello.move_left(distance)
        print("无人机 向左飞 {} 厘米".format(distance))

    def _move_right(self, command):
        distance = self._get_finger_tip_distance()
        distance = (distance * self.rate)
        if self.has_connect:
            self.tello.move_right(distance)
        print("无人机 向右飞 {} 厘米".format(distance))

    def _move_forward(self, command):
        distance = self._get_finger_tip_distance()
        distance = (distance * self.rate)
        if self.has_connect:
            self.tello.move_forward(distance)
        print("无人机 向前飞 {} 厘米".format(distance))

    def _move_back(self, command):
        distance = self._get_finger_tip_distance()
        distance = (distance * self.rate)
        if self.has_connect:
            self.tello.move_back(distance)
        print("无人机 向后飞 {} 厘米".format(distance))

    def _move_up(self, command):
        distance = self._get_finger_tip_distance()
        distance = (distance * self.rate)
        if self.has_connect:
            self.tello.move_up(distance)
        print("无人机 向上飞 {} 厘米".format(distance))

    def _move_down(self, command):
        distance = self._get_finger_tip_distance()
        distance = (distance * self.rate)
        if self.has_connect:
            self.tello.move_down(distance)
        print("无人机 向下飞 {} 厘米".format(distance))

    def _rotate_clockwise(self, command):
        distance = self._get_finger_tip_distance()
        if self.has_connect:
            self.tello.rotate_clockwise(distance)
        print("无人机 顺时针旋转 {} 度".format(distance))

    def _rotate_counter_clockwise(self, command):
        distance = self._get_finger_tip_distance()
        if self.has_connect:
            self.tello.rotate_counter_clockwise(distance)
        print("无人机 逆时针旋转 {} 度".format(distance))

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

    def control(self):
        self.coordinate_info = self.tello_params.mpresults.multi_hand_landmarks
        command = self.tello_params.command
        print("interval_frames: ", self.tello_params.inter_frame)
        print("distance: ", self.tello_params.distance)
        print("command: ", command)
        if command in self.takeoff:
            self._take_off(command)
        elif command in self.land:
            self._land(command)
        elif command in self.move_left:
            self._move_left(command)
        elif command in self.move_right:
            self._move_right(command)
        elif command in self.move_forward:
            self._move_forward(command)
        elif command in self.move_back:
            self._move_back(command)
        elif command in self.move_up:
            self._move_up(command)
        elif command in self.move_down:
            self._move_down(command)
        elif command in self.flip_left:
            self._flip_left()
        elif command in self.flip_right:
            self._flip_right()
        elif command in self.flip_forward:
            self._flip_forward()
        elif command in self.flip_back:
            self._flip_back()
        elif command in self.rotate_clockwise:
            self._rotate_clockwise(command)
        elif command in self.rotate_counter_clockwise:
            self._rotate_counter_clockwise(command)

