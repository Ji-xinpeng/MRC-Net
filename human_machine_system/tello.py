from djitellopy import Tello


class TelloController:
    def __init__(self) -> None:
        self._registered_command()
        self.tello = Tello()
        self.coordinate_info = None

    def tello_connect(self):
        self.tello.connect()

    def _registered_command(self):
        self.takeoff = [24, 56, 62]
        self.land = [57, 63]
        self.move_left = [58, 70]
        self.move_left_unstop = [80]
        self.move_right = [59, 71]
        self.move_right_unstop = [81]
        self.move_back = [60]
        self.move_forward = [61]
        self.move_foward_unstop = [82]
        self.move_up = [68]
        self.move_down = [69]
        self.flip_right = [0, 77]
        self.flip_left = [1, 76]
        self.flip_forward = [4]
        self.flip_back = [5]
        self.stop = [33, 35]

    def _get_distance(self, mp_results):
        pass
        return 1

    def _take_off(self, command, mp_results):
        # self.tello.takeoff()
        print("无人机 起飞")

    def _land(self, command, mp_results):
        # self.tello.land()
        print("无人机 降落")
    
    def _move_left(self, command, mp_results):
        distance = self._get_distance(mp_results)
        # self.tello.move_left(distance)
        print("无人机 向左飞 {} 厘米".format(distance))

    def _move_right(self, command, mp_results):
        distance = self._get_distance(mp_results)
        # self.tello.move_right(distance)
        print("无人机 向右飞 {} 厘米".format(distance))

    def _move_forward(self, command, mp_results):
        distance = self._get_distance(mp_results)
        # self.tello.move_forward(distance)
        print("无人机 向前飞 {} 厘米".format(distance))

    def _move_back(self, command, mp_results):
        distance = self._get_distance(mp_results)
        # self.tello.move_back(distance)
        print("无人机 向后飞 {} 厘米".format(distance))

    def control(self, command: int, mp_results):
        self.coordinate_info = mp_results.multi_hand_landmarks
        if command in self.takeoff:
            self._take_off(command, mp_results)
        elif command in self.land:
            self._land(command, mp_results)
        elif command in self.move_left:
            self._move_left(command, mp_results)
        elif command in self.move_right:
            self._move_right(command, mp_results) 
        elif command in self.move_forward:
            self._move_forward(command, mp_results)
        elif command in self.move_back:
            self._move_back(command, mp_results) 

