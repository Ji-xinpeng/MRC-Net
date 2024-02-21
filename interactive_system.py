import cv2
import os
import time
import math
from PIL import Image
import mediapipe as mp
from inference_utils.inference_class import Inference
from human_machine_system.tello import TelloController

class InteractivaSystem:
    def __init__(self) -> None:
        self._init_check_camera()
        self._parse_weights_path()
        self._init_inference_model()
        self._init_mediapipe()
        self._init_varbile()
        self._init_tello()

    def _init_tello(self):
        self.tello_controller = TelloController()
        self.tello_controller.tello_connect()

    def _parse_weights_path(self):
        current_file = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file)
        self.checkpoint_path_classify = current_directory + '/weights/mobilenetv2classify.pth'
        self.checkpoint_path_detect = current_directory + '/weights/mobilenetv3_smalldetect.pth'

    def _init_inference_model(self):
        self.inference = Inference(self.checkpoint_path_classify, self.checkpoint_path_detect)

    def _init_check_camera(self):
        # 0代表电脑自带的摄像头
        self.cap = cv2.VideoCapture(0) 
        if not self.cap.isOpened():
            raise Exception("Could not open camera.")
        
    def _init_mediapipe(self):
        self.mpHands = mp.solutions.hands  #接收方法
        self.hands = self.mpHands.Hands(static_image_mode=False, #静态追踪，低于0.5置信度会再一次跟踪
                            max_num_hands=2, # 最多有2只手
                            min_detection_confidence=0.5, # 最小检测置信度
                            min_tracking_confidence=0.5)  # 最小跟踪置信度 
        # 创建检测手部关键点和关键点之间连线的方法
        self.mpDraw = mp.solutions.drawing_utils
        self.first_index_finger_tip = None
        self.prev_index_finger_tip = None
        self.prev_mp_results = None

    def _init_varbile(self):
        self.frame = None
        self.mp_results = None
        self.last_detect_result = 0
        self.command = None

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def _show_mediapipe(self):
        if self.mp_results.multi_hand_landmarks:
            for handlms in self.mp_results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(self.frame, handlms, self.mpHands.HAND_CONNECTIONS)

    def _get_index_finger_tip(self):
        landmarks = self.mp_results.multi_hand_landmarks[0].landmark
        # print(f"Index finger displacement in 1 second: X={landmarks[8].x}, Y={landmarks[8].y}, Z={landmarks[8].z}")
        return landmarks[8].x, landmarks[8].y, landmarks[8].z

    def _get_distance(self):
        prev_x, prev_y, prev_z = self.first_index_finger_tip
        current_x, current_y, current_z = self.prev_index_finger_tip
        distance = math.sqrt((prev_x - current_x)**2 + (prev_y - current_y)**2 + (prev_z - current_z)**2)
        return distance

    def send_params_to_tello(self):
        self.tello_controller.tello_params.distance = self._get_distance()
        self.tello_controller.tello_params.inter_frame = len(self.inference.need_classify_video)
        self.tello_controller.tello_params.mpresults = self.mp_results
        self.tello_controller.tello_params.command = self.command

    def _inference_system(self, frame_rgb):
        rgb_image = Image.fromarray(frame_rgb)
        rgb_cache = rgb_image.convert("RGB")
        detect_result = self.inference.inference_detect_pth(rgb_cache)
        if detect_result == 1 and self.mp_results.multi_hand_landmarks is not None:
            if self.first_index_finger_tip is None:
                self.first_index_finger_tip = self._get_index_finger_tip()
            self.inference.need_classify_video.append(rgb_cache)
            self.prev_index_finger_tip = self._get_index_finger_tip()
            self.prev_mp_results = self.mp_results.multi_hand_landmarks
        elif detect_result == 0 and self.last_detect_result == 1 and len(self.inference.need_classify_video) > 8 and self.prev_mp_results is not None:
            print("----------------------------- classify -----------------------------")
            # 进行手势识别推理
            self.command = self.inference.inference_pth()
            # 控制无人机
            self.send_params_to_tello()
            self.tello_controller.control()
            self.inference.need_classify_video = []
            self.first_index_finger_tip = None
        self.last_detect_result = detect_result

    def system_run(self):
        while True:
            success, self.frame = self.cap.read()
            if success:
                frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                self.mp_results = self.hands.process(frame_rgb)
                self._show_mediapipe()
                self._inference_system(frame_rgb)
                cv2.imshow('frame', self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break




if __name__ == '__main__':
    interactiva_system = InteractivaSystem()
    interactiva_system.system_run()

    
