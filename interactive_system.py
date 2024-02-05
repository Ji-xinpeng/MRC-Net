import cv2
import os
import time
from PIL import Image
import mediapipe as mp
from inference_utils.inference_class import Inference
from human_machine_system.tello import TelloController

class InteractivaSystem:
    def __init__(self) -> None:
        # 0代表电脑自带的摄像头
        self.cap = cv2.VideoCapture(0) 
        self._parse_weights_path()
        self._init_inference_model()
        self._init_mediapipe()
        self._init_varbile()
        self.tello = TelloController()

    def _init_tello(self):
        self.tello = TelloController()
        # self.tello.tello_connect()

    def _parse_weights_path(self):
        current_file = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file)
        self.checkpoint_path_classify = current_directory + '/weights/mobilenetv2classify.pth'
        self.checkpoint_path_detect = current_directory + '/weights/mobilenetv2detect.pth'

    def _init_inference_model(self):
        self.inference = Inference(self.checkpoint_path_classify, self.checkpoint_path_detect)

    def _init_check_camera(self):
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

    def _init_varbile(self):
        self.frame = None
        self.mp_results = None
        self.last_detect_result = 0

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def _show_mediapipe(self):
        # print(f'\r预测的类别是: {self.mp_results.multi_hand_landmarks}', end='')
        if self.mp_results.multi_hand_landmarks:
            for handlms in self.mp_results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(self.frame, handlms, self.mpHands.HAND_CONNECTIONS)

    def _inference_system(self, frame_rgb):
        rgb_image = Image.fromarray(frame_rgb)
        rgb_cache = rgb_image.convert("RGB")
        detect_result = self.inference.inference_detect_pth(rgb_cache)
        if detect_result == 1 and self.mp_results.multi_hand_landmarks != None:
            self.inference.need_classify_video.append(rgb_cache)
        elif detect_result == 0 and self.last_detect_result == 1 and len(self.inference.need_classify_video) > 8:
            print("----------------------------- classify -----------------------------")
            # 进行手势识别推理
            command = self.inference.inference_pth()
            # 控制无人机
            self.tello.control(command, self.mp_results)
            self.inference.need_classify_video = []
        self.last_detect_result = detect_result

    def system_run(self):
        self._init_check_camera()
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




def main():
    interactiva_system = InteractivaSystem()
    interactiva_system.system_run()

if __name__ == '__main__':
    main()
    
