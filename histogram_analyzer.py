import os
import cv2
import numpy as np
from tqdm import tqdm

class HistogramSceneAnalyzer:
    def __init__(self, similarity_threshold=0.7):
        """
        �������������� ���������� �� ������ ����������
        param similarity_threshold: ����� �������� (�� 0 �� 1). ��� ����, ��� ������ ���������
        """
        self.threshold = similarity_threshold
        print(f"������������� HistogramSceneAnalyzer � ������� ��������: {self.threshold}")

    def _get_shot_histogram(self, shot_path):

        cap = cv2.VideoCapture(shot_path)
        if not cap.isOpened():
            return None
        
        middle_frame_index = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
            
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_frame], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def analyze_shots(self, shots_dir):
        """
        ��������� ����� � ������ � ���������� ������ ������������
        """
        print("\n--- ������ �� ������������ ---")
        shot_histograms = {}
        shot_files = sorted([f for f in os.listdir(shots_dir) if f.endswith(".mp4")])

        for filename in tqdm(shot_files, desc="���������� ����������"):
            shot_path = os.path.join(shots_dir, filename)
            shot_number = int(filename.split('-')[-1].split('.')[0])
            
            histogram = self._get_shot_histogram(shot_path)
            if histogram is not None:
                shot_histograms[shot_number] = histogram
        
        if len(shot_histograms) < 2:
            return np.array([])

        print("\n--- ������ �� ������������: ��������� �������� ����� ---")
        continuation_probabilities = []
        # �������� �� ���� �������� ����� ������
        for shot_num in range(1, len(shot_histograms)):
            # ������ ����� ���������� � 1, � � ������� ����� ���� � 1
            prev_hist = shot_histograms.get(shot_num)
            current_hist = shot_histograms.get(shot_num + 1)
            
            if prev_hist is not None and current_hist is not None:
                similarity = cv2.compareHist(prev_hist, current_hist, cv2.HISTCMP_CORREL)
                
                # �������� �������� (�� -1 �� 1) � ����������� ����������� (�� 0 �� 1)
                # ���� similarity = 1 (��������� ������) -> prob = 1.0
                # ���� similarity = -1 (��������� ������) -> prob = 0.0
                prob = (similarity + 1) / 2
                continuation_probabilities.append(prob)
            else:
                continuation_probabilities.append(0.5) # ���� ������ ���, ���� ����������� ������

        return np.array(continuation_probabilities)




