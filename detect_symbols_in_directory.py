
import os
import cv2
import sys
from tqdm import tqdm

# 'deepscores_workspace'가 상위 디렉터리에 있을 경우를 대비해 경로 추가
# 이 스크립트를 프로젝트 루트에서 실행한다고 가정합니다.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from deepscores_workspace.symbol_detector import SymbolDetector
except ImportError:
    print("오류: SymbolDetector를 deepscores_workspace/symbol_detector.py에서 찾을 수 없습니다.")
    print("스크립트를 ScoreEye 프로젝트의 루트 디렉터리에서 실행하고 있는지 확인하세요.")
    sys.exit(1)

def detect_symbols_in_directory(model_path, input_dir, output_dir):
    """
    지정된 디렉터리의 모든 PNG 파일에서 악보 기호를 검출하고 결과를 저장합니다.

    :param model_path: 학습된 YOLO 모델 파일 경로 (.pt)
    :param input_dir: 입력 PNG 파일들이 있는 디렉터리
    :param output_dir: 결과 이미지를 저장할 디렉터리
    """
    # 모델 파일 존재 여부 확인
    if not os.path.exists(model_path):
        print(f"오류: 모델 파일을 찾을 수 없습니다 - '{model_path}'")
        print("학습된 .pt 파일을 해당 경로에 위치시켜 주세요.")
        return

    # 입력 디렉터리 존재 여부 확인
    if not os.path.isdir(input_dir):
        print(f"오류: 입력 디렉터리를 찾을 수 없습니다 - '{input_dir}'")
        return

    # 출력 디렉터리 생성
    os.makedirs(output_dir, exist_ok=True)
    print(f"결과를 '{output_dir}' 디렉터리에 저장합니다.")

    # SymbolDetector 초기화
    try:
        detector = SymbolDetector(model_path)
    except Exception as e:
        print(f"SymbolDetector 초기화 중 오류 발생: {e}")
        return

    # 입력 디렉터리 내의 PNG 파일 목록 가져오기
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
    if not image_files:
        print(f"입력 디렉터리 '{input_dir}'에서 PNG 파일을 찾을 수 없습니다.")
        return

    print(f"총 {len(image_files)}개의 이미지 파일에 대해 기호 검출을 시작합니다...")

    # 각 이미지 파일에 대해 기호 검출 수행
    for filename in tqdm(image_files, desc="기호 검출 진행 중"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"detected_{filename}")

        try:
            # 이미지 로드
            image = cv2.imread(input_path)
            if image is None:
                print(f"경고: '{filename}' 파일을 읽을 수 없습니다. 건너뜁니다.")
                continue

            # 기호 검출
            symbols = detector.detect(image)

            # 결과 시각화
            for sym in symbols:
                x1, y1, x2, y2 = sym['box']
                label = f"{sym['class_name']} {sym['confidence']:.2f}"
                
                # 바운딩 박스 그리기
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2) # 빨간색 사각형
                
                # 라벨 텍스트 배경 그리기
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(image, (x1, y1 - h - 10), (x1 + w, y1), (0, 0, 255), -1)
                
                # 라벨 텍스트 쓰기
                cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) # 흰색 텍스트

            # 결과 이미지 저장
            cv2.imwrite(output_path, image)

        except Exception as e:
            print(f"'{filename}' 처리 중 오류 발생: {e}")

    print("모든 이미지 처리가 완료되었습니다.")


if __name__ == "__main__":
    # --- 설정 변수 ---
    #MODEL_PATH = "models/deepscores_yolov8s.pt"  # 학습된 모델 파일 경로
    MODEL_PATH = "deepscores_workspace/scoreeye-yolov8/stem_fixed_2048_batch22/weights/best.pt"  # 학습된 모델 파일 경로
    INPUT_DIRECTORY = "gazza_png_pages"           # 변환된 PNG 페이지들이 있는 폴더
    OUTPUT_DIRECTORY = "detection_results"        # 결과물을 저장할 폴더
    # -----------------

    # 명령줄 인수로 경로를 받을 수도 있습니다 (선택 사항)
    if len(sys.argv) == 4:
        MODEL_PATH = sys.argv[1]
        INPUT_DIRECTORY = sys.argv[2]
        OUTPUT_DIRECTORY = sys.argv[3]
    elif len(sys.argv) != 1:
        print("사용법: python detect_symbols_in_directory.py")
        print("또는: python detect_symbols_in_directory.py <모델_경로> <입력_디렉터리> <출력_디렉터리>")
        sys.exit(1)

    detect_symbols_in_directory(MODEL_PATH, INPUT_DIRECTORY, OUTPUT_DIRECTORY)
