import os
import csv
from PIL import Image # 이미지 처리를 위한 Pillow 라이브러리

def create_csv_from_image_folders(root_dir, output_csv_file):
    """
    숫자별로 정리된 이미지 폴더를 읽어 하나의 CSV 파일로 변환합니다.
    CSV 형식: label, pixel_0, pixel_1, ..., pixel_783

    Args:
        root_dir (str): '0', '1', ... 폴더가 들어있는 최상위 폴더 경로.
        output_csv_file (str): 저장될 CSV 파일 경로.
    """
    print(f"'{root_dir}' 디렉토리에서 이미지 변환을 시작합니다...")

    # CSV 파일에 쓰기 위한 준비
    try:
        with open(output_csv_file, 'w', newline='') as f_csv:
            writer = csv.writer(f_csv)

            # CSV 헤더 작성
            header = ['label'] + [f'pixel_{i}' for i in range(28 * 28)]
            writer.writerow(header)

            # 루트 디렉토리의 하위 폴더들을 순회 (0, 1, 2, ... 순으로 정렬)
            # os.listdir()는 순서를 보장하지 않으므로 정렬해주는 것이 좋습니다.
            dir_list = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

            total_images = 0
            for label in dir_list:
                if not label.isdigit():
                    print(f"폴더 '{label}'는 숫자 이름이 아니므로 건너뜁니다.")
                    continue

                label_dir = os.path.join(root_dir, label)
                print(f"폴더 '{label}' 처리 중...")

                # 폴더 내의 모든 파일 순회
                for filename in os.listdir(label_dir):
                    # 이미지 파일인지 간단히 확장자로 확인 (필요시 .jpeg, .bmp 등 추가)
                    if filename.lower().endswith(('.png', '.jpg')):
                        image_path = os.path.join(label_dir, filename)
                        
                        try:
                            # 이미지 열기 -> 흑백 변환 -> 28x28 리사이즈
                            img = Image.open(image_path).convert('L').resize((28, 28))
                            
                            # 픽셀 데이터 추출 (0~255 값의 리스트)
                            pixels = list(img.getdata())
                            
                            # [레이블, 픽셀값들...] 형태의 행 데이터 생성
                            row = [int(label)] + pixels
                            
                            # CSV 파일에 한 줄 쓰기
                            writer.writerow(row)
                            total_images += 1

                        except Exception as e:
                            print(f"  - 파일 '{filename}' 처리 중 오류 발생: {e}")
            
            print(f"\n변환 성공! 총 {total_images}개의 이미지를 '{output_csv_file}' 파일에 저장했습니다.")

    except FileNotFoundError:
        print(f"오류: '{root_dir}' 폴더를 찾을 수 없습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")


if __name__ == '__main__':
    # --- 설정 ---
    # '0', '1', '2'... 폴더가 들어있는 상위 폴더 경로를 지정하세요.
    # 예: 'mnist_images' 폴더 안에 '0', '1' 등의 폴더가 있다면 'mnist_images'를 입력
    image_root_folder = 'mnist_images' # <--- 이 부분을 실제 폴더 이름으로 변경하세요.

    # 저장될 CSV 파일의 이름을 지정하세요.
    output_csv_filename = 'mnist_dataset.csv'

    # --- 실행 ---
    if os.path.isdir(image_root_folder):
        create_csv_from_image_folders(image_root_folder, output_csv_filename)
    else:
        print(f"입력 폴더 '{image_root_folder}'을(를) 찾을 수 없습니다.")
        print("스크립트의 'image_root_folder' 변수를 올바른 폴더 경로로 수정해주세요.")
        print("예상 폴더 구조:")
        print(f"{image_root_folder}/")
        print("├── 0/")
        print("│   ├── 1.png")
        print("│   └── 2.png")
        print("├── 1/")
        print("│   ├── 3.png")
        print("│   └── 4.png")
        print("...")
