#object 데이터 위치 : /home/nuninu98/Junseo/CaLiD Dataset/data/object   (사진+ros 파일 함께)
#viewer 데이터 위치 : /home/nuninu98/Junseo/CaLiD Dataset/data/viewer   (사진만)
#총 갯수 : 8328개

import os
import sys
import shutil
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow
from PyQt5.QtGui import QPixmap

# (1) object 폴더 경로
SOURCE_OBJECT_PATH = r'/home/nuninu98/Junseo/CaLiD Dataset/data/object'

# (2) viewer 폴더 경로
SOURCE_VIEWER_PATH = r'/home/nuninu98/Junseo/CaLiD Dataset/data/viewer'

# (3) classfied object 폴더들이 저장될 위치
DESTINATION_OBJECT_PATH = r'/home/nuninu98/Junseo/CaLiD Dataset/data/object_classfied'

# (4) classfied viewer 이미지(viewer_*.png)들이 저장될 위치
DESTINATION_VIEWER_PATH = r'/home/nuninu98/Junseo/CaLiD Dataset/data/viewer_classfied'

# (5) 매칭 이미지 파일의 접두사와 확장자
MATCHING_IMAGE_PREFIX = 'viewer_'
MATCHING_IMAGE_EXTENSION = '.png'

class ImageViewer(QMainWindow):
    def __init__(self, image_path):
        super().__init__()
        self.setWindowTitle("Image Classifier")
        
        pixmap = QPixmap(image_path)
        screen_resolution = QApplication.primaryScreen().availableGeometry()
        # 이미지를 화면의 절반, 비율은 유지
        pixmap = pixmap.scaled(
            screen_resolution.width() // 2, 
            screen_resolution.height() - 100, 
            1 
        )

        label = QLabel(self)
        label.setPixmap(pixmap)
        self.setCentralWidget(label)
        self.adjustSize()
        # 창을 화면 정중앙에 뜨도록..
        self.move(
            (screen_resolution.width() - self.width()) // 2,
            (screen_resolution.height() - self.height()) // 2
        )

def find_image_in_folder(folder_path): #이미지 폴더 찾기
    for item in os.listdir(folder_path):
        if item.lower().endswith(('.png')):
            return os.path.join(folder_path, item)
    return None

def main():
    print("--- 이미지 분류 시작~ ---")
    
    for path in [DESTINATION_OBJECT_PATH, DESTINATION_VIEWER_PATH]: #각각 object_classfied, viewer_classfied 폴더!
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"저장 폴더 생성: {path}") #새로운 종류가 나오면 새 폴더 생성

    classified_folders = set()
    for _, dirs, _ in os.walk(DESTINATION_OBJECT_PATH): #object_classfied 폴더 검색
        classified_folders.update(d for d in dirs)

    subfolders_to_classify = sorted([
        d for d in os.listdir(SOURCE_OBJECT_PATH)
        if os.path.isdir(os.path.join(SOURCE_OBJECT_PATH, d)) and d not in classified_folders 
    ], key=int) #같은 부유물이 연속적이므로 숫자는 순차적으로
    
    if not subfolders_to_classify:
        print("끝났다. 수고했다.")
        return

    total_folders = len(subfolders_to_classify)
    print(f"아직 {total_folders}개나 남았어. 서둘러라")

    app = QApplication(sys.argv)
    viewer = None

    for i, folder_name in enumerate(subfolders_to_classify):
        current_folder_path = os.path.join(SOURCE_OBJECT_PATH, folder_name)
        
        print(f"\n---  진행 상황: {i + 1} / {total_folders} ---")
        print(f"현재 폴더: {folder_name}")

        image_path = find_image_in_folder(current_folder_path)
        if not image_path:
            print(f"'{folder_name}' 폴더에서는 이미지 없음..")
            continue
        
        
        if viewer:
            viewer.close() #렉 방지

        
        viewer = ImageViewer(image_path) #그 다음 이미지
        viewer.show()
        QApplication.processEvents() # 무한루프에서 응답성 유지

        category_name = input("카테고리 이름 입력! (건너뛰기: Enter, 종료: 'gman'): ").strip()
        
        if category_name.lower() == 'gman':
            print("프로그램 그만!.")
            break
        
        if not category_name:
            print("건너뜁니다.")
            continue

        try:
            # object_classfied 폴더로 이동
            dest_folder_object = os.path.join(DESTINATION_OBJECT_PATH, category_name) #경로
            os.makedirs(dest_folder_object, exist_ok=True)
            shutil.move(current_folder_path, os.path.join(dest_folder_object, folder_name))
            print(f"  '{dest_folder_object}' (object 이동 완료)")

            # viewer_classfied 폴더로 함께 이동
            dest_folder_viewer = os.path.join(DESTINATION_VIEWER_PATH, category_name)
            os.makedirs(dest_folder_viewer, exist_ok=True)
            
            matching_img_file = f"{MATCHING_IMAGE_PREFIX}{folder_name}{MATCHING_IMAGE_EXTENSION}" #viewer 파일명 설정
            matching_img_path = os.path.join(SOURCE_VIEWER_PATH, matching_img_file)
            
            if os.path.exists(matching_img_path):
                shutil.move(matching_img_path, dest_folder_viewer)
                print(f" '{dest_folder_viewer}' (viewer 이동 완료)")
            else:
                print(f" 매칭 이미지를 찾지 못했습니다 ({matching_img_path})")

        except Exception as e:
            print(f"오류발생!!! {e}")

    if viewer:
        viewer.close()
    print("\n--- 끝!!!!!---")

if __name__ == "__main__":
    main()