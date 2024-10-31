import cv2
import numpy as np
import exFunc

classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
def train_by_camera(save_img = True):
    name = input("Nhập tên của bạn: ")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    face = (0,0,0,0)
    f_list = []
    auto_capture = False
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)
        faces = sorted(faces, key = lambda x: x[2] * x[3], reverse = True)
        faces = faces[:1]
        if len(faces) >= 1:
            face = faces[0]
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        # Với x,y là tọa độ khuôn mặt, w, h là chiều dài và chiều rộng
        x, y, w, h = face
        im_face = frame[y : y + h, x : x + w]
        # Tiến hành làm xám khuôn mặt bằng OpenCV
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
        if not ret:
            continue
        cv2.imshow("Face-Recognition", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            break
        elif key & 0xFF == ord("c") or auto_capture:
            if len(faces) == 1:
                gray_face = cv2.cvtColor(im_face, cv2.COLOR_BGR2GRAY)
                # Thay đổi kích thước khuôn mặt về dạng 100 x 100
                gray_face = cv2.resize(gray_face, (100, 100))
                print(len(f_list), type(gray_face), gray_face.shape)
                f_list.append(gray_face.reshape(-1))
                if (save_img):
                    exFunc.save_image(gray_face, name, len(f_list))
            else:
                print("Không tìm thấy khuôn mặt")
            if len(f_list) == 20:
                if (f_list): exFunc.write(name, np.array(f_list))
                f_list = []
                auto_capture = False
        elif key & 0xFF == ord("s"):
            auto_capture = True
    cap.release()
    cv2.destroyAllWindows()

def train_by_image(save_img = False):
    for i, name in enumerate(names):
        f_list = []
        image_path = images_path[i]
        image = cv2.imread(image_path)
        image = exFunc.image_resize(image, height = 600)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)
        faces = sorted(faces, key = lambda x: x[2] * x[3], reverse = True)
        faces = faces[:1]
        if len(faces) >= 1:
            face = faces[0]
            x, y, w, h = face
            im_face = image[y : y + h, x : x + w]
            if len(faces) == 1:
                gray_face = cv2.cvtColor(im_face, cv2.COLOR_BGR2GRAY)
                gray_face = cv2.resize(gray_face, (100, 100))
                data = gray_face.reshape(-1)
                if (save_img):
                    exFunc.save_image(gray_face, name, i)
                f_list.append(data)
                if (f_list): exFunc.write(name, np.array(f_list))
                print(name, data)
            else:
                print("Không tìm thấy khuôn mặt")
train_by_camera()