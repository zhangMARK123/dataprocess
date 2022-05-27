import cv2
import os
import shutil



###复制该文件夹下文件到另一目录
def move_file(orgin_path,moved_path1,moved_path2):
    dir_files=os.listdir(orgin_path)            #得到该文件夹下所有的文件
    for file in  dir_files:
        file_path=os.path.join(orgin_path,file)   #路径拼接成绝对路径 
        if os.path.isfile(file_path):           #如果是文件，就打印这个文件路径
            if file.endswith(".jpg"):
                if os.path.exists(os.path.join(moved_path1,file)):
                    continue
                else:
                    shutil.copy(file_path, moved_path1)  
            elif file.endswith(".json"):
                if os.path.exists(os.path.join(moved_path2,file)):
                    continue
                else:
                    shutil.copy(file_path, moved_path2)  
        if os.path.isdir(file_path):  #如果目录，就递归子目录
            move_file(file_path,moved_path1,moved_path2)
    
if __name__ == '__main__':
    orgin_path = r'C:\Users\zxpan\Desktop\daily_report\data\lightwithdire\W_20210629'      
    moved_path1 = r'C:\Users\zxpan\Desktop\daily_report\data\lightwithdire\images'    
    moved_path2=r"C:\Users\zxpan\Desktop\daily_report\data\lightwithdire\labels"  
    move_file(orgin_path,moved_path1,moved_path2)

