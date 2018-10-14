import cv2
import numpy as np
import glob
import os

#dataset_name="Bisque/"
window=220
slide=2
threshold=210
classf="Non_cancerous"
path="/home/vindhya/Documents/Siamese_Project/Dataset/BreakHis/Train/label0"
dest="/home/vindhya/Documents/Siamese_Project/Dataset/BreakHis/"+classf+"_patches_220/"
files = sorted(glob.glob(os.path.join(path, "*.png")))
if not os.path.exists(dest):
    os.makedirs(dest)
for file in files:
	img_name=file.split("/")[-1]
	img_name=img_name.replace('.tif','')
	img=cv2.imread(file)
	row,col= img.shape[:2]
	for i in range(0,row-window,np.int(window/slide)):
		for j in range(0,col-window,np.int(window/slide)):
			patch=img[i:i+window,j:j+window]
			channel_1=patch[:,:,0]>threshold
			channel_2=patch[:,:,1]>threshold
			channel_3=patch[:,:,2]>threshold
			vfunc=np.vectorize(np.logical_and)
			pixel_and=vfunc(vfunc(channel_1,channel_2),channel_3)
			pixel_and_count=np.count_nonzero(pixel_and)
			# print(pixel_and_count)
			ratio_white_pixel=float(pixel_and_count*100/(200*200))
			# print(ratio_white_pixel)-+
			if (ratio_white_pixel<40):
			 	cv2.imwrite(dest+img_name+"_"+str(i)+"_"+str(j)+".tif",patch)
			# # prob0=patch[:,:,0].mean()
			# prob1=patch[:,:,1].mean()
			# prob2=patch[:,:,2].mean()
			# if(prob0 < threshold and prob1 < threshold and prob2 < threshold):
			# 	cv2.imwrite(dest+img_name+"_"+str(i)+"_"+str(j)+".tif",patch)
 
			# print(patch.shape)
			# cv2.imshow("patch",patch)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()
			
		
	
	


