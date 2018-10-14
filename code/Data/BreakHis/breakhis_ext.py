import glob
import os
import shutil

class_name='malignant'
path='/home/vindhya/Documents/Siamese_Project/BreaKHis_v1/histology_slides/breast/'+class_name+'/SOB'
dest='/home/vindhya/Documents/Siamese_Project/'+class_name
if not os.path.exists(dest):
	os.mkdir(dest)
for (root,dirc,files) in os.walk(path,topdown='True'):
	if (root.split('/')[-1]=='400X'):
		for file in files:
			shutil.copy(os.path.join(root,file),dest)
	