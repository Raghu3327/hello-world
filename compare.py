
from skimage import measure 
import matplotlib.pyplot as plt
import numpy as np
import cv2

def mse(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	return err
 
def compare_images(imageA, imageB, title):
	
	m = mse(imageA, imageB)
	s = measure.compare_ssim(imageA, imageB)
 
	
	fig = plt.figure(title)
	plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
 
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, cmap = plt.cm.gray)
	plt.axis("off")
 
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, cmap = plt.cm.gray)
	plt.axis("off")
 

	plt.show()
    
original = cv2.imread("R:\\New folder (2)\\Modificada.jpg")
person1 = cv2.imread("R:\\New folder (2)\\sulai.jpg")
person2 = cv2.imread("R:\\New folder (2)\\uthra.jpg")
 

original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
person1 = cv2.cvtColor(person1, cv2.COLOR_BGR2GRAY)
person2 = cv2.cvtColor(person2, cv2.COLOR_BGR2GRAY)


fig = plt.figure("Images")
images = ("Original", original), ("person1", person1), ("person2", person2)
 
for (i, (name, image)) in enumerate(images):

	ax = fig.add_subplot(1, 3, i + 1)
	ax.set_title(name)
	plt.imshow(image, cmap = plt.cm.gray)
	plt.axis("off")
 

plt.show()

 
compare_images(original, original, "Original vs. Original")
print('true')
compare_images(original,person1, "Original vs.person1")
print('false')
compare_images(original, person2, "Original vs. person2")
print('false')

