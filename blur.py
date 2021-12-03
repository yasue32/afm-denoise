import cv2
import matplotlib.pyplot as plt
import glob
import os 

filepath ="afm_dataset4/20211126/"

files = [line.rstrip() for line in open((filepath+"sep_trainlist.txt"))]
files = glob.glob("orig_img/20211112/*")
def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()


gt_fm = []
input_fm = []
for i, file in enumerate(files):
    image = cv2.imread(file)
    # image = cv2.imread(filepath + file)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(image)
    input_fm.append(fm)

    # file_gt = "/".join(file.split("/")[:-1] + ["gt.png"])
    # image = cv2.imread(filepath + file_gt)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # fm = variance_of_laplacian(image)
    # gt_fm.append(fm)

    # if (i+1)%25==0:
    if fm < 500:
        text = "Blurry"
    elif fm>2000:
        text = "Noisy"
    else:
        text = "Not blurry"
	# show the image
    os.makedirs("blur/"+file[:-9], exist_ok=True)
    cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.imwrite("blur/"+file, image)
    # fig = plt.figure()
    # plt.imshow(image)
    # fig.savefig("blur/"+file)
print("iter", i)
# print("gt:", sum(gt_fm)/len(gt_fm))
print("input:", sum(input_fm)/len(input_fm))

fig = plt.figure()
plt.scatter(list(range(len(input_fm))), input_fm)
# plt.scatter(list(range(len(gt_fm))), gt_fm)
fig.savefig("img_1126.png")

# print("gt:", sum(gt_fm)/len(gt_fm))
# print("input:", sum(input_fm)/len(input_fm))
# print(len(gt_fm))