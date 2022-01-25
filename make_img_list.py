import glob

dates = ["20211022", 
        "20211023",
        "20211029",
        "20211030",
        "20211106",
        "20211108",
        "20211109",
        "20211112",
        "20211115",
        "20211116",
        "20211122",
        "20211126"]
filepath = "clearn_img/"
n_burst = 10

with open(filepath + "sep_trainlist.txt", mode='w') as f:
    pass
lines = 0
for d in dates:
    path = filepath + d + "/*"
    files = glob.glob(path)
    with open(filepath + "sep_trainlist.txt", mode='a') as f:
        for i in range(0,len(files), n_burst):
            if lines!=0:
                f.write("\n")
            f.write("/".join(files[i].split("/")[1:]))
            lines += 1