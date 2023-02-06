import os

# Update classes.txt
trainDirs = os.listdir('pytorch-inceptionv3/data/hymenoptera_data/train')
valDirs = os.listdir('pytorch-inceptionv3/data/hymenoptera_data/val')

if trainDirs != valDirs:
    print("Train directory and validation directory are different. Using train directory as class labels")

print(trainDirs)

with open("pytorch-inceptionv3/classes.txt", "w") as f:
    for i in trainDirs:
        f.write(i)
        if i != trainDirs[-1]:
            f.write("\n")
    print(f"{len(trainDirs)} classes labeled successfully!")