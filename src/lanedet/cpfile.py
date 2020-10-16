import os
import random
import glob

def main():
    # with open('/space/data/lane/finallist.txt', 'r') as f:
    #     i = 0
    #     # lines = f.readlines()
    #     # pickedLines = random.sample(lines, 500)
    #     for line in f.readlines():
    #         # i = random.randint(0,10000)
    #         srcImgPath = line[:-1]
    #         srcLabelPath = srcImgPath[:-3] + 'lines.txt'
    #         dstImgPath = '/space/data/lane/finaldataset/' + str(i) + '.jpg'
    #         dstLabelPath = '/space/data/lane/finaldataset/' + str(i) + '.lines.txt'
    #         i+=1

    #         # print(srcImgPath)
    #         # print(srcLabelPath)
    #         # print(dstImgPath)
    #         # print(dstLabelPath)
    #         cmd = 'cp {} {}'.format(srcImgPath, dstImgPath)
    #         os.system(cmd)
    #         cmd = 'cp {} {}'.format(srcLabelPath, dstLabelPath)
    #         os.system(cmd)

    imgs = glob.glob('/space/data/lane/daystraight/*.jpg')
    pickedLines = random.sample(imgs, 100)
    i=1
    for line in pickedLines:
            # i = random.randint(0,10000)
            srcImgPath = line[:]
            srcLabelPath = srcImgPath[:-3] + 'lines.txt'
            imgNum = srcImgPath[srcImgPath.rfind('/')+1:-4]
            dstImgPath = '/space/data/lane/final100/' + str(i) + '.jpg'
            dstLabelPath = '/space/data/lane/final100/' + str(i) + '.lines.txt'
            i+=1

            # print(srcImgPath)
            # print(srcLabelPath)
            # print(dstImgPath)
            # print(dstLabelPath)
            cmd = 'cp {} {}'.format(srcImgPath, dstImgPath)
            os.system(cmd)
            cmd = 'cp {} {}'.format(srcLabelPath, dstLabelPath)
            os.system(cmd)



if __name__ == "__main__":
    main()