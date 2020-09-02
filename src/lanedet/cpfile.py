import os 

def main():
    with open('/space/data/lane/ret_acc20_test.txt', 'r') as f:
        i = 0
        for line in f.readlines():
            srcImgPath = line[:-1]
            srcLabelPath = srcImgPath[:-3] + 'lines.txt'
            dstImgPath = '/space/data/lane/dataset/' + str(i) + '.jpg'
            dstLabelPath = '/space/data/lane/dataset/' + str(i) + '.lines.txt'
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