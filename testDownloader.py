import shutil
import sys

file = open('filetocopy.txt')

content = file.readlines()
for i in range(0, len(content)):
    content[i] = content[i].strip('\n')
shutil.copy(content[int(sys.argv[1])], "./tmp/newData")

