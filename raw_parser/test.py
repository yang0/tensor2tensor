
import os

#生成测试数据

_data = [
    ["你好", "你好"],
    ["吃饭了吗", "吃了"]
]



class Test:
    def splitTrainTestFile(self):
        srcFilePath = "data/aligned/test_src.txt"
        targetFilePath = "data/aligned/test_target.txt"

        if os.path.exists(srcFilePath):
            print("文件已存在")
            return

        srcFile = open(srcFilePath, "w")
        targetFile = open(targetFilePath, "w")

        for l in _data:
            srcFile.write(l[0] + "\n")
            targetFile.write(l[1] + "\n")
