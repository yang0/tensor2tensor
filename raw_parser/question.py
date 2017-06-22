#coding=utf-8

from __future__ import print_function

import json

from os.path import basename
import os

RAW_FILE = "data/raw/diagnosis/question.json"


#提问句最多40个字，回答最多50
MAX_SRC_SIZE = 40
MAX_TARGET_SIZE = 50


class Question:
    def splitTrainTestFile(self):
        _ROOTDIR = os.path.split(os.path.realpath(__file__))[0]
        base = basename(RAW_FILE)
        filePrefix = os.path.splitext(base)[0]
        srcFilePath = "data/aligned/" + filePrefix + "_train_src.txt"
        targetFilePath = "data/aligned/" + filePrefix + "_train_target.txt"

        if os.path.exists(srcFilePath):
            print("文件已存在")
            return

        srcFile = open(srcFilePath, "w")
        targetFile = open(targetFilePath, "w")

        with open(RAW_FILE) as f:
            i= 0
            for line in f:
                data = json.loads(line)
                question = data["description"].strip().replace(' ', '').replace("\n", "")
                answer = data["answer"].strip().replace(' ', '').replace("\n", "")
                if question != '' and answer != '' and len(question) < MAX_SRC_SIZE and len(answer) < MAX_TARGET_SIZE:
                    srcFile.write(" ".join(question) + "\n")
                    targetFile.write(" ".join(answer) + "\n")

                    print("[%s] 生成会话 %d" % (base, i))
                    i += 1

