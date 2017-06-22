#coding=utf-8

from __future__ import print_function
from os.path import basename
import os
from raw_parser.cutwords_helper import cutWords

RAW_FILE="data/raw/multi100w/multi100w.data"

#提问句最多40个字，回答最多50
MAX_SRC_SIZE = 40
MAX_TARGET_SIZE = 50


#这个库来自购买的多轮聊天数据
class Multi100w:
    """
    """

    def __init__(self):
        """
        Args:
            lightweightFile (string): file containing our lightweight-formatted corpus
        """
        self.CONVERSATION_SEP = ""

    # 把一组连续会话拆成question,anwser对，拆开会话的同时也做了分词，只不过是按照一个字一个词分的
    def splitConversations(self, conversations, userJieba=False):
        convertionPairs = []
        for i in range(len(conversations)):
            if i >= len(conversations) - 1:
                break

            question = conversations[i].replace(' ', '')
            answer = conversations[i+1].replace(' ', '')

            if userJieba:
                question = cutWords(question).split()
                answer = cutWords(answer).split()


            if question != '' and answer != '' and len(question) < MAX_SRC_SIZE and len(answer) < MAX_TARGET_SIZE:
                convertionPairs.append([" ".join(question), " ".join(answer)])

        return convertionPairs



    def conversationIterator(self, fileName):
        print("从 %s 中读取会话" % fileName)
        with open(fileName, "r") as f:
            linesBuffer = []
            for line in f:
                l = line.strip()
                if l == self.CONVERSATION_SEP:
                    yield self.splitConversations(linesBuffer, userJieba=True)
                    linesBuffer = []
                else:
                    linesBuffer.append(l)



    def splitTrainTestFile(self):
        """
        把cutfile分割成训练和测试文件
        :return:
        """
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


        i = 0
        for conversations in self.conversationIterator(RAW_FILE):
            for conv in conversations:
                srcFile.write(conv[0] + "\n")
                targetFile.write(conv[1] + "\n")

                print("[%s] 生成会话 %d" % (base, i))
                i += 1

        srcFile.close()
        targetFile.close()







