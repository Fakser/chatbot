from __future__ import absolute_import, division, print_function, unicode_literals
import os
import json
from copy import deepcopy
import unicodedata
import re

def unicode_to_ascii(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn')

def repair_string(string):
    znaki = [['Ä\x85','a'],['Ä\x99','e'],['Ä\x87','c'],
            ['Å\x84','n'],['Å¼','z'],['Åº','z'],['Å\x82','l'],
            ['Å\x9b','s'],['Ã³','o']]
    for znak in znaki:
        string = string.replace(znak[0], znak[1])
    return string

def max_length(tensor):
    return max(len(t) for t in tensor)



def preprocess_sentence(w):
    w = unicode_to_ascii(repair_string(w).lower().strip())

    # creating a space between a word and the punctuation following it
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w


def load_convos():
    directories = [x for x in os.listdir('./data/')]
    print(directories)
    conversations = []
    for directory in directories:
        conversation = []
        i = 1
        while True:
            try:
                file_dir = './data/' + directory + '/message_' + chr(i+48) + '.json'
                with open(file_dir, 'r', encoding='utf-8', errors='ignore') as json_file:
                    data = json.load(json_file)
                    for message in data['messages']:
                        if 'content' in list(message.keys()) and 'sender_name' in list(message.keys()):
                            conversation.append([message['sender_name'], message['content']])
                print(file_dir)
                i += 1
            except:
                break
        conversations.append(deepcopy(conversation))
        del conversation
    return conversations


def merge_convos(conversations):
    conversations_merged = []
    for conversation in conversations:
        merged_conversation = []
        merged_message = ''
        sender = conversation[-1][0]
        for message in reversed(conversation):
            if message[0] == sender:
                merged_message += ' '
                merged_message += message[1]
            else:
                merged_conversation.append([sender, preprocess_sentence(merged_message)])
                sender = message[0]
                merged_message = message[1]
        conversations_merged.append(deepcopy(merged_conversation))
        del merged_conversation  
    return conversations_merged

def convos_to_questions_and_answers(conversations_merged, user_name = 'Krzysztof Kramarz'):
    questions = []
    answers = []
    max_len = 20
    min_len = 2
    for conversation in conversations_merged:
        for i in range(len(conversation) - 1):
            if conversation[i+1][0] == user_name and len(conversation[i][1].split(' ')) >= min_len and len(conversation[i][1].split(' ')) <= max_len and len(conversation[i+1][1].split(' ')) >= min_len and len(conversation[i+1][1].split(' ')) <= max_len:
                questions.append(deepcopy(conversation[i][1]))
                answers.append(deepcopy(conversation[i+1][1]))
    return questions, answers
