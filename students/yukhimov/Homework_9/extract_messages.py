from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import os
import json

request_messages = []

for filename in os.listdir('1551/'):
    category = filename.split('.')[0]
    messages = []
    with open('1551/' + filename, "r") as input_file:
        for line in input_file:
            if line.strip().isnumeric() or not line.strip():
                messages.append(line)
            else:
                messages.append(line.strip())
    splitted_messages = ''.join(messages).split('\n\n')
    for item in splitted_messages:
        message = item.split('\n')
        if message[0].isnumeric():
            text = ' '.join(message[1:])
            try:
                if detect(text) == 'uk':
                    request_messages.append({'id': message[0], 'category': category, 'message': text})
            except LangDetectException:
                pass

with open('request_messages.txt', 'w') as outfile:
    json.dump({'messages': request_messages}, outfile, ensure_ascii=False, indent=4)

