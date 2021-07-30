#!/usr/bin/env python
# coding: utf-8

# In[1]:


import textwrap
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from PIL import Image, ImageDraw, ImageFont
import re
import pandas as pd
import base64

def centered_text(image_path,csv_path,cap_inp):

    a = range(1,15)
    print(csv_path)
    heading_text=list(map(str,a))

    df = pd.read_csv(csv_path,error_bad_lines=False)
    print(df.head())
    df=df['TEXT']
    df1=df.drop_duplicates()
    df1.dropna(inplace=True)
    df = df1.to_frame().reset_index()
    list_text = df['TEXT'].astype(str).values.tolist()
    list_text=[s for s in list_text if s != 'Answer']

#     cap_options = ["WITH DIGIT", "WITHOUT DIGIT"]
#     print("1: " + cap_options[0] + " 2: " + cap_options[1])
#     cap_inp = input("with digit or without digit: choose option? ")
    if cap_inp == "1":

        texts=[]
        for sents in list_text:
            sents=sents.split("\n")
            texts.append(sents)

#         options=["LEFT","CENTER"]
#         print("1: "+ options[0] + " 2: "+ options[1])
#         inp = input("Heading should be on which side, enter a number: ")
        inp="right"
        head_size=120
        body_size=60
        stopwords_list = [line.rstrip('\n') for line in open("C:\\dev\\Marketing work\\stopwords")]

        font1 = ImageFont.truetype("C:\\dev\\Marketing work\\font3.ttf", head_size)
        font2 = ImageFont.truetype("C:\\dev\\Marketing work\\font3.ttf", body_size)
        counter = 1
        i = 1

        for sents in texts:
            path0 = 'folder {}'.format(i)
            path = os.mkdir(path0)
            print(len(heading_text))
            print(len(sent))

            for text1,text2 in zip(heading_text,sents):
               
                image = open(image_path, 'rb')
                image_read = image.read()
                image_64_encode = base64.encodestring(image_read)
                image_64_decode = base64.decodestring(image_64_encode)
                image_result = open('decoded_img.png', 'wb')  # create a writable image and write the decoding result
                image_result.write(image_64_decode)
                image = Image.open('decoded_img.png')
                draw = ImageDraw.Draw(image)
                text2_len=len(text2)

                if text2_len <= 50:
                    startHeight=500
                    breather=250

                if text2_len >=50:
                    startHeight= 350
                    breather= 175


                if text2_len >=85:
                    startHeight=200
                    breather=100


                lowercase_words = re.split(" ", text2.lower())
                text2 = [lowercase_words[0].capitalize()]
                text2 += [word if word in stopwords_list else word.capitalize() for word in lowercase_words[1:]]
                text2 = " ".join(text2)
                iw, ih = image.size

                w1, h1 = font1.getsize(text1)
                w2, h2 = font2.getsize(text2)

                textX1 = int((iw - w1) / 2)
                textX2 = int((iw - w2) / 2)

                lines1 = textwrap.wrap(text1, width=15)
                lines2 = textwrap.wrap(text2, width=26)

                y_text1 = h1
                for line in lines1:
                    width, height = font1.getsize(line)
                    if inp == 'right':
                        pos_value = 170
                        pos_value = int(pos_value)
                    if inp == 'left':
                        pos_value = int((iw - width) / 2)
                    draw.text((pos_value, startHeight - breather +y_text1), line, font=font1, align="left", color="red")
                    y_text1 += height

                y_text2 = y_text1+70
                for line in lines2:
                    width, height = font2.getsize(line)
                    draw.text((170, startHeight - breather +y_text2), line, font=font2, align="center" , color="black")
                    y_text2 += height

                image.save(path0 + "/" + str(counter) + '.jpg')
                counter += 1
                i += 1

    if cap_inp == "2":

        texts = []
        for sents in list_text:
            sents = sents.split("\n")
            texts.append(sents)

        body_size = 60
        stopwords_list = [line.rstrip('\n') for line in open("C:\\dev\\Marketing work\\stopwords")]

        font2 = ImageFont.truetype("C:\\dev\\Marketing work\\font3.ttf", body_size)
        counter = 1
        i = 1

        for sents in texts:
            path0 = 'folder {}'.format(i)
            path = os.mkdir(path0)

            for text2 in (sents):
                image = open(image_path, 'rb')
                image_read = image.read()
                image_64_encode = base64.encodestring(image_read)
                image_64_decode = base64.decodestring(image_64_encode)
                image_result = open('decoded_img.png', 'wb')  # create a writable image and write the decoding result
                image_result.write(image_64_decode)
                image = Image.open('decoded_img.png')
                draw = ImageDraw.Draw(image)
                text2_len = len(text2)

                if text2_len <= 50:
                    startHeight=650
                    breather=250

                if text2_len >=50:
                    startHeight= 500
                    breather= 175


                if text2_len >=85:
                    startHeight=420
                    breather=100

                lowercase_words = re.split(" ", text2.lower())
                text2 = [lowercase_words[0].capitalize()]
                text2 += [word if word in stopwords_list else word.capitalize() for word in lowercase_words[1:]]
                text2 = " ".join(text2)
                iw, ih = image.size

                #w1, h1 = font1.getsize(text1)
                w2, h2 = font2.getsize(text2)

                textX2 = int((iw - w2) / 2)

                lines2 = textwrap.wrap(text2, width=26)

                #y_text1 = h1
                # for line in lines1:
                #     width, height = font1.getsize(line)
                #     if inp == '1':
                #         pos_value = 170
                #         pos_value = int(pos_value)
                #     if inp == '2':
                #         pos_value = int((iw - width) / 2)
                #     draw.text((pos_value, startHeight - breather + y_text1), line, font=font1, align="left",
                #               color="red")
                #     y_text1 += height

                y_text2 = h2
                for line in lines2:
                    width, height = font2.getsize(line)
                    draw.text((int((iw - width) / 2), startHeight - breather + y_text2), line, font=font2, align="center", color="black")
                    y_text2 += height

                image.save(path0 + "/" + str(counter) + '.jpg')
                counter += 1
                i += 1
# centered_text(image_path="1.png",csv_path="y.csv",cap_inp='1')


# In[2]:


from flask import Flask, render_template, request
import os
app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = "C:\\dev\\Marketing work\\centered"
app.config["FILE_UPLOADS"] = "C:\\dev\\Marketing work\\centered"

@app.route("/")
def hello():
    return render_template("index1.html")
@app.route("/get_image", methods=['GET', 'POST'])
def test():
#     print("hello")
    if request.method == "POST":
        text = request.form.get('status')
        img = request.files["media"]
        print(img)
        img.save(os.path.join(app.config["IMAGE_UPLOADS"], img.filename))
        img = request.files["media"]
        print("-------------error-----------")
        myfile = request.files["filess"]
        print(myfile)
        myfile.save(os.path.join(app.config["FILE_UPLOADS"], myfile.filename))

        print(myfile)

        opt3 = request.form.get("option_boxxx")
        

        print(opt3)
#         r1, g1, b1  = tuple(int(first.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
#         second = request.form.get("second")
#         r2, g2, b2  = tuple(int(second.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        
        
#         solidcolorr = request.form.get("solidcolorr")
#         r, g, b  = tuple(int(solidcolorr.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
#             #(inp1,r1,g1,b1,r2,g2,b2,img_path,text)
#             #print(img.filename)
#         if colors=='solid':
#             all_input2(opt1,r,g,b,app.config["IMAGE_UPLOADS"]+'/'+img.filename,text)
            
#         elif colors=='gradient':
#             all_inputs(opt1,r1, g1, b1, r2, g2, b2,app.config["IMAGE_UPLOADS"]+'/'+img.filename, text)
            

        centered_text(app.config["IMAGE_UPLOADS"]+'/'+img.filename,app.config["FILE_UPLOADS"]+'/'+myfile.filename,opt3 )


# In[ ]:


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8006)