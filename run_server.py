#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from PIL import Image, ImageDraw, ImageFont
import re

import base64


# In[2]:


def all_inputs(inp1,r1,g1,b1,r2,g2,b2,img_path,text):
    

    #if ask_grad=="no":
        # ask_side=input("In which side you want gradient? ")
        # ask_side=ask_side.lower()
#         options1 = ["LEFT","RIGHT"]
#         print("1: "+ options1[0] + " 2: "+ options1[1])
#         inp1 = input("Enter a number: ")


#         r1=input("Input RGB colors of top color; r: ")
        r1=int(r1)
#         g1=input("Input RGB colors of top color; g: ")
        g1=int(g1)
#         b1=input("Input RGB colors of top color; b: ")
        b1=int(b1)

#         r2=input("Input RGB colors of bottom color; r: ")
        r2=int(r2)
#         g2=input("Input RGB colors of bottom color; g: ")
        g2=int(g2)
#         b2=input("Input RGB colors of bottom color; b: ")
        b2=int(b2)

        mask_data = []
        width= 1080
        height= 1080
        #print((r1,g1,b1), (r2,g2,b2) )
        base = Image.new('RGB', (width, height), (r1,g1,b1))
        #base.save("base.jpg")
        top = Image.new('RGB', (width, height),(r2,g2,b2))
        #top.save("top.jpg")
        mask = Image.new('L', (width, height))
        for y in range(height):
            for x in range(width):
                mask_data.append(int(300 * (y / height)))
        mask.putdata(mask_data)
        base.paste(top, (0, 0), mask)
        #base.save("gguu.jpg")

#         img_path=  #input("enter image path: ")
#         image = open(img_path, 'rb')
#         image_read = image.read()
#         image_64_encode = base64.encodestring(image_read)
#         image_64_decode = base64.decodestring(image_64_encode)
#         image_result = open('decoded_img.png', 'wb') # create a writable image and write the decoding result
#         image_result.write(image_64_decode)
        img = cv2.imread(img_path)

        hi, wi, c = img.shape
        modelFile = "c:/dev/Marketing work/res10_300x300_ssd_iter_140000.caffemodel"
        configFile = "c:/dev/Marketing work/deploy.prototxt.txt"
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

        #frame = cv2.imread(img_path)
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                texte = "{:.2f}%".format(confidence * 100)
                y = y1 - 10 if y1 - 10 > 10 else y1 + 10
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 0, 255), 2)

                cv2.putText(frame, texte, (x1, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 3)

        try:
            print("Face Detecting Points")
            left_side = x1
            print(left_side)
            right_side = wi - x2
            print(right_side)
            face_crop_limit=x1+x2
            limit=face_crop_limit
        except NameError:
            print("Body Detecting Points")
            net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
            classes = []
            with open("coco.names", "r") as f:
                classes = [line.strip() for line in f.readlines()]

            layers_names = net.getLayerNames()
            output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            colors = np.random.uniform(0, 255, size=(len(classes), 3))

            img = cv2.imread(img_path)
            img = cv2.resize(img, None, fx=0.4, fy=0.4)
            height, width, channels = img.shape

            blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
            net.setInput(blob)
            outputs = net.forward(output_layers)

            boxes = []
            confs = []
            class_ids = []
            for output in outputs:
                for detect in output:
                    scores = detect[5:]
                    class_id = np.argmax(scores)
                    conf = scores[class_id]
                    if conf > 0.3:
                        center_x = int(detect[0] * width)
                        center_y = int(detect[1] * height)
                        wb = int(detect[2] * width)
                        hb = int(detect[3] * height)
                        xb = int(center_x - wb / 2)
                        yb = int(center_y - hb / 2)
                        boxes.append([xb, yb, wb, hb])
                        confs.append(float(conf))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
            font = cv2.FONT_HERSHEY_PLAIN
            for i in range(len(boxes)):
                if i in indexes:
                    x1, y1, x2, y2 = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = colors[i]
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    if label == "person":
                        upper_left = (x1, y1)
                        lower_right = (x2, y2)
                        cv2.putText(img, label, (x1, y1 - 5), font, 1, color, 1)

            left_side=x1
            print(left_side)
            right_side=wi-x2
            print(right_side)
            body_crop_limit=wi - x2 + x1
            limit=body_crop_limit

        if right_side > left_side:
            input_img = Image.open(img_path)
            box = (0, 0, limit, 1080)
            cropped_img = input_img.crop(box)
            #cropped_img.save("bbb.jpg")
            numpy_image = np.array(cropped_img)
            cropped_img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

        else:
            img = Image.open(img_path)
            left = left_side - right_side
            top = 0
            width = wi
            height = 1080
            box = (left, top, left + width, top + height)
            cropped_img = img.crop(box)
            #cropped_img.save("bbb.jpg")
            numpy_image = np.array(cropped_img)
            
            cropped_img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

        # *****REMOVE BLACK BACKGROUND*****
        print(cropped_img)
        #cropped_img.save("bbb.jpg")
#         cv2.save(cropped_img, "cc.jpg")
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        hh, ww = thresh.shape

        thresh[hh - 3:hh, 0:ww] = 0
            
        white = np.where(thresh == 255)
        xmin, ymin, xmax, ymax = np.min(white[1]), np.min(white[0]), np.max(white[1]), np.max(white[0])

        final_crop = cropped_img[ymin:ymax + 3, xmin:xmax]
        #rint("asd",final_crop)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        color_coverted = cv2.cvtColor(final_crop, cv2.COLOR_BGR2RGB)
        final_crop=Image.fromarray(color_coverted)
        #final_crop.save("bbbb.jpg")
        def crop_center(pil_img, crop_width, crop_height):
            img_width, img_height = pil_img.size
            return pil_img.crop(((img_width - crop_width) // 2,
                                 (img_height - crop_height) // 2,
                                 (img_width + crop_width) // 2,
                                 (img_height + crop_height) // 2))

        exact_size = crop_center(final_crop, 540, 1079)
        #exact_size.save("sss.jpg")
        print(inp1)

        if inp1=="right":

        #if ask_side=="right":
            image1 = exact_size
            image2 = base
            print(image1.size)
            print(image2.size)
            x1 = 600
            y1 = 350

        if inp1=="left":

        # if ask_side=="left":
            image1 = base
            image2 = exact_size
            print(image1.size)
            print(image2.size)
            
            x1 = 45
            y1 = 350

        image1 = image1.resize((540, 1080))
        image1_size = image1.size
        image2_size = image2.size
        new_image = Image.new('RGB',(2*image1_size[0], image1_size[1]), (250,250,250))
        new_image.paste(image1,(0,0))
        new_image.paste(image2,(image1_size[0],0))
        #new_image.show()

        def text_wrap(text, font, max_width):
            lines = []

            if font.getsize(text)[0] <= max_width:
                lines.append(text)
            else:
                words = text.split(' ')
                i = 0
                while i < len(words):
                    line = ''
                    while i < len(words) and font.getsize(line + words[i])[0] <= max_width:
                        line = line + words[i] + " "
                        i += 1
                    if not line:
                        line = words[i]
                        i += 1
                    lines.append(line)
            return lines

        stopwords_list = [line.rstrip('\n') for line in open("c:/dev/Marketing work/stopwords")]
        cap_words=["AI","NBI","IEEE"]
        font_path = 'c:/dev/Marketing work/font3.ttf'
        font = ImageFont.truetype(font=font_path, size=50)
        #text = "64 PROJECTS FOR EVERY DATA SCIENCE PROFESSIONAL"
#         text=input("write text for image: ")
        
        lowercase_words = re.split(" ", text.lower())
        final_words_low = [lowercase_words[0].capitalize()]
        final_words_low += [word if word in stopwords_list else word.capitalize() for word in lowercase_words[1:]]
        final_title_low = " ".join(final_words_low)

        lines = text_wrap(final_title_low, font, 540)
        line_height = font.getsize('hg')[1]
        lines="\n ".join(lines)

        draw = ImageDraw.Draw(new_image)
        color = 'hsl(0,0%,100%)'
        draw.text((x1, y1), lines, fill=color, font=font,align="center")

        new_image.save("static/solid.jpg")


# all_inputs("2","1","21","21","21","77","21","23","d.png","jhsuiiiabduygb")


# In[3]:


def all_input2(inp1,r,g,b,img_path,text):    

    #if ask_grad=="yes":
        # ask_side=input("In which side you want gradient? ")
        # ask_side=ask_side.lower()
#         options1 = ["LEFT","RIGHT"]
#         print("1: "+ options1[0] + " 2: "+ options1[1])
#         inp1 = input("Enter a number: ")
#         r2=input("Input RGB colors ; r: ")
        r=int(r)
#         g2=input("Input RGB colors; g: ")
        g=int(g)
#         b2=input("Input RGB colors; b: ")
        b=int(b)

#         r2=r1
#         g2=g1
#         b2=b1

        mask_data = []
        width= 1080
        height= 1080
        base = Image.new('RGB', (width, height), (r,g,b))
        top = Image.new('RGB', (width, height),(r,g,b))
        mask = Image.new('L', (width, height))
        for y in range(height):
            for x in range(width):
                mask_data.append(int(300 * (y / height)))
        mask.putdata(mask_data)
        base.paste(top, (0, 0), mask)

#         img_path = input("enter image path: ")
#         image = open(img_path, 'rb')
#         image_read = image.read()
#         image_64_encode = base64.encodestring(image_read)
#         image_64_decode = base64.decodestring(image_64_encode)
#         image_result = open('decoded_img.png', 'wb') # create a writable image and write the decoding result
#         image_result.write(image_64_decode)
        img = cv2.imread(img_path)

        hi, wi, c = img.shape
        modelFile = "c:/dev/Marketing work/res10_300x300_ssd_iter_140000.caffemodel"
        configFile = "c:/dev/Marketing work/deploy.prototxt.txt"
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

        #frame = cv2.imread(img_path)
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                text1 = "{:.2f}%".format(confidence * 100)
                y = y1 - 10 if y1 - 10 > 10 else y1 + 10
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 0, 255), 2)

                cv2.putText(frame, text1, (x1, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 3)

        try:
            print("Face Detecting Points")
            left_side = x1
            print(left_side)
            right_side = wi - x2
            print(right_side)
            face_crop_limit=x1+x2
            limit=face_crop_limit
        except NameError:
            print("Body Detecting Points")
            net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
            classes = []
            with open("coco.names", "r") as f:
                classes = [line.strip() for line in f.readlines()]

            layers_names = net.getLayerNames()
            output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            colors = np.random.uniform(0, 255, size=(len(classes), 3))

            img = cv2.imread(img_path)
            img = cv2.resize(img, None, fx=0.4, fy=0.4)
            height, width, channels = img.shape

            blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
            net.setInput(blob)
            outputs = net.forward(output_layers)

            boxes = []
            confs = []
            class_ids = []
            for output in outputs:
                for detect in output:
                    scores = detect[5:]
                    class_id = np.argmax(scores)
                    conf = scores[class_id]
                    if conf > 0.3:
                        center_x = int(detect[0] * width)
                        center_y = int(detect[1] * height)
                        wb = int(detect[2] * width)
                        hb = int(detect[3] * height)
                        xb = int(center_x - wb / 2)
                        yb = int(center_y - hb / 2)
                        boxes.append([xb, yb, wb, hb])
                        confs.append(float(conf))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
            font = cv2.FONT_HERSHEY_PLAIN
            for i in range(len(boxes)):
                if i in indexes:
                    x1, y1, x2, y2 = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = colors[i]
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    if label == "person":
                        upper_left = (x1, y1)
                        lower_right = (x2, y2)
                        cv2.putText(img, label, (x1, y1 - 5), font, 1, color, 1)

            left_side=x1
            print(left_side)
            right_side=wi-x2
            print(right_side)
            body_crop_limit=wi - x2 + x1
            limit=body_crop_limit

        if right_side > left_side:
            input_img = Image.open(img_path)
            box = (0, 0, limit, 1080)
            cropped_img = input_img.crop(box)
            numpy_image = np.array(cropped_img)
            cropped_img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

        else:
            img = Image.open(img_path)
            left = left_side - right_side
            top = 0
            width = wi
            height = 1080
            box = (left, top, left + width, top + height)
            cropped_img = img.crop(box)
            numpy_image = np.array(cropped_img)
            cropped_img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

        # *****REMOVE BLACRightK BACKGROUND*****
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        hh, ww = thresh.shape

        thresh[hh - 3:hh, 0:ww] = 0

        white = np.where(thresh == 255)
        xmin, ymin, xmax, ymax = np.min(white[1]), np.min(white[0]), np.max(white[1]), np.max(white[0])

        final_crop = cropped_img[ymin:ymax + 3, xmin:xmax]
        print(final_crop)

        color_coverted = cv2.cvtColor(final_crop, cv2.COLOR_BGR2RGB)
        final_crop=Image.fromarray(color_coverted)
        
        def crop_center(pil_img, crop_width, crop_height):
            img_width, img_height = pil_img.size
            return pil_img.crop(((img_width - crop_width) // 2,
                                 (img_height - crop_height) // 2,
                                 (img_width + crop_width) // 2,
                                 (img_height + crop_height) // 2))

        exact_size = crop_center(final_crop, 540, 1079)

        if inp1 == "right":

        #if ask_side=="right":
            image1 = exact_size
            image2 = base
            x1 = 600
            y1 = 350

        if inp1=="left":
        #if ask_side=="left":
            image1 = base
            image2 = exact_size
            x1 = 45
            y1 = 350

        image1 = image1.resize((540, 1080))
        image1_size = image1.size
        image2_size = image2.size
        new_image = Image.new('RGB',(2*image1_size[0], image1_size[1]), (250,250,250))
        new_image.paste(image1,(0,0))
        new_image.paste(image2,(image1_size[0],0))
        #new_image.show()

        def text_wrap(text, font, max_width):
            lines = []

            if font.getsize(text)[0] <= max_width:
                lines.append(text)
            else:
                words = text.split(' ')
                i = 0
                while i < len(words):
                    line = ''
                    while i < len(words) and font.getsize(line + words[i])[0] <= max_width:
                        line = line + words[i] + " "
                        i += 1
                    if not line:
                        line = words[i]
                        i += 1
                    lines.append(line)
            return lines

        stopwords_list = [line.rstrip('\n') for line in open("c:/dev/Marketing work/stopwords")]
        cap_words=["AI","NBI","IEEE"]
        font_path = 'c:/dev/Marketing work/font3.ttf'
        font = ImageFont.truetype(font=font_path, size=60)
        #text = "64 PROJECTS FOR EVERY DATA SCIENCE PROFESSIONAL"
#         text=input("write text for image: ")
        lowercase_words = re.split(" ", text.lower())
        final_words_low = [lowercase_words[0].capitalize()]
        final_words_low += [word if word in stopwords_list else word.capitalize() for word in lowercase_words[1:]]
        final_title_low = " ".join(final_words_low)

        lines = text_wrap(final_title_low, font, 450)
        line_height = font.getsize('hg')[1]
        lines="\n ".join(lines)

        draw = ImageDraw.Draw(new_image)
        color = 'hsl(0,0%,100%)'
        draw.text((x1, y1), lines, fill=color, font=font,align="center")

        new_image.save("static/solid.jpg")



# In[ ]:

from PIL import Image, ImageDraw, ImageFont
import textwrap
import re


def grad_blend(img_path,r1,g1,b1,r2,g2,b2,text,obesity):
    width= 1080
    height= 1080
    base = Image.new('RGB', (width, height), (r1,g1,b1))
    top = Image.new('RGB', (width, height), (r2,g2,b2))
    mask = Image.new('L', (width, height))
    mask_data = []
    for y in range(height):
        for x in range(width):
            mask_data.append(int(300 * (y / height)))
    mask.putdata(mask_data)
    base.paste(top, (0, 0), mask)

    Im = Image.open(img_path)
#     Im.show()

    newIm = Image.new ("RGBA", (1080, 1080), (255, 0, 0))
    Im2 = base.convert(Im.mode)
    Im2 = Im2.resize(Im.size)
#     Im2.show()

    img = Image.blend(Im,Im2,obesity)
#     img.show()
    img.save("blend.jpg")

    stopwords_list = [line.rstrip('\n') for line in open("c:/dev/Marketing work/stopwords")]
    lowercase_words = re.split(" ", text.lower())
    text = [lowercase_words[0].capitalize()]
    text += [word if word in stopwords_list else word.capitalize() for word in lowercase_words[1:]]
    text = " ".join(text)
    iw, ih = img.size

    font=ImageFont.truetype("c:/dev/Marketing work/font3.ttf", 60)
    w1, h1 = font.getsize(text)
    draw = ImageDraw.Draw(img)
    textX1 = int((iw - w1) / 2)
    lines1 = textwrap.wrap(text, width=35)
    startHeight=630
    breather=250
    y_text1 = h1
    for line in lines1:
        width, height = font.getsize(line)
        draw.text((int((iw - width) / 2), startHeight - breather +y_text1), line, font=font, align="left", color="red")
        y_text1 += height

    img.save("static/solid.jpg")



from flask import Flask, render_template, request, send_file
from flask import url_for

import os
app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = "c:/dev/Marketing work/images_path"
@app.route("/")
def hello():
    return render_template("index.html")
@app.route("/get_image", methods=['GET', 'POST'])
def test():
    print("hello")
    if request.method == "POST":
        text = request.form.get('status')
        img = request.files["media"]
        img.save(os.path.join(app.config["IMAGE_UPLOADS"], img.filename))
        fontsize = float(request.form.get("fontsize"))
        colors = request.form.get("colors")
        opt1 = request.form.get("option_box")
        opt2 = request.form.get("option_boxx")
        
        first = request.form.get("first")
        r1, g1, b1  = tuple(int(first.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        second = request.form.get("second")
        r2, g2, b2  = tuple(int(second.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

        print(colors)
        solidcolorr = request.form.get("solidcolorr")
        r, g, b  = tuple(int(solidcolorr.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            #(inp1,r1,g1,b1,r2,g2,b2,img_path,text)
            #print(img.filename)
        if colors=='solid':
            all_input2(opt1,r,g,b,app.config["IMAGE_UPLOADS"]+'/'+img.filename,text)
            return '<img src=' + url_for('static',filename='solid.jpg') + '>'
        elif colors=='gradient':
            all_inputs(opt1,r1, g1, b1, r2, g2, b2,app.config["IMAGE_UPLOADS"]+'/'+img.filename, text)
            return '<img src=' + url_for('static',filename='solid.jpg') + '>'
        elif colors=='centere':
            grad_blend(app.config["IMAGE_UPLOADS"]+'/'+img.filename,r1,g1,b1,r2,g2,b2,text,fontsize)
            return '<img src=' + url_for('static',filename='solid.jpg') + '>' 
            #return send_file('solid.jpg', mimetype='image/JPG')
        else:
             return "NOT image"
	


# In[ ]:





# In[ ]:


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8004)


# In[ ]:


