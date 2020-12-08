import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tkinter as tk
import os
import subprocess
import io
from tkinter.ttk import Notebook
from PIL import ImageTk, Image
from LSTM import *
from tkinter import filedialog, Scale
from datetime import date
from random import *

# from Kmean import *
# try:
#     nltk.download('stopwords')
#     nltk.download('punkt')
#     nltk.download('all')
# except:
#     pass

X_test_LSTM = []
y_test_LSTM = []
X_test_RBNB = []
y_test_RBNB = []

root = tk.Tk()
root.title('FAKE_REAL_NEWS_DEMO')
background_img = img = ImageTk.PhotoImage(Image.open("background.jpg"))
canvas = tk.Canvas(root, height=background_img.height(), width=background_img.width())
canvas.pack(side='top', fill='both', expand='yes')
canvas.create_image(0, 0, image=background_img, anchor='nw')


def combine_funcs(*funcs):
    def combined_func(*args, **kwargs):
        for fun in funcs:
            fun(*args, **kwargs)

    return combined_func


def browse():
    file = filedialog.askopenfile()
    if file:
        filename = file.name
        link.set(filename)
        try:
            xl_file = pd.read_csv(filename)
        except:
            xl_file = pd.ExcelFile(filename)
        global X_test_LSTM
        X_test_LSTM = getDataBeforeTrain(xl_file)


def openFolder():
    path = './output'
    subprocess.Popen(f'explorer {os.path.realpath(path)}')


def run():
    score = GetPred(X_test_LSTM)
    # addResult(1, LSTM)
    if score >= 0.5:
        label_ans_fake.pack_forget()
        label_ans_real.pack(side="bottom", fill="both", expand="yes")
    else:
        label_ans_real.pack_forget()
        label_ans_fake.pack(side="bottom", fill="both", expand="yes")

def run_predict():
    global X_test_LSTM, y_test_LSTM

    today = date.today()
    d = today.strftime("%d-%m-%y")
    link_ = entry1.get()
    if link_ != '':
        run()
        return

    title_data = title.get()
    text_data = text.get()
    subject_data = subject.get()
    if title_data != '' or text_data != '' or subject_data != '':
        if title_data == '':
            title_data = " "
        if text_data == '':
            text_data = " "
        if subject_data == '':
            subject_data = " "

        dataFile = pd.DataFrame([[title_data, text_data, subject_data, d]],
                                columns=['title', 'text', 'subject', 'date'])
        dataFile.to_csv('./demo.csv', index=False)
        dataFile = pd.read_csv('./demo.csv')
        os.remove('./demo.csv')
        global X_test_LSTM
        X_test_LSTM = getDataBeforeTrain(dataFile)
        run()


def clear():
    link.set("")
    title.set("")
    text.set("")
    subject.set("")


link = tk.StringVar()
title = tk.StringVar()
text = tk.StringVar()
subject = tk.StringVar()
# frame0
frame0 = tk.Frame(root, bg='#80c1ff', bd=5)
frame0.place(relx=0.5, rely=0.01, relwidth=0.7, relheight=0.06, anchor='n')
label0 = tk.Label(frame0, text="FAKE OR REAL DEMO", font="bold")
label0.place(relwidth=1, relheight=1)

# frame 1
frame1 = tk.Frame(root, bg='#80c1ff', bd=5)
frame1.place(relx=0.5, rely=0.08, relwidth=0.7, relheight=0.2, anchor='n')
label1 = tk.Label(frame1, text="Check a File", font="bold")
label1.place(relwidth=1, relheight=0.35)
# add link in frame 1
label1 = tk.Label(frame1, text="Link", font="bold")
label1.place(relwidth=0.23, relheight=0.3, rely=0.37)
entry1 = tk.Entry(frame1, font=40, textvariable=link)
entry1.place(relx=0.24, relwidth=0.55, relheight=0.3, rely=0.37)
button1 = tk.Button(frame1, text='Import', font=40, command=browse)
button1.place(relx=0.8, rely=0.37, relheight=0.3, relwidth=0.2)

# frame 2
frame2 = tk.Frame(root, bg='#80c1ff', bd=5)
frame2.place(relx=0.5, rely=0.22, relwidth=0.7, relheight=0.4, anchor='n')
label2 = tk.Label(frame2, text="Check a Text", font="bold")
label2.place(relwidth=1, relheight=0.2)
# add title in frame2
label2 = tk.Label(frame2, text="Title", font="bold")
label2.place(relwidth=0.22, relheight=0.2, rely=0.25)
entry2 = tk.Entry(frame2, font=40, textvariable=title)
entry2.place(relx=0.23, relwidth=0.76, relheight=0.2, rely=0.25)
# add text in frame2
label3 = tk.Label(frame2, text="Text", font="bold")
label3.place(relwidth=0.22, relheight=0.2, rely=0.5)
entry3 = tk.Entry(frame2, font=40, textvariable=text)
entry3.place(relx=0.23, relwidth=0.76, relheight=0.2, rely=0.5)
# add subject in frame2
label4 = tk.Label(frame2, text="Subject", font="bold")
label4.place(relwidth=0.22, relheight=0.2, rely=0.75)
entry4 = tk.Entry(frame2, font=40, textvariable=subject)
entry4.place(relx=0.23, relwidth=0.76, relheight=0.2, rely=0.75)

# add button run in root
ClearButoon = tk.Button(root, text='Clear Data', font=40, command=clear)
ClearButoon.place(relx=0.3, rely=0.6, relheight=0.08, relwidth=0.2)
RunTrain = tk.Button(root, text='Run', font=40, command=run_predict)
RunTrain.place(relx=0.52, rely=0.6, relheight=0.08, relwidth=0.2)

# frame 3
frame3 = tk.Frame(root)
frame3.place(relx=0.38, rely=0.7)
img1 = ImageTk.PhotoImage(Image.open("./real_news.png"))
img2 = ImageTk.PhotoImage(Image.open("./fake_news.jpg"))
label_ans_real = tk.Label(frame3, image=img1)
label_ans_fake = tk.Label(frame3, image=img2)

# #table in frame 3
# frameTable = tk.Frame(frame3, bd=10).place(relx = 0, rely = 0.5, relwidth = 1, relheight=0.65)
# tabLayout = Notebook(frameTable)
# tabLayout.place(relx = 0.16, rely = 0.5, relwidth = 0.68, relheight = 0.37)
# #tab in table
# tab1 = tk.Frame(tabLayout, bd=10)

# def make_entry(row, column, width, text, state):
#     e = tk.Entry(tab1, width=width)
#     if text:
#         e.insert(0, text)
#     if not state:
#         e.configure(font = ( "bold"))
#     e['state'] = tk.NORMAL if state else tk.DISABLED
#     e.coords = (row - 1, column - 1)
#     e.grid(row=row, column=column)

# make_entry(0, 1, 10, "METHOD", False)
# make_entry(0, 2, 10, "ACCURACY-SCORE", False)
# make_entry(0, 3, 10, "PRECISION-SCORE", False)
# make_entry(0, 4, 10, "RECALL-SCORE", False)
# make_entry(0, 5, 10, "F1-SCORE", False)
# make_entry(0, 6, 10, "RESULT", False)
#
# def addResult(i, data):
#     for column in range(6):
#         make_entry(i, column + 1, 10, data[column], False)
#
# tabLayout.add(tab1, text = "Compare LSTM + MNB + RB")
# def show_cells():
#     for e in tab1.children:
#         v = tab1.children[e]
#         print(f'{v.get()}', end=', ')
#     print()

# lower_frame = tk.Frame(root, bg='#80c1ff',bd=10)
# lower_frame.place(relx=0.5,rely=0.25,relwidth=0.75,relheight=0.6,anchor='n')

# label = tk.Label(lower_frame,font=('Courier',15))
# label.place(relwidth=1,relheight=1)

# OpenFolder = tk.Button(root, text='OpenFolder!', command = openFolder)

root.mainloop()
