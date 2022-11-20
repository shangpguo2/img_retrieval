import glob
import time
import PySimpleGUI as sg
import os
import cv2
import retrieve
import subprocess


# store all the records
record = []
out_dir = "out/"
threshold = 50


def formatimg(img):
    orh = img.shape[0]
    width = int(img.shape[1] * 300 / orh)
    height = int(img.shape[0] * 300 / orh)
    dim = (width, height)
    return dim


def show_img(filename, window_name):
    img = cv2.imread(filename)
    dim = formatimg(img)
    # resize image
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    imgbytes = cv2.imencode(".png", img)[1].tobytes()
    window[window_name].update(data=imgbytes)


def output():
    # sort the records
    record.sort(key=lambda x: x[1])
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    files = glob.glob(out_dir+'*')
    for f in files:
        os.remove(f)
    for i in range(threshold):
        cv2.imwrite(out_dir + record[i][0][19:], cv2.imread(record[i][0]))


file_list_column = [
    [
        sg.Text("Image Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(initial_folder="image.orig/examples"),
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
        )
    ],
]

# For now will only show the name of the file that was chosen
image_column = [
    [sg.Text("Choose an image from list on left:")],
    [sg.Text(size=(40, 1), key="-TINPUT-")],
    [sg.Image(key="-IMAGE-")],
]

similar_column = [
    [sg.Text("The most similar image:"),
     sg.Text(size=(20, 1), key="-TOUT-"),
     sg.Text(size=(20, 1), key="-TTIME-")],
    [
        sg.Text("Number of images to retrieve:"),
        sg.Slider(
            (0, 100),
            50,
            1,
            orientation="h",
            size=(30, 10),
            key="-SLIDER-",
        ),
        sg.Button("Export")],
    [sg.Image(key="-SIMILAR-")],
]

# ----- Full layout -----
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_column),
        sg.VSeperator(),
        sg.Column(similar_column)
    ]
]

window = sg.Window("Image Viewer", layout)

# Run the Event Loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    if values["-SLIDER-"]:
        threshold = int(values["-SLIDER-"])
    # Folder name was filled in, make a list of files in the folder
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
               and f.lower().endswith(".jpg")
        ]
        window["-FILE LIST-"].update(fnames)
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            filename = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]
            )
            window["-TINPUT-"].update(filename)
            show_img(filename, "-IMAGE-")

            choice = values["-FILE LIST-"][0]
            start = time.time()
            similar_filename, record = retrieve.dl(choice)
            window["-TTIME-"].update(str(time.time() - start) + "ms")
            window["-TOUT-"].update(similar_filename)
            show_img(similar_filename, "-SIMILAR-")
        except:
            pass
    elif event == "Export":
        if len(record) != 0:
            output()
            alter_layout = [[sg.Text("Success!")], [sg.Button("OK"), sg.Button("OPEN")]]
            alert = sg.Window("Alert", alter_layout, element_justification="center")
            e, v = alert.read()
            if e == "OK" or event == sg.WIN_CLOSED:
                alert.close()
            elif e == "OPEN":
                path = os.path.abspath(out_dir)
                subprocess.Popen(f'explorer /select,"{path}"')
                alert.close()
        else:
            alter_layout = [[sg.Text("Choose a image first!")], [sg.Button("OK")]]
            alert = sg.Window("Alert", alter_layout, element_justification="center")
            e, v = alert.read()
            if e == "OK" or event == sg.WIN_CLOSED:
                alert.close()

window.close()
