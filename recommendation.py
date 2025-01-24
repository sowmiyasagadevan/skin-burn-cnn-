import tkinter as tk 
from tkinter import filedialog, Label, Button
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json

win = tk.Tk()

lbl = Label(win, text="", fg='black')
lbl.pack()

recommendation_lbl = Label(win, text="", fg='black')
recommendation_lbl.pack()

def get_recommendation(degree):
    recommendations = {
       "DEGREE1": "Mild burn.\n"
                    "1. Apply aloe vera gel and keep the area clean.\n"
                    "2. Soak the wound in cool water for five minutes or longer, take acetaminophen or ibuprofen for pain relief.\n"
                    "3. Apply lidocaine (an anesthetic) with aloe vera gel or cream to soothe the skin, use an antibiotic ointment, and loose gauze to protect the affected area.",
        "DEGREE2": "Moderate burn.\n"
                    "1. Use a sterile dressing and seek medical attention.\n"
                    "2. Run the skin under cool water for 15 minutes or longer.\n"
                    "3. Apply antibiotic cream to blisters.",
        "DEGREE3": "Severe burn.\n"
                    "1. Seek immediate medical attention. Do not apply anything.\n"
                    "2. Highly recommended for plastic surgery.",
        "HEALTHY SKIN": "No burn detected. Keep your skin healthy with regular care."
    }
    return recommendations.get(degree, "Unknown degree. Consult a healthcare professional.")

def b1_click():
    global path2
    try:
        json_file = open(r"c:\Users\sowmiyasagadevan\Downloads\model.json", 'r', encoding='utf-8')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(r"C:\Users\sowmiyasagadevan\Downloads\model_weights.h5")

        label = ["DEGREE1", "DEGREE2", "DEGREE3", "HEALTHY SKIN"]

        path2 = filedialog.askopenfilename()
        print(path2)

        test_image = image.load_img(path2, target_size=(128, 128))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = loaded_model.predict(test_image)
        print(result)
        degree = label[result.argmax()]
        print(degree)
        lbl.configure(text=degree)

        recommendation_text = get_recommendation(degree)
        recommendation_lbl.configure(text=recommendation_text)

    except Exception as e:
        print(f"An error occurred: {e}")

label1 = Label(win, text="GUI For skin burn Detection using OpenCV", fg='blue')
label1.pack()

b1 = Button(win, text="browse image", width=25, height=3, fg='red', command=b1_click)
b1.pack()

win.geometry("550x300")
win.title("Skin Burn Detection and Recommendations")
win.bind("<Return>", b1_click)
win.mainloop()
