from tkinter import *
from chat import get_response, bot_name, save_context, load_context
import speech_recognition as sr
import pyttsx3

BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOUR = "#EAECEE"

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

class ChatApplication:
    def __init__(self):
        self.window = Tk()
        self._setup_main_window()
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        self.context = load_context()

    def run(self):
        self.window.mainloop()

    def _setup_main_window(self):
        self.window.title("Chat")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=600, height=750, bg=BG_COLOR)

        # head label
        head_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOUR,
                           text="Welcome", font=FONT_BOLD, pady=10)
        head_label.place(relwidth=1)

        # tiny divider
        line = Label(self.window, width=450, bg=BG_GRAY)
        line.place(relwidth=1, rely=0.07, relheight=0.012)

        # text widget
        self.text_widget = Text(self.window, width=20, height=2, bg=BG_COLOR, fg=TEXT_COLOUR,
                                font=FONT, padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        # bottom label
        bottom_label = Label(self.window, bg=BG_GRAY, height=80)
        bottom_label.place(relwidth=1, rely=0.825)

        # message entry box
        self.msg_entry = Entry(bottom_label, bg="#2C3E50", fg=TEXT_COLOUR, font=FONT)
        self.msg_entry.place(relwidth=0.6, relheight=0.06, rely=0.008, relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)

        # send button
        send_button = Button(bottom_label, text="Send", font=FONT_BOLD, width=20, bg=BG_GRAY,
                             command=lambda: self._on_enter_pressed(None))
        send_button.place(relx=0.62, rely=0.008, relheight=0.06, relwidth=0.18)
        
        # voice input button
        voice_button = Button(bottom_label, text="Voice", font=FONT_BOLD, width=20, bg=BG_GRAY,
                              command=self._on_voice_pressed)
        voice_button.place(relx=0.81, rely=0.008, relheight=0.06, relwidth=0.18)

    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        self._insert_message(msg, "You")

    def _on_voice_pressed(self):
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source)
                print("Listening...")
                audio = self.recognizer.listen(source)

            msg = self.recognizer.recognize_google(audio)
            self._insert_message(msg, "You")
        except Exception as e:
            print(f"Error: {e}")
            self._insert_message("Sorry, I did not catch that.", "You")

    def _insert_message(self, msg, sender):
        if not msg:
            return

        self.msg_entry.delete(0, END)
        msg1 = f"{sender}: {msg}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state=DISABLED)

        response = get_response(msg, self.context)
        msg2 = f"{bot_name}: {response}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg2)
        self.text_widget.configure(state=DISABLED)

        self._speak_response(response)
        self.text_widget.see(END)

        save_context(self.context)

    def _speak_response(self, response):
        self.engine.say(response)
        self.engine.runAndWait()

if __name__ == "__main__":
    app = ChatApplication()
    app.run()
