import pyttsx3

engine = pyttsx3.init()

while True:
    text = input("Enter text (or 'exit'): ")
    if text.lower() == "exit":
        break
    engine.say(text)
    engine.runAndWait()