import speech_recognition as sr

recognizer = sr.Recognizer()

def detect_audio():

    try:
        with sr.Microphone() as source:

            print("Listening...")

            recognizer.adjust_for_ambient_noise(source, duration=0.5)

            audio = recognizer.listen(source, timeout=3)

            text = recognizer.recognize_google(audio)

            print("Audio detected:", text)

            return text

    except sr.WaitTimeoutError:
        return None

    except sr.UnknownValueError:
        return None

    except sr.RequestError:
        return None

    except Exception as e:
        print("Audio error:", e)
        return None