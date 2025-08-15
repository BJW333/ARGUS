# core/bootstrap.py
import sys
import threading
from PyQt5.QtWidgets import QApplication
from config import script_dir
from core.startup import wishme, print_banner
from gui.main_window import MainWindow
from speech.listen import listen_for_wake_word

def initialize_argus():
    #print("Booting ARGUS...") 
    print_banner()
    wishme()    
    
    #create and display the PyQt application
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    
    #GUI INITATION
    #start wake word listener in a background thread
    wake_word_thread = threading.Thread(target=listen_for_wake_word, daemon=True)
    wake_word_thread.start()
     
    sys.exit(app.exec_())
    
    
        
        