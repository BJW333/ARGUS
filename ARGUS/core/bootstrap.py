import threading
from core.startup import wishme, print_banner
from speech.listen import listen_for_wake_word

def initialize_argus_backend():
    """
    Initialize ARGUS backend services without starting the GUI.
    This is useful for running ARGUS in a headless mode or for testing.
    """
    print_banner()
    wishme()

    #Start wake word listener in background thread
    wake_thread = threading.Thread(
        target=listen_for_wake_word,
        daemon=True,
        name="ListenWakeWordThread",
    )
    wake_thread.start()

# Backwards-compat: if anything else still calls initialize_argus(),
# just route it to the backend-only startup.
# def initialize_argus() -> None:
#     initialize_argus_backend()