from threading import Thread, Event
from time import sleep

class UtilTimer(Thread):
    """Basic Timer for plant"""
    _id_counter = 1

    def __init__(self, interval=10, name=""):
        super().__init__()
        self.interval = interval
        self.event = Event()
        self.force_shutdown = False
        self.set_name(name)

    def set_name(self, name_in: str):
        if name_in != "":
            self.name = name_in
        else:
            self.name = f"Timer {UtilTimer._id_counter}"
            UtilTimer._id_counter += 1

    def wait(self):
        return self.event.wait()

    def end_of_thread(self):
        print(f"{self.name} ended")

    def run(self):
        try:
            while not self.force_shutdown:
                sleep(self.interval)
                if self.force_shutdown:  
                    break
                self.event.set()
                # Here you can do your periodic task if needed
                self.event.clear()
        finally:
            self.end_of_thread()  # This ensures it is called when the thread exits
            self.stop()

    def stop(self):
        """Force the timer to stop."""
        self.force_shutdown = True
        self.join()  # Wait for the thread to finish
