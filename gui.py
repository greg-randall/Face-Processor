import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import face_processor  # Import our processing script
import threading
import queue
from tkinter import ttk
import sys
import os

class FaceProcessorApp:
    """The main application class for the GUI."""

    def __init__(self, root):
        self.root = root
        self.root.title("Face Processor")
        self.root.geometry("650x350")
        self.root.resizable(False, False)
        self.log_visible = False

        # --- NEW --- Handle window closing correctly
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Queues for communicating with the worker thread
        self.log_queue = queue.Queue()
        self.progress_queue = queue.Queue()

        # --- Main Layout ---
        main_frame = tk.Frame(root, padx=15, pady=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Input Path Section ---
        input_frame = tk.LabelFrame(main_frame, text="Input", padx=10, pady=10)
        input_frame.pack(fill=tk.X, pady=(0, 10))

        self.input_path_var = tk.StringVar()
        self.input_entry = tk.Entry(input_frame, textvariable=self.input_path_var, width=50)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=2)
        self.browse_file_button = tk.Button(input_frame, text="Select File...", command=self.browse_file)
        self.browse_file_button.pack(side=tk.LEFT, padx=(5, 0))
        self.browse_folder_button = tk.Button(input_frame, text="Select Folder...", command=self.browse_folder)
        self.browse_folder_button.pack(side=tk.LEFT, padx=(5, 0))

        # --- Mode & Options Section ---
        mode_options_frame = tk.Frame(main_frame)
        mode_options_frame.pack(fill=tk.X, pady=(0, 10))
        
        # --- Mode Selection ---
        mode_frame = tk.LabelFrame(mode_options_frame, text="Mode", padx=10, pady=10)
        mode_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        self.mode_var = tk.StringVar(value="crop")
        tk.Radiobutton(mode_frame, text="Crop Images", variable=self.mode_var, value="crop", command=self.toggle_options).pack(anchor=tk.W)
        tk.Radiobutton(mode_frame, text="Analyze Folder", variable=self.mode_var, value="analyze", command=self.toggle_options).pack(anchor=tk.W)

        # --- Cropping Options ---
        self.options_frame = tk.LabelFrame(mode_options_frame, text="Cropping Options", padx=10, pady=5)
        self.options_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Size
        tk.Label(self.options_frame, text="Size (width):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.size_var = tk.StringVar(value="600")
        self.size_entry = tk.Entry(self.options_frame, textvariable=self.size_var, width=10)
        self.size_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        # Aspect Ratio
        tk.Label(self.options_frame, text="Aspect Ratio:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.aspect_ratio_var = tk.StringVar(value="1:1")
        self.aspect_ratio_entry = tk.Entry(self.options_frame, textvariable=self.aspect_ratio_var, width=10)
        self.aspect_ratio_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        # Content Fill Checkbox
        self.content_fill_var = tk.BooleanVar(value=True)
        self.content_fill_check = tk.Checkbutton(self.options_frame, text="Use Content-Aware Fill", variable=self.content_fill_var)
        self.content_fill_check.grid(row=0, column=2, rowspan=2, sticky='w', padx=10)
        
        # --- Progress Bar ---
        progress_frame = tk.LabelFrame(main_frame, text="Progress", padx=10, pady=5)
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        self.progress_bar = ttk.Progressbar(progress_frame, orient='horizontal', mode='determinate')
        self.progress_bar.pack(fill=tk.X, ipady=4)

        # --- Output Log ---
        self.log_frame = tk.LabelFrame(main_frame, text="Log Output", padx=10, pady=10)
        # self.log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10)) # Keep it hidden initially
        self.log_text = scrolledtext.ScrolledText(self.log_frame, state='disabled', height=10, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # --- Action Buttons ---
        action_frame = tk.Frame(main_frame)
        action_frame.pack(fill=tk.X)
        self.toggle_log_button = tk.Button(action_frame, text="Show Log", command=self.toggle_log_visibility)
        self.toggle_log_button.pack(side=tk.LEFT, ipadx=10)

        self.run_button = tk.Button(action_frame, text="Run", command=self.start_processing, bg="#4CAF50", fg="white", font=("Helvetica", 10, "bold"))
        self.run_button.pack(side=tk.RIGHT, ipadx=20, ipady=5)

        # Start the queue processor
        self.process_queues()

    def on_closing(self):
        """Handle the window close event."""
        self.root.destroy()

    def browse_file(self):
        """Opens a file dialog to select a single image file."""
        path = filedialog.askopenfilename(
            title="Select an Image File",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")]
        )
        if path:
            self.input_path_var.set(path)

    def browse_folder(self):
        """Opens a directory dialog."""
        path = filedialog.askdirectory(title="Select a Folder")
        if path:
            self.input_path_var.set(path)

    def toggle_log_visibility(self):
        """Shows or hides the log frame and resizes the window."""
        if self.log_visible:
            self.log_frame.pack_forget()
            self.root.geometry("650x350")
            self.toggle_log_button.config(text="Show Log")
            self.log_visible = False
        else:
            self.log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
            self.root.geometry("650x550")
            self.toggle_log_button.config(text="Hide Log")
            self.log_visible = True

    def toggle_options(self):
        """Enables or disables cropping options based on the selected mode."""
        if self.mode_var.get() == "crop":
            for child in self.options_frame.winfo_children():
                child.configure(state='normal')
        else: # Analyze mode
            for child in self.options_frame.winfo_children():
                child.configure(state='disabled')

    def log_message(self, message):
        """Inserts a message into the log text widget."""
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message)
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def start_processing(self):
        """Gathers options and starts the processing in a new thread."""
        for q in [self.log_queue, self.progress_queue]:
            while not q.empty():
                try: q.get_nowait()
                except queue.Empty: continue
        
        input_path = self.input_path_var.get()
        if not input_path:
            messagebox.showerror("Error", "Please select an input path.")
            return

        image_files = face_processor.get_image_paths(input_path)
        total_files = len(image_files)

        if total_files == 0:
            messagebox.showinfo("Info", "No new images found to process.")
            return

        self.progress_bar['value'] = 0
        self.progress_bar['maximum'] = total_files
        
        self.run_button.config(state="disabled", text="Processing...")
        self.log_text.config(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state='disabled')

        def progress_callback():
            self.progress_queue.put(1)

        options = {
            'input_path': input_path, 'output': None, 'debug': False, 
            'disable_progress_bar': True, 'progress_callback': progress_callback
        }

        if self.mode_var.get() == 'crop':
            options.update({
                'config': "median_landmarks.json", 'size': int(self.size_var.get()),
                'aspect_ratio': self.aspect_ratio_var.get(), 'eye_y': None, 'face_height': None,
                'content_fill': self.content_fill_var.get()
            })
            target_func = face_processor.run_cropping
        else:
            target_func = face_processor.run_analysis
        
        # --- MODIFIED --- Make the thread a daemon
        self.thread = threading.Thread(target=self.run_task_in_thread, args=(target_func, options))
        self.thread.daemon = True
        self.thread.start()

    def run_task_in_thread(self, target_func, options):
        """The worker function that runs in a separate thread."""
        sys.stdout = self
        sys.stderr = self
        try:
            target_func(options)
        except Exception as e:
            import traceback
            self.write(f"\n--- A CRITICAL ERROR OCCURRED ---\n{traceback.format_exc()}\n")
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            self.log_queue.put("--- All tasks complete ---")
            self.log_queue.put(None)

    def write(self, text):
        self.log_queue.put(text)

    def flush(self):
        pass

    def process_queues(self):
        """Checks the log and progress queues for messages from the worker thread."""
        try:
            while True:
                message = self.log_queue.get_nowait()
                if message is None:
                    self.run_button.config(state="normal", text="Run")
                    # Clear any remaining items in the progress queue
                    while not self.progress_queue.empty():
                        try:
                            self.progress_queue.get_nowait()
                        except queue.Empty:
                            break
                    # Reset progress bar to 0
                    self.progress_bar['value'] = 0
                    messagebox.showinfo("Success", "Processing has finished!")
                else:
                    self.log_message(message)
        except queue.Empty:
            pass

        try:
            while True:
                self.progress_queue.get_nowait()
                self.progress_bar.step(1)
        except queue.Empty:
            pass
        
        self.root.after(100, self.process_queues)

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceProcessorApp(root)
    root.mainloop()

