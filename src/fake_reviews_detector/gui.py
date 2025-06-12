# src/fake_reviews_detector/gui.py


from utils import load_yaml_config
from preprocessing import create_processed_csv
# import data_loader
# import modelx
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import pandas as pd
import time

# Загрузка конфига
config = load_yaml_config("../../config/gui_config.yaml")

model = None
model_trained = False
train_file_path = ""
test_file_path = ""


def create_gui(root, status_bar=None) -> None:
    # Настройка главного окна из конфига
    root.title(config['app']['title'])
    root.geometry(config['app']['geometry'])

    # Применение стилей
    style = ttk.Style()
    style.configure('TFrame', background=config['app']['background'])
    style.configure('TButton', font=config['styles']['button']['font'],
                    padding=config['styles']['button']['padding'])
    style.configure('TLabel', background=config['styles']['label']['background'],
                    font=config['styles']['label']['font'])

    # Основные окна
    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Окно загрузки данных
    data_frame = ttk.LabelFrame(main_frame, text=config['ui_text']['main']['load_data'], padding=10)
    data_frame.pack(fill=tk.X, pady=5)

    # Кнопка и поле для файла обучения
    ttk.Label(data_frame, text=config['ui_text']['labels']['train_file']).grid(row=0, column=0, sticky=tk.W)
    train_file_entry = ttk.Entry(data_frame, width=87)
    train_file_entry.grid(row=0, column=1, padx=5)
    ttk.Button(data_frame, text=config['ui_text']['buttons']['browse'],
               command=lambda: browse_train_file(train_file_entry, status_bar)).grid(row=0, column=2)

    # Окно обучения модели
    train_frame = ttk.LabelFrame(main_frame, text=config['ui_text']['main']['model_training'], padding=10)
    train_frame.pack(fill=tk.X, pady=5)

    progress = ttk.Progressbar(train_frame, orient=tk.HORIZONTAL, length=300, mode='determinate')
    progress.pack(pady=5)

    ttk.Button(train_frame, text=config['ui_text']['buttons']['train'],
               command=lambda: train_model(progress, status_bar)).pack(pady=5)

    # Фрейм предсказания
    predict_frame = ttk.LabelFrame(main_frame, text=config['ui_text']['main']['prediction'], padding=10)
    predict_frame.pack(fill=tk.BOTH, expand=True, pady=5)

    # Ввод текста для предсказания
    ttk.Label(predict_frame, text=config['ui_text']['labels']['input_text']).pack(anchor=tk.W)
    predict_text = tk.Text(predict_frame, height=5, width=80)
    predict_text.pack(fill=tk.X, pady=5)

    # Или загрузка файла для предсказания
    ttk.Label(predict_frame, text=config['ui_text']['labels']['input_file']).pack(anchor=tk.W)

    file_predict_frame = ttk.Frame(predict_frame)
    file_predict_frame.pack(fill=tk.X, pady=5)

    predict_file_entry = ttk.Entry(file_predict_frame, width=110)
    predict_file_entry.pack(side=tk.LEFT, padx=5)
    ttk.Button(file_predict_frame, text=config['ui_text']['buttons']['browse'],
               command=lambda: browse_predict_file(predict_file_entry, status_bar)).pack(side=tk.LEFT)

    # Кнопка предсказания
    ttk.Button(predict_frame, text=config['ui_text']['buttons']['predict'],
               command=lambda: make_prediction(predict_text, predict_file_entry, results_text, status_bar)).pack(pady=5)

    # Метка и область результатов
    ttk.Label(predict_frame, text=config['ui_text']['labels']['results']).pack(anchor=tk.W)
    results_text = tk.Text(predict_frame, height=10, width=80, state=tk.DISABLED)
    results_text.pack(fill=tk.BOTH, expand=True)

    # Статус бар
    status_bar = ttk.Label(main_frame, text=config['ui_text']['status']['ready'], relief=tk.SUNKEN)
    status_bar.pack(fill=tk.X, pady=5)


# Поиск файла для обучения
def browse_train_file(entry_widget, status_bar) -> None:
    global train_file_path
    initial_dir = config['files']['default_train_dir'] if os.path.exists(config['files']['default_train_dir']) else None
    file_path = filedialog.askopenfilename(
        initialdir=initial_dir,
        filetypes=[(f"{ext.upper()} files", f"*.{ext}") for ext in config['files']['allowed_extensions']] + [
            ("All files", "*.*")]
    )
    if file_path:
        train_file_path = file_path
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, file_path)
        update_status(f"File loaded: {os.path.basename(file_path)}", status_bar)


# Поиск файла для предсказания
def browse_predict_file(entry_widget, status_bar) -> None:
    global test_file_path
    initial_dir = config['files']['default_predict_dir'] if os.path.exists(
        config['files']['default_predict_dir']) else None
    file_path = filedialog.askopenfilename(
        initialdir=initial_dir,
        filetypes=[(f"{ext.upper()} files", f"*.{ext}") for ext in config['files']['allowed_extensions']] + [
            ("All files", "*.*")]
    )
    if file_path:
        test_file_path = file_path
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, file_path)
        update_status(f"Loaded predict file: {os.path.basename(file_path)}", status_bar)


# Сделать предсказание
def make_prediction(text_widget, file_entry, results_text, status_bar) -> None:
    global model, model_trained
    if not model_trained:
        messagebox.showerror("Error", "Please, train model")
        return
    input_text = text_widget.get("1.0", tk.END).strip()
    file_path = file_entry.get().strip()
    if not input_text and not file_path:
        messagebox.showerror("Error", "Please enter text or select a file for prediction.")
        return
    try:
        results_text.config(state=tk.NORMAL)
        results_text.delete("1.0", tk.END)
        if input_text:
            update_status("Processing input string...", status_bar)
            try:
                data = [list(map(float, input_text.split(',')))]
                # prediction = model.predict(data)
                # results_text.insert(tk.END, f"Prediction result: {prediction[0]}\n")
                update_status("Prediction fulfilled for the input string", status_bar)
            except Exception as e:
                results_text.insert(tk.END, f"Error processing string: {e}\n")
                update_status("Error in format of entered string", status_bar)
        if file_path:
            update_status(f"Processing file {os.path.basename(file_path)}...", status_bar)
            try:
                test_data = pd.read_csv(file_path)
                predictions = model.predict(test_data)
                results_text.insert(tk.END, "Prediction result:\n")
                for i, pred in enumerate(predictions, 1):
                    results_text.insert(tk.END, f"String {i}: {pred}\n")
                update_status(f"Prediction fulfilled for file {os.path.basename(file_path)}", status_bar)
            except Exception as e:
                results_text.insert(tk.END, f"Error processing file: {str(e)}\n")
                update_status("Error processing file", status_bar)
        results_text.config(state=tk.DISABLED)
    except Exception as e:
        update_status("Error in prediction execution", status_bar)
        messagebox.showerror("Error", f"Failed to fulfill prediction: {str(e)}")


def update_status(message, status_bar) -> None:
    status_bar.config(text=message)
    status_bar.master.update_idletasks()


def train_model(progress_bar, status_bar) -> None:
    global model, model_trained, train_file_path
    if not train_file_path:
        messagebox.showerror("Error", "Choose learning file")
        return
    try:
        progress_bar['value'] = 0
        update_status("Loading data...", status_bar)
        raw_data = pd.read_csv(train_file_path)
        progress_bar['value'] = 10
        update_status("Preprocessing data...", status_bar)
        processed_data = create_processed_csv(raw_data)
        if processed_data is None or processed_data.empty:
            raise ValueError("Preprocessing returned empty data")
        progress_bar['value'] = 50
        update_status("Learning model...", status_bar)

        # Обучение модели

        progress_bar['value'] = 80
        update_status("Finalization of the model...", status_bar)
        time.sleep(1)
        model_trained = True
        progress_bar['value'] = 100
        update_status(f"The model is trained!", status_bar)
        messagebox.showinfo("Success!", "The model is trained successfully!")
    except Exception as e:
        model_trained = False
        progress_bar['value'] = 0
        update_status("Error during training", status_bar)
        messagebox.showerror("Error during training",
                             f"Failed to train model:\n{str(e)}\n\nDetails in the console")


def main():
    root = tk.Tk()
    create_gui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
