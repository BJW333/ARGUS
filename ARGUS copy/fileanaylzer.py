import os
import pytesseract
from PIL import Image
import PyPDF2
from docx import Document
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import re
from transformers import pipeline
import tkinter as tk
from tkinter import filedialog
import threading

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def process_file(file_path, file_type=None):
    """
    Processes a file based on its type (image, PDF, Word document, CSV, TXT, or visual input).

    Args:
        file_path (str): Path to the file to be processed.
        file_type (str): Optional. Specify the type of file ('image', 'pdf', 'word', 'csv', 'txt', 'chart', or 'receipt').

    Returns:
        str: Result of the processing (extracted text, summary, analysis, or visualization).
    """
    if not os.path.exists(file_path):
        return "File not found. Please check the path and try again."

    try:
        #automatically determine file type if not provided
        if file_type is None:
            _, ext = os.path.splitext(file_path)
            file_type = ext.lower().replace(".", "")  #get the file extension

        if file_type in ['png', 'jpg', 'jpeg', 'image']:
            return process_image(file_path)
        elif file_type == 'pdf':
            return process_pdf(file_path)
        elif file_type in ['docx', 'word']:
            return process_word(file_path)
        elif file_type == 'csv':
            return process_csv(file_path)
        elif file_type == 'txt':
            return process_txt(file_path)
        elif file_type == 'chart':
            return analyze_chart(file_path)
        elif file_type == 'receipt':
            return analyze_receipt(file_path)
        else:
            return "Unsupported file type. Please provide an image, PDF, Word document, CSV, TXT file, or specify 'chart' or 'receipt' analysis."
    except Exception as e:
        return f"An error occurred while processing the file: {e}"

def process_image(image_path):
    """extracts text from an image using OCR"""
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return f"Extracted text from image:\n{text}"
    except Exception as e:
        return f"Error processing image: {e}"

def process_pdf(pdf_path):
    """extracts text from a PDF and summarizes it"""
    try:
        text = ""
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        if not text.strip():
            return "No text found in the PDF file."
        summary = summarize_text(text)
        return f"PDF Text Summary:\n{summary}"
    except Exception as e:
        return f"Error processing PDF: {e}"

def process_word(docx_path):
    """extracts text from a Word document and summarizes it"""
    try:
        doc = Document(docx_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        if not text.strip():
            return "No text found in the Word document."
        summary = summarize_text(text)
        return f"Word Document Text Summary:\n{summary}"
    except Exception as e:
        return f"Error processing Word document: {e}"

def process_csv(csv_path):
    """
    Processes a CSV file to extract data and provide insights or visualizations.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        str: Summary of the CSV file or a notification about the generated visualization.
    """
    try:
        #load CSV file into a pandas dataframe
        data = pd.read_csv(csv_path)

        #basic summary
        summary = f"CSV File Summary:\nColumns: {', '.join(data.columns)}\nNumber of Rows: {len(data)}\n\n"
        
        #display basic statistics
        stats = data.describe().to_string()
        summary += f"Statistics:\n{stats}\n"

        #generate a visualization
        plt.figure(figsize=(8, 6))
        if len(data.columns) > 1:
            data.hist(bins=15, figsize=(12, 8))
            plt.suptitle("Histograms for Numeric Columns", fontsize=14)
        else:
            plt.plot(data[data.columns[0]])
            plt.title(f"Plot of {data.columns[0]}")
        plt.savefig("csv_visualization.png")
        plt.close()
        
        summary += "A visualization has been saved as 'csv_visualization.png'."
        return summary
    except Exception as e:
        return f"Error processing CSV file: {e}"

def process_txt(txt_path):
    """extracts text from a TXT file and summarizes it"""
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            text = file.read()
        if not text.strip():
            return "No text found in the TXT file."
        summary = summarize_text(text)
        return f"TXT File Text Summary:\n{summary}"
    except Exception as e:
        return f"Error processing TXT file: {e}"

def analyze_chart(image_path):
    """analyzes a chart image and identifies key components."""
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            return "Chart detected. Do you want me to analyze specific data points?"
        else:
            return "No chart detected in the image."
    except Exception as e:
        return f"Error analyzing chart: {e}"

def analyze_receipt(image_path):
    """extracts data like total amount from a receipt image"""
    try:
        text_result = process_image(image_path)
        text = text_result.replace("Extracted text from image:\n", "")
        if text:
            total_match = re.search(r"Total:\s*\$?([\d.,]+)", text, re.IGNORECASE)
            if total_match:
                total_amount = total_match.group(1)
                return f"Total amount on the receipt is ${total_amount}."
            else:
                return "I couldn't find the total amount on the receipt."
        return "No text found in the receipt image."
    except Exception as e:
        return f"Error analyzing receipt: {e}"

def summarize_text(text):
    """summarizes the given text using a summarization model"""
    try:
        if len(text.split()) > 100:  #avoid summarizing very short texts
            #break the text into chunks if it's too long
            max_chunk = 500
            text_chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
            summaries = []
            for chunk in text_chunks:
                summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
                summaries.append(summary[0]["summary_text"])
            return ' '.join(summaries)
        return text  #if text is short return it as is
    except Exception as e:
        return f"Error summarizing text: {e}"

def input_file_command():
    file_path = filedialog.askopenfilename(title="Select a File")
    if not file_path:
        return  #user cancelled the file dialog

    #automatically determine file type based on file extension
    _, ext = os.path.splitext(file_path)
    file_type = ext.lower().replace(".", "")  #get the file extension

    #start processing in a separate thread to keep the GUI responsive
    threading.Thread(target=process_and_display, args=(file_path, file_type)).start()

def process_and_display(file_path, file_type):
    #process the file
    result = process_file(file_path, file_type)

    #display the result in a new window
    result_window = tk.Toplevel()
    result_window.title("Processing Result")
    result_text = tk.Text(result_window, wrap=tk.WORD)
    result_text.pack(expand=True, fill=tk.BOTH)
    result_text.insert(tk.END, result)
    result_text.config(state=tk.DISABLED)  #make the text read only

    #if a visualization image was saved display it
    if os.path.exists("csv_visualization.png") and file_type == 'csv':
        img_window = tk.Toplevel()
        img_window.title("CSV Visualization")
        img = tk.PhotoImage(file="csv_visualization.png")
        img_label = tk.Label(img_window, image=img)
        img_label.image = img  #keep a reference
        img_label.pack()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("File Analyzer")

    #set the window size
    root.geometry("300x150")

    #create a frame for the upload button
    frame = tk.Frame(root)
    frame.pack(expand=True)

    #create and pack the upload button
    upload_button = tk.Button(frame, text="Upload and Analyze File", command=input_file_command)
    upload_button.pack(pady=20)

    root.mainloop()
