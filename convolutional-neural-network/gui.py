import customtkinter as ctk
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

df1 = pd.DataFrame({
    'year': [2001, 2002, 2003],
    'value': [1, 3, 2],
    'personal': [9, 1, 5],
})

df2 = pd.DataFrame({
    'year': [2001, 2002, 2003],
    'value': [1, 3, 2],
    'personal': [9, 1, 5],
})

root = ctk.CTk()
root.geometry("500x350")

def login():
    print("Login button clicked")

frame = ctk.CTkFrame(master=root)
frame.pack(pady=20, padx=60, fill="both", expand=True)

label = ctk.CTkLabel(master=frame, text="Login")
label.pack(pady=10)

entry = ctk.CTkEntry(master=frame)
entry.pack(pady=10)

button = ctk.CTkButton(master=frame, text="Login", command=login)
button.pack(pady=10)


figure1 = plt.Figure(figsize=(2,2), dpi=100)

scatter1 = FigureCanvasTkAgg(figure1, root)
# scatter1.get_tk_widget().grid(row=0, column=0, sticky='news')
scatter1.get_tk_widget().pack(fill='both', expand=True)

ax1 = figure1.add_subplot(111)
ax1.plot(df1['year'], df1['personal'], color='red')

ax1.legend([''])
ax1.set_xlabel('valeur de personals')
ax1.set_title('ev de personal ')

# --- 

figure2 = plt.Figure(figsize=(2,2), dpi=100)

scatter2 = FigureCanvasTkAgg(figure2, root)
# scatter2.get_tk_widget().grid(row=0, column=1, sticky='news')
scatter2.get_tk_widget().pack(side='right', fill='both', expand=True)

ax2 = figure2.add_subplot(111)
ax2.plot(df2['year'], df2['value'], color='red')

ax2.legend([''])
ax2.set_xlabel('valeur BSA')
ax2.set_title('Evolutiion des valeurs BSA depuis 1990 ')

# --- 

figure3 = plt.Figure(figsize=(2,2), dpi=100)

scatter3 = FigureCanvasTkAgg(figure3, root)
# scatter3.get_tk_widget().grid(row=1, column=0, sticky='news')
scatter3.get_tk_widget().pack(fill='both', expand=True)

ax3 = figure3.add_subplot(111)
ax3.plot(df1['year'], df1['personal'], color='red')

ax3.legend([''])
ax3.set_xlabel('valeur de personals')
ax3.set_title('ev de personal ')

# --- 

figure4 = plt.Figure(figsize=(2,2), dpi=100)

scatter4 = FigureCanvasTkAgg(figure4, root)
# scatter4.get_tk_widget().grid(row=1, column=1, sticky='news')
scatter4.get_tk_widget().pack(fill='both', expand=True)

ax4 = figure4.add_subplot(111)
ax4.plot(df2['year'], df2['value'], color='red')

ax4.legend([''])
ax4.set_xlabel('valeur BSA')
ax4.set_title('Evolutiion des valeurs BSA depuis 1990 ')

# ---

root.mainloop()


