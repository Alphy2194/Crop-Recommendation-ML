import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Load the dataset (adjust the file path)
data = pd.read_csv('C:/Users/AL4NS 0T13N0/OneDrive/Desktop/Crop_recommendation.csv')

# Create a Tkinter window for the login page
login_window = tk.Tk()
login_window.title("Login Page")
login_window.geometry("400x300")

# Configure a custom color scheme
login_window.configure(bg="#5F9EA0")

# Define the default login credentials
default_username = "admin"
default_password = "admin"


# Function to check login credentials
def check_login():
    entered_username = username_entry.get()
    entered_password = password_entry.get()

    if entered_username == default_username and entered_password == default_password:
        # Login successful, create the main page for crop recommendation
        create_main_page()
    else:
        login_result_label.config(text="Invalid login credentials")


# Create labels and entry widgets for the login page
username_label = ttk.Label(login_window, text="Username:", font=("Arial", 12), background="#5F9EA0")
username_label.grid(row=0, column=0, padx=10, pady=10)
username_entry = ttk.Entry(login_window)
username_entry.grid(row=0, column=1, padx=10, pady=10)
password_label = ttk.Label(login_window, text="Password:", font=("Arial", 12), background="#5F9EA0")
password_label.grid(row=1, column=0, padx=10, pady=10)
password_entry = ttk.Entry(login_window, show="*")  # Hide the password
password_entry.grid(row=1, column=1, padx=10, pady=10)

# Create a login button
login_button = ttk.Button(login_window, text="Login", command=check_login)
login_button.grid(row=2, columnspan=2, pady=20)

# Label to display login result
login_result_label = ttk.Label(login_window, text="", foreground="red", background="#5F9EA0")
login_result_label.grid(row=3, columnspan=2, pady=10)


# Function to create the main page for crop recommendation
def create_main_page():
    # Close the login window
    login_window.destroy()

    # Create a Tkinter window for the main page
    main_page = tk.Tk()
    main_page.title("Crop Recommendation System")
    main_page.geometry("800x600")

    # Configure a custom color scheme
    main_page.configure(bg="#5F9EA0")

    # Create a label for the title
    title_label = ttk.Label(main_page, text="Crop Recommendation System", font=("Arial", 16, "bold"),
                            background="#5F9EA0")
    title_label.pack(fill="x", padx=10, pady=10)

    # Create labels and entry widgets for user input
    input_labels = ["Nitrogen (N) value:", "Phosphorus (P) value:", "Potassium (K) value:",
                    "Temperature value:", "Humidity value:", "pH value:", "Rainfall value:"]
    input_entries = []

    for label_text in input_labels:
        label = ttk.Label(main_page, text=label_text)
        label.configure(background="#5F9EA0")
        label.pack()
        entry = ttk.Entry(main_page)
        entry.pack()
        input_entries.append(entry)

    # Create a button to submit user input and get recommendations
    submit_button = ttk.Button(main_page, text="Get Crop Recommendations",
                               command=lambda: recommend_crop(input_entries, recommendation_label, main_page))
    submit_button.pack(pady=20)

    # Label to display recommendations
    recommendation_label = ttk.Label(main_page, text="")
    recommendation_label.pack()

    # Create a button to view the graph
    view_graph_button = ttk.Button(main_page, text="View Graph", command=lambda: show_graph(input_entries))
    view_graph_button.pack()

    # Start the Tkinter main loop for the main page
    main_page.mainloop()


# Function to make crop recommendations based on user input
def recommend_crop(input_entries, recommendation_label, main_page):
    user_input = [float(entry.get()) for entry in input_entries]
    N, P, K, temperature, humidity, ph, rainfall = user_input

    user_input_df = pd.DataFrame({'N': [N], 'P': [P], 'K': [K], 'temperature': [temperature],
                                  'humidity': [humidity], 'ph': [ph], 'rainfall': [rainfall]})

    X = data.drop('label', axis=1)
    y = data['label']

    # Random Forest Classifier
    classifier_rf = RandomForestClassifier(n_estimators=10, criterion="entropy")
    classifier_rf.fit(X, y)
    user_pred_rf = classifier_rf.predict(user_input_df)

    # Display the recommended crop
    recommendation_label.config(text=f"Crop Recommendation (Random Forest Classifier): {user_pred_rf[0]}")


# Function to show the graph
def show_graph(input_entries):
    user_input = [float(entry.get()) for entry in input_entries]
    N, P, K, temperature, humidity, ph, rainfall = user_input

    user_input_df = pd.DataFrame({'N': [N], 'P': [P], 'K': [K], 'temperature': [temperature],
                                  'humidity': [humidity], 'ph': [ph], 'rainfall': [rainfall]})

    X = data.drop('label', axis=1)
    y = data['label']

    # Random Forest Classifier
    classifier_rf = RandomForestClassifier(n_estimators=10, criterion="entropy")
    classifier_rf.fit(X, y)
    y_pred_rf = classifier_rf.predict(X)

    # Calculate Recall, F1-Score, and Accuracy
    precision = precision_score(y, y_pred_rf, average='macro')
    recall = recall_score(y, y_pred_rf, average='macro')
    f1 = f1_score(y, y_pred_rf, average='macro')
    accuracy = accuracy_score(y, y_pred_rf)

    # Create a Tkinter window for the graph
    graph_window = tk.Tk()
    graph_window.title("Crop Recommendation Comparison")
    graph_window.geometry("600x400")

    # Configure a custom color scheme
    graph_window.configure(bg="#5F9EA0")

    graph_data = {
        'Metrics': ['Precision', 'Recall', 'F1-Score', 'Accuracy'],
        'Values': [precision, recall, f1, accuracy]
    }

    # Create a figure for the graph
    graph_fig = Figure(figsize=(6, 4), facecolor="#5F9EA0")
    graph_plot = graph_fig.add_subplot(111)
    graph_plot.bar(graph_data['Metrics'], graph_data['Values'], color=['blue', 'green', 'orange', 'red'])
    graph_plot.set_title('Crop Recommendation Metrics')
    graph_plot.set_xlabel('Metrics')
    graph_plot.set_ylabel('Values')

    # Create a canvas for the graph
    graph_canvas = FigureCanvasTkAgg(graph_fig, master=graph_window)
    graph_canvas.get_tk_widget().pack(fill="both", expand=True)
    graph_canvas.draw()

    # Start the Tkinter main loop for the graph window
    graph_window.mainloop()


# Start the Tkinter main loop for the login page
login_window.mainloop()

