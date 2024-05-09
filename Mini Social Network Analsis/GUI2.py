import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import Functions
import math
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk
import matplotlib.pyplot as plt
import networkx as nx
import random
import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedStyle




def load_csv_file(entry_var, dropdown_menu):
    filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    entry_var.set(filename)
    if filename:
        df = pd.read_csv(filename)

        columns = df.columns.tolist()
        dropdown_menu["menu"].delete(0, "end")
        for column in columns:
            dropdown_menu["menu"].add_command(label=column, command=lambda col=column: class_label_var.set(col))


apply_pressed = False
social_graph = nx.Graph()  # Initialize social_graph with an empty graph


def apply_preferences():
    global social_graph  # Access the global social_graph variable
    global apply_pressed
    apply_pressed = True
    node_size = node_size_var.get()  # Get the selected node size
    class_label_column = class_label_var.get()
    edge_color = edge_color_var.get()
    print("Node Size:", node_size)
    print("Class Label Column:", class_label_column)
    print("Edge Color:", edge_color)

    nodes_file = nodes_file_var.get()
    edges_file = edges_file_var.get()

    if nodes_file and edges_file:
        directed = var.get()
        nodes_df = pd.read_csv(nodes_file)
        edges_df = pd.read_csv(edges_file)

        # Get unique class labels from the specified column
        class_labels = nodes_df.set_index('ID')[class_label_column].to_dict()

        # Generate random colors for each unique class label
        color_map = {label: mcolors.to_hex(random.choice(list(mcolors.CSS4_COLORS.values()))) for label in
                     class_labels.values()}

        social_graph = Functions.create_graph(nodes_df, edges_df, directed)
        social_graph = Functions.customize_node_attributes(social_graph, node_size, class_label_column, color_map)
        social_graph = Functions.customize_edge_attributes(social_graph, edge_color)

        # Assign node colors based on their labels
        node_color = [color_map[label] for label in nx.get_node_attributes(social_graph, class_label_column).values()]

        # Draw the network
        control_view(social_graph, node_size, node_color, edge_color)


def control_view(social_graph, node_size, node_color, edge_color):
    plt.figure(figsize=(8, 6))

    # if fruchterman_reingold_layout:
    #     pos = nx.fruchterman_reingold_layout(social_graph)
    # elif apply_radial_layout:
    #     pos = nx.nx_agraph.graphviz_layout(social_graph, prog="twopi")
    # elif apply_tree_layout:
    #     pos = nx.nx_agraph.graphviz_layout(social_graph, prog="dot")
    # else:
    #     pos = nx.spring_layout(social_graph)

    pos = nx.spring_layout(social_graph)
    nx.draw(social_graph, pos, with_labels=False, node_size=node_size, node_color=node_color,
            edge_color=edge_color, linewidths=1, font_size=10)
    plt.title("Social Network Graph")
    plt.suptitle("Network Graph : with user prefrences", fontsize=16)
    plt.show()


fruchterman_reingold_layout = False
apply_tree_layout = False
apply_radial_layout = False
circular_layout = False


def apply_fruchterman_reingold():
    global fruchterman_reingold_layout, apply_tree_layout, apply_radial_layout, circular_layout
    fruchterman_reingold_layout = True
    apply_tree_layout = False
    apply_radial_layout = False
    circular_layout = False
    view_network()


def apply_circular_layout():
    global fruchterman_reingold_layout, apply_tree_layout, apply_radial_layout, circular_layout
    fruchterman_reingold_layout = False
    apply_tree_layout = False
    apply_radial_layout = False
    circular_layout = True
    draw_network(social_graph)


def apply_hierarchical_layout():
    # Apply hierarchical layout algorithm (e.g., tree or radial layouts)
    pos = nx.kamada_kawai_layout(social_graph)

    # Create a new top-level window for displaying the graph
    graph_window = tk.Toplevel()
    graph_window.title("Hierarchical Layout")

    # Draw the graph with the specified layout positions in the new window
    draw_graph_with_layout(graph_window, pos)


def draw_graph_with_layout(root, positions):
    # Create a new figure for the graph
    fig, ax = plt.subplots()

    # Draw the graph with the specified layout positions
    nx.draw(social_graph, positions, with_labels=True, ax=ax)

    # Create a canvas widget to display the graph in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)



def apply_radial():
    fruchterman_reingold_layout, apply_tree_layout, apply_radial_layout, circular_layout
    fruchterman_reingold_layout = False
    circular_layout = False
    apply_tree_layout = False
    apply_radial_layout = True
    view_network()


def draw_network(graph):
    plt.figure(figsize=(8, 6))
    label = ""
    if fruchterman_reingold_layout:
        label = "fruchterman_reingold_layout"
        pos = nx.fruchterman_reingold_layout(graph)
    elif apply_radial_layout:
        label = "apply_radial_layout"
        pos = nx.nx_agraph.graphviz_layout(graph, prog="twopi")
    elif apply_tree_layout:
        label = "apply_tree_layout"
        pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")
    elif circular_layout:
        label = "circular_layout"
        pos = nx.circular_layout(graph)
    else:
        pos = nx.spring_layout(graph)

    nx.draw(graph, pos, with_labels=True, node_size=400, node_color='skyblue',
            edge_color='black', linewidths=1, font_size=10)

    plt.title("Social Network Graph")
    plt.suptitle("Network Graph : " + label, fontsize=16)
    plt.show()


def view_network():
    global social_graph
    directed = var.get()
    nodes_file = nodes_file_var.get()
    edges_file = edges_file_var.get()

    if nodes_file and edges_file:
        nodes_df = pd.read_csv(nodes_file)
        edges_df = pd.read_csv(edges_file)
        social_graph = Functions.create_graph(nodes_df, edges_df, directed)
        draw_network(social_graph)


# Function to compute and display graph metrics
def display_graph_metrics():
    metrics = Functions.compute_graph_metrics(social_graph)
    for key, value in metrics.items():
        label_text = f"{key}: {value}"
        tk.Label(metrics_frame, text=label_text).pack(anchor='w')


def apply_filters():
    global social_graph

    # Get the selected centrality measure
    centrality_measure = centrality_var.get()

    # Get the threshold/range value
    threshold = threshold_entry.get()

    # Check if community membership filtering is enabled
    filter_by_community = community_var.get()

    # Validate the threshold/range input
    try:
        threshold = float(threshold)
    except ValueError:
        # Invalid threshold value
        messagebox.showerror("Error", "Invalid threshold value. Please enter a valid numerical value.")
        return

    # Check if a centrality measure is selected
    if not centrality_measure and not filter_by_community:
        # No filter options selected
        messagebox.showerror("Error", "Please select at least one filtering option.")
        return

    # Apply the selected filters
    if centrality_measure:
        social_graph = Functions.filter_by_centrality(social_graph, centrality_measure, threshold)

    if filter_by_community:
        social_graph = Functions.filter_by_community_membership(social_graph)

    # Redraw the network with the applied filters
    draw_network(social_graph)
    # Provide feedback to the user
    messagebox.showinfo("Success", "Filters applied successfully.")


def compare_communities():
    print("Compare button pressed")
    global social_graph

    selected_algorithms = []

    # Check which algorithms are selected
    if girvan_newman_var.get():
        selected_algorithms.append("Girvan-Newman")

    if louvain_var.get():
        selected_algorithms.append("Louvain")

    if FGreedy_var.get():
        selected_algorithms.append("Fast Greedy")

    print("Selected algorithms:", selected_algorithms)

    # If no algorithm is selected, show an error message
    if not selected_algorithms:
        messagebox.showerror("Error", "Please select at least one community detection algorithm.")
        return

    # Compute communities for each selected algorithm
    results = {}
    for algorithm in selected_algorithms:
        if algorithm == "Girvan-Newman":
            community_graph, communities_count, modularity_score, Communities = Functions.Girvan_Newman_Algorithm(
                social_graph)
        elif algorithm == "Louvain":
            community_graph, communities_count, modularity_score, Communities = Functions.Louvain_Algorithm(
                social_graph)
        elif algorithm == "Fast Greedy":
            community_graph, communities_count, modularity_score, Communities = Functions.Fast_Greedy_Algorithm(
                social_graph)

        results[algorithm] = (community_graph, communities_count, modularity_score, Communities)

    # Update the GUI with the results for each algorithm
    for algorithm, (community_graph, communities_count, modularity_score, Communities) in results.items():
        if algorithm == "Girvan-Newman":
            girvan_newman_communities_label.config(text=f"Girvan-Newman Communities: {communities_count}")
            girvan_newman_modularity_label.config(text=f"Girvan-Newman Modularity Score: {modularity_score}")
            draw_community_graph(community_graph, algorithm, Communities)
        elif algorithm == "Louvain":
            louvain_communities_label.config(text=f"Louvain Communities: {communities_count}")
            louvain_modularity_label.config(text=f"Louvain Modularity Score: {modularity_score}")
            draw_community_graph(community_graph, algorithm, Communities)
        elif algorithm == "Fast Greedy":
            FGreedy_communities_label.config(text=f"Fast Greedy Communities: {communities_count}")
            FGreedy_modularity_label.config(text=f"Fast Greedy Modularity Score: {modularity_score}")
            draw_community_graph(community_graph, algorithm, Communities)



import matplotlib.pyplot as plt
import networkx as nx
import random


def draw_community_graph(graph, Algo, Communities):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph)

    # Generate a list of unique colors for each community
    if isinstance(Communities, dict):
        unique_communities = list(set(Communities.values()))
        community_colors = {community: f'#{random.randint(0, 0xFFFFFF):06x}' for community in unique_communities}
        node_colors = [community_colors.get(Communities.get(node, None), 'gray') for node in graph.nodes]
    elif isinstance(Communities, (list, tuple)):
        # Assume Communities is a list or tuple of communities
        unique_communities = list(set(Communities))
        community_colors = {community: f'#{random.randint(0, 0xFFFFFF):06x}' for community in unique_communities}
        node_colors = [community_colors.get(community, 'gray') for community in Communities for node in community]
    elif isinstance(Communities, set):
        # Handle sets differently
        unique_communities = list(Communities)
        community_colors = {community: f'#{random.randint(0, 0xFFFFFF):06x}' for community in unique_communities}
        node_colors = [community_colors.get(node, 'gray') for node in graph.nodes]
    else:
        # If Communities is of an unsupported type, raise an error
        raise ValueError("Unsupported type for Communities")

    # Draw nodes with different colors based on their community membership
    nx.draw(graph, pos, with_labels=True, node_size=300, node_color=node_colors)

    # Draw community labels
    labels = nx.get_node_attributes(graph, 'community')
    nx.draw_networkx_labels(graph, pos, labels, font_color='red')

    plt.suptitle("Communities by: " + Algo, fontsize=16)
    plt.show()


def partition_and_visualize():
    global social_graph
    attribute_name = attribute_var.get()
    clusters = Functions.partition_graph_based_on_attribute(social_graph, attribute_name)

    num_partitions = len(clusters)
    cols = 3  # Number of columns for subplots
    rows = (num_partitions + cols - 1) // cols  # Calculate number of rows needed

    plt.figure(figsize=(15, 5 * rows))

    for i, (partition, nodes) in enumerate(clusters.items(), start=1):
        plt.subplot(rows, cols, i)
        subgraph = social_graph.subgraph(nodes)
        pos = nx.spring_layout(subgraph)

        # Generate a random color for this partition
        color = [random.random(), random.random(), random.random()]
        nx.draw(subgraph, pos, with_labels=True, node_color=color, node_size=300)
        plt.title("Partition " + str(partition))

    plt.suptitle("Graph Partitioning based on Attribute: " + attribute_name, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust subplot layout
    plt.show()


def select_algorithm(root):
    ty = "Directed" if social_graph.is_directed() else "Undirected"
    print("Type of the graph is " + ty)
    # Create a new top-level window
    new_window = tk.Toplevel(root)
    new_window.geometry("500x300")
    new_window.title("Techniques")
    # =====================================================

    label = tk.Label(new_window, text="Select an option", font=12)
    label.pack()

    # Create a list of options for the combobox
    options = ["Louvain algorithm", "Modularity", "Conductance",
               "NMI", "Page rank", "Degree centrality", "Closeness centrality",
               "Betweenness centrality", "Adjust graph", "Gervan New-Man"]

    # Create a StringVar object to store the selected option
    selected_option = tk.StringVar()

    # Create the combobox widget and pack it onto the window
    combobox = ttk.Combobox(new_window, values=options, textvariable=selected_option)
    combobox.pack()

    # Create a function to handle combobox selection events
    def combobox_selected(event):
        if selected_option.get() == options[0]:
            Functions.Louvain_algorithm(social_graph)
        elif selected_option.get() == options[8]:
            Functions.adjust_graph(social_graph)
        elif selected_option.get() == options[4]:
            Functions.Page_Rank(social_graph)
        elif selected_option.get() == options[5]:
            Functions.Degree_Centrality(social_graph)
        elif selected_option.get() == options[6]:
            Functions.Closeness_Centrality(social_graph)
        elif selected_option.get() == options[7]:
            Functions.Betweenness_Centrality(social_graph)
        elif selected_option.get() == options[1]:
            Functions.Modularity(social_graph)
        elif selected_option.get() == options[2]:
            Functions.Conductance(social_graph)
        elif selected_option.get() == options[3]:
            Functions.NMI(social_graph)
        elif selected_option.get() == options[9]:
            Functions.Girvan_Newman_Algorithm(social_graph)

    # Bind the combobox selection event to the combobox_selected function
    combobox.bind("<<ComboboxSelected>>", combobox_selected)
    combobox.pack(pady=60)

    # Set the default option
    selected_option.set(options[0])



# GUI setup
root = tk.Tk()
root.title("Social Network Analysis Tool")
root.geometry("900x700")  # Set window size

# Apply a Bootstrap-like theme
style = ThemedStyle(root)
style.set_theme("radiance")  # Choose a theme similar to Bootstrap, e.g., Radiance

# Create a canvas with a scrollbar
canvas = tk.Canvas(root)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Add a scrollbar to the canvas
scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill="y")

# Configure the canvas to scroll vertically with the scrollbar
canvas.configure(yscrollcommand=scrollbar.set)

# Create a frame inside the canvas to hold all the widgets
frame = ttk.Frame(canvas)
canvas.create_window((0, 0), window=frame, anchor="nw")

# Function to update the scroll region when the frame size changes
def on_frame_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

frame.bind("<Configure>", on_frame_configure)

# Frame for data loading segment
data_loading_frame = ttk.Frame(frame)
data_loading_frame.pack(side=tk.TOP, padx=10, pady=10, anchor='nw' , fill='x')

# Data Loading Segment
# Radio buttons for graph direction
var = tk.BooleanVar()
var.set(False)  # Default to undirected graph
directed_frame = tk.Frame(data_loading_frame)
directed_frame.pack(side=tk.TOP, pady=(0, 10),fill='x')
tk.Label(directed_frame, text="Graph Direction:").pack(side=tk.LEFT, padx=(0, 5))
tk.Radiobutton(directed_frame, text="Undirected", variable=var, value=False).pack(side=tk.LEFT)
tk.Radiobutton(directed_frame, text="Directed", variable=var, value=True).pack(side=tk.LEFT)

# Buttons and labels for nodes and edges files
nodes_file_var = tk.StringVar()
edges_file_var = tk.StringVar()

# Nodes File Entry
nodes_label = ttk.Label(data_loading_frame, text="Nodes CSV File:")
nodes_label.pack(anchor='w', pady=(0, 5), fill='x')
nodes_entry = ttk.Entry(data_loading_frame, textvariable=nodes_file_var, state='disabled')
nodes_entry.pack(anchor='w', fill='x')
nodes_button = ttk.Button(data_loading_frame, text="Select Nodes File", command=lambda: load_csv_file(nodes_file_var, class_label_dropdown))
nodes_button.pack(anchor='w', pady=(5, 0))

# Edges File Entry
edges_label = ttk.Label(data_loading_frame, text="Edges CSV File:")
edges_label.pack(anchor='w', pady=(5, 0), fill='x')
edges_entry = ttk.Entry(data_loading_frame, textvariable=edges_file_var, state='disabled')
edges_entry.pack(anchor='w', fill='x')
edges_button = ttk.Button(data_loading_frame, text="Select Edges File", command=lambda: load_csv_file(edges_file_var, None))
edges_button.pack(anchor='w')

# Button to view network
view_button = ttk.Button(data_loading_frame, text="View Network", command=view_network)
view_button.pack(anchor='w', pady=(5, 0))

# Frame for view control segment
view_control_frame = ttk.Frame(frame)
view_control_frame.pack(side=tk.TOP, padx=10, pady=5, anchor='nw', fill='x')

# Frame for view control segment
view_control_frame = ttk.Frame(frame)
view_control_frame.pack(side=tk.TOP, padx=10, pady=5, anchor='nw', fill='x')

# View Control Segment
# Node Size
node_size_label = ttk.Label(view_control_frame, text="Node Size:")
node_size_label.grid(row=0, column=0, padx=(0, 5))
node_size_var = tk.IntVar()
node_size_scale = ttk.Scale(view_control_frame, variable=node_size_var, from_=100, to=800, orient=tk.HORIZONTAL)
node_size_scale.grid(row=0, column=1, padx=(0, 5))

# Class Label Column Dropdown
class_label_label = ttk.Label(view_control_frame, text="Class Label Column:")
class_label_label.grid(row=0, column=2, padx=(0, 5))
class_label_var = tk.StringVar()
class_label_dropdown = ttk.OptionMenu(view_control_frame, class_label_var, "")
class_label_dropdown.grid(row=0, column=3, padx=(0, 5))

# Edge Color Dropdown
edge_color_label = ttk.Label(view_control_frame, text="Edge Color:")
edge_color_label.grid(row=0, column=4, padx=(0, 5))
edge_color_var = tk.StringVar()
edge_color_options = ["Red", "Green", "Blue", "Yellow", "Orange", "grey", "Pink", "Black"]
edge_color_dropdown = ttk.OptionMenu(view_control_frame, edge_color_var, *edge_color_options)
edge_color_dropdown.grid(row=0, column=5, padx=(0, 5))

# Apply button for customizing attributes
apply_button = ttk.Button(view_control_frame, text="Apply", command=apply_preferences)
apply_button.grid(row=1, column=0, columnspan=6, pady=(5, 8))

# Frame for layout control buttons
layout_control_frame = ttk.Frame(frame)
layout_control_frame.pack(side=tk.TOP, anchor='nw', fill='x')

# Layout Control Buttons
fr_button = ttk.Button(layout_control_frame, text="Fruchterman-Reingold", command=apply_fruchterman_reingold)
fr_button.pack(side=tk.LEFT, padx=(10, 5))
tree_button = ttk.Button(layout_control_frame, text="Tree Layout", command=apply_hierarchical_layout)
tree_button.pack(side=tk.LEFT, padx=5)
circular_button = ttk.Button(layout_control_frame, text="Circular Layout", command=apply_circular_layout)
circular_button.pack(side=tk.LEFT, padx=5)

metrics_frame = ttk.Frame(frame)
metrics_frame.pack(side=tk.TOP, padx=10, pady=5, anchor='nw', fill='x')

# Graph Metrics Segment
metrics_label = ttk.Label(metrics_frame, text="Graph Metrics and Statistics:")
metrics_label.pack(anchor='w', pady=(0, 5), fill='x')

# Button to compute and display graph metrics
metrics_button = ttk.Button(metrics_frame, text="Compute Metrics", command=display_graph_metrics)
metrics_button.pack(anchor='w')

# GUI setup for filtering options
filter_options_frame = ttk.Frame(root)
filter_options_frame.pack(side=tk.TOP, padx=10, pady=5, anchor='nw', fill='x')

# Filtering Options Segment
filter_options_label = ttk.Label(filter_options_frame, text="Filtering Options:")
filter_options_label.pack(anchor='w', pady=(0, 5), fill='x')

# Centrality Measures Dropdown
#centrality_label = ttk.Label(filter_options_frame, text="Centrality Measures:")
#centrality_label.pack(anchor='w', pady=(0, 5), fill='x')
#centrality_var = tk.StringVar()
#centrality_options = ["Degree Centrality", "Betweenness Centrality", "Closeness Centrality"]  # Add more as needed
#centrality_dropdown = ttk.OptionMenu(filter_options_frame, centrality_var, *centrality_options)
#centrality_dropdown.pack(anchor='w', pady=(0, 5), fill='x')

# Threshold/Ranges Input Fields
threshold_label = ttk.Label(filter_options_frame, text="Threshold/Range:")
threshold_label.pack(anchor='w', pady=(0, 5), fill='x')
threshold_entry = ttk.Entry(filter_options_frame)
threshold_entry.pack(anchor='w', pady=(0, 5), fill='x')

# Community Membership Checkbox
community_var = tk.BooleanVar()
community_checkbox = ttk.Checkbutton(filter_options_frame, text="Filter by Community Membership", variable=community_var)
community_checkbox.pack(anchor='w', pady=(0, 5), fill='x')

# Apply Filters Button
apply_filters_button = ttk.Button(filter_options_frame, text="Apply Filters", command=apply_filters)
apply_filters_button.pack(anchor='w', pady=(0, 5), fill='x')

# GUI setup for community detection comparison
community_comparison_frame = ttk.Frame(root)
community_comparison_frame.pack(side=tk.TOP, padx=10, pady=5, anchor='nw', fill='x')

# Community Detection Comparison Segment
community_comparison_label = ttk.Label(community_comparison_frame, text="Community Detection Comparison:")
community_comparison_label.pack(anchor='w', pady=(0, 5), fill='x')

# Checkbox for Girvan-Newman algorithm
girvan_newman_var = tk.BooleanVar()
girvan_newman_checkbox = ttk.Checkbutton(community_comparison_frame, text="Girvan-Newman Algorithm", variable=girvan_newman_var)
girvan_newman_checkbox.pack(anchor='w', pady=(0, 5), fill='x')

# Checkbox for Louvain algorithm
louvain_var = tk.BooleanVar()
louvain_checkbox = ttk.Checkbutton(community_comparison_frame, text="Louvain Algorithm", variable=louvain_var)
louvain_checkbox.pack(anchor='w', pady=(0, 5), fill='x')

# Checkbox for Louvain algorithm
FGreedy_var = tk.BooleanVar()
FGreedy_checkbox = ttk.Checkbutton(community_comparison_frame, text="Fast Greedy Algorithm", variable=FGreedy_var)
FGreedy_checkbox.pack(anchor='w', pady=(0, 5), fill='x')

# Button to perform community detection comparison
compare_communities_button = ttk.Button(community_comparison_frame, text="Compare Communities", command=compare_communities)
compare_communities_button.pack(anchor='w', pady=(0, 5), fill='x')

# Frame to display results
results_frame = ttk.Frame(root)
results_frame.pack(side=tk.TOP, padx=10, pady=5, anchor='nw', fill='x')

# Results Segment
results_label = ttk.Label(results_frame, text="Community Detection Comparison Results:")
results_label.pack(anchor='w', pady=(0, 5), fill='x')

# Label to display number of communities detected for Girvan-Newman algorithm
girvan_newman_communities_label = ttk.Label(results_frame, text="Girvan-Newman Communities: ")
girvan_newman_communities_label.pack(anchor='w', fill='x')

# Label to display modularity score for Girvan-Newman algorithm
girvan_newman_modularity_label = ttk.Label(results_frame, text="Girvan-Newman Modularity Score: ")
girvan_newman_modularity_label.pack(anchor='w', fill='x')

# Label to display number of communities detected for Louvain algorithm
louvain_communities_label = ttk.Label(results_frame, text="Louvain Communities: ")
louvain_communities_label.pack(anchor='w', fill='x')

# Label to display modularity score for Louvain algorithm
louvain_modularity_label = ttk.Label(results_frame, text="Louvain Modularity Score: ")
louvain_modularity_label.pack(anchor='w', fill='x')

# Label to display number of communities detected for Fast Greedy algorithm
FGreedy_communities_label = ttk.Label(results_frame, text="Fast Greedy Communities: ")
FGreedy_communities_label.pack(anchor='w', fill='x')

# Label to display modularity score for Fast Greedy algorithm
FGreedy_modularity_label = ttk.Label(results_frame, text="Fast Greedy Modularity Score: ")
FGreedy_modularity_label.pack(anchor='w', fill='x')

# Frame for attribute selection
attribute_frame = ttk.Frame(root)
attribute_frame.pack(side=tk.TOP, padx=10, pady=5, anchor='nw', fill='x')

# Label for attribute selection
attribute_label = ttk.Label(attribute_frame, text="Select Attribute for Graph Partitioning:")
attribute_label.pack(side=tk.LEFT, padx=5, pady=5, fill='x')

# Variable to store selected attribute
attribute_var = tk.StringVar(root)
attribute_var.set("Class")  # Set default value

# Dropdown menu for attribute selection
attribute_options = ["Class", "Gender"]  # Options for attribute selection
attribute_dropdown = ttk.OptionMenu(attribute_frame, attribute_var, *attribute_options)
attribute_dropdown.pack(side=tk.LEFT, padx=5, pady=5, fill='x')

# Button to partition graph and visualize clusters
partition_button = ttk.Button(root, text="Partition and Visualize", command=partition_and_visualize)
partition_button.pack(side=tk.TOP, padx=10, pady=5, fill='x')



root.mainloop()
