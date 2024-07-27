import json
import pandas as pd
import streamlit as st
import os
import matplotlib.pyplot as plt
from io import BytesIO

# Function to load loss data from a JSON file
def load_loss_data_from_file(filename='placement_loss_data2.json'):
    current_path = os.getcwd()
    with open(os.path.join(current_path, filename), 'r') as f:
        return json.load(f)

# Function to plot actual vs predicted values for a placement
def plot_actual_vs_predicted_for_placement(placement_loss_data, placement_id):
    if placement_id not in placement_loss_data:
        print(f"Placement ID {placement_id} not found in the data.")
        return None
    
    losses = placement_loss_data[placement_id]
    
    figs = []
    for repetition, loss_data in enumerate(losses, start=1):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(loss_data['actual'], label='Actual Loss', marker='o')
        ax.plot(loss_data['predictions'], label='Predicted Loss', linestyle='--', marker='x')
        ax.set_xlabel('Days')
        ax.set_ylabel('Loss')
        ax.set_title(f'Placement ID: {placement_id} - Last Age: {loss_data["last_age"]}')
        ax.legend()
        ax.grid(True)
        figs.append(fig)
    
    return figs

def main():
    st.set_page_config(layout="wide")  # Set the layout to wide mode
    st.title("Model Prediction Loss Dashboard")

    placement_loss_data = load_loss_data_from_file()

    placement_ids = list(placement_loss_data.keys())
    selected_placement_id = st.sidebar.selectbox("Select Placement ID", placement_ids)

    st.subheader(f"Placement ID: {selected_placement_id}")
    df = pd.DataFrame(placement_loss_data[selected_placement_id])
    df['Difference (%)'] = abs(df['total_predicted_loss'] - df['total_actual_loss']) / df['live_birds'] * 100
    df = df[['last_age', 'live_birds', 'total_actual_loss', 'total_predicted_loss', 'Difference (%)']]
    
    st.write(
        f"<div style='height:500px;overflow:auto;'>{df.to_html(index=False)}</div>", 
        unsafe_allow_html=True
    )

    # Plotting actual vs predicted values for each repetition
    st.subheader("Actual vs Predicted Loss for Each Repetition")
    figs = plot_actual_vs_predicted_for_placement(placement_loss_data, selected_placement_id)
    
    if figs:
        for i, fig in enumerate(figs):
            st.write(f"Repetition {i + 1}")
            buf = BytesIO()
            fig.savefig(buf, format="png")
            st.image(buf, caption=f'Actual vs Predicted Loss for Repetition {i + 1}')
            plt.close(fig)  # Close the figure after saving it to buffer

if __name__ == "__main__":
    main()
