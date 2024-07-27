import json
import pandas as pd
import streamlit as st
import os

def load_loss_data_from_file(filename='placement_loss_data.json'):
    current_path = os.getcwd()
    with open(os.path.join(current_path, filename), 'r') as f:
        return json.load(f)

def main():
    st.set_page_config(layout="wide")  # Set the layout to wide mode
    st.title("Model Prediction Loss Dashboard")

    placement_loss_data = load_loss_data_from_file()

    placement_ids = list(placement_loss_data.keys())
    selected_placement_id = st.sidebar.selectbox("Select Placement ID", placement_ids)

    st.subheader(f"Placement ID: {selected_placement_id}")
    df = pd.DataFrame(placement_loss_data[selected_placement_id])
    df['Difference (%)'] = abs(df['percentage_predicted_loss'] - df['percentage_actual_loss'])
    df = df[['last_age', 'live_birds', 'total_actual_loss', 'total_predicted_loss', 'percentage_actual_loss', 'percentage_predicted_loss', 'Difference (%)']]
    
    st.write(
        f"<div style='height:1000px;'>{df.to_html(index=False)}</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
