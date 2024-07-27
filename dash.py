# import streamlit as st
# import torch
# import pandas as pd
# import numpy as np

# # Function to make predictions
# def make_predictions(data_loader, model, device):
#     model.eval()
#     all_data = []

#     with torch.no_grad():
#         for batch in data_loader:
#             features = batch['features'].to(device).float()
#             placement_ids = batch['placement_id'].to(device).float()
#             ages = batch['age'].to(device).float()
#             actual = batch['targets'].to(device).float()
#             live_birds_start = batch['live_birds_start'].to(device).float()

#             predictions = model(features)

#             batch_data = {
#                 'placement_ids': placement_ids.cpu().numpy(),
#                 'ages': ages.cpu().numpy(),
#                 'predictions': predictions.cpu().numpy(),
#                 'actual': actual.cpu().numpy(),
#                 'live_birds_start': live_birds_start.cpu().numpy()
#             }

#             all_data.append(batch_data)

#     return all_data

# # Function to extract the last non-zero value before padding
# def get_last_non_zero(sequence):
#     for value in reversed(sequence):
#         if value != 0:
#             return value
#     return 0

# # Function to extract the last age before padding
# def get_last_age_before_padding(age_sequence):
#     for value in reversed(age_sequence):
#         if value != 0:
#             return value
#     return 0

# # Function to calculate total loss and its percentage of live birds
# def calculate_losses(all_data):
#     placement_loss_data = {}

#     for data in all_data:
#         placement_ids = data['placement_ids']
#         predictions = data['predictions']
#         actual = data['actual']
#         live_birds_start = data['live_birds_start']
#         ages = data['ages']

#         for i in range(len(placement_ids)):
#             placement_id = int(placement_ids[i])
#             total_predicted_loss = predictions[i].sum()
#             total_actual_loss = actual[i].sum()
#             live_birds = get_last_non_zero(live_birds_start[i])
#             last_age = get_last_age_before_padding(ages[i])

#             if placement_id not in placement_loss_data:
#                 placement_loss_data[placement_id] = []

#             placement_loss_data[placement_id].append({
#                 'total_predicted_loss': total_predicted_loss,
#                 'total_actual_loss': total_actual_loss,
#                 'live_birds': live_birds,
#                 'last_age': last_age,
#                 'percentage_predicted_loss': total_predicted_loss / live_birds * 100 if live_birds != 0 else 0,
#                 'percentage_actual_loss': total_actual_loss / live_birds * 100 if live_birds != 0 else 0
#             })

#     return placement_loss_data






# import json

# def load_loss_data_from_file(filename='placement_loss_data.json'):
#     with open(filename, 'r') as f:
#         return json.load(f)



# # Streamlit app
# def main():
#     st.title("Model Prediction Loss Dashboard")

#     # Load placement_loss_data from file
#     placement_loss_data = load_loss_data_from_file()

#     # Display results
#     for placement_id, losses in placement_loss_data.items():
#         st.subheader(f"Placement ID: {placement_id}")
#         df = pd.DataFrame(losses)
#         df['Difference (%)'] = abs(df['percentage_predicted_loss'] - df['percentage_actual_loss'])
#         st.dataframe(df, use_container_width=True)

# if __name__ == "__main__":
#     main()



import streamlit as st
import torch
import pandas as pd
import numpy as np
import json

# # Function to make predictions
# def make_predictions(data_loader, model, device):
#     model.eval()
#     all_data = []

#     with torch.no_grad():
#         for batch in data_loader:
#             features = batch['features'].to(device).float()
#             placement_ids = batch['placement_id'].to(device).float()
#             ages = batch['age'].to(device).float()
#             actual = batch['targets'].to(device).float()
#             live_birds_start = batch['live_birds_start'].to(device).float()

#             predictions = model(features)

#             batch_data = {
#                 'placement_ids': placement_ids.cpu().numpy(),
#                 'ages': ages.cpu().numpy(),
#                 'predictions': predictions.cpu().numpy(),
#                 'actual': actual.cpu().numpy(),
#                 'live_birds_start': live_birds_start.cpu().numpy()
#             }

#             all_data.append(batch_data)

#     return all_data

# # Function to extract the last non-zero value before padding
# def get_last_non_zero(sequence):
#     for value in reversed(sequence):
#         if value != 0:
#             return value
#     return 0

# # Function to extract the last age before padding
# def get_last_age_before_padding(age_sequence):
#     for value in reversed(age_sequence):
#         if value != 0:
#             return value
#     return 0

# # Function to calculate total loss and its percentage of live birds
# def calculate_losses(all_data):
#     placement_loss_data = {}

#     for data in all_data:
#         placement_ids = data['placement_ids']
#         predictions = data['predictions']
#         actual = data['actual']
#         live_birds_start = data['live_birds_start']
#         ages = data['ages']

#         for i in range(len(placement_ids)):
#             placement_id = int(placement_ids[i])
#             total_predicted_loss = predictions[i].sum()
#             total_actual_loss = actual[i].sum()
#             live_birds = get_last_non_zero(live_birds_start[i])
#             last_age = get_last_age_before_padding(ages[i])

#             if placement_id not in placement_loss_data:
#                 placement_loss_data[placement_id] = []

#             placement_loss_data[placement_id].append({
#                 'total_predicted_loss': total_predicted_loss,
#                 'total_actual_loss': total_actual_loss,
#                 'live_birds': live_birds,
#                 'last_age': last_age,
#                 'percentage_predicted_loss': total_predicted_loss / live_birds * 100 if live_birds != 0 else 0,
#                 'percentage_actual_loss': total_actual_loss / live_birds * 100 if live_birds != 0 else 0
#             })

#     return placement_loss_data

import json
import pandas as pd
import streamlit as st

def load_loss_data_from_file(filename='placement_loss_data.json'):
    with open(filename, 'r') as f:
        return json.load(f)

def main():
    st.set_page_config(layout="wide")  # Set the layout to wide mode
    st.title("Model Prediction Loss Dashboard")

    # Load placement_loss_data from file
    placement_loss_data = load_loss_data_from_file()

    # Sidebar to select placement
    placement_ids = list(placement_loss_data.keys())
    selected_placement_id = st.sidebar.selectbox("Select Placement ID", placement_ids)

    # Display results for selected placement
    st.subheader(f"Placement ID: {selected_placement_id}")
    df = pd.DataFrame(placement_loss_data[selected_placement_id])
    df['Difference (%)'] = abs(df['percentage_predicted_loss'] - df['percentage_actual_loss'])
    # df.columns = ['last_age', 'live_birds', 'total_actual_loss', 'total_predicted_loss', 'percentage_actual_loss', 'percentage_predicted_loss', 'Difference (%)']
    df = df[['last_age', 'live_birds', 'total_actual_loss', 'total_predicted_loss', 'percentage_actual_loss', 'percentage_predicted_loss', 'Difference (%)']]
    
    # Display the dataframe with a larger height
    st.write(
        f"<div style='height:1000px;'>{df.to_html(index=False)}</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
