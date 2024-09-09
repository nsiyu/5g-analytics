import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from faker import Faker
import time

# Set page config
st.set_page_config(page_title="5G Analytics Dashboard", layout="wide")

# Load custom CSS
with open('.streamlit/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Initialize Faker for generating mock data
fake = Faker()

# Function to generate mock 5G network data
def generate_mock_data(num_records=100):
    data = {
        'timestamp': [fake.date_time_this_year() for _ in range(num_records)],
        'cell_id': [fake.unique.random_int(min=1000, max=9999) for _ in range(num_records)],
        'signal_strength': [np.random.uniform(-120, -70) for _ in range(num_records)],
        'throughput': [np.random.uniform(100, 1000) for _ in range(num_records)],
        'latency': [np.random.uniform(1, 20) for _ in range(num_records)],
        'connected_devices': [np.random.randint(1, 100) for _ in range(num_records)],
        'packet_loss': [np.random.uniform(0, 5) for _ in range(num_records)]
    }
    return pd.DataFrame(data)

# Function to update data (simulating real-time updates)
@st.cache_data(ttl=5)
def update_data():
    return generate_mock_data()

# Main function to run the Streamlit app
def main():
    # Top navigation bar
    st.markdown('<div class="top-nav">5G Network Analytics Dashboard</div>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("", ["Dashboard", "Raw Data"])

    # Get updated data
    df = update_data()

    # Sidebar for filtering and parameter selection
    st.sidebar.header("Filters and Parameters")
    
    # Date range filter
    date_range = st.sidebar.date_input("Select Date Range", [df['timestamp'].min(), df['timestamp'].max()])
    
    # Cell ID multiselect
    cell_ids = st.sidebar.multiselect("Select Cell IDs", options=sorted(df['cell_id'].unique()))

    # Apply filters
    mask = (df['timestamp'].dt.date >= date_range[0]) & (df['timestamp'].dt.date <= date_range[1])
    if cell_ids:
        mask &= df['cell_id'].isin(cell_ids)
    filtered_df = df[mask]

    if page == "Dashboard":
        # Display key metrics
        st.header("Key Performance Indicators")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📶 Avg Signal Strength", f"{filtered_df['signal_strength'].mean():.2f} dBm")
        with col2:
            st.metric("🚀 Avg Throughput", f"{filtered_df['throughput'].mean():.2f} Mbps")
        with col3:
            st.metric("⏱️ Avg Latency", f"{filtered_df['latency'].mean():.2f} ms")
        with col4:
            st.metric("📱 Avg Connected Devices", f"{filtered_df['connected_devices'].mean():.0f}")

        # Visualizations
        st.header("Network Performance Visualizations")

        # Expandable sections for visualizations
        with st.expander("Signal Strength Analysis", expanded=True):
            fig_signal = px.line(filtered_df, x='timestamp', y='signal_strength', color='cell_id',
                                 title='Signal Strength over Time')
            st.plotly_chart(fig_signal, use_container_width=True)

        with st.expander("Throughput Analysis", expanded=False):
            avg_throughput = filtered_df.groupby('cell_id')['throughput'].mean().reset_index()
            fig_throughput = px.bar(avg_throughput, x='cell_id', y='throughput',
                                    title='Average Throughput by Cell ID')
            st.plotly_chart(fig_throughput, use_container_width=True)

        with st.expander("Correlation Analysis", expanded=False):
            correlation_matrix = filtered_df[['signal_strength', 'throughput', 'latency', 'connected_devices', 'packet_loss']].corr()
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='Viridis'))
            fig_heatmap.update_layout(title='Correlation Heatmap of Network Metrics')
            st.plotly_chart(fig_heatmap, use_container_width=True)

    else:  # Raw Data page
        st.title("Raw 5G Network Data")
        st.dataframe(filtered_df)

    # Error handling and data validation
    if filtered_df.empty:
        st.error("No data available for the selected filters. Please adjust your selection.")
    
    # Add a placeholder for real-time updates
    placeholder = st.empty()
    
    # Simulating real-time updates
    while True:
        time.sleep(5)  # Update every 5 seconds
        df = update_data()
        placeholder.text(f"Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
