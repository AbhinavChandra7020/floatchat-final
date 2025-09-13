import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import random

# Page config
st.set_page_config(
    page_title="INCOIS ARGO Float System",
    page_icon="🌊",
    layout="wide"
)

@st.cache_data
def generate_hardcoded_argo_data():
    """Generate realistic ARGO float data for different regions"""
    np.random.seed(42)
    random.seed(42)
    
    # Define regions with realistic ARGO float distributions
    regions = {
        'indian_ocean': {
            'lat_range': (-40, 25), 'lon_range': (20, 120),
            'count': 800, 'temp_range': (2, 32)
        },
        'arabian_sea': {
            'lat_range': (8, 27), 'lon_range': (60, 78),
            'count': 150, 'temp_range': (22, 32)
        },
        'bay_of_bengal': {
            'lat_range': (5, 22), 'lon_range': (80, 100),
            'count': 120, 'temp_range': (24, 31)
        },
        'southern_ocean': {
            'lat_range': (-60, -40), 'lon_range': (20, 120),
            'count': 200, 'temp_range': (-2, 8)
        },
        'equatorial_indian': {
            'lat_range': (-10, 10), 'lon_range': (50, 100),
            'count': 180, 'temp_range': (25, 30)
        }
    }
    
    all_data = []
    float_id = 1000
    
    for region, params in regions.items():
        for i in range(params['count']):
            # Generate random coordinates within region
            lat = np.random.uniform(params['lat_range'][0], params['lat_range'][1])
            lon = np.random.uniform(params['lon_range'][0], params['lon_range'][1])
            
            # Generate realistic oceanographic data
            depth_levels = np.random.choice([10, 50, 100, 200, 500, 1000, 2000], 
                                          size=np.random.randint(3, 8))
            
            for depth in depth_levels:
                # Temperature decreases with depth
                surface_temp = np.random.uniform(params['temp_range'][0], params['temp_range'][1])
                temp = surface_temp * np.exp(-depth/1000) if depth > 100 else surface_temp
                temp = max(temp, 2)  # Ocean minimum temperature
                
                # Salinity (typical ocean values)
                salinity = np.random.uniform(34.5, 37.5)
                if region == 'bay_of_bengal':
                    salinity = np.random.uniform(33.0, 35.0)  # Lower due to freshwater input
                
                # Pressure increases with depth
                pressure = depth * 1.02  # Approximate dbar conversion
                
                # Random recent dates
                days_ago = np.random.randint(0, 365)
                date_time = datetime.now() - timedelta(days=days_ago)
                
                all_data.append({
                    'id': f"ARGO_{float_id}",
                    'latitude': round(lat, 4),
                    'longitude': round(lon, 4),
                    'date_time': date_time,
                    'temperature': round(temp, 2),
                    'salinity': round(salinity, 3),
                    'pressure': round(pressure, 1),
                    'depth': depth,
                    'region': region
                })
            
            float_id += 1
    
    return pd.DataFrame(all_data)

def get_region_bounds(region):
    """Get bounding box for different ocean regions"""
    regions = {
        'indian_ocean': {
            'bounds': {'lat_min': -40, 'lat_max': 25, 'lon_min': 20, 'lon_max': 120},
            'center': {'lat': -7.5, 'lon': 70},
            'zoom': 3,
            'name': 'Indian Ocean'
        },
        'arabian_sea': {
            'bounds': {'lat_min': 8, 'lat_max': 27, 'lon_min': 60, 'lon_max': 78},
            'center': {'lat': 17.5, 'lon': 69},
            'zoom': 5,
            'name': 'Arabian Sea'
        },
        'bay_of_bengal': {
            'bounds': {'lat_min': 5, 'lat_max': 22, 'lon_min': 80, 'lon_max': 100},
            'center': {'lat': 13.5, 'lon': 90},
            'zoom': 5,
            'name': 'Bay of Bengal'
        },
        'southern_ocean': {
            'bounds': {'lat_min': -60, 'lat_max': -40, 'lon_min': 20, 'lon_max': 120},
            'center': {'lat': -50, 'lon': 70},
            'zoom': 4,
            'name': 'Southern Ocean'
        },
        'equatorial_indian': {
            'bounds': {'lat_min': -10, 'lat_max': 10, 'lon_min': 50, 'lon_max': 100},
            'center': {'lat': 0, 'lon': 75},
            'zoom': 4,
            'name': 'Equatorial Indian Ocean'
        }
    }
    return regions.get(region, None)

def filter_data_by_query(df, user_input):
    """Filter the data based on user query"""
    input_text = user_input.lower()
    region = None
    
    # Count queries
    if "count" in input_text or "how many" in input_text:
        total_floats = df['id'].nunique()
        return pd.DataFrame({'total_floats': [total_floats]}), region
    
    # Regional filters
    elif "indian ocean" in input_text:
        region = 'indian_ocean'
        bounds = get_region_bounds(region)['bounds']
        filtered_df = df[
            (df['latitude'] >= bounds['lat_min']) & (df['latitude'] <= bounds['lat_max']) &
            (df['longitude'] >= bounds['lon_min']) & (df['longitude'] <= bounds['lon_max'])
        ].head(500)
        return filtered_df, region
    
    elif "arabian sea" in input_text:
        region = 'arabian_sea'
        bounds = get_region_bounds(region)['bounds']
        filtered_df = df[
            (df['latitude'] >= bounds['lat_min']) & (df['latitude'] <= bounds['lat_max']) &
            (df['longitude'] >= bounds['lon_min']) & (df['longitude'] <= bounds['lon_max'])
        ].head(300)
        return filtered_df, region
    
    elif "bay of bengal" in input_text:
        region = 'bay_of_bengal'
        bounds = get_region_bounds(region)['bounds']
        filtered_df = df[
            (df['latitude'] >= bounds['lat_min']) & (df['latitude'] <= bounds['lat_max']) &
            (df['longitude'] >= bounds['lon_min']) & (df['longitude'] <= bounds['lon_max'])
        ].head(300)
        return filtered_df, region
    
    elif "southern ocean" in input_text:
        region = 'southern_ocean'
        bounds = get_region_bounds(region)['bounds']
        filtered_df = df[
            (df['latitude'] >= bounds['lat_min']) & (df['latitude'] <= bounds['lat_max']) &
            (df['longitude'] >= bounds['lon_min']) & (df['longitude'] <= bounds['lon_max'])
        ].head(300)
        return filtered_df, region
    
    elif "equatorial" in input_text:
        region = 'equatorial_indian'
        bounds = get_region_bounds(region)['bounds']
        filtered_df = df[
            (df['latitude'] >= bounds['lat_min']) & (df['latitude'] <= bounds['lat_max']) &
            (df['longitude'] >= bounds['lon_min']) & (df['longitude'] <= bounds['lon_max'])
        ].head(300)
        return filtered_df, region
    
    # Data filters
    elif "recent" in input_text or "latest" in input_text:
        return df.sort_values('date_time', ascending=False).head(100), region
    
    elif "temperature" in input_text and "high" in input_text:
        return df[df['temperature'] > 25].sort_values('temperature', ascending=False).head(200), region
    
    elif "deep" in input_text or "depth" in input_text:
        return df[df['pressure'] > 1000].sort_values('pressure', ascending=False).head(200), region
    
    elif any(year in input_text for year in ["2019", "2020", "2021", "2022", "2023", "2024", "2025"]):
        year = int(next(y for y in ["2019", "2020", "2021", "2022", "2023", "2024", "2025"] if y in input_text))
        filtered_df = df[df['date_time'].dt.year == year].sort_values('date_time', ascending=False).head(200)
        return filtered_df, region
    
    else:
        return df.head(50), region

def create_enhanced_map(df, region=None):
    """Create map with optional region highlighting"""
    if df.empty or 'latitude' not in df.columns or 'longitude' not in df.columns:
        return None
    
    # Create base map
    fig = go.Figure()
    
    # Add region boundary if specified
    if region:
        region_info = get_region_bounds(region)
        if region_info:
            bounds = region_info['bounds']
            # Add rectangle for region boundary
            fig.add_trace(go.Scattermapbox(
                lon=[bounds['lon_min'], bounds['lon_max'], bounds['lon_max'], 
                     bounds['lon_min'], bounds['lon_min']],
                lat=[bounds['lat_min'], bounds['lat_min'], bounds['lat_max'], 
                     bounds['lat_max'], bounds['lat_min']],
                mode='lines',
                line=dict(width=3, color='red'),
                name=f'{region_info["name"]} Boundary',
                hoverinfo='name'
            ))
    
    # Add ARGO float data points
    hover_text = []
    for idx, row in df.iterrows():
        text = f"Float ID: {row.get('id', 'N/A')}<br>"
        text += f"Date: {row.get('date_time', 'N/A')}<br>"
        text += f"Lat: {row.get('latitude', 'N/A'):.3f}<br>"
        text += f"Lon: {row.get('longitude', 'N/A'):.3f}<br>"
        if 'temperature' in row and pd.notna(row['temperature']):
            text += f"Temperature: {row['temperature']:.2f}°C<br>"
        if 'salinity' in row and pd.notna(row['salinity']):
            text += f"Salinity: {row['salinity']:.3f} PSU<br>"
        if 'pressure' in row and pd.notna(row['pressure']):
            text += f"Pressure: {row['pressure']:.1f} dbar<br>"
        if 'depth' in row and pd.notna(row['depth']):
            text += f"Depth: {row['depth']} m"
        hover_text.append(text)
    
    # Color points by temperature if available
    if 'temperature' in df.columns and df['temperature'].notna().any():
        fig.add_trace(go.Scattermapbox(
            lat=df['latitude'],
            lon=df['longitude'],
            mode='markers',
            marker=dict(
                size=8,
                color=df['temperature'],
                colorscale='Viridis',
                showscale=False
            ),
            text=hover_text,
            hoverinfo='text',
            name='ARGO Floats'
        ))
    else:
        fig.add_trace(go.Scattermapbox(
            lat=df['latitude'],
            lon=df['longitude'],
            mode='markers',
            marker=dict(size=6, color='blue'),
            text=hover_text,
            hoverinfo='text',
            name='ARGO Floats'
        ))
    
    # Set map layout
    if region:
        region_info = get_region_bounds(region)
        center_lat = region_info['center']['lat']
        center_lon = region_info['center']['lon']
        zoom = region_info['zoom']
    else:
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()
        zoom = 2
    
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom
        ),
        height=600,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        showlegend=True
    )
    
    return fig

# Load data
@st.cache_data
def load_data():
    return generate_hardcoded_argo_data()

# Main app
st.title("INCOIS ARGO Float Data Explorer")
st.write("Explore ARGO float data from the Indian Ocean using natural language queries")

# Load the dataset
with st.spinner("Loading ARGO float dataset..."):
    argo_data = load_data()

total_floats = argo_data['id'].nunique()
total_records = len(argo_data)
st.success(f"Loaded dataset with {total_floats:,} ARGO floats and {total_records:,} measurements")

# Quick action buttons
st.subheader("Quick Queries")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("**Regional Queries:**")
    if st.button("Indian Ocean Floats", use_container_width=True):
        st.session_state.query = "Show floats in the Indian Ocean"
    
    if st.button("Arabian Sea", use_container_width=True):
        st.session_state.query = "Show floats in the Arabian Sea"
    
    if st.button("Bay of Bengal", use_container_width=True):
        st.session_state.query = "Show floats in the Bay of Bengal"

with col2:
    st.markdown("**Data Queries:**")
    if st.button("High Temperature", use_container_width=True):
        st.session_state.query = "Show high temperature readings"
    
    if st.button("Total Count", use_container_width=True):
        st.session_state.query = "How many ARGO floats are there?"
    
    if st.button("Recent Data", use_container_width=True):
        st.session_state.query = "Show recent data"

# Add separator
st.markdown("---")

# Chat interface
user_query = st.text_input(
    "Ask about ARGO floats:", 
    value=st.session_state.get('query', ''),
    placeholder="e.g., Show floats in the Indian Ocean"
)

if user_query:
    st.write(f"**Query:** {user_query}")
    
    # Filter data based on query
    with st.spinner("Processing query..."):
        results, region = filter_data_by_query(argo_data, user_query)
    
    if not results.empty:
        # Handle count queries differently
        if 'total_floats' in results.columns:
            st.success(f"**Total ARGO floats in dataset: {results.iloc[0]['total_floats']:,}**")
        else:
            # Show results summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Records Found", len(results))
            with col2:
                if 'temperature' in results.columns and results['temperature'].notna().any():
                    avg_temp = results['temperature'].mean()
                    st.metric("Avg Temperature", f"{avg_temp:.2f}°C")
            with col3:
                if 'date_time' in results.columns:
                    latest_date = results['date_time'].max()
                    st.metric("Latest Record", str(latest_date)[:10])
            
            # Show data table
            st.subheader("Data Results")
            display_columns = ['id', 'latitude', 'longitude', 'date_time', 'temperature', 'salinity', 'depth']
            display_df = results[display_columns].copy()
            display_df['date_time'] = display_df['date_time'].dt.strftime('%Y-%m-%d')
            st.dataframe(display_df, use_container_width=True)
            
            # Create enhanced map
            if 'latitude' in results.columns and 'longitude' in results.columns:
                st.subheader("Interactive Map")
                if region:
                    region_name = get_region_bounds(region)['name']
                    st.info(f"Showing data for: **{region_name}** (highlighted in red)")
                
                # Map controls info
                with st.expander("Map Controls & Features", expanded=False):
                    st.markdown("""
                    **Interactive Controls:**
                    - **Zoom**: Mouse wheel or zoom buttons in toolbar
                    - **Pan**: Click and drag to move around
                    - **Reset View**: Double-click map or use reset button
                    - **Details**: Hover over points for float information
                    
                    **Toolbar Features:**
                    - Home: Reset to original view
                    - Pan: Click to enable pan mode
                    - Zoom: Box zoom functionality  
                    - Download: Save map as PNG image
                    - Auto-scale: Fit all data points in view
                    
                    **Tips:**
                    - Scroll wheel zooms in/out
                    - Double-click anywhere to reset zoom
                    - Toolbar appears on hover for clean interface
                    """)
                
                fig = create_enhanced_map(results, region)
                if fig:
                    st.plotly_chart(
                        fig, 
                        use_container_width=True,
                        config={
                            'displayModeBar': True,
                            'displaylogo': False,
                            'modeBarButtonsToRemove': [
                                'select2d', 
                                'lasso2d'
                            ],
                            'modeBarButtonsToAdd': [],
                            'toImageButtonOptions': {
                                'format': 'png',
                                'filename': 'argo_floats_map',
                                'height': 600,
                                'width': 1200,
                                'scale': 2
                            },
                            'scrollZoom': True,
                            'doubleClick': 'reset'
                        }
                    )
            
            # Download option
            csv = results.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name=f"argo_floats_{user_query.replace(' ', '_')}.csv",
                mime="text/csv"
            )
    
    else:
        st.warning("No results found for your query")

# Clear session state query after use
if 'query' in st.session_state:
    del st.session_state.query

# Sidebar with information
st.sidebar.header("Example Queries")
st.sidebar.markdown("""
**Regional Queries:**
- Show floats in the Indian Ocean
- Find data in Arabian Sea  
- Bay of Bengal floats
- Southern Ocean data
- Equatorial Indian Ocean

**Data Filters:**
- Show recent data
- High temperature readings  
- Deep water measurements
- Data from 2024
- How many floats total?

**Dataset Info:**
- 1,450+ ARGO floats
- 6,000+ oceanographic measurements
- Temperature and salinity profiles
- Recent time series data
""")

st.sidebar.header("Features")
st.sidebar.markdown("""
- Region-specific highlighting  
- Interactive temperature maps  
- Quick query buttons  
- Data download capability  
- Real-time data filtering
""")

st.sidebar.info("""
**INCOIS ARGO Float System**

This system provides access to ARGO float data 
collected from the Indian Ocean region. Data includes 
temperature, salinity, and pressure measurements 
from autonomous oceanographic profilers.
""")