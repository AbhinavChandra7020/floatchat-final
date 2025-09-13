import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import random
import os
from dotenv import load_dotenv
import google.generativeai as genai
import json
import re

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="INCOIS ARGO Float System",
    page_icon="🌊",
    layout="wide"
)

# Configure Gemini
@st.cache_resource
def configure_gemini():
    """Initialize Gemini API"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        st.error("GEMINI_API_KEY not found in environment variables!")
        return None
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    return model

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

def create_gemini_prompt(user_query, data_summary):
    """Create a structured prompt for Gemini to understand the query"""
    prompt = f"""
You are an expert oceanographic data analyst working with ARGO float data from the Indian Ocean region. 
Your task is to interpret user queries and return structured JSON responses for data filtering.

DATASET CONTEXT:
- Total ARGO floats: {data_summary['total_floats']}
- Total measurements: {data_summary['total_records']}
- Date range: {data_summary['date_range']}
- Available regions: Indian Ocean, Arabian Sea, Bay of Bengal, Southern Ocean, Equatorial Indian Ocean
- Available measurements: temperature, salinity, pressure, depth, latitude, longitude, date_time

USER QUERY: "{user_query}"

Please analyze this query and return a JSON response with the following structure:
{{
    "query_type": "count|regional|data_filter|temporal|general",
    "intent": "Brief description of what user wants",
    "filters": {{
        "region": "indian_ocean|arabian_sea|bay_of_bengal|southern_ocean|equatorial_indian|null",
        "temperature_range": {{"min": number, "max": number}} or null,
        "depth_range": {{"min": number, "max": number}} or null,
        "salinity_range": {{"min": number, "max": number}} or null,
        "pressure_range": {{"min": number, "max": number}} or null,
        "date_filter": {{"type": "recent|year|range", "value": "value"}} or null,
        "coordinate_bounds": {{"lat_min": number, "lat_max": number, "lon_min": number, "lon_max": number}} or null
    }},
    "sort_by": "temperature|salinity|pressure|depth|date_time|null",
    "sort_order": "asc|desc",
    "limit": number (default 200),
    "response_message": "Natural language response to user explaining what data will be shown"
}}

EXAMPLE INTERPRETATIONS:
- "Show floats in Arabian Sea" → region: "arabian_sea", query_type: "regional"
- "High temperature readings" → temperature_range: {{"min": 25, "max": null}}, query_type: "data_filter"
- "Deep water measurements" → depth_range: {{"min": 500, "max": null}}, query_type: "data_filter"
- "Recent data from 2024" → date_filter: {{"type": "year", "value": "2024"}}, query_type: "temporal"
- "How many floats" → query_type: "count"

Return ONLY the JSON response, no additional text.
"""
    return prompt

def query_gemini(model, user_query, data_summary):
    """Query Gemini for intelligent data filtering"""
    try:
        prompt = create_gemini_prompt(user_query, data_summary)
        response = model.generate_content(prompt)
        
        # Extract JSON from response
        response_text = response.text.strip()
        
        # Try to extract JSON if wrapped in code blocks
        if "```json" in response_text:
            json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
        elif "```" in response_text:
            json_match = re.search(r'```\n(.*?)\n```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
        
        # Parse JSON
        parsed_response = json.loads(response_text)
        return parsed_response
        
    except Exception as e:
        st.error(f"Error querying Gemini: {str(e)}")
        # Fallback response
        return {
            "query_type": "general",
            "intent": "Show general data",
            "filters": {},
            "sort_by": None,
            "sort_order": "desc",
            "limit": 100,
            "response_message": f"Showing general ARGO float data. (Note: AI processing failed, using fallback)"
        }

def apply_gemini_filters(df, gemini_response):
    """Apply filters based on Gemini's interpretation"""
    filtered_df = df.copy()
    region = None
    
    filters = gemini_response.get('filters', {})
    
    # Handle count queries
    if gemini_response.get('query_type') == 'count':
        total_floats = df['id'].nunique()
        return pd.DataFrame({'total_floats': [total_floats]}), None
    
    # Regional filter
    if filters.get('region'):
        region = filters['region']
        region_info = get_region_bounds(region)
        if region_info:
            bounds = region_info['bounds']
            filtered_df = filtered_df[
                (filtered_df['latitude'] >= bounds['lat_min']) & 
                (filtered_df['latitude'] <= bounds['lat_max']) &
                (filtered_df['longitude'] >= bounds['lon_min']) & 
                (filtered_df['longitude'] <= bounds['lon_max'])
            ]
    
    # Temperature filter
    temp_range = filters.get('temperature_range')
    if temp_range:
        if temp_range.get('min') is not None:
            filtered_df = filtered_df[filtered_df['temperature'] >= temp_range['min']]
        if temp_range.get('max') is not None:
            filtered_df = filtered_df[filtered_df['temperature'] <= temp_range['max']]
    
    # Depth filter
    depth_range = filters.get('depth_range')
    if depth_range:
        if depth_range.get('min') is not None:
            filtered_df = filtered_df[filtered_df['depth'] >= depth_range['min']]
        if depth_range.get('max') is not None:
            filtered_df = filtered_df[filtered_df['depth'] <= depth_range['max']]
    
    # Salinity filter
    salinity_range = filters.get('salinity_range')
    if salinity_range:
        if salinity_range.get('min') is not None:
            filtered_df = filtered_df[filtered_df['salinity'] >= salinity_range['min']]
        if salinity_range.get('max') is not None:
            filtered_df = filtered_df[filtered_df['salinity'] <= salinity_range['max']]
    
    # Pressure filter
    pressure_range = filters.get('pressure_range')
    if pressure_range:
        if pressure_range.get('min') is not None:
            filtered_df = filtered_df[filtered_df['pressure'] >= pressure_range['min']]
        if pressure_range.get('max') is not None:
            filtered_df = filtered_df[filtered_df['pressure'] <= pressure_range['max']]
    
    # Date filter
    date_filter = filters.get('date_filter')
    if date_filter:
        if date_filter.get('type') == 'recent':
            days_back = 30
            cutoff_date = datetime.now() - timedelta(days=days_back)
            filtered_df = filtered_df[filtered_df['date_time'] >= cutoff_date]
        elif date_filter.get('type') == 'year':
            year = int(date_filter.get('value', datetime.now().year))
            filtered_df = filtered_df[filtered_df['date_time'].dt.year == year]
    
    # Coordinate bounds filter
    coord_bounds = filters.get('coordinate_bounds')
    if coord_bounds:
        filtered_df = filtered_df[
            (filtered_df['latitude'] >= coord_bounds['lat_min']) & 
            (filtered_df['latitude'] <= coord_bounds['lat_max']) &
            (filtered_df['longitude'] >= coord_bounds['lon_min']) & 
            (filtered_df['longitude'] <= coord_bounds['lon_max'])
        ]
    
    # Sorting
    sort_by = gemini_response.get('sort_by')
    sort_order = gemini_response.get('sort_order', 'desc')
    if sort_by and sort_by in filtered_df.columns:
        ascending = sort_order == 'asc'
        filtered_df = filtered_df.sort_values(sort_by, ascending=ascending)
    
    # Limit results
    limit = gemini_response.get('limit', 200)
    filtered_df = filtered_df.head(limit)
    
    return filtered_df, region

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
                showscale=True,
                colorbar=dict(title="Temperature (°C)")
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
st.title("🌊 INCOIS ARGO Float Data Explorer")
st.markdown("**AI-Powered Oceanographic Data Analysis** - Ask questions in natural language!")

# Initialize Gemini
gemini_model = configure_gemini()
if not gemini_model:
    st.stop()

# Load the dataset
with st.spinner("Loading ARGO float dataset..."):
    argo_data = load_data()

total_floats = argo_data['id'].nunique()
total_records = len(argo_data)
date_range = f"{argo_data['date_time'].min().strftime('%Y-%m-%d')} to {argo_data['date_time'].max().strftime('%Y-%m-%d')}"

data_summary = {
    'total_floats': total_floats,
    'total_records': total_records,
    'date_range': date_range
}

st.success(f"🤖 AI-Enhanced system loaded with {total_floats:,} ARGO floats and {total_records:,} measurements")

# Quick action buttons
st.subheader("Quick Queries")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("**Regional Queries:**")
    if st.button("Indian Ocean Floats", use_container_width=True):
        st.session_state.query = "Show me all ARGO floats in the Indian Ocean"
    
    if st.button("Arabian Sea", use_container_width=True):
        st.session_state.query = "Find floats in the Arabian Sea region"
    
    if st.button("Bay of Bengal", use_container_width=True):
        st.session_state.query = "Show data from Bay of Bengal"

with col2:
    st.markdown("**Data Analysis:**")
    if st.button("High Temperature Waters", use_container_width=True):
        st.session_state.query = "Show me areas with high water temperature"
    
    if st.button("Total Float Count", use_container_width=True):
        st.session_state.query = "How many ARGO floats are in the dataset?"
    
    if st.button("Recent Measurements", use_container_width=True):
        st.session_state.query = "Show me the most recent data"

# Add separator
st.markdown("---")

# Enhanced chat interface
st.subheader("AI Chat Interface")
st.markdown("*Ask questions about ARGO floats in natural language - the AI will understand and filter the data accordingly*")

# Example queries
with st.expander("Example Queries", expanded=False):
    st.markdown("""
    **Try these natural language queries:**
    - "Show me warm water areas above 28°C in the Arabian Sea"
    - "Find deep measurements below 1000 meters from 2024"
    - "What's the average salinity in Bay of Bengal?"
    - "Show recent data from the equatorial region"
    - "Find floats with low salinity readings"
    - "Display temperature patterns in Southern Ocean"
    """)

user_query = st.text_input(
    "Ask about ARGO floats:", 
    value=st.session_state.get('query', ''),
    placeholder="e.g., Show me warm water areas in the Indian Ocean with temperature above 25°C"
)

if user_query:
    st.markdown(f"**Your Query:** *{user_query}*")
    
    # Process query with Gemini
    with st.spinner("AI is analyzing your query..."):
        gemini_response = query_gemini(gemini_model, user_query, data_summary)
    
    # Show AI interpretation
    with st.expander("AI Query Interpretation", expanded=False):
        st.json(gemini_response)
    
    # Apply filters based on Gemini's interpretation
    with st.spinner("🔍 Filtering data based on AI analysis..."):
        results, region = apply_gemini_filters(argo_data, gemini_response)
    
    # Show AI response message
    if gemini_response.get('response_message'):
        st.info(f"**AI Response:** {gemini_response['response_message']}")
    
    if not results.empty:
        # Handle count queries differently
        if 'total_floats' in results.columns:
            st.success(f"**Total ARGO floats in dataset: {results.iloc[0]['total_floats']:,}**")
        else:
            # Show results summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Records Found", len(results))
            with col2:
                if 'temperature' in results.columns and results['temperature'].notna().any():
                    avg_temp = results['temperature'].mean()
                    st.metric("Avg Temperature", f"{avg_temp:.2f}°C")
            with col3:
                if 'salinity' in results.columns and results['salinity'].notna().any():
                    avg_salinity = results['salinity'].mean()
                    st.metric("Avg Salinity", f"{avg_salinity:.2f} PSU")
            with col4:
                if 'date_time' in results.columns:
                    latest_date = results['date_time'].max()
                    st.metric("Latest Record", str(latest_date)[:10])
            
            # Show data table
            st.subheader("Filtered Results")
            display_columns = ['id', 'latitude', 'longitude', 'date_time', 'temperature', 'salinity', 'depth']
            display_df = results[display_columns].copy()
            display_df['date_time'] = display_df['date_time'].dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(
                display_df, 
                use_container_width=True,
                height=300
            )
            
            # Create enhanced map
            if 'latitude' in results.columns and 'longitude' in results.columns:
                st.subheader("🗺️ Interactive Map Visualization")
                if region:
                    region_name = get_region_bounds(region)['name']
                    st.success(f"**Focused Region:** {region_name} (highlighted in red)")
                
                fig = create_enhanced_map(results, region)
                if fig:
                    st.plotly_chart(
                        fig, 
                        use_container_width=True,
                        config={
                            'displayModeBar': True,
                            'displaylogo': False,
                            'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
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
            
            # Enhanced analytics
            if len(results) > 1:
                st.subheader("Quick Analytics")
                
                # Temperature distribution
                if 'temperature' in results.columns and results['temperature'].notna().any():
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_temp = px.histogram(
                            results, 
                            x='temperature', 
                            nbins=20,
                            title='Temperature Distribution',
                            labels={'temperature': 'Temperature (°C)', 'count': 'Frequency'}
                        )
                        fig_temp.update_layout(height=300)
                        st.plotly_chart(fig_temp, use_container_width=True)
                    
                    with col2:
                        if 'depth' in results.columns and results['depth'].notna().any():
                            fig_depth = px.scatter(
                                results, 
                                x='temperature', 
                                y='depth',
                                color='salinity',
                                title='Temperature vs Depth Profile',
                                labels={'temperature': 'Temperature (°C)', 'depth': 'Depth (m)'}
                            )
                            fig_depth.update_yaxis(autorange="reversed")  # Depth increases downward
                            fig_depth.update_layout(height=300)
                            st.plotly_chart(fig_depth, use_container_width=True)
            
            # Download option
            csv = results.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name=f"argo_floats_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    else:
        st.warning("No results found for your query. Try rephrasing or broadening your search terms.")

# Clear session state query after use
if 'query' in st.session_state:
    del st.session_state.query

# Enhanced sidebar with AI features
st.sidebar.header("AI-Enhanced Features")
st.sidebar.markdown("""
**Natural Language Processing:**
- Intelligent query interpretation
- Context-aware data filtering  
- Multi-parameter analysis
- Smart region detection

**Query Examples:**
- "Warm waters above 25°C"
- "Deep measurements in Arabian Sea"
- "Recent salinity data from 2024"
- "Temperature patterns near equator"
- "Count floats by region"
""")

st.sidebar.header("Dataset Information")
st.sidebar.markdown(f"""
**Current Dataset:**
- **Total Floats:** {total_floats:,}
- **Measurements:** {total_records:,}
- **Date Range:** {date_range}
- **Regions:** 5 ocean areas
- **Parameters:** Temp, Salinity, Depth, Pressure
""")