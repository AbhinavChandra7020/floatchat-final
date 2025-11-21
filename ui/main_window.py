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
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import sql

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
    model = genai.GenerativeModel('gemini-flash-latest')
    return model

# Database connection
@st.cache_resource
def get_db_connection():
    """Create PostgreSQL database connection"""
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            port=os.getenv('DB_PORT', 5432)
        )
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        return None

def execute_query(query, params=None):
    """Execute SQL query and return results as DataFrame"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if params:
                cur.execute(query, params)
            else:
                cur.execute(query)
            
            results = cur.fetchall()
            df = pd.DataFrame(results)
            return df
    except Exception as e:
        st.error(f"Query execution failed: {str(e)}")
        return pd.DataFrame()
    finally:
        conn.commit()

def get_database_schema():
    """Get database schema information for Gemini"""
    schema_query = """
    SELECT 
        column_name, 
        data_type, 
        is_nullable
    FROM 
        information_schema.columns
    WHERE 
        table_name = 'argo_profiles'
    ORDER BY 
        ordinal_position;
    """
    
    schema_df = execute_query(schema_query)
    return schema_df

def get_data_summary():
    """Get summary statistics from database"""
    summary_queries = {
        'total_profiles': "SELECT COUNT(*) as count FROM argo_profiles",
        'total_platforms': "SELECT COUNT(DISTINCT platform_number) as count FROM argo_profiles",
        'date_range': """
            SELECT 
                MIN(juld) as min_date, 
                MAX(juld) as max_date 
            FROM argo_profiles
            WHERE juld IS NOT NULL
        """,
        'data_centres': "SELECT DISTINCT data_centre FROM argo_profiles WHERE data_centre IS NOT NULL",
        'platform_types': "SELECT DISTINCT platform_type FROM argo_profiles WHERE platform_type IS NOT NULL"
    }
    
    summary = {}
    
    for key, query in summary_queries.items():
        result = execute_query(query)
        if not result.empty:
            if key == 'total_profiles' or key == 'total_platforms':
                summary[key] = int(result.iloc[0]['count'])
            elif key == 'date_range':
                min_date = result.iloc[0]['min_date']
                max_date = result.iloc[0]['max_date']
                summary[key] = f"{min_date} to {max_date}"
            elif key in ['data_centres', 'platform_types']:
                summary[key] = result.iloc[:, 0].tolist()
    
    return summary

def create_gemini_sql_prompt(user_query, schema_info, data_summary):
    """Create a prompt for Gemini to generate SQL query"""
    
    schema_description = "\n".join([
        f"- {row['column_name']}: {row['data_type']}" 
        for _, row in schema_info.iterrows()
    ])
    
    prompt = f"""
You are an expert SQL query generator for oceanographic ARGO float profile data stored in PostgreSQL.

DATABASE SCHEMA (table name: argo_profiles):
{schema_description}

COLUMN DESCRIPTIONS:
- id: Primary key
- platform_number: ARGO float platform identifier
- cycle_number: Profile cycle number
- direction: Profile direction (A=Ascending, D=Descending)
- data_centre: Data collection centre code
- dc_reference: Data centre reference
- data_state_indicator: Quality state of data
- data_mode: Data mode (R=Real-time, D=Delayed, A=Adjusted)
- platform_type: Type of platform/float
- float_serial_no: Serial number of the float
- firmware_version: Firmware version
- wmo_inst_type: WMO instrument type
- project_name: Project name
- pi_name: Principal Investigator name
- juld: Julian date (main timestamp)
- juld_qc: Quality control for julian date
- juld_location: Julian date of location
- latitude: Latitude coordinate
- longitude: Longitude coordinate
- position_qc: Position quality control
- positioning_system: GPS/positioning system used
- profile_pres_qc: Profile pressure quality control
- profile_temp_qc: Profile temperature quality control
- profile_psal_qc: Profile salinity quality control
- vertical_sampling_scheme: Vertical sampling method
- config_mission_number: Configuration mission number
- description: Profile description
- embedding: Vector embedding (ignore for queries)

DATASET CONTEXT:
- Total profiles: {data_summary.get('total_profiles', 'N/A')}
- Total platforms: {data_summary.get('total_platforms', 'N/A')}
- Date range: {data_summary.get('date_range', 'N/A')}
- Available data centres: {', '.join(data_summary.get('data_centres', [])[:5])}
- Platform types: {', '.join(data_summary.get('platform_types', [])[:5])}

IMPORTANT SQL GUIDELINES:
- Table name is 'argo_profiles'
- Always use proper PostgreSQL syntax
- Use LIMIT clause to prevent overwhelming results (default LIMIT 200)
- For count queries, use COUNT() aggregation
- Date filtering should use 'juld' column (Julian date timestamp)
- Location uses 'latitude' and 'longitude' columns
- Platform identification uses 'platform_number' column
- Use 'data_centre' for filtering by data collection centre
- Use 'platform_type' for filtering by float type
- Quality control columns end with '_qc'
- Use ORDER BY for sorting results
- Always return relevant columns for display
- Do NOT query the 'embedding' column as it's for vector operations only

USER QUERY: "{user_query}"

Please analyze this query and return a JSON response with the following structure:
{{
    "query_type": "count|profile|data_filter|temporal|general|aggregate|platform",
    "intent": "Brief description of what user wants",
    "sql_query": "SELECT ... FROM argo_profiles WHERE ... ORDER BY ... LIMIT ...",
    "response_message": "Natural language response explaining what data will be shown",
    "visualization_type": "map|chart|table|summary",
    "region_focus": null
}}

EXAMPLE INTERPRETATIONS:
- "Show profiles from platform 1901234" → 
  sql_query: "SELECT id, platform_number, cycle_number, latitude, longitude, juld, data_centre FROM argo_profiles WHERE platform_number = '1901234' ORDER BY juld DESC LIMIT 200"
  
- "How many profiles are there?" → 
  sql_query: "SELECT COUNT(*) as total_profiles FROM argo_profiles"
  
- "Show recent profiles from 2024" → 
  sql_query: "SELECT id, platform_number, latitude, longitude, juld, data_centre, platform_type FROM argo_profiles WHERE EXTRACT(YEAR FROM juld) = 2024 ORDER BY juld DESC LIMIT 200"
  
- "Find profiles in Arabian Sea region" → 
  sql_query: "SELECT id, platform_number, latitude, longitude, juld, data_centre FROM argo_profiles WHERE latitude BETWEEN 8 AND 27 AND longitude BETWEEN 60 AND 78 ORDER BY juld DESC LIMIT 200"

- "Count profiles by data centre" → 
  sql_query: "SELECT data_centre, COUNT(*) as profile_count FROM argo_profiles GROUP BY data_centre ORDER BY profile_count DESC"

- "Show profiles with good quality position data" → 
  sql_query: "SELECT id, platform_number, latitude, longitude, juld, position_qc FROM argo_profiles WHERE position_qc = '1' ORDER BY juld DESC LIMIT 200"

- "List all platform types" →
  sql_query: "SELECT DISTINCT platform_type, COUNT(*) as count FROM argo_profiles GROUP BY platform_type ORDER BY count DESC"

- "Recent ascending profiles" →
  sql_query: "SELECT id, platform_number, cycle_number, direction, latitude, longitude, juld FROM argo_profiles WHERE direction = 'A' ORDER BY juld DESC LIMIT 200"

- "Profiles from Indian Ocean" →
  sql_query: "SELECT id, platform_number, latitude, longitude, juld, data_centre FROM argo_profiles WHERE latitude BETWEEN -40 AND 25 AND longitude BETWEEN 20 AND 120 ORDER BY juld DESC LIMIT 200"

Return ONLY the JSON response, no additional text or code blocks.
"""
    return prompt

def query_gemini_for_sql(model, user_query, schema_info, data_summary):
    """Query Gemini to generate SQL query"""
    try:
        prompt = create_gemini_sql_prompt(user_query, schema_info, data_summary)
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
            "sql_query": "SELECT id, platform_number, latitude, longitude, juld, data_centre FROM argo_profiles ORDER BY juld DESC LIMIT 100",
            "response_message": f"Showing general ARGO profile data. (Note: AI processing failed, using fallback)",
            "visualization_type": "table",
            "region_focus": None
        }

def validate_sql_query(sql_query):
    """Basic SQL injection prevention and validation"""
    # Convert to lowercase for checking
    query_lower = sql_query.lower()
    
    # Blocked keywords (SQL injection prevention)
    blocked_keywords = [
        'drop', 'delete', 'truncate', 'insert', 'update',
        'alter', 'create', 'grant', 'revoke', '--', ';--'
    ]
    
    for keyword in blocked_keywords:
        if keyword in query_lower:
            st.error(f"Query contains blocked keyword: {keyword}")
            return False
    
    # Must be a SELECT query
    if not query_lower.strip().startswith('select'):
        st.error("Only SELECT queries are allowed")
        return False
    
    # Must reference argo_profiles table
    if 'argo_profiles' not in query_lower:
        st.error("Query must reference argo_profiles table")
        return False
    
    return True

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
        text = f"Platform: {row.get('platform_number', 'N/A')}<br>"
        if 'cycle_number' in row:
            text += f"Cycle: {row.get('cycle_number', 'N/A')}<br>"
        text += f"Date: {row.get('juld', 'N/A')}<br>"
        text += f"Lat: {row.get('latitude', 'N/A'):.3f}<br>"
        text += f"Lon: {row.get('longitude', 'N/A'):.3f}<br>"
        if 'data_centre' in row:
            text += f"Data Centre: {row.get('data_centre', 'N/A')}<br>"
        if 'platform_type' in row:
            text += f"Platform Type: {row.get('platform_type', 'N/A')}<br>"
        if 'direction' in row:
            text += f"Direction: {row.get('direction', 'N/A')}"
        hover_text.append(text)
    
    # Color points by data centre or platform if available
    if 'data_centre' in df.columns and df['data_centre'].notna().any():
        # Create color mapping for data centres
        unique_centres = df['data_centre'].unique()
        color_map = {centre: idx for idx, centre in enumerate(unique_centres)}
        colors = [color_map[centre] if pd.notna(centre) else -1 for centre in df['data_centre']]
        
        fig.add_trace(go.Scattermapbox(
            lat=df['latitude'],
            lon=df['longitude'],
            mode='markers',
            marker=dict(
                size=8,
                color=colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Data Centre")
            ),
            text=hover_text,
            hoverinfo='text',
            name='ARGO Profiles'
        ))
    else:
        fig.add_trace(go.Scattermapbox(
            lat=df['latitude'],
            lon=df['longitude'],
            mode='markers',
            marker=dict(size=6, color='blue'),
            text=hover_text,
            hoverinfo='text',
            name='ARGO Profiles'
        ))
    
    # Set map layout - FIXED THIS SECTION
    if region:
        region_info = get_region_bounds(region)
        if region_info:  # Check if region_info is not None
            center_lat = region_info['center']['lat']
            center_lon = region_info['center']['lon']
            zoom = region_info['zoom']
        else:
            # Fallback to data-based center if region not found
            center_lat = df['latitude'].mean()
            center_lon = df['longitude'].mean()
            zoom = 2
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

# Main app
st.title("🌊 INCOIS ARGO Float Profile Explorer")
st.markdown("**AI-Powered Oceanographic Data Analysis** - Ask questions in natural language!")

# Initialize Gemini
gemini_model = configure_gemini()
if not gemini_model:
    st.stop()

# Test database connection
conn = get_db_connection()
if not conn:
    st.error("Failed to connect to database. Please check your .env configuration.")
    st.stop()
else:
    st.success("✅ Connected to PostgreSQL database")

# Get database schema and summary
with st.spinner("Loading database schema and summary..."):
    schema_info = get_database_schema()
    data_summary = get_data_summary()

if not schema_info.empty:
    st.success(f"🤖 AI-Enhanced system connected to database with {data_summary.get('total_platforms', 0):,} ARGO platforms and {data_summary.get('total_profiles', 0):,} profiles")
else:
    st.warning("Database schema could not be loaded. Please ensure the 'argo_profiles' table exists.")

# Quick action buttons
st.subheader("Quick Queries")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("**Regional Queries:**")
    if st.button("Indian Ocean Profiles", use_container_width=True):
        st.session_state.query = "Show me profiles from the Indian Ocean"
    
    if st.button("Arabian Sea", use_container_width=True):
        st.session_state.query = "Find profiles in the Arabian Sea region"
    
    if st.button("Bay of Bengal", use_container_width=True):
        st.session_state.query = "Show profiles from Bay of Bengal"

with col2:
    st.markdown("**Data Analysis:**")
    if st.button("Recent Profiles", use_container_width=True):
        st.session_state.query = "Show me the most recent profiles from 2024"
    
    if st.button("Profile Count", use_container_width=True):
        st.session_state.query = "How many profiles are in the database?"
    
    if st.button("Platform Types", use_container_width=True):
        st.session_state.query = "Show me all platform types and their counts"

# Add separator
st.markdown("---")

# Enhanced chat interface
st.subheader("AI Chat Interface")
st.markdown("*Ask questions about ARGO profiles in natural language - the AI will generate SQL queries automatically*")

# Example queries
with st.expander("Example Queries", expanded=False):
    st.markdown("""
    **Try these natural language queries:**
    - "Show profiles from platform 2902746"
    - "Find recent profiles from Arabian Sea"
    - "How many profiles by data centre?"
    - "Show ascending profiles from 2024"
    - "List all profiles with good position quality"
    - "What platform types are available?"
    - "Show profiles from Indian Ocean in last 6 months"
    - "Count profiles by direction"
    """)

user_query = st.text_input(
    "Ask about ARGO profiles:", 
    value=st.session_state.get('query', ''),
    placeholder="e.g., Show me recent profiles from the Arabian Sea"
)

if user_query:
    st.markdown(f"**Your Query:** *{user_query}*")
    
    # Process query with Gemini to generate SQL
    with st.spinner("🤖 AI is generating SQL query..."):
        gemini_response = query_gemini_for_sql(gemini_model, user_query, schema_info, data_summary)
    
    # Show AI interpretation and generated SQL
    with st.expander("AI Query Interpretation & Generated SQL", expanded=True):
        st.json(gemini_response)
        st.code(gemini_response.get('sql_query', ''), language='sql')
    
    # Validate and execute SQL query
    sql_query = gemini_response.get('sql_query', '')
    
    if validate_sql_query(sql_query):
        with st.spinner("🔍 Executing SQL query on database..."):
            results = execute_query(sql_query)
        
        # Show AI response message
        if gemini_response.get('response_message'):
            st.info(f"**AI Response:** {gemini_response['response_message']}")
        
        if not results.empty:
            # Check if this is an aggregate query
            is_aggregate = any(col in results.columns for col in ['count', 'total_profiles', 'profile_count', 'avg', 'sum', 'min', 'max'])
            
            if is_aggregate:
                # Display aggregate results
                st.subheader("📊 Query Results")
                
                if len(results) == 1:
                    # Single result metrics
                    cols = st.columns(len(results.columns))
                    for idx, col in enumerate(results.columns):
                        with cols[idx]:
                            value = results.iloc[0][col]
                            if isinstance(value, (int, float)):
                                st.metric(col.replace('_', ' ').title(), f"{value:,.0f}")
                            else:
                                st.metric(col.replace('_', ' ').title(), value)
                else:
                    # Multiple rows table
                    st.dataframe(results, use_container_width=True, height=400)
            
            else:
                # Show results summary for regular queries
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Records Found", len(results))
                with col2:
                    if 'platform_number' in results.columns:
                        unique_platforms = results['platform_number'].nunique()
                        st.metric("Unique Platforms", unique_platforms)
                with col3:
                    if 'data_centre' in results.columns and results['data_centre'].notna().any():
                        unique_centres = results['data_centre'].nunique()
                        st.metric("Data Centres", unique_centres)
                with col4:
                    if 'juld' in results.columns:
                        latest_date = pd.to_datetime(results['juld']).max()
                        st.metric("Latest Profile", str(latest_date)[:10])
                
                # Show data table
                st.subheader("Filtered Results")
                display_columns = [col for col in ['id', 'platform_number', 'cycle_number', 'latitude', 'longitude', 'juld', 'data_centre', 'platform_type', 'direction'] if col in results.columns]
                display_df = results[display_columns].copy()
                
                if 'juld' in display_df.columns:
                    display_df['juld'] = pd.to_datetime(display_df['juld']).dt.strftime('%Y-%m-%d %H:%M')
                
                st.dataframe(
                    display_df, 
                    use_container_width=True,
                    height=300
                )
                
                # Create enhanced map if geographic data is present
                
        if 'latitude' in results.columns and 'longitude' in results.columns:
            st.subheader("🗺️ Interactive Map Visualization")
            
            region_focus = gemini_response.get('region_focus')
            if region_focus:
                region_info = get_region_bounds(region_focus)
                if region_info:  # Only show success message if region is valid
                    st.success(f"**Focused Region:** {region_info['name']} (highlighted in red)")
            
            fig = create_enhanced_map(results, region_focus)
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
                            'filename': 'argo_profiles_map',
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
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Platform distribution
                        if 'platform_number' in results.columns:
                            platform_counts = results['platform_number'].value_counts().head(10)
                            fig_platforms = px.bar(
                                x=platform_counts.index,
                                y=platform_counts.values,
                                title='Top 10 Platforms by Profile Count',
                                labels={'x': 'Platform Number', 'y': 'Profile Count'}
                            )
                            fig_platforms.update_layout(height=300)
                            st.plotly_chart(fig_platforms, use_container_width=True)
                    
                    with col2:
                        # Data centre distribution
                        if 'data_centre' in results.columns and results['data_centre'].notna().any():
                            centre_counts = results['data_centre'].value_counts()
                            fig_centres = px.pie(
                                values=centre_counts.values,
                                names=centre_counts.index,
                                title='Distribution by Data Centre'
                            )
                            fig_centres.update_layout(height=300)
                            st.plotly_chart(fig_centres, use_container_width=True)
                
                # Download option
                csv = results.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name=f"argo_profiles_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        else:
            st.warning("No results found for your query. Try rephrasing or broadening your search terms.")
    else:
        st.error("❌ SQL query validation failed. This may be due to security restrictions or invalid query structure.")

# Clear session state query after use
if 'query' in st.session_state:
    del st.session_state.query

# Enhanced sidebar with AI features
st.sidebar.header("🤖 AI-Enhanced Features")
st.sidebar.markdown("""
**Natural Language to SQL:**
- Intelligent query interpretation
- Automatic SQL generation
- Security validation
- Multi-parameter analysis
- Smart region detection

**Query Examples:**
- "Profiles from platform X"
- "Recent data from Arabian Sea"
- "Count by data centre"
- "Ascending profiles from 2024"
- "Platform types available"
""")

st.sidebar.header("📊 Dataset Information")
if data_summary:
    st.sidebar.markdown(f"""
    **Current Dataset:**
    - **Total Profiles:** {data_summary.get('total_profiles', 'N/A'):,}
    - **Total Platforms:** {data_summary.get('total_platforms', 'N/A'):,}
    - **Date Range:** {data_summary.get('date_range', 'N/A')}
    - **Data Centres:** {len(data_summary.get('data_centres', []))} centres
    - **Platform Types:** {len(data_summary.get('platform_types', []))} types
    """)


st.sidebar.header("🔍 Available Columns")
if not schema_info.empty:
    with st.sidebar.expander("View Schema"):
        st.dataframe(schema_info[['column_name', 'data_type']], use_container_width=True)