# Find this section in your code where the map layers are defined and replace it with this code:

# Map showing stores and warehouses with icons
st.subheader("Map Visualization")

# Create the IconLayer for warehouses
warehouse_layer = pdk.Layer(
    "IconLayer",
    data=warehouse_data_for_map,
    get_position=["longitude", "latitude"],
    get_icon=lambda x: {
        "url": "https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/icon-atlas.png",
        "width": 128,
        "height": 128,
        "anchorY": 128,
        "mask": True,
        "x": 0,
        "y": 0
    },
    get_size=20,  # Size of the icon
    get_color=["color_r", "color_g", "color_b"],
    pickable=True,
    opacity=1.0,
)

# Create a text layer for warehouse labels
warehouse_text_layer = pdk.Layer(
    "TextLayer",
    data=warehouse_data_for_map,
    get_position=["longitude", "latitude"],
    get_text="warehouse_id",
    get_size=16,
    get_color=[0, 0, 0],
    get_angle=0,
    get_text_anchor="middle",
    get_alignment_baseline="center",
    pickable=True,
)

# Create layers for stores and connections
store_layer = pdk.Layer(
    "ScatterplotLayer",
    data=store_data_for_map,
    get_position=["longitude", "latitude"],
    get_radius=[100, 300],  # Increased radius for better visibility
    get_fill_color=["color_r", "color_g", "color_b", 200],  # Color based on warehouse assignment
    pickable=True,
    opacity=0.8,
    stroked=True,
    filled=True,
)

line_layer = pdk.Layer(
    "LineLayer",
    data=lines_df,
    get_source_position=["start_lon", "start_lat"],
    get_target_position=["end_lon", "end_lat"],
    get_color=["color_r", "color_g", "color_b", 150],  # Increased opacity, colors match warehouse
    get_width=2,  # Increased line width
    pickable=True,
)

# If IconLayer doesn't work well in Streamlit, use this alternative enhanced ScatterplotLayer for warehouses
alternative_warehouse_layer = pdk.Layer(
    "ScatterplotLayer",
    data=warehouse_data_for_map,
    get_position=["longitude", "latitude"],
    get_radius=1000,  # Large radius for warehouses
    get_fill_color=["color_r", "color_g", "color_b", 250],
    get_line_color=[0, 0, 0, 200],  # Black border
    get_line_width=5,  # Thick border
    pickable=True,
    opacity=0.9,
    stroked=True,
    filled=True,
)

# Create the map with the layers
st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state=pdk.ViewState(
        latitude=np.mean(df["latitude"]),
        longitude=np.mean(df["longitude"]),
        zoom=3,
        pitch=0,
    ),
    # Try IconLayer first, but have alternative ready
    layers=[line_layer, store_layer, alternative_warehouse_layer, warehouse_text_layer],
    tooltip={
        "html": "<b>ID:</b> {store_id or warehouse_id}<br><b>Type:</b> {type}<br><b>Demand:</b> {yearly_demand_lbs} lbs<br><b>Cost:</b> ${transportation_cost}",
        "style": {"background": "white", "color": "black", "font-family": '"Helvetica Neue", Arial', "z-index": "10000"},
    },
))

# Add a legend to explain the visualization
st.markdown("""
### Map Legend
- **Large Circles with Labels**: Warehouses (optimized locations)
- **Small Dots**: Stores (colored by their assigned warehouse)
- **Lines**: Connections between stores and their assigned warehouses
""")
