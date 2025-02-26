import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title('NBA Player Stats Explorer')

st.markdown("""
This app performs simple webscraping of NBA player stats data!
* **Python libraries:** base64, pandas, streamlit
* **Data source:** [Basketball-reference.com](https://www.basketball-reference.com/).
""")

st.sidebar.header('User Input Features')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1950, 2025))))

# Web scraping of NBA player stats
@st.cache_data
def load_data(year):
    url = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_per_game.html"
    try:
        html = pd.read_html(url, header=0, match="Per Game")  # Ensure the correct table is scraped
        df = html[0]
        print("Raw data columns:", df.columns)  # Debug column names
        raw = df.drop(df[df.Age == 'Age'].index)  # Deletes repeating headers in content
        raw = raw.fillna(0)
        if 'Team' not in raw.columns:
            st.error(f"'Team' column not found in the data. Columns available: {raw.columns}")
            return pd.DataFrame()  # Return an empty DataFrame if the column is missing
        # Ensure the 'Team' column contains only strings
        raw['Team'] = raw['Team'].astype(str)
        # Ensure the 'Awards' column contains only strings
        if 'Awards' in raw.columns:
            raw['Awards'] = raw['Awards'].astype(str)
        playerstats = raw.drop(['Rk'], axis=1)
        return playerstats
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if scraping fails

playerstats = load_data(selected_year)

# Check if data is loaded correctly
if playerstats.empty:
    st.error("Failed to load data. Please check the website structure or try another year.")
else:
    # Sidebar - Team selection
    if 'Team' in playerstats.columns:
        sorted_unique_team = sorted(playerstats.Team.unique())
        selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team)
    else:
        st.warning("Team data not available. Skipping team filtering.")
        selected_team = []

    # Sidebar - Position selection
    unique_pos = ['C', 'PF', 'SF', 'PG', 'SG']
    selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos)

    # Filtering data
    if selected_team:
        df_selected_team = playerstats[(playerstats.Team.isin(selected_team)) & (playerstats.Pos.isin(selected_pos))]
    else:
        df_selected_team = playerstats[playerstats.Pos.isin(selected_pos)]

    st.header('Display Player Stats of Selected Team(s)')
    st.write('Data Dimension: ' + str(df_selected_team.shape[0]) + ' rows and ' + str(df_selected_team.shape[1]) + ' columns.')
    st.dataframe(df_selected_team)

    # Download NBA player stats data
    def filedownload(df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
        return href

    st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)

    # Heatmap
    if st.button('Intercorrelation Heatmap'):
        st.header('Intercorrelation Matrix Heatmap')
        df_selected_team.to_csv('output.csv', index=False)
        df = pd.read_csv('output.csv')

        # Select only numeric columns for correlation
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            corr = numeric_df.corr()
            mask = np.zeros_like(corr)
            mask[np.triu_indices_from(mask)] = True
            with sns.axes_style("white"):
                fig, ax = plt.subplots(figsize=(7, 5))  # Create a figure explicitly
                sns.heatmap(corr, mask=mask, vmax=1, square=True, ax=ax)
            st.pyplot(fig)  # Pass the figure to st.pyplot()
        else:
            st.warning("No numeric data available for correlation heatmap.")
