from League_of_Legends import data_preprocess, model_development, plot_importance
import streamlit as st
from PIL import Image


st.write("""
# Riot Games: Project D

Good Games Well Played!

""")
#image = Image.open('https://reshape.sport1.de/c/t/dff8b7bb-1ae8-11e6-a9b7-f80f41fc6a62/1200x675.jpeg')

lgbm = dt = rf = logreg = None

st.sidebar.title('🦉 Mert Bozkir')
st.sidebar.header('Choose Game')

cross_val = False
grid_search_cv = False

game = st.sidebar.selectbox(
    'Which game you want to analyze?',
    ('League-of-Legends', 'Team-fight-Tactics', 'Valorant', 'League-of-Runeterra')
)

if 'Valorant' in game or 'Runeterra'in game:
    """## Under development process!"""
else: 
    if game[0] == 'L':
        st.image('League Of Legends Challenger Rank Game-10min,15min/asset.png')

        min = st.sidebar.radio("Which dataset you want to choose?", ["10 min", "15 min"])
        
        st.sidebar.header('Model')
        model = st.sidebar.selectbox(
        'Which model you want to use for League of Legends?',
        ('LightGBM', 'Random_Forest', 'Logistic_Regression', 'Decision_Tree'))
    else:
        st.image('League Of Legends Challenger Rank Game-10min,15min/asset2.png')

        min = st.sidebar.radio("How many classes you want to predict?", ["4-class", "8-class"])
        min = st.sidebar.radio("Which rank you want to choose?", ["Challenger", "Diamond", 'GrandMaster', 'Master', 'Platinum'])

        st.sidebar.header('Model')
        model = st.sidebar.selectbox(
            'Which model you want to use for Team-fight-Tactics',
            ('LightGBM', 'Xgboost')
        )

    cross_val = st.sidebar.checkbox("Cross-Validation")
    
    grid_search_cv = st.sidebar.checkbox("Grid Search Cross-Validation")


model_dict = {'LightGBM': lgbm, 'Decision_Tree': dt, 'Logistic_Regression': logreg, 'Random_Forest': rf}

dataframe = data_preprocess()
features, model = model_development(dataframe, model, cv = cross_val, grid_cv = grid_search_cv)

st.dataframe(dataframe.groupby('blueWins').agg({
    'blue-red_kill_diff' : 'mean',
    'blue-red_death_diff' : 'mean',
    'blue-red_assist_diff' : 'mean',
    'blue-red_tower_diff': 'mean',
    'blue-red_dragon_diff' : 'mean'
}))

st.dataframe(dataframe.groupby('blueWins').agg({
    'blue-red_total_golds_diff' : 'mean',
    'blue-red_current_golds_diff' : 'mean',
    'blue-red_total_level_diff' : 'mean',
    'blue-red_avg_level_diff' : 'mean',
    'blue-red_total_minion_diff' : 'mean',
    'blue-red_jungle_minion_diff' : 'mean'
}))

st.pyplot(plot_importance(model, features, num = 5))
