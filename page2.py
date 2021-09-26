import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, plot, iplot
from plotly import tools
from PIL import Image
from warnings import filterwarnings
filterwarnings('ignore')


def app():

    modelTraining = st.container()

    with modelTraining:

        st.title(' IPL Analysis and Visualization')

        image = Image.open('IPL-2021-fb.jpg')
        st.image(image, caption='indian premier league')


        #----------------------import the data-----------------------
        deliveries_data = pd.read_csv('IPL Ball-by-Ball 2008-2020.csv')

        match_data = pd.read_csv('IPL Matches 2008-2020.csv')
        print("Data ready for exploration")


        #----------------------preprossesing----------------------
        match_data.isnull().sum()

        #-----------------------total mathces-----------------------
        print('Total Matches Played:',match_data.shape[0])
        print(' \n Venues Played At:',match_data['city'].unique())     
        print(' \n Teams :',match_data['team1'].unique())


        #---------------------Number of matches played in various seasons--------------------
        match_data['Season'] = pd.DatetimeIndex(match_data['date']).year

        match_per_season=match_data.groupby(['Season'])['id'].count().reset_index().rename(columns={'id':'matches'})
        match_per_season.style.background_gradient(cmap='PuBu')


        #--------------------------------------------plot 1------------------------------------------
        colors = ['turquoise',] * 13
        colors[5] = 'crimson'

        fig=px.bar(data_frame=match_per_season,x=match_per_season.Season,y=match_per_season.matches,labels=dict(x="Season",y="Count"),)
        fig.update_layout(title="Number of matches played in different seasons ",
                        titlefont={'size': 26},template='simple_white'     
                        )
        fig.update_traces(marker_line_color='black',
                        marker_line_width=2.5, opacity=1,marker_color=colors)
        #fig.show()

        st.write(fig)

       


        #--------------------Total number of runs scored across seasons :------------------
        season_data=match_data[['id','Season']].merge(deliveries_data, left_on = 'id', right_on = 'id', how = 'left').drop('id', axis = 1)

        #--------------------------------------------------plot 2-----------------------------------------------
        season=season_data.groupby(['Season'])['total_runs'].sum().reset_index()
        p=season.set_index('Season')
        fig = px.line(p, x=p.index, y="total_runs")
        fig.update_layout(title="Total Runs Across the Seasons ",
                        titlefont={'size': 26},template='simple_white'     
                        )
        #fig.show()
        st.write(fig)





        #-----------------------Runs scored per match across seasons :--------------------------------
        runs_per_season=pd.concat([match_per_season,season.iloc[:,1]],axis=1)
        runs_per_season['Runs scored per match']=runs_per_season['total_runs']/runs_per_season['matches']
        runs_per_season.set_index('Season',inplace=True)
        runs_per_season.style.background_gradient(cmap='PuBu',subset=['Runs scored per match'])

        #-------------------------------------------plot 3-------------------------------------------------
        fig = px.line(runs_per_season, x=runs_per_season.index, y="Runs scored per match")
        fig.update_layout(title="Runs scored per match across seasons",
                        titlefont={'size': 26},template='simple_white'     
                        )
        #fig.show()
        st.write(fig)






         #------------------------------Total number of fours in each season :------------------------

        data_4 = match_data['Season'].unique()

        fours_list = []
        for var in data_4:
            new_df = match_data[match_data['Season']==var]
            total_fours = 0
            for i in new_df['id'].values:
                temp_df = deliveries_data[deliveries_data['id']==i]
                fours = temp_df[temp_df['batsman_runs']==4]['batsman_runs'].count()
                total_fours+=fours
            fours_list.append(total_fours)
            

        colors = ['turquoise',] * 14
        colors[5] = 'crimson'
        fig=px.bar(x=data_4, y=fours_list,labels=dict(x="Season",y="Total Fours"),)
        fig.update_layout(title="Total number of Fours in each season",
                        titlefont={'size': 26},template='simple_white'     
                        )
        fig.update_traces(marker_line_color='black',
                        marker_line_width=2.5, opacity=1,marker_color=colors)
        #fig.show()
        st.write(fig)





        #--------------------------- Total number of Six in each season :-------------------------

        data_6 = match_data['Season'].unique()

        # Getting total sixes from each season by check id from matches dataset in deliveries dataset.
        sixes_list = []
        for var in data_6:
            new_df = match_data[match_data['Season']==var]
            total_sixes = 0
            for i in new_df['id'].values:
                temp_df = deliveries_data[deliveries_data['id']==i]
                sixes = temp_df[temp_df['batsman_runs']==6]['batsman_runs'].count()
                total_sixes+=sixes
            sixes_list.append(total_sixes)
            
        colors = ['turquoise',] * 14
        colors[-4] = 'crimson'
        fig=px.bar(x=data_4, y=sixes_list,labels=dict(x="Season",y="Total Sixes"),)
        fig.update_layout(title="Total number of Sixes in each season",
                        titlefont={'size': 26},template='simple_white'     
                        )
        fig.update_traces(marker_line_color='black',
                        marker_line_width=2.5, opacity=1,marker_color=colors)
        #fig.show()
        st.write(fig)







        #--------------------------------------------plot 4-----------------------------------------------

        #--------------------------Number of tosses won by teams :----------------------
        toss=match_data['toss_winner'].value_counts()
        colors = ['turquoise',] * 15
        colors[0] = 'crimson'
        fig=px.bar( y=toss,x=toss.index,labels=dict(x="Season",y="Count"),)
        fig.update_layout(title="No. of tosses won by each team",
                        titlefont={'size': 26},template='simple_white'     
                        )
        fig.update_traces(marker_line_color='black',
                        marker_line_width=2.5, opacity=1,marker_color=colors)
        #fig.show()
        st.write(fig)


        #---------------------Decision made after winning the toss :-----------------

        temp_series = match_data.toss_decision.value_counts()
        labels = (np.array(temp_series.index))
        values = (np.array((temp_series / temp_series.sum())*100))
        colors = ['turquoise', 'crimson']
        fig = go.Figure(data=[go.Pie(labels=labels,
                                    values=values,hole=.3)])
        fig.update_traces(hoverinfo='label+percent', textinfo='label+percent', textfont_size=20,
                        marker=dict(colors=colors, line=dict(color='#000000', width=3)))
        fig.update_layout(title="Toss decision percentage",
                        titlefont={'size': 30},
                        )
        #fig.show()
        st.write(fig)


        #----------------------------Winning toss implies winning game ?---------------------------

        match_data['toss_win_game_win'] = np.where((match_data.toss_winner == match_data.winner),'Yes','No')

        labels =["Yes",'No']
        values = match_data['toss_win_game_win'].value_counts()
        colors = ['turquoise', 'crimson']
        fig = go.Figure(data=[go.Pie(labels=labels,
                                    values=values,hole=.3)])
        fig.update_traces(hoverinfo='label+percent', textinfo='label+percent', textfont_size=20,
                        marker=dict(colors=colors, line=dict(color='#000000', width=3)))
        fig.update_layout(title="Winning toss implies winning macthes?",
                        titlefont={'size': 30},
                        )
        #fig.show()
        st.write(fig)




        #---------------Number of times team have won the tournament :-----------

        winning_teams = match_data[['Season','winner']]

        #dictionaries to get winners to each season
        winners_team = {}
        for i in sorted(winning_teams.Season.unique()):
            winners_team[i] = winning_teams[winning_teams.Season == i]['winner'].tail(1).values[0]
            
        winners_of_IPL = pd.Series(winners_team)
        winners_of_IPL = pd.DataFrame(winners_of_IPL, columns=['team'])

        
        colors = ['turquoise',] * 6
        colors[0] = 'crimson'
        fig=px.bar( y=winners_of_IPL['team'].value_counts(),x=winners_of_IPL['team'].value_counts().index,labels=dict(x="Team Name",y="Count"),)
        fig.update_layout(title="Winners of IPL",
                        titlefont={'size': 26},template='simple_white'     
                        )
        fig.update_traces(marker_line_color='black',
                        marker_line_width=2.5, opacity=1,marker_color=colors)
        #fig.show()
        st.write(fig)








        #----------------Top 10 run scorer of all time :----------------------
        runs=deliveries_data.groupby(['batsman'])['batsman_runs'].sum().reset_index()
        runs.columns=['Batsman','runs']
        y=runs.sort_values(by='runs',ascending=False).head(10).reset_index().drop('index',axis=1)
        y.style.background_gradient(cmap='PuBu')


        #---------------------------------------------plot 5----------------------------------------------------

        colors = ['turquoise',] * 13
        colors[0] = 'crimson'
        fig=px.bar(x=y['Batsman'],y=y['runs'],labels=dict(x="Player",y="Total Runs"),)
        fig.update_layout(title="Top 10 leading run-scrorer",
                        titlefont={'size': 26},template='simple_white'     
                        )
        fig.update_traces(marker_line_color='black',
                        marker_line_width=2.5, opacity=1,marker_color=colors)
        #fig.show()
        st.write(fig)





        #---------------------------------------------plot 6----------------------------------------------

        #------------------------------------Highest wicket-taker :----------------------------------------
        deliveries_data['dismissal_kind'].unique()
        dismissal_kinds = ['caught', 'bowled', 'lbw', 'caught and bowled',
            'stumped', 'hit wicket']
        hwt=deliveries_data[deliveries_data["dismissal_kind"].isin(dismissal_kinds)]
        bo=hwt['bowler'].value_counts()

        colors = ['turquoise',] * 13
        colors[0] = 'crimson'
        fig=px.bar(x=bo[:10].index,y=bo[:10],labels=dict(x="Bowler",y="Total Wickets"),)
        fig.update_layout(title="Leading wicket-takers",
                        titlefont={'size': 26},template='simple_white'     
                        )
        fig.update_traces(marker_line_color='black',
                        marker_line_width=2.5, opacity=1,marker_color=colors)
        #fig.show()
        st.write(fig)




        #------------------------------Most number of 4's :--------------------------------------
        balls_played=deliveries_data.groupby(['batsman'])['ball'].count().reset_index()
        balls_played=balls_played.merge(runs,left_on='batsman',right_on='Batsman',how='outer')
        four=deliveries_data[deliveries_data['batsman_runs']==4]
        runs_4=four.groupby('batsman')['batsman_runs'].count().reset_index()
        runs_4.columns=['Batsman','4s']
        runs_4.sort_values(by='4s',ascending=False).head(10).reset_index().drop('index',axis=1).style.background_gradient(cmap='PuBu')

        six_title = '<p style="font-family:sans-serif; color:white; font-size: 25px;">The Players Who Scored The Most Number Of 4\'s</p>'
        st.markdown(six_title,unsafe_allow_html=True)
        st.write(runs_4)



        #------------------------------Most number of 6's :--------------------------------------
        six=deliveries_data.groupby('batsman')['batsman_runs'].agg(lambda x: (x==6).sum()).reset_index()
        six.columns=['Batsman','6s']
        six.sort_values(by='6s',ascending=False).head(10).reset_index().drop('index',axis=1).style.background_gradient(cmap='PuBu')
        
        six_title = '<p style="font-family:sans-serif; color:white; font-size: 25px;">The Players Who Scored The Most Number Of 6\'s</p>'
        st.markdown(six_title,unsafe_allow_html=True)
        st.write(six)
        





        #----------------------------------------------plot 7----------------------------------------

        #-----------------------------------Man of the Match award :----------------------------
        colors = ['turquoise',] * 11
        colors[0] = 'crimson'
        fig=px.bar(x=match_data.player_of_match.value_counts()[:10].index,y=match_data.player_of_match.value_counts()[:10],labels=dict(x="Players",y="Count"),)
        fig.update_layout(title="Top 10 Man Of the Match award",
                        titlefont={'size': 26},template='simple_white'     
                        )
        fig.update_traces(marker_line_color='black',
                        marker_line_width=2.5, opacity=1,marker_color=colors)
        #fig.show()
        st.write(fig)
