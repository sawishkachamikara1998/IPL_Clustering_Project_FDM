
from contextlib import nullcontext
import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from PIL import Image

def app():

    modelTraining = st.container()

    with modelTraining:

        st.title('Predict With IPL (2008-2020)')

        image = Image.open('IPL_image.jfif')
        st.image(image, caption='indian premier league')

        match = st.text_input("Enter Number of matches : ")
        if not match:
            st.warning('Please input number of matches.')
            st.stop()
        st.success('Successfully insered number of matches.')

        batAverage = st.text_input("Enter Number of bat average : ")
        if not batAverage:
            st.warning('Please input a batting average.') 
            st.stop()
        st.success('Successfully insered batting average.')

        economy_rate = st.text_input("Enter Number of Economy rate : ")
        if not economy_rate:
            st.warning('Please input a econoomic rate.')
            st.stop()
        st.success('Successfully insered economic Rate.')

        butn = st.button("Enter")

        
        balldf=pd.read_csv("IPL Ball-by-Ball 2008-2020.csv")
        matches=pd.read_csv("IPL Matches 2008-2020.csv")
        balldf.drop_duplicates()

        batgroup = balldf.groupby(['batsman'])

        # Create a batting dataframe with a summary statistics for each batsman
        batdf = pd.DataFrame(batgroup['ball'].count()).rename(columns={'ball':'balls_faced'})
        batdf['innings'] = batgroup['id'].nunique()
        batdf['runs'] = batgroup['batsman_runs'].sum()
        batdf['4s'] = balldf[balldf['batsman_runs'] == 4].groupby('batsman')['batsman_runs'].count()
        batdf['4s'].fillna(0,inplace=True)
        batdf['6s'] = balldf[balldf['batsman_runs'] == 6].groupby('batsman')['batsman_runs'].count()
        batdf['6s'].fillna(0,inplace=True)

        # Batting average = total run_scored / innings
        batdf['bat_average'] = round(batdf['runs']/batdf['innings'],2)

        # Strike Rate = (Runs Scored / Balls faced) * 100
        batdf['bat_strike'] = round(batdf['runs']/batdf['balls_faced']*100,2)

        # display bowler details
        bowlgroup = balldf.groupby(['bowler'])

        # Create a bowling dataframe (bowldf) with a summary statistics for each batsman
        bowldf = pd.DataFrame(bowlgroup['ball'].count()).rename(columns={'ball':'balls_bowled'})


        # Get no. of wickets taken by each bowler
        bwl_wkts = balldf[balldf['dismissal_kind'].isin(['caught','bowled', 'lbw','stumped', 'caught and bowled', 'hit wicket'])]
        bowldf['wickets'] = bwl_wkts.groupby(['bowler'])['ball'].count()
        bowldf['wickets'].fillna(0,inplace=True)
        
        # total number of overs for each bowler
        overs = pd.DataFrame(balldf.groupby(['bowler','id'])['over'].nunique())
        bowldf['overs'] = overs.groupby(['bowler'])['over'].sum()    

        # Calculate the runs conceded (total number of runs for each bowler)
        bowldf['runs_conceded'] = balldf.groupby('bowler')['batsman_runs'].sum()
        bowldf['runs_conceded'] = bowldf['runs_conceded'].fillna(0) 

        # Add the runs conceded through wide and noball / extra runs for each bowler
        bowldf['runs_conceded'] = bowldf['runs_conceded'].add(balldf[balldf['extras_type'].isin(['wides','noballs'])].groupby('bowler')['extra_runs'].sum(),fill_value=0)

        # Note - roughly apprx to overs.  Should be runs_conceded/overs.balls
        bowldf['bowler economic rate'] = round(bowldf['runs_conceded']/bowldf['overs'],2)

        all_players_dict = {}

        # Add each player to the final all players list
        def update_player_with_match(player_name, id):
            if player_name in all_players_dict:
                all_players_dict[player_name].add(id)
            else:
                all_players_dict[player_name] = {id}

        # Consider players listed as batsman, non striker or bowler
        def update_player_list(x):
            update_player_with_match(x['batsman'],x['id'])
            update_player_with_match(x['non_striker'],x['id'])
            update_player_with_match(x['bowler'],x['id'])

        out_temp = balldf.apply(lambda x: update_player_list(x),axis=1)
        all_df = pd.DataFrame({'Players':list(all_players_dict.keys())})
        all_df['matches'] = all_df['Players'].apply(lambda x: len(all_players_dict[x]))
        all_df=all_df.set_index('Players')

        # Combine the batting and bowling dataframes to create a merged players dataframe
        players = pd.merge(all_df,batdf, left_index=True, right_index=True,how='outer')
        players = pd.merge(players,bowldf, left_index=True, right_index=True,how='outer')
        players.fillna(0,inplace=True)

        players = pd.merge(players,matches['player_of_match'].value_counts(), left_index=True, right_index=True,how='left')
        players['player_of_match']  = players[['player_of_match']].fillna(0)

        def elbow_plot(min_k, max_k, k_max_iter):
            # Elbow-curve/sum of squared distances
            sum_squared_distances = []
            k_range = range(min_k, max_k+1)
            for k in k_range:
                kmeans = KMeans(n_clusters=k, max_iter=k_max_iter)
                kmeans.fit(players)

                sum_squared_distances.append(kmeans.inertia_)

            # Plot the score for each value of k
            plt.plot(k_range, sum_squared_distances, 'bx-')
            plt.xlabel('k')
            plt.ylabel('Sum of squared distances')
            plt.title('Elbow Method For Optimal k')
            plt.show()

        # Method to compute the silhouette score for the given input
        def compute_silhouette_score(input_df, min_k, max_k, k_max_iter):
            # silhouette analysis
            k_range = range(min_k, max_k+1)

            for k in k_range :
                # intialise kmeans
                kmeans = KMeans(n_clusters=k, max_iter=k_max_iter)
                kmeans.fit(input_df)
                cluster_labels = kmeans.labels_

                # silhouette score
                silhouette_avg = silhouette_score(input_df, cluster_labels)
                print(f"For k={k}, silhouette score = {silhouette_avg}")

        compute_silhouette_score(players, 2,12,300)
        n_cluster = 5

        kmeans = KMeans(n_clusters = n_cluster, init = 'k-means++',random_state=0)
        identified_clusters = kmeans.fit_predict(players)

        #Make a copy of original dataframe in player 
        Player_Cluster=players.copy()

        #merge the new coloumn with cluster number for the table
        Player_Cluster['clusterNo']= identified_clusters

        kmeans = KMeans(n_clusters=5)
        kmeans.fit(Player_Cluster[["matches", "bat_average","bowler economic rate"]])
        Player_Cluster["clusterNo"] = kmeans.labels_

        # fig = plt.figure(figsize=(12,6))

        # colors = ["blue", "lightgreen", "black", "red","orange"]

        # ax = fig.add_subplot(111, projection='3d')

        # for r in range(0,5):
        #     clustered_Player = Player_Cluster[Player_Cluster["clusterNo"] == r]
        #     #plt.scatter(clustered_Player["runs"], clustered_Player["bat_average"], color=colors[r-1])
            
        #     ax.scatter(clustered_Player['matches'],clustered_Player['bat_average'],clustered_Player['bowler economic rate'],color=colors[r-1])

            
        # plt.title("IPL Players", fontsize=16)
        # plt.xlabel("matches", fontsize=14)
        # plt.ylabel("Bat average", fontsize=14)
        # ax.set_zlabel("blower economic rate", fontsize=14)
            
        # plt.show()

        #add the player coloumn to the dataframe
        Player_Cluster.insert(0, 'name', Player_Cluster.index)

        #Make a copy of original dataframe in player cluster to build classification
        Player_Classify1=Player_Cluster.copy()

        #remove coloumns instead of matches,bat avg, blower economic and cluster no
        Player_Classify1.drop(['name','balls_faced','innings','runs','4s','6s','bat_strike','balls_bowled','wickets'
                            ,'overs','runs_conceded','player_of_match'],axis=1,inplace=True)

        x=Player_Classify1.iloc[:,0:3]
        y=Player_Classify1.iloc[:,3]

        Player_classifier = DecisionTreeClassifier(random_state=0)



    #Split the dataset as training set and testing set. Training set 90% and Testing set 10%
        # x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.10)

        Player_classifier.fit(x,y)

        #this function is returned the predicted cluster, related to the given input by us(inputs: matches, bat_avg, blw_econ_rate)
        def get_new_input_dashboard(matches, bat_avg, blw_econ_rate):
            new_input = pd.DataFrame({'matches':[matches],'bat_average':[bat_avg],'bowler economic rate':[blw_econ_rate]})
            return Player_classifier.predict(new_input)



        #access the value as a integer point

        #this function is display the final output(related data points with each cluster)
        def display_details(cluster_number):
            y = Player_Cluster.loc[Player_Cluster['clusterNo'] == cluster_number]
            z = y[['matches','balls_faced','innings','runs','bat_average','wickets','bowler economic rate','player_of_match']]
            return z


        if butn:
            cluster_no = get_new_input_dashboard(float(match),float(batAverage),float(economy_rate))
            cluster_Num = cluster_no[0]
            st.text(str(cluster_Num))
            disp = display_details(cluster_Num)
            new_title = '<p style="font-family:sans-serif; color:Green; font-size: 28px;margin-left:30px;">According To The Clusters Player Details</p>'
            st.markdown(new_title,unsafe_allow_html=True)
            #st.text(new_title)
            tab = st.table(disp)
      
    