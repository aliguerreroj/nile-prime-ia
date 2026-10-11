import numpy as np
import pandas as pd

# Actualizar si hace falta
movies_df = pd.read_csv('./Metadatos Películas.csv')
user_interactions_df = pd.read_csv('./interacciones.csv')


class SimulationEnvironment:
    def __init__(self, movies_df, user_interactions_df):
        self.movies_df = movies_df
        self.user_interactions_df = user_interactions_df
        self.current_user = None
        self.done = False

    def reset(self):
        self.current_user = self.user_interactions_df.sample(1).iloc[0]

    def step(self, action):
        recommended_movie = self.movies_df.iloc[action]
        user_watched = self._simulate_user_watch(recommended_movie)
        # Recompensa 1 si el usuario ve la película, 0 si no
        reward = 1 if user_watched else 0
        self.done = True
        return self._get_state(), reward, self.done

    def _simulate_user_watch(self, recommended_movie):
        user_age_group = self.current_user['Age_Group']
        movie_genre = recommended_movie['Genre']

        likelihood_not_watching_to_end = {
            'youth': {'Horror': 0.3, 'Drama': 0.3},
            'adults': {'Action': 0.2, 'Sci-Fi': 0.2},
            'seniors': {'Comedy': 0.25, 'Romance': 0.25}
        }

        default_probability_watching_to_end = 0.8

        if user_age_group in likelihood_not_watching_to_end:
            probability_not_watching = likelihood_not_watching_to_end[user_age_group].get(movie_genre, 0)
            probability_watching_to_end = 1 - probability_not_watching
        else:
            probability_watching_to_end = default_probability_watching_to_end

        user_watched_to_end = np.random.rand() < probability_watching_to_end
        return user_watched_to_end

# env = SimulationEnvironment(movies_df, user_interactions_df)
# env.reset()
# env.current_user <-- Devuelve un usuario aleatorio
# env.current_user['Age_Group'] <-- Devuelve el grupo de edad del usuario
# env.step(0) <-- Recomendar la película con índice 0
# ...
