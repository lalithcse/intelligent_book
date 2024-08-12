import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np


class Recommender:
    def __init__(self, books: pd.DataFrame, reviews: pd.DataFrame):
        self.books = books
        self.reviews = reviews
        self.model = NearestNeighbors(metric='cosine', algorithm='brute')

    def fit(self):
        book_ratings = self.reviews.pivot_table(index='book_id', columns='user_id',
                                                values='rating').fillna(0)
        self.book_matrix = book_ratings.values
        self.model.fit(self.book_matrix)

    def recommend(self, book_id: int, n_recommendations: int = 5):
        book_idx = self.books[self.books['id'] == book_id].index[0]
        distances, indices = self.model.kneighbors(
            self.book_matrix[book_idx].reshape(1, -1),
            n_neighbors=n_recommendations + 1)
        recommendations = [self.books.iloc[i]['id'] for i in indices.flatten()]
        recommendations.remove(book_id)
        return recommendations
