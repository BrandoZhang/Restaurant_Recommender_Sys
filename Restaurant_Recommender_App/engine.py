import os

import pyspark
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_counts_and_averages(ID_and_ratings_tuple):
    """Given a tuple (restaurant_id, ratings_iterable)
    returns (restaurant_id, (ratings_count, ratings_avg))
    """
    nratings = len(ID_and_ratings_tuple[1])
    return ID_and_ratings_tuple[0], (nratings, float(sum(x for x in ID_and_ratings_tuple[1])) / nratings)


class RecommendationEngine:
    """A restaurant recommendation engine
    """

    def __count_and_average_ratings(self):
        """Updates the restaurants ratings counts from
        the current data self.ratings_RDD
        """
        logger.info("Counting restaurant ratings...")
        restaurant_id_with_ratings_RDD = self.ratings_RDD.map(lambda x: (x[1], x[2])).groupByKey()
        restaurant_id_with_avg_ratings_RDD = restaurant_id_with_ratings_RDD.map(get_counts_and_averages)
        self.restaurant_rating_counts_RDD = restaurant_id_with_avg_ratings_RDD.map(lambda x: (x[0], x[1][0]))

    def train_model(self):
        """Train the ALS model with the current dataset
        """
        logger.info("Training the ALS model...")
        self.model = ALS.train(self.ratings_RDD, self.rank, seed=self.seed,
                               iterations=self.iterations, lambda_=self.regularization_parameter)
        logger.info("ALS model built!")

    def test_model(self):
        """Test the model based on test dataset and return the mean square error (MSE)
        """
        self.predictions = self.model.predictAll(self.test_data).map(lambda r: ((r[0], r[1]), r[2]))
        self.ratesAndPreds = self.test_RDD.map(lambda r: ((r[0], r[1]), r[2])).join(self.predictions)
        MSE = self.ratesAndPreds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
        return MSE

    def get_all_test_users(self):
        """Return a list of all the users'id in test dataset
        """
        self.test_users_id_RDD = self.test_RDD.map(lambda x: x[0])
        self.test_user_id_list = self.test_users_id_RDD.collect()
        return self.test_user_id_list

    def random_test_user(self):
        """Randomly choose a user_id from user_list
        """
        user_id1, user_id2 = self.ratings_RDD.map(lambda x: x[0]).takeSample(False, 2)
        if not self.test_user_id_list:
            self.get_all_test_users()
        if user_id1 in self.test_user_id_list:
            return user_id1
        elif user_id2 in self.test_user_id_list:
            return user_id2
        else:
            return self.random_test_user()

    def get_user_prev_rating(self, user_id):
        """Return the previous ratings of a given user.
        """
        # Get pairs of (user_id, restaurant_id, stars) for user_id rated restaurants
        user_rated_restaurant_RDD = self.ratings_RDD.filter(lambda x: x[0] == user_id).map(
            lambda x: (user_id, x[1], x[2]))
        user_rated_restaurants = user_rated_restaurant_RDD.collect()
        return user_rated_restaurants

    def test_recommend(self, user_id):
        """Return the top recommend restaurant formatted as (restaurant_id, real_stars, predicted_stars).
        """
        if not self.ratesAndPreds:
            self.test_model()
        logger.info("*********************** raratesAndPreds: {} ***********************".
                    format(self.ratesAndPreds.filter(lambda rating: rating[0][0] == user_id
                                                     ).map(lambda x: (x[0][1], x[1][0], x[1][1])).first()))
        return self.ratesAndPreds.filter(lambda rating: rating[0][0] == user_id).map(
            lambda x: (x[0][1], x[1][0], x[1][1])). \
            takeOrdered(1, key=lambda r: -r[2])

    def __predict_ratings(self, user_and_restaurant_RDD):
        """Gets predictions for a given (user_id, restaurant_id) formatted RDD
        Returns: an RDD with format (restaurant_id, restaurant_rating, num_ratings)
        """
        predicted_RDD = self.model.predictAll(user_and_restaurant_RDD)
        predicted_rating_RDD = predicted_RDD.map(lambda x: (x.product, x.rating))
        predicted_rating_title_and_count_RDD = \
            predicted_rating_RDD.join(self.restaurant_titles_RDD).join(self.restaurant_rating_counts_RDD)
        predicted_rating_title_and_count_RDD = \
            predicted_rating_title_and_count_RDD.map(lambda r: (r[1][0][1], r[1][0][0], r[1][1]))

        return predicted_rating_title_and_count_RDD

    def add_ratings(self, ratings):
        """Add additional restaurant ratings in the format (user_id, restaurant_id, rating)
        """
        # Convert ratings to an RDD
        new_ratings_RDD = self.sc.parallelize(ratings)
        # Add new ratings to the existing ones
        self.ratings_RDD = self.ratings_RDD.union(new_ratings_RDD)
        # Re-compute restaurant ratings count
        self.__count_and_average_ratings()
        # Re-train the ALS model with the new ratings
        self.train_model()

        return ratings

    def get_ratings_for_restaurant_ids(self, user_id, restaurant_id):
        """Given a user_id and a list of restaurant_id, predict ratings for them
        """
        requested_restaurants_RDD = self.sc.parallelize(restaurant_id).map(lambda x: (user_id, x))
        # Get predicted ratings
        ratings = self.__predict_ratings(requested_restaurants_RDD).collect()

        return ratings

    def get_top_ratings(self, user_id, recommends_count):
        """Recommends up to recommends_count top unrated restaurants to user_id
        """
        # Get pairs of (user_id, restaurant_id) for user_id unrated restaurants
        user_unrated_restaurant_RDD = self.ratings_RDD.filter(lambda rating: not rating[0] == user_id) \
            .map(lambda x: (user_id, x[1])).distinct()
        # Get predicted ratings
        ratings = self.__predict_ratings(user_unrated_restaurant_RDD).takeOrdered(recommends_count, key=lambda x: -x[1])
        return ratings

    def __init__(self, sc, dataset_path, train_ratio):
        """Init the recommendation engine given a Spark context and a dataset path
        """

        logger.info("Starting up the Recommendation Engine: ")

        self.sc = sc

        # Load ratings data for later use
        logger.info("Loading Ratings data...")

        self.df_RDD = sc.textFile(dataset_path).map(lambda x: x.split("::"))
        self.ratings = self.df_RDD.map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
        self.ratings_RDD, self.test_RDD = self.ratings.randomSplit([train_ratio, 1 - train_ratio])
        self.ratings_RDD.cache()
        self.test_RDD.cache()
        self.test_data = self.test_RDD.map(lambda p: (p[0], p[1]))
        self.test_user_id_list = None
        self.predictions = None
        self.ratesAndPreds = None
        # Load restaurants data for later use
        logger.info("Loading Yelp Restaurant data...")
        self.restaurant_titles_RDD = self.ratings_RDD.map(lambda x: (int(x[1]), x[1])).cache()
        # Pre-calculate restaurants ratings counts
        self.__count_and_average_ratings()

        # Train model preference
        self.rank = 8
        self.seed = 5
        self.iterations = 10
        self.regularization_parameter = 0.1
