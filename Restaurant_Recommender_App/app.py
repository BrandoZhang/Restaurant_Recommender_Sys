
import json
import logging
from flask import Flask, request, Blueprint

from Restaurant_Recommender_App.engine import RecommendationEngine


main = Blueprint('main', __name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@main.route("/train_model", methods=["GET"])
def train_model():
    logger.info("**************************** Train start!! ****************************")
    recommendation_engine.train_model()
    logger.info("**************************** Model built!! ****************************")
    return json.dumps({"status": 1})


@main.route("/test_model", methods=["GET"])
def test_model():
    """ Test the model and return its MSE.
        Return format: {    "MSE": (double) Square Error    }
    """
    logger.info("**************************** Test start!! ****************************")
    MSE = recommendation_engine.test_model()
    rv = {"MSE": MSE}
    logger.info("**************************** Model MSE: {} ****************************".format(MSE))
    return json.dumps(rv)


@main.route("/random_user", methods=["GET"])
def random_user():
    """ Return an user_id in test dataset randomly.
        Return format: {    "user_id": user_id  }
    """
    logger.info("*********************** Random user_id start!! ***********************")
    user_id = recommendation_engine.random_test_user()
    rv = {"user_id": user_id}
    logger.info("*********************** Sent user_id: {}!! ***********************".format(user_id))
    return json.dumps(rv)


@main.route("/<int:user_id>/prev_ratings", methods=["GET"])
def get_user_prev_rating(user_id):
    """Return format: {{ "user_id": user_id1,
                        "restaurant_id": restaurant_id1,
                        "stars": stars1
                        },
                        { "user_id": user_id2,
                        "restaurant_id": restaurant_id2,
                        "stars": stars2
                        }}
    """
    logger.info("******************** Searching previous ratings!! ********************")
    user_prev_ratings = recommendation_engine.get_user_prev_rating(user_id)
    logger.info("******************** User previous ratings sent!! ********************")
    return json.dumps(user_prev_ratings)


@main.route("/<int:user_id>/test_recommend", methods=["GET"])
def test_recommend(user_id):
    """Return format: { "restaurant_id": restaurant_id,
                        "real_stars": real_stars,
                        "predicted_stars": predicted_stars
                        }
    """
    logger.info("******************** Test recommend starts!! ********************")
    rv = recommendation_engine.test_recommend(user_id)
    logger.info("******************** Test recommend sends!! ********************")
    return json.dumps(rv)


@main.route("/<int:user_id>/ratings/top/<int:count>", methods=["GET"])
def top_ratings(user_id, count):
    logger.debug("User %s TOP ratings requested", user_id)
    top_ratings = recommendation_engine.get_top_ratings(user_id, count)
    return json.dumps(top_ratings)


@main.route("/<int:user_id>/ratings/<int:restaurant_id>", methods=["GET"])
def restaurant_ratings(user_id, restaurant_id):
    logger.debug("User %s rating requested for restaurant %s", user_id, restaurant_id)
    ratings = recommendation_engine.get_ratings_for_restaurant_ids(user_id, [restaurant_id])
    return json.dumps(ratings)


@main.route("/<int:user_id>/ratings", methods=["POST"])
def add_ratings(user_id):
    # get the ratings from the Flask POST request object
    ratings_list = request.form.keys()[0].strip().split("\n")
    ratings_list = map(lambda x: x.split(","), ratings_list)
    # create a list with the format required by the engine (user_id, restaurant_id, rating)
    ratings = map(lambda x: (user_id, int(x[0]), float(x[1])), ratings_list)
    # add them to the model using then engine API
    recommendation_engine.add_ratings(ratings)

    return json.dumps(ratings)


def create_app(spark_context, dataset_path, train_ratio):
    global recommendation_engine

    recommendation_engine = RecommendationEngine(spark_context, dataset_path, train_ratio)

    app = Flask(__name__)
    app.register_blueprint(main)
    return app
