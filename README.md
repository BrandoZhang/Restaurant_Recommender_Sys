# Restaurant Recommender Based on Yelp Dataset and Spark

## Discription

In this project, we design a simple restaurant recommendation system with Matrix Factorization (ALS) and Yelp Dataset. The basic idea is to port the algorithm in Movielens Recommendation to restaurant recommendation problem. This repository is only the backend of this system, and it was tested on AWS EMR with pySpark.

### Catalogue

```bash
.
├── README.md
├── Restaurant_Recommender_App
│   ├── __init__.py
│   ├── app.py					# api
│   ├── engine.py				# class of recommender
│   └── server.py				# server
├── data_preprocess
│   └── fm_data_generator.py	# script for processing data
└── requirements.txt			# backend environment specification
```



## How to deploy

If you start from very beginning, download the dataset from [Yelp](https://www.yelp.com/dataset/download) and convert the review JSON data to CSV file, then upload it to your S3 public bucket.

Upload the `data_preprocess/fm_data_generator.py` to the same directory of your CSV file, and modify the corresponding path settings in the script.

Copy the URL of your `.dat` data file, and modify the data path settings in `Restaurant_Recommender_App/server.py`:

```python
    dataset_path = 's3://yelpdataset-cc2018/fm_data_int.dat'  # change the data path according to your own S3 URL
```

Then, upload the whole repository to your EMR master node, type the following commend to install dependencies:

```bash
$ pip install -r requirements.txt --user
```

To launch server, tap:

```bash
$ cd Restaurant_Recommender_App
$ spark-submit server.py
```



## Reference

[Simple Matrix Factorization example on the Movielens dataset using Pyspark](https://medium.com/@connectwithghosh/simple-matrix-factorization-example-on-the-movielens-dataset-using-pyspark-9b7e3f567536)

[Spark-movie-lens: An on-line movie recommender using Spark, Python Flask, and the MovieLens dataset](https://github.com/jadianes/spark-movie-lens)

